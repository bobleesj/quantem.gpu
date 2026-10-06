import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels
import XCTest

final class MetalTANSDecoderLiteralTests: XCTestCase {
  /// Independent small high-count oracle; never an app raw-resident fallback.
  func testEveryExperimentalTopologyPreservesFullUInt16LiteralCounts() throws {
    try verifyEveryTopology(literal: true)
  }

  func testPairedReductionPreservesEscapesAndSignedSixBitCounts() throws {
    try verifyEveryTopology(literal: false)
  }

  private func verifyEveryTopology(literal: Bool) throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let baselineFunction = try XCTUnwrap(library.makeFunction(name: "tans_detector_batch"))
    let arguments = baselineFunction.makeArgumentEncoder(bufferIndex: 0)
    XCTAssertEqual(arguments.encodedLength, 40)
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
      try values.withUnsafeBytes { bytes in
        try XCTUnwrap(
          device.makeBuffer(
            bytes: bytes.baseAddress!, length: bytes.count,
            options: .storageModeShared))
      }
    }
    func value(_ scan: Int, _ col: Int) -> UInt32 {
      if !literal { return UInt32((scan * 13 + col * 7) & 63) }
      if (scan + col) % 7 == 0 { return 65535 }
      return UInt32((scan * 131 + col * 8191) & 65535)
    }
    let columns = 32
    let scans = 16384
    var payload: [UInt32] = []
    for packet in 0..<32 {
      for col in 0..<columns {
        if literal {
          for pair in 0..<256 {
            let scan = packet * 512 + pair * 2
            payload.append(value(scan, col) | (value(scan + 1, col) << 16))
          }
        } else {
          // State zero, zero transition bits, every symbol is a 12-bit escape.
          // Independently encode the bitstream after the initial 10 state bits.
          var words = [UInt32](repeating: 0, count: 97)
          for pair in 0..<256 {
            let scan = packet * 512 + pair * 2
            let symbol = value(scan, col) | (value(scan + 1, col) << 6)
            let bit = 10 + pair * 12
            let word = bit / 32
            let shift = bit % 32
            words[word] |= symbol << shift
            if shift > 20 { words[word + 1] |= symbol >> (32 - shift) }
          }
          payload += words
        }
      }
    }
    // Dense format stores 32-stream checkpoints plus length-minus-one bytes.
    let wordsPerStream = literal ? 256 : 97
    let lengthByte = UInt32(wordsPerStream - 1)
    let offsets =
      (0..<32).map { UInt32($0 * columns * wordsPerStream) }
      + [UInt32](repeating: lengthByte * 0x0101_0101, count: 32 * 8)
    let encoded = try buffer(payload)
    let seeks = try buffer(offsets)
    let empty = try buffer([UInt32(0)])
    let decoding = try buffer([UInt32](repeating: literal ? 0 : 4095, count: 1024))
    let models = try buffer([UInt8](repeating: literal ? 255 : 0, count: 36864))
    let map = try buffer([Int32](repeating: -1, count: 36864))
    let ranks = try buffer((0..<36864).map(UInt32.init))
    let modelOffsets = try buffer([UInt32(0)])
    let descriptors = try buffer([UInt32(literal ? 255 : 0), 0])
    let counts = try buffer([UInt32(1)])
    let table = try XCTUnwrap(device.makeBuffer(length: 40, options: .storageModeShared))
    var query: [UInt32] = [32, 0, 0, 32, 1]
    for mode in [0, 1, 2, 4, 8, 32, 64, 65, 66, 67, 68, 69, 70] {
      let function: MTLFunction
      if mode == 0 {
        function = baselineFunction
      } else if mode >= 32 {
        var pair = [64, 66, 67, 68, 69, 70].contains(mode)
        var word32 = [65, 66, 67].contains(mode)
        var threads = UInt32([67, 70].contains(mode) ? 512 : 128)
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&pair, type: .bool, index: 2)
        constants.setConstantValue(&word32, type: .bool, index: 3)
        constants.setConstantValue(&threads, type: .uint, index: 4)
        var pairs: UInt32 = mode >= 69 ? 4 : 1
        constants.setConstantValue(&pairs, type: .uint, index: 6)
        function = try library.makeFunction(
          name: mode >= 68 ? "tans_detector_cuda_funnel_batch" : "tans_detector_shared_model_batch",
          constantValues: constants)
      } else {
        var streams = UInt32(mode == 8 ? 1 : mode)
        var coalesced = mode == 8
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&streams, type: .uint, index: 0)
        constants.setConstantValue(&coalesced, type: .bool, index: 1)
        function = try library.makeFunction(
          name: "tans_detector_interleaved_batch", constantValues: constants)
      }
      let pipeline = try device.makeComputePipelineState(function: function)
      for removeHalf in [false, true] {
        // Group uniform negative signs and mark the other lanes inactive for paired reduction.
        let selected = try buffer(
          (0..<32).map {
            UInt32(removeHalf && $0 >= 16 && mode >= 64 && mode != 65 ? UInt32.max : UInt32($0))
          })
        let expected: [UInt32] = (0..<scans).map { scan in
          (removeHalf ? 16..<32 : 0..<32).reduce(UInt32(0)) { $0 + value(scan, $1) }
        }
        let seed: [UInt32] = (0..<scans).map { scan in
          removeHalf ? (0..<32).reduce(UInt32(0)) { $0 + value(scan, $1) } : 0
        }
        let signs = try buffer((0..<32).map { Int32(removeHalf ? ($0 < 16 ? -1 : 0) : 1) })
        let output = try buffer(seed)
        arguments.setArgumentBuffer(table, offset: 0)
        for (index, resource) in [encoded, seeks, empty, empty, output].enumerated() {
          arguments.setBuffer(resource, offset: 0, index: index)
        }
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.useResources([encoded, seeks, empty], usage: .read)
        encoder.useResource(output, usage: [.read, .write])
        for (index, resource) in [table, decoding, models, map, ranks, selected, signs].enumerated()
        {
          encoder.setBuffer(resource, offset: 0, index: index)
        }
        encoder.setBytes(&query, length: 20, index: 7)
        encoder.setBuffer(modelOffsets, offset: 0, index: 8)
        if mode >= 32 {
          for (index, resource) in [descriptors, selected, signs, modelOffsets, counts].enumerated()
          {
            encoder.setBuffer(resource, offset: 0, index: index + 9)
          }
        }
        encoder.dispatchThreadgroups(
          MTLSize(width: 1, height: mode >= 32 ? ([67, 70].contains(mode) ? 2 : 8) : 32, depth: 1),
          threadsPerThreadgroup: MTLSize(
            width: mode >= 32 ? ([67, 70].contains(mode) ? 512 : 128) : 32, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(mode): \(String(describing: command.error))")
        let actual = Array(
          UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: UInt32.self),
            count: scans))
        XCTAssertEqual(actual, expected, "mode \(mode), signed removal \(removeHalf)")
        if literal { XCTAssertGreaterThan(actual.max()!, 65535) }
      }
    }
  }
}
