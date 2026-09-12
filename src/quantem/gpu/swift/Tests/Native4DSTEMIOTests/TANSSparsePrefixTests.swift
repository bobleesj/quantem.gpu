import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Independent tiny encoded fixtures, not a production raw-resident path.
final class TANSSparsePrefixTests: XCTestCase {
  func testExplicitMetal4CompilesTheUnchangedIntegerKernelsWhenAvailable() throws {
    guard #available(macOS 26.0, iOS 26.0, *) else { throw XCTSkip("Requires Metal4") }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    guard device.supportsFamily(.metal4) else { throw XCTSkip("Requires Metal4") }
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let context = try TANSMetal4Query(device: device, library: library, sources: [])
    withExtendedLifetime(context) {}
  }
  func testSequentialPrefixPreservesBoundariesSignsAndOutputCanaries() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let recordFunction = try XCTUnwrap(library.makeFunction(name: "tans_detector_batch"))
    let arguments = recordFunction.makeArgumentEncoder(bufferIndex: 0)
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
      try values.withUnsafeBytes {
        try XCTUnwrap(
          device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared))
      }
    }
    let columns = 35
    let streamCount = columns * 32
    let seed: UInt32 = 0x1357_2468
    let canary: UInt32 = 0xDECA_FBAD
    for scenario in 0..<4 {
      var events: [(position: UInt32, count: UInt32)] = []
      var lengths: [UInt32] = []
      var checkpoints: [UInt32] = []
      var expected = [UInt32](repeating: seed, count: 16384)
      for stream in 0..<streamCount {
        if stream % 32 == 0 { checkpoints.append(UInt32(events.count)) }
        let length = scenario == 3 ? 255 : (stream * 67 + scenario * 19) % 256
        lengths.append(UInt32(length))
        let col = stream % columns
        let sign: UInt32 = col % 3 == 0 ? UInt32.max : 1
        for event in 0..<length {
          let position = UInt32((event * 2 + col) % 512)
          let flagged =
            scenario == 0
            ? false
            : (scenario == 1 ? true : (scenario == 2 ? event > 190 : (events.count % 11 < 5)))
          let count: UInt32 = flagged ? UInt32(2 + (stream * 17 + event * 31) % 254) : 1
          events.append((position, count))
          expected[stream / columns * 512 + Int(position)] &+= count &* sign
        }
      }
      checkpoints.append(UInt32(events.count))
      XCTAssertEqual(checkpoints.count, columns + 1)
      var lengthWords = [UInt32](repeating: 0, count: streamCount / 4)
      for i in lengths.indices { lengthWords[i / 4] |= lengths[i] << ((i % 4) * 8) }
      let sparseOffsets = try buffer(checkpoints + lengthWords)
      var positions = [UInt32](repeating: 0, count: (events.count * 9 + 31) / 32)
      var flags = [UInt32](repeating: 0, count: (events.count + 31) / 32)
      var ranks: [UInt32] = []
      var values: [UInt8] = []
      for (i, event) in events.enumerated() {
        if i % 256 == 0 { ranks.append(UInt32(values.count)) }
        let bit = i * 9
        let word = bit / 32
        let shift = bit % 32
        positions[word] |= event.position << shift
        if shift > 23 { positions[word + 1] |= event.position >> (32 - shift) }
        if event.count != 1 {
          flags[i / 32] |= 1 << (i % 32)
          values.append(UInt8(event.count))
        }
      }
      var valueWords = [UInt32](repeating: 0, count: max(1, (values.count + 3) / 4))
      for i in values.indices { valueWords[i / 4] |= UInt32(values[i]) << ((i % 4) * 8) }
      let header = [
        UInt32(events.count), UInt32(positions.count), UInt32(flags.count), UInt32(ranks.count),
      ]
      let payload = try buffer(header + positions + flags + ranks + valueWords)
      let dummy = try buffer([UInt32(0)])
      let cacheMap = try buffer((0..<columns).map(Int32.init))
      let selected = try buffer((0..<columns).reversed().map(UInt32.init))
      let signs = try buffer((0..<columns).reversed().map { Int32($0 % 3 == 0 ? -1 : 1) })
      for enabled in [false, true] {
        let constants = MTLFunctionConstantValues()
        var enabled = enabled
        constants.setConstantValue(&enabled, type: .bool, index: 22)
        let function = try library.makeFunction(
          name: "tans_detector_sparse_batch", constantValues: constants)
        let pipeline = try device.makeComputePipelineState(function: function)
        let output = try buffer(
          [UInt32](repeating: canary, count: 4) + [UInt32](repeating: seed, count: 16384)
            + [UInt32](repeating: canary, count: 4))
        let table = try XCTUnwrap(
          device.makeBuffer(length: arguments.encodedLength, options: .storageModeShared))
        arguments.setArgumentBuffer(table, offset: 0)
        arguments.setBuffer(dummy, offset: 0, index: 0)
        arguments.setBuffer(dummy, offset: 0, index: 1)
        arguments.setBuffer(payload, offset: 0, index: 2)
        arguments.setBuffer(sparseOffsets, offset: 0, index: 3)
        arguments.setBuffer(output, offset: 16, index: 4)
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.useResources([dummy, payload, sparseOffsets], usage: .read)
        encoder.useResource(output, usage: [.read, .write])
        encoder.setBuffer(table, offset: 0, index: 0)
        encoder.setBuffer(cacheMap, offset: 0, index: 1)
        encoder.setBuffer(selected, offset: 0, index: 2)
        encoder.setBuffer(signs, offset: 0, index: 3)
        var parameters: [UInt32] = [UInt32(columns), UInt32(columns)]
        encoder.setBytes(&parameters, length: 8, index: 4)
        encoder.dispatchThreads(
          MTLSize(width: columns * 32, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        let actual = output.contents().bindMemory(to: UInt32.self, capacity: 16392)
        XCTAssertEqual(
          Array(UnsafeBufferPointer(start: actual + 4, count: 16384)), expected,
          "scenario=\(scenario), prefix=\(enabled)")
        XCTAssertEqual(
          Array(UnsafeBufferPointer(start: actual, count: 4)), [UInt32](repeating: canary, count: 4)
        )
        XCTAssertEqual(
          Array(UnsafeBufferPointer(start: actual + 16388, count: 4)),
          [UInt32](repeating: canary, count: 4))
      }
    }
  }
}
