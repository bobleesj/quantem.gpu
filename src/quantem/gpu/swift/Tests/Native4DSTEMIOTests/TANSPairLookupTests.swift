import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels
import XCTest

@testable import Metal4DSTEMStreamingIO

final class TANSPairLookupTests: XCTestCase {
  func testGPUEntryTableMatchesIndependentTwoStepReaderAndCanaries() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let function = try XCTUnwrap(library.makeFunction(name: "tans_prepare_pair_lookup"))
    let pipeline = try device.makeComputePipelineState(function: function)
    let codes: [UInt32] = (0..<2048).map { index in
      let bits = [0, 1, 2, 4, 6, 10][index % 6]
      let base = (index * 13) % (1025 - (1 << bits))
      let pair = index % 9 == 0 ? 4095 : (index * 31) % 4095
      return UInt32(base << 16 | bits << 12 | pair)
    }
    let source = try codes.withUnsafeBytes { raw in
      try XCTUnwrap(device.makeBuffer(bytes: raw.baseAddress!, length: raw.count))
    }
    for width in [4, 6] {
      let count = codes.count << width
      let sentinel = UInt32(0xABCD_FEDC)
      let poison = [UInt32](repeating: sentinel, count: count + 16)
      let output = try poison.withUnsafeBytes { raw in
        try XCTUnwrap(device.makeBuffer(bytes: raw.baseAddress!, length: raw.count))
      }
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(source, offset: 0, index: 0)
      encoder.setBuffer(output, offset: 0, index: 1)
      var info = [UInt32(count), UInt32(width)]
      encoder.setBytes(&info, length: 8, index: 2)
      encoder.dispatchThreads(
        MTLSize(width: count + 64, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      XCTAssertEqual(command.status, .completed)
      let actual = output.contents().assumingMemoryBound(to: UInt32.self)
      for index in 0..<count {
        let entry = index >> width
        var reservoir = index % (1 << width)
        var remaining = width
        var state = entry % 1024
        var pairs: [UInt32] = []
        for _ in 0..<2 {
          let code = codes[(entry / 1024) * 1024 + state]
          let count = Int((code >> 12) & 15)
          if count > remaining || code & 4095 == 4095 { break }
          pairs.append(code & 4095)
          state = Int(code >> 16) + reservoir % (1 << count)
          reservoir /= 1 << count
          remaining -= count
        }
        let expected: UInt32 =
          pairs.count == 2
          ? pairs[1] | UInt32(state) << 12
            | UInt32(width - remaining) << 22 | UInt32(1) << 28 : 0
        XCTAssertEqual(actual[index], expected, "width=\(width), entry=\(index)")
      }
      for index in count..<(count + 16) { XCTAssertEqual(actual[index], sentinel) }
      XCTAssertThrowsError(
        try TANSPairLookup.make(
          device: device, queue: queue, library: library, decoding: source,
          bits: width, maximumBytes: UInt64(count * 4 - 1)))
    }
  }
}
