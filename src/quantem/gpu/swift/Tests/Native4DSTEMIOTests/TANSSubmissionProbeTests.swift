import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels
import XCTest

final class TANSSubmissionProbeTests: XCTestCase {
  func testIndependentDenseLengthsChecksumsBindingMarkersAndCanaries() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let function = try XCTUnwrap(library.makeFunction(name: "tans_submission_probe"))
    let pipeline = try device.makeComputePipelineState(function: function)
    let arguments = function.makeArgumentEncoder(bufferIndex: 0)
    XCTAssertEqual(arguments.encodedLength, 40)
    func buffer(_ words: [UInt32]) throws -> MTLBuffer {
      try words.withUnsafeBytes {
        try XCTUnwrap(device.makeBuffer(bytes: $0.baseAddress!, length: $0.count))
      }
    }
    let columns = 5
    let ranks: [UInt32] = [3, 4, 1, 0, 2]
    let rankBuffer = try buffer(ranks)
    let sentinel: UInt32 = 0xABCD_FEDC
    let untouched = try buffer([UInt32](repeating: sentinel, count: 16))
    var resources: [MTLBuffer] = []
    var streamWords: [[[UInt32]]] = []
    let table = try XCTUnwrap(device.makeBuffer(length: 2 * arguments.encodedLength))
    for record in 0..<2 {
      var bases: [UInt32] = []
      var lengths = [UInt32](repeating: 0, count: columns * 8)
      var payload: [UInt32] = []
      var streams: [[UInt32]] = []
      for stream in 0..<(columns * 32) {
        if stream % 32 == 0 { bases.append(UInt32(payload.count)) }
        let count = stream % 31 == 0 ? 256 : 1 + stream % 7
        lengths[stream / 4] |= UInt32(count - 1) << ((stream % 4) * 8)
        let words = (0..<count).map {
          UInt32(truncatingIfNeeded: (record + 1) * 1_000_003 + stream * 97 + $0) &* 65_537
        }
        streams.append(words)
        payload += words
      }
      streamWords.append(streams)
      let payloadBuffer = try buffer(payload)
      let offsets = try buffer(bases + lengths)
      resources += [payloadBuffer, offsets]
      arguments.setArgumentBuffer(table, offset: record * arguments.encodedLength)
      arguments.setBuffer(payloadBuffer, offset: 0, index: 0)
      arguments.setBuffer(offsets, offset: 0, index: 1)
      for binding in 2...4 { arguments.setBuffer(untouched, offset: 0, index: binding) }
    }
    for selected: [UInt32] in [[4, 0, 2], []] {
      let selection = try buffer(selected.isEmpty ? [0] : selected)
      for mode: UInt32 in [1, 2] {
        let initial =
          [UInt32](repeating: sentinel, count: 8)
          + [UInt32](repeating: 0, count: 8) + [UInt32](repeating: sentinel, count: 8)
        let output = try buffer(initial)
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.useResources(resources + [untouched], usage: .read)
        encoder.setComputePipelineState(pipeline)
        for (index, value) in [table, rankBuffer, selection, output].enumerated() {
          encoder.setBuffer(value, offset: 0, index: index)
        }
        var parameters: [UInt32] = [UInt32(columns), UInt32(selected.count), 2, 2, mode]
        encoder.setBytes(&parameters, length: 20, index: 4)
        encoder.dispatchThreads(
          MTLSize(width: max(1, selected.count * 32) + 7, height: 3, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed)
        var expected = initial
        for record in 0..<2 {
          let base = (record + 2) * 4
          if mode == 1 {
            expected[base + 3] = 0x51A7_0000 + UInt32(record + 2)
          } else {
            for packet in 0..<32 {
              for column in selected {
                let words = streamWords[record][packet * columns + Int(ranks[Int(column)])]
                expected[base] = words.reduce(expected[base], &+)
                expected[base + 1] += UInt32(words.count)
                expected[base + 2] += 1
              }
            }
          }
        }
        XCTAssertEqual(
          Array(
            UnsafeBufferPointer(
              start: output.contents().assumingMemoryBound(to: UInt32.self), count: 24)), expected)
        XCTAssertEqual(
          Array(
            UnsafeBufferPointer(
              start: untouched.contents().assumingMemoryBound(to: UInt32.self), count: 16)),
          [UInt32](repeating: sentinel, count: 16))
      }
    }
  }
}
