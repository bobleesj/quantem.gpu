import Metal
import Metal4DSTEMKernels
import XCTest

final class CompactH5SIMDDecodeTests: XCTestCase {
  private struct Block {
    let input: [UInt8]
    let expected: [UInt8]
    var error: UInt32 = 0
  }

  func testSIMDDecoderMatchesScalarBytesAndExistingGPUDecoder() throws {
    var blocks: [Block] = []
    for count in stride(from: 4, through: 128, by: 4) {
      let literals = (0..<count).map { UInt8(truncatingIfNeeded: $0 * 73 + count) }
      blocks.append(Block(input: sequence(literals: literals), expected: literals))
    }
    // Periods cross byte, word and SIMD-lane boundaries. Overlapping matches
    // require repeating the pre-match prefix, not reading unwritten output.
    for prefixCount in 1...119 {
      for period in [1, 2, 3, 4, 5, 7, 8, 13, 16, 31, 32, 63, 64, prefixCount]
      where period <= prefixCount {
        let prefix = (0..<prefixCount).map { UInt8(truncatingIfNeeded: $0 * 79 + period) }
        let tail: [UInt8] = [5, 4, 3, 2, 1]
        let match = 128 - prefixCount - tail.count
        var expected = prefix
        appendMatch(&expected, count: match, period: period)
        expected += tail
        let input =
          sequence(literals: prefix, match: match, period: period)
          + sequence(literals: tail)
        blocks.append(Block(input: input, expected: expected))
      }
    }
    for period in 1...3 {
      var expected: [UInt8] = [17, 34, 51]
      var input = sequence(literals: expected, match: 57, period: period)
      appendMatch(&expected, count: 57, period: period)
      expected += [68, 85]
      input += sequence(literals: [68, 85], match: 61, period: 17)
      appendMatch(&expected, count: 61, period: 17)
      expected += [102, 119, 136, 153, 170]
      input += sequence(literals: [102, 119, 136, 153, 170])
      blocks.append(Block(input: input, expected: expected))
    }
    // Exercise every possible active-block tail in the eight-SIMD dispatch.
    for count in 1...9 { try verify(Array(blocks.prefix(count))) }
    try verify(blocks)
  }

  func testSIMDDecoderPreservesMalformedStreamErrors() throws {
    let malformed: [([UInt8], UInt32)] = [
      ([], 2), ([0xf0], 4), ([0x40, 1], 4), ([0x10, 1, 0], 5),
      ([0x10, 1, 0, 0], 6), ([0x10, 1, 2, 0], 6),
      ([0x1f, 1, 1, 0], 7), ([0x1f, 1, 1, 0, 255], 7),
      ([0x1f, 1, 1, 0, 200], 8), ([0x10, 1], 9),
      ([0x10, 1, 1, 0], 9), (sequence(literals: [UInt8](repeating: 5, count: 128)) + [0], 9),
    ]
    try verify(
      malformed.map { Block(input: $0.0, expected: [UInt8](repeating: 0, count: 128), error: $0.1) }
    )
  }

  private func appendMatch(_ bytes: inout [UInt8], count: Int, period: Int) {
    for _ in 0..<count { bytes.append(bytes[bytes.count - period]) }
  }

  private func sequence(literals: [UInt8], match: Int = 0, period: Int = 0) -> [UInt8] {
    var result = [UInt8(min(literals.count, 15) << 4 | min(max(match - 4, 0), 15))]
    if literals.count >= 15 { result += extensionBytes(literals.count - 15) }
    result += literals
    if match > 0 {
      result += [UInt8(period & 255), UInt8(period >> 8)]
      if match >= 19 { result += extensionBytes(match - 19) }
    }
    return result
  }

  private func extensionBytes(_ count: Int) -> [UInt8] {
    [UInt8](repeating: 255, count: count / 255) + [UInt8(count % 255)]
  }

  private func verify(_ blocks: [Block]) throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeCompactH5Library(device: device)
    var compressed: [UInt8] = []
    var records: [UInt32] = []
    var outputBytes = 0
    for block in blocks {
      records += [
        UInt32(compressed.count), UInt32(block.input.count),
        UInt32(outputBytes / 4), UInt32(block.expected.count),
      ]
      compressed += block.input
      outputBytes += block.expected.count
    }
    var parameters = [UInt32(blocks.count), UInt32(compressed.count)]
    compressed += [UInt8](repeating: 0, count: 4 - compressed.count % 4)
    let input = try XCTUnwrap(device.makeBuffer(bytes: compressed, length: compressed.count))
    let chunks = try XCTUnwrap(device.makeBuffer(bytes: records, length: records.count * 4))
    for name in ["compact_h5_lz4_decode", "compact_h5_lz4_decode_simd32"] {
      let pipeline = try device.makeComputePipelineState(
        function: XCTUnwrap(library.makeFunction(name: name)))
      let output = try XCTUnwrap(
        device.makeBuffer(length: outputBytes + 16, options: .storageModeShared))
      let status = try XCTUnwrap(
        device.makeBuffer(length: blocks.count * 4, options: .storageModeShared))
      memset(output.contents(), 0xa5, output.length)
      memset(status.contents(), 0xff, status.length)
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(input, offset: 0, index: 0)
      encoder.setBuffer(chunks, offset: 0, index: 1)
      encoder.setBuffer(output, offset: 0, index: 2)
      encoder.setBuffer(status, offset: 0, index: 3)
      encoder.setBytes(&parameters, length: 8, index: 4)
      let simd = name.hasSuffix("simd32")
      encoder.dispatchThreadgroups(
        MTLSize(width: simd ? (blocks.count + 7) / 8 : blocks.count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: simd ? 256 : 64, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      XCTAssertEqual(command.status, .completed, command.error?.localizedDescription ?? name)
      let statuses = status.contents().bindMemory(to: UInt32.self, capacity: blocks.count)
      let decoded = output.contents().bindMemory(to: UInt8.self, capacity: output.length)
      var offset = 0
      for (index, block) in blocks.enumerated() {
        XCTAssertEqual(statuses[index], block.error, "\(name) block \(index)")
        if block.error == 0 {
          XCTAssertEqual(
            Array(UnsafeBufferPointer(start: decoded + offset, count: block.expected.count)),
            block.expected, "\(name) block \(index)")
        }
        offset += block.expected.count
      }
      XCTAssertEqual(
        Array(UnsafeBufferPointer(start: decoded + outputBytes, count: 16)),
        [UInt8](repeating: 0xa5, count: 16), "Decoder wrote past final output")
    }
  }
}
