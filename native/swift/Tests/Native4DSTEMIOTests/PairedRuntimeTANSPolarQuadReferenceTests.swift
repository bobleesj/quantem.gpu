import XCTest

final class PairedRuntimeTANSPolarQuadReferenceTests: XCTestCase {
  func testContiguousQuadWindowMatchesScalarPackingForEveryWidth() {
    for width in 0...32 {
      let valueMask: UInt32 =
        width == 32
        ? .max
        : width == 0 ? 0 : (UInt32(1) << width) - 1
      let values = (0..<512).map { scan in
        (UInt32(scan) &* 2_654_435_761 ^ UInt32(scan >> 2)) & valueMask
      }
      let words = pack(values, width: width)
      let base: UInt32 = 0xffff_fff0

      for lane in 0..<128 {
        let firstScan = lane * 4
        let expected = (firstScan..<(firstScan + 4)).map { scan in
          base &+ scalarValue(words, width: width, scan: scan)
        }
        XCTAssertEqual(
          contiguousQuadValues(words, width: width, firstScan: firstScan, base: base),
          expected,
          "width \(width), lane \(lane)")
      }
    }
  }

  private func pack(_ values: [UInt32], width: Int) -> [UInt32] {
    guard width > 0 else { return [] }
    var words = [UInt32](repeating: 0, count: (values.count * width + 31) / 32)
    for (scan, value) in values.enumerated() {
      for bit in 0..<width where value & (1 << bit) != 0 {
        let position = scan * width + bit
        words[position >> 5] |= 1 << (position & 31)
      }
    }
    return words
  }

  private func scalarValue(_ words: [UInt32], width: Int, scan: Int) -> UInt32 {
    guard width > 0 else { return 0 }
    var value: UInt32 = 0
    for bit in 0..<width {
      let position = scan * width + bit
      value |= ((words[position >> 5] >> (position & 31)) & 1) << bit
    }
    return value
  }

  private func contiguousQuadValues(
    _ words: [UInt32], width: Int, firstScan: Int, base: UInt32
  ) -> [UInt32] {
    guard width > 0 else { return [base, base, base, base] }
    let firstBit = firstScan * width
    let firstWord = firstBit >> 5
    let firstShift = firstBit & 31
    let windowWordCount = (firstShift + 4 * width + 31) >> 5
    let window = Array(words[firstWord..<(firstWord + windowWordCount)])
    let valueMask = width == 32 ? UInt32.max : (1 << width) - 1
    return (0..<4).map { sample in
      let bit = firstShift + sample * width
      let word = bit >> 5
      let shift = bit & 31
      var packed = UInt64(window[word])
      if shift + width > 32 {
        packed |= UInt64(window[word + 1]) << 32
      }
      let value = UInt32(truncatingIfNeeded: packed >> shift) & valueMask
      return base &+ value
    }
  }
}
