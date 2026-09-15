import CryptoKit
import Metal
import Metal4DSTEMKernels
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO
import XCTest

final class MetalPairedRuntimeTANSCodecTests: XCTestCase {
  private struct Fixture {
    let name: String
    let values: [UInt16]
    let mode: UInt8
    let payloadBytes: Int
    let payloadSHA256: String
  }

  func testMetalBytesAndCountsMatchFrozenPairedOracle() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("A physical Metal device is required for paired tANS parity")
    }
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)

    for (dtype, fixtures) in [
      (Metal4DSTEMIntegerDType.uint8, uint8Fixtures()),
      (Metal4DSTEMIntegerDType.uint16, uint16Fixtures()),
    ] {
      let result = try codec.roundTrip(
        streams: fixtures.map(\.values), logicalDtype: dtype)
      XCTAssertEqual(result.decodedStreams, fixtures.map(\.values))
      XCTAssertEqual(result.modes, fixtures.map(\.mode))
      for (index, fixture) in fixtures.enumerated() {
        let lower = Int(result.offsets[index])
        let upper = Int(result.offsets[index + 1])
        XCTAssertEqual(upper - lower, fixture.payloadBytes, fixture.name)
        let digest = SHA256.hash(data: Data(result.payload[lower..<upper]))
          .map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(digest, fixture.payloadSHA256, fixture.name)
      }
    }
  }

  func testInterleavedStatesRoundTripEvenAndOddUInt16Escapes() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("A physical Metal device is required for paired tANS parity")
    }
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)

    for count in [512, 509] {
      let values = interleavedEscapeFixture(count: count)
      let result = try codec.roundTrip(
        streams: [values], logicalDtype: .uint16, useInterleavedStates: true)

      XCTAssertEqual(result.decodedStreams, [values], "count: \(count)")
      XCTAssertEqual(result.modes.count, 1)
      XCTAssertTrue((96..<128).contains(result.modes[0]), "count: \(count)")
      XCTAssertGreaterThanOrEqual(result.payload.count, 3)
      XCTAssertEqual(result.payload[2] & 0x80, 0, "reserved header bit, count: \(count)")
    }
  }

  func testInterleavedStatesRejectReservedHeaderBit() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("A physical Metal device is required for paired tANS parity")
    }
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)

    XCTAssertThrowsError(
      try codec.roundTrip(
        streams: [interleavedEscapeFixture(count: 512)], logicalDtype: .uint16,
        useInterleavedStates: true, corruptInterleavedHeaderForTesting: true))
  }

  private func uint8Fixtures() -> [Fixture] {
    let entropy = (0..<512).map { index in UInt16((index * 17 + index / 7) % 11) }
    return [
      Fixture(
        name: "zero", values: [UInt16](repeating: 0, count: 512), mode: 253,
        payloadBytes: 0,
        payloadSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
      Fixture(
        name: "constant", values: [UInt16](repeating: 255, count: 512), mode: 255,
        payloadBytes: 2,
        payloadSHA256: "ea5dbf9596d187e9500f23e9a680109475341cf4e81f7e043f7d97152c10772f"),
      Fixture(
        name: "entropy", values: entropy, mode: 89, payloadBytes: 255,
        payloadSHA256: "d62c1f09f44fe91d3b9b9f95ad9f87b6c5b8ff11f2f6ff979daca5751c667d5"),
      Fixture(
        name: "six-bit escape",
        values: (0..<512).map { index in UInt16((index * 37 + 11) % 64) },
        mode: 94, payloadBytes: 541,
        payloadSHA256: "47f2b897bcb7394a0f54b631df6a34696865c472542ce2988f157c25683a7b84"),
    ]
  }

  private func interleavedEscapeFixture(count: Int) -> [UInt16] {
    (0..<count).map { index in
      if index == count - 1 { return 65_535 }
      if index == count / 2 { return 32_768 }
      if index == count / 4 { return 256 }
      return UInt16((index * 17 + index / 7) % 11)
    }
  }

  private func uint16Fixtures() -> [Fixture] {
    let entropy = (0..<512).map { index in UInt16((index * 17 + index / 7) % 11) }
    var sparse = [UInt16](repeating: 0, count: 512)
    sparse[13] = 128
    sparse[511] = 1
    var wide = entropy
    wide[3] = 40
    wide[127] = 63
    wide[255] = 256
    wide[383] = 32_768
    wide[511] = 65_535
    var state = UInt32(481_516)
    let literal: [UInt16] = (0..<512).map { _ in
      state = 1_664_525 &* state &+ 1_013_904_223
      return UInt16(truncatingIfNeeded: state >> 8)
    }
    return [
      Fixture(
        name: "constant sentinel", values: [UInt16](repeating: 65_535, count: 512), mode: 255,
        payloadBytes: 2,
        payloadSHA256: "ca2fd00fa001190744c15c317643ab092e7048ce086a243e2be9437c898de1bb"),
      Fixture(
        name: "sparse", values: sparse, mode: 252, payloadBytes: 4,
        payloadSHA256: "8eae0aa67f373fa805517b67e1891898e403dfd33578e8d4b5ed229803097838"),
      Fixture(
        name: "entropy", values: entropy, mode: 89, payloadBytes: 255,
        payloadSHA256: "d62c1f09f44fe91d3b9b9f95ad9f87b6c5b8ff11f2f6ff979daca5751c667d5"),
      Fixture(
        name: "wide escape", values: wide, mode: 89, payloadBytes: 276,
        payloadSHA256: "5c58b3bf2637e37478c3813dce7b285ae3ef9ba66b429459fa0c1834e18a6c77"),
      Fixture(
        name: "literal", values: literal, mode: 254, payloadBytes: 1_024,
        payloadSHA256: "db594deee53f8966db4c1b089ef4edfae85db47b2c7ecf618b0ce07df666fe11"),
    ]
  }
}
