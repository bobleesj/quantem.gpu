import CryptoKit
import Foundation
import Metal
import Metal4DSTEMKernels
import Metal4DSTEMStreamingIO
import XCTest

private struct ANSFixture: Decodable {
  struct Arrays: Decodable {
    var shape: [Int]
    var blockFrames: Int
    var scale: Int
    var payload: [UInt8]
    var offsets: [UInt64]
    var modelIds: [UInt32]
    var contextOffsets: [UInt32]
    var symbols: [UInt16]
    var cumulative: [UInt16]
    var frequencies: [UInt16]
    var literal: [UInt8]

    func admitted(_ dtype: Metal4DSTEMIntegerDType) throws -> MetalANSCountArrays {
      try MetalANSCountArrays(
        shape: shape, blockFrames: blockFrames, scale: scale, logicalDtype: dtype,
        payload: payload, offsets: offsets, modelIDs: modelIds,
        contextOffsets: contextOffsets, symbols: symbols, cumulative: cumulative,
        frequencies: frequencies, literal: literal)
    }
  }
  struct Case: Decodable {
    let logicalDtype: String
    let arrays: Arrays
    let expectedCounts: [[UInt32]]
    let mask: [UInt8]
    let expectedSums: [UInt64]
    var dtype: Metal4DSTEMIntegerDType { logicalDtype == "uint8" ? .uint8 : .uint16 }
  }
  let schema: String
  let cases: [Case]
}

final class MetalANSCountTests: XCTestCase {
  private func fixture() throws -> ANSFixture {
    let url = try XCTUnwrap(
      Bundle.module.url(
        forResource: "ans_counts_v1", withExtension: "json", subdirectory: "Fixtures"))
    let bytes = try Data(contentsOf: url)
    XCTAssertEqual(
      SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(),
      "c0fc5e6a06a2d015b6306788cc93f5f9cce49148e6481b9e7cf6cca74e8ac960")
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let result = try decoder.decode(ANSFixture.self, from: bytes)
    XCTAssertEqual(result.schema, "quantem.gpu.count-ans-native-fixture/v1")
    return result
  }

  private func device() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("A physical Metal device is required for native ANS conformance.")
    }
    return device
  }

  func testTableAdmissionRejectsAmbiguousGeometryAndProbabilityModels() throws {
    let original = try XCTUnwrap(fixture().cases.last).arrays
    XCTAssertNoThrow(try original.admitted(.uint16))
    XCTAssertThrowsError(try original.admitted(.uint32))
    var changed = original
    changed.shape = [Int.max, 2, 2, 3]
    XCTAssertThrowsError(try changed.admitted(.uint16))
    changed = original
    changed.offsets[0] = 1
    XCTAssertThrowsError(try changed.admitted(.uint16))
    changed = original
    changed.modelIds[0] = 3
    XCTAssertThrowsError(try changed.admitted(.uint16))
    changed = original
    changed.contextOffsets[2] = 6
    XCTAssertThrowsError(try changed.admitted(.uint16))
    changed = original
    changed.frequencies[0] = 0
    XCTAssertThrowsError(try changed.admitted(.uint16))
    changed = original
    changed.cumulative[1] = 6
    XCTAssertThrowsError(try changed.admitted(.uint16))
    changed = original
    changed.literal[0] = 2
    XCTAssertThrowsError(try changed.admitted(.uint16))
  }

  func testRawDPAndBoundedMaskSumsMatchIndependentUInt8AndUInt16Oracle() throws {
    let device = try device()
    for record in try fixture().cases {
      let source = try MetalANSResidentSource(
        arrays: record.arrays.admitted(record.dtype), device: device)
      defer { source.releaseResidentStorage() }
      XCTAssertEqual(source.shape, [3, 5, 2, 3])
      XCTAssertEqual(source.logicalDtype, record.dtype)
      let residentBytes = source.residentBytes
      for scan in record.expectedCounts.indices {
        XCTAssertEqual(
          try source.extractRawDiffraction(scanRow: scan / 5, scanColumn: scan % 5),
          record.expectedCounts[scan])
      }
      // Cross entropy block boundaries, the three-frame tail, and repeat a DP.
      for scan in [14, 0, 3, 4, 5, 14] {
        var actual = try source.extractRawDiffraction(scanRow: scan / 5, scanColumn: scan % 5)
        XCTAssertEqual(actual, record.expectedCounts[scan])
        actual[0] = .max
        XCTAssertEqual(
          try source.extractRawDiffraction(scanRow: scan / 5, scanColumn: scan % 5),
          record.expectedCounts[scan])
      }
      let scratch = 2 * 6 * (record.dtype == .uint8 ? 1 : 2)
      XCTAssertEqual(
        try source.sumVirtualDetector(mask: record.mask, maximumDecodedBytes: scratch),
        record.expectedSums)
      XCTAssertEqual(source.lastOperationScratchBytes, scratch + 6 + 15 * 8)
      XCTAssertEqual(source.residentBytes, residentBytes)
      XCTAssertEqual(
        try source.sumVirtualDetector(
          mask: [UInt8](repeating: 0, count: 6), maximumDecodedBytes: scratch),
        [UInt64](repeating: 0, count: 15))
      XCTAssertEqual(
        try source.sumVirtualDetector(
          mask: [UInt8](repeating: 1, count: 6), maximumDecodedBytes: scratch),
        record.expectedCounts.map { $0.reduce(UInt64(0)) { $0 + UInt64($1) } })
      XCTAssertThrowsError(try source.extractRawDiffraction(scanRow: 3, scanColumn: 0))
      XCTAssertThrowsError(try source.sumVirtualDetector(mask: [2, 0, 0, 0, 0, 0]))
      XCTAssertThrowsError(
        try source.sumVirtualDetector(mask: record.mask, maximumDecodedBytes: 0))
      XCTAssertEqual(
        try source.sumVirtualDetector(mask: record.mask, maximumDecodedBytes: scratch),
        record.expectedSums)
      source.releaseResidentStorage()
      XCTAssertEqual(source.residentBytes, 0)
      XCTAssertThrowsError(try source.extractRawDiffraction(scanRow: 0, scanColumn: 0))
      XCTAssertThrowsError(try source.sumVirtualDetector(mask: record.mask))
    }
  }

  func testCompleteValidationRejectsMalformedStreamsAndFalseUInt8Declaration() throws {
    let device = try device()
    let original = try XCTUnwrap(fixture().cases.last).arrays
    XCTAssertThrowsError(
      try MetalANSResidentSource(
        arrays: original.admitted(.uint16), device: device, maximumAdditionalBytes: 0))
    // uint8 cannot be admitted merely because the model tables look small:
    // complete validation must also visit the literal stream containing 65535.
    XCTAssertThrowsError(
      try MetalANSResidentSource(arrays: original.admitted(.uint8), device: device))
    var changed = original
    changed.payload.replaceSubrange(0..<4, with: [0, 0, 0, 0])
    XCTAssertThrowsError(
      try MetalANSResidentSource(arrays: changed.admitted(.uint16), device: device))
    changed = original
    changed.payload.append(0)
    changed.offsets[changed.offsets.count - 1] += 1
    XCTAssertThrowsError(
      try MetalANSResidentSource(arrays: changed.admitted(.uint16), device: device))
    // A failed constructor cannot poison a subsequent independently valid source.
    let source = try MetalANSResidentSource(arrays: original.admitted(.uint16), device: device)
    XCTAssertEqual(try source.extractRawDiffraction(scanRow: 0, scanColumn: 1)[3], 65535)
    source.releaseResidentStorage()
  }
}
