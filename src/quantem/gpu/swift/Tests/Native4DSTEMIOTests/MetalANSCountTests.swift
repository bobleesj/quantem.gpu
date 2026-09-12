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
  private func littleEndianData<T: FixedWidthInteger>(_ values: [T]) -> Data {
    var result = Data()
    result.reserveCapacity(values.count * MemoryLayout<T>.stride)
    for value in values {
      var little = value.littleEndian
      withUnsafeBytes(of: &little) { result.append(contentsOf: $0) }
    }
    return result
  }

  private func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private func writeQGANSFixture(_ record: ANSFixture.Case) throws -> (URL, String) {
    var logical = Data()
    for scan in record.expectedCounts {
      if record.logicalDtype == "uint8" {
        logical.append(contentsOf: scan.map { UInt8(truncatingIfNeeded: $0) })
      } else {
        logical.append(littleEndianData(scan.map { UInt16(truncatingIfNeeded: $0) }))
      }
    }
    let sectionValues: [(String, Data, String, Int)] = [
      ("payload", Data(record.arrays.payload), "u1", 1),
      ("offsets", littleEndianData(record.arrays.offsets), "<u8", 8),
      ("model_ids", littleEndianData(record.arrays.modelIds), "<u4", 4),
      ("context_offsets", littleEndianData(record.arrays.contextOffsets), "<u4", 4),
      ("symbols", littleEndianData(record.arrays.symbols), "<u2", 2),
      ("cumulative", littleEndianData(record.arrays.cumulative), "<u2", 2),
      ("frequencies", littleEndianData(record.arrays.frequencies), "<u2", 2),
      ("literal", Data(record.arrays.literal), "u1", 1),
    ]
    var file = Data(repeating: 0, count: 65_536)
    var sections = [String: [String: Any]]()
    for (name, bytes, dtype, stride) in sectionValues {
      while file.count % 8 != 0 { file.append(0) }
      let offset = file.count
      file.append(bytes)
      sections[name] = [
        "offset": offset, "count": bytes.count / stride, "dtype": dtype,
        "sha256": digest(bytes),
      ]
    }
    let manifest: [String: Any] = [
      "schema": "quantem.gpu.count-ans.v1",
      "codec": "block-column-rans-byte-v1",
      "order": "scan_row,scan_column,detector_row,detector_column",
      "shape": record.arrays.shape,
      "dtype": record.logicalDtype,
      "block_frames": record.arrays.blockFrames,
      "scale": record.arrays.scale,
      "metadata": [String: Any](),
      "sections": sections,
      "logical_sha256": digest(logical),
      "encoder": "native-test-fixture-v1",
    ]
    let manifestBytes = try JSONSerialization.data(
      withJSONObject: manifest, options: [.sortedKeys])
    XCTAssertLessThan(manifestBytes.count, 65_536 - 24)
    file.replaceSubrange(0..<8, with: [0x51, 0x47, 0x41, 0x4e, 0x53, 0, 1, 0])
    func store(_ value: UInt64, at offset: Int) {
      var little = value.littleEndian
      withUnsafeBytes(of: &little) { file.replaceSubrange(offset..<(offset + 8), with: $0) }
    }
    store(UInt64(manifestBytes.count), at: 8)
    store(65_536, at: 16)
    file.replaceSubrange(24..<(24 + manifestBytes.count), with: manifestBytes)
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("qgans-native-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("fixture.qgans")
    try file.write(to: url, options: .atomic)
    return (url, digest(file))
  }

  private func literalArrays(
    shape: [Int], blockFrames: Int = 512, value: UInt16 = 7
  ) throws -> MetalANSCountArrays {
    precondition(shape.count == 4)
    let scans = shape[0] * shape[1]
    let pixels = shape[2] * shape[3]
    let blocks = (scans + blockFrames - 1) / blockFrames
    var payload = [UInt8]()
    payload.reserveCapacity(scans * pixels * 2)
    var offsets = [UInt64](repeating: 0, count: blocks * pixels + 1)
    let modelIDs = [UInt32](repeating: 0, count: blocks * pixels)
    var stream = 0
    for block in 0..<blocks {
      let frames = min(blockFrames, scans - block * blockFrames)
      for _ in 0..<pixels {
        for _ in 0..<frames {
          payload.append(UInt8(truncatingIfNeeded: value))
          payload.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        stream += 1
        offsets[stream] = UInt64(payload.count)
      }
    }
    return try MetalANSCountArrays(
      shape: shape, blockFrames: blockFrames, scale: 1, logicalDtype: .uint16,
      payload: payload, offsets: offsets, modelIDs: modelIDs,
      contextOffsets: [0, 0], symbols: [], cumulative: [], frequencies: [], literal: [1])
  }

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

  func testGenericANSPathSupports512And1024SquareScanShapes() throws {
    let device = try device()
    for shape in [[512, 512, 1, 1], [1024, 1024, 1, 1]] {
      let arrays = try literalArrays(shape: shape)
      let source = try MetalANSResidentSource(arrays: arrays, device: device)
      defer { source.releaseResidentStorage() }
      XCTAssertEqual(source.shape, shape)
      let columns = shape[1]
      for scan in [0, 1, columns - 1, columns, shape[0] * columns - 1] {
        XCTAssertEqual(
          try source.extractRawDiffraction(scanRow: scan / columns, scanColumn: scan % columns),
          [7])
      }
    }
  }

  func testFileBackedQGANSMatchesFixtureAndWholeFileAuthentication() throws {
    let device = try device()
    let record = try XCTUnwrap(try fixture().cases.first(where: { $0.logicalDtype == "uint16" }))
    let (url, fileSHA256) = try writeQGANSFixture(record)
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
    let source = try MetalANSResidentSource(
      sourceURL: url, device: device, expectedSHA256: fileSHA256, verifyChecksums: true)
    defer { source.releaseResidentStorage() }
    XCTAssertEqual(source.shape, record.arrays.shape)
    XCTAssertEqual(source.logicalDtype, .uint16)
    let columns = record.arrays.shape[1]
    for scan in record.expectedCounts.indices {
      XCTAssertEqual(
        try source.extractRawDiffraction(scanRow: scan / columns, scanColumn: scan % columns),
        record.expectedCounts[scan])
    }
    XCTAssertEqual(
      try source.sumVirtualDetector(mask: record.mask, maximumDecodedBytes: 2 * 6 * 2),
      record.expectedSums)
    XCTAssertThrowsError(
      try MetalANSResidentSource(
        sourceURL: url, device: device, expectedSHA256: String(repeating: "0", count: 64)))
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
