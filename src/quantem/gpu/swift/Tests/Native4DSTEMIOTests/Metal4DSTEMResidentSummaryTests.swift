import Foundation
import XCTest

@testable import Native4DSTEMIO

final class Metal4DSTEMResidentSummaryTests: XCTestCase {
  func testExactIntegerSummaryRoundTripAndIntegrity() throws {
    let fixture = try makeResidentFixture()
    addTeardownBlock { try? FileManager.default.removeItem(at: fixture.root) }
    let detectorBands = Data([1, 2, 4, 0])
    let artifacts = exactArtifacts(scanCount: 2, detectorPixels: 4)
    let summaryURL = fixture.root.appendingPathComponent("summary", isDirectory: true)

    let metadata = try Metal4DSTEMResidentSummaryIO.write(
      to: summaryURL,
      residentMetadata: fixture.metadata,
      detectorBands: detectorBands,
      selectedScanRow: 1,
      selectedScanColumn: 0,
      artifacts: artifacts
    )

    XCTAssertEqual(metadata.schema, "quantem.gpu.resident-summary/v1")
    XCTAssertEqual(
      Set(metadata.artifacts.map(\.role)),
      Set(Metal4DSTEMResidentSummaryRole.allCases)
    )
    XCTAssertEqual(
      metadata.artifacts.first(where: { $0.role == .totalIntensity })?.dtype,
      "uint64"
    )
    XCTAssertEqual(
      try FileManager.default.contentsOfDirectory(atPath: fixture.root.path).filter {
        $0.hasSuffix(".tmp")
      },
      []
    )
    let manifest = try String(
      contentsOf: summaryURL.appendingPathComponent("summary.json"),
      encoding: .utf8
    )
    XCTAssertTrue(manifest.contains("quantem.gpu.resident-summary"))
    XCTAssertFalse(manifest.contains("live4dstem"))

    let decoded = try Metal4DSTEMResidentSummaryIO.read(
      from: summaryURL,
      residentMetadata: fixture.metadata,
      detectorBands: detectorBands
    )
    XCTAssertEqual(decoded.metadata, metadata)
    for role in Metal4DSTEMResidentSummaryRole.allCases {
      XCTAssertEqual(decoded.artifacts[role], artifacts[role])
    }

    XCTAssertThrowsError(
      try Metal4DSTEMResidentSummaryIO.write(
        to: summaryURL,
        residentMetadata: fixture.metadata,
        detectorBands: detectorBands,
        selectedScanRow: 1,
        selectedScanColumn: 0,
        artifacts: artifacts
      )
    ) { error in
      guard case Metal4DSTEMResidentSummaryError.destinationExists = error else {
        return XCTFail("Expected destinationExists, received \(error)")
      }
    }
  }

  func testSummaryRejectsChangedBandsAndArtifactBytes() throws {
    let fixture = try makeResidentFixture()
    addTeardownBlock { try? FileManager.default.removeItem(at: fixture.root) }
    let detectorBands = Data([1, 2, 4, 0])
    let artifacts = exactArtifacts(scanCount: 2, detectorPixels: 4)
    let summaryURL = fixture.root.appendingPathComponent("summary", isDirectory: true)
    _ = try Metal4DSTEMResidentSummaryIO.write(
      to: summaryURL,
      residentMetadata: fixture.metadata,
      detectorBands: detectorBands,
      selectedScanRow: 0,
      selectedScanColumn: 0,
      artifacts: artifacts
    )

    XCTAssertThrowsError(
      try Metal4DSTEMResidentSummaryIO.read(
        from: summaryURL,
        residentMetadata: fixture.metadata,
        detectorBands: Data([1, 2, 0, 4])
      )
    ) { error in
      guard case Metal4DSTEMResidentSummaryError.invalidMetadata = error else {
        return XCTFail("Expected invalidMetadata, received \(error)")
      }
    }

    let bfURL = summaryURL.appendingPathComponent(
      Metal4DSTEMResidentSummaryRole.brightField.fileName
    )
    var changed = try Data(contentsOf: bfURL)
    changed[0] ^= 0xFF
    try changed.write(to: bfURL)
    XCTAssertThrowsError(
      try Metal4DSTEMResidentSummaryIO.read(
        from: summaryURL,
        residentMetadata: fixture.metadata,
        detectorBands: detectorBands
      )
    ) { error in
      guard
        case Metal4DSTEMResidentSummaryError.artifactDigestMismatch(
          .brightField,
          _,
          _
        ) = error
      else {
        return XCTFail("Expected bright-field digest mismatch, received \(error)")
      }
    }
  }

  func testSummaryRejectsPotentialUInt32VirtualDetectorOverflow() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Metal4DSTEMResidentSummaryOverflowTests-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let sourceURL = root.appendingPathComponent("source.h5")
    try Data([1]).write(to: sourceURL)
    let metadata = Metal4DSTEMResidentCacheMetadata(
      datasetID: "overflow-fixture",
      sourceIdentitySHA256: String(repeating: "a", count: 64),
      sources: [try Metal4DSTEMSourceIdentity(url: sourceURL)],
      sourceScanRows: 1,
      sourceScanColumns: 1,
      sourceDetectorRows: 1,
      sourceDetectorColumns: 65_538,
      sourceDtype: "uint16",
      outputScanRows: 1,
      outputScanColumns: 1,
      outputDetectorRows: 1,
      outputDetectorColumns: 65_538,
      outputDtype: "uint16",
      scanRowStart: 0,
      scanRowStop: 1,
      scanColumnStart: 0,
      scanColumnStop: 1,
      scanBin: 1,
      detectorBin: 1,
      badPixelIndices: [],
      maxCount: UInt32(UInt16.max),
      pixelsAbove255: 1,
      payloadBytes: 131_076
    )
    let payloadURL = root.appendingPathComponent("resident.bin")
    let sealed = try [UInt32](repeating: 0, count: 32_769).withUnsafeBytes { bytes in
      try Metal4DSTEMResidentCacheIO.write(
        pointer: try XCTUnwrap(bytes.baseAddress),
        length: bytes.count,
        payloadURL: payloadURL,
        metadataURL: root.appendingPathComponent("resident.json"),
        metadata: metadata
      )
    }

    XCTAssertThrowsError(
      try Metal4DSTEMResidentSummaryIO.write(
        to: root.appendingPathComponent("summary"),
        residentMetadata: sealed,
        detectorBands: Data(repeating: 1, count: 65_538),
        selectedScanRow: 0,
        selectedScanColumn: 0,
        artifacts: exactArtifacts(scanCount: 1, detectorPixels: 65_538)
      )
    ) { error in
      guard case Metal4DSTEMResidentSummaryError.invalidMetadata(let reason) = error else {
        return XCTFail("Expected invalidMetadata, received \(error)")
      }
      XCTAssertTrue(reason.contains("virtual-detector sums do not fit uint32"))
    }
  }

  private func makeResidentFixture() throws -> (
    root: URL,
    metadata: Metal4DSTEMResidentCacheMetadata
  ) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Metal4DSTEMResidentSummaryTests-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let sourceURL = root.appendingPathComponent("source.h5")
    try Data([1, 2, 3, 4]).write(to: sourceURL)
    let metadata = Metal4DSTEMResidentCacheMetadata(
      datasetID: "fixture",
      sourceIdentitySHA256: String(repeating: "a", count: 64),
      sources: [try Metal4DSTEMSourceIdentity(url: sourceURL)],
      sourceScanRows: 2,
      sourceScanColumns: 1,
      sourceDetectorRows: 2,
      sourceDetectorColumns: 2,
      sourceDtype: "uint16",
      outputScanRows: 2,
      outputScanColumns: 1,
      outputDetectorRows: 2,
      outputDetectorColumns: 2,
      outputDtype: "uint16",
      scanRowStart: 0,
      scanRowStop: 2,
      scanColumnStart: 0,
      scanColumnStop: 1,
      scanBin: 1,
      detectorBin: 1,
      badPixelIndices: [],
      maxCount: 4,
      pixelsAbove255: 0,
      payloadBytes: 16
    )
    let sealed = try [UInt32(0), 0, 0, 0].withUnsafeBytes { bytes in
      try Metal4DSTEMResidentCacheIO.write(
        pointer: try XCTUnwrap(bytes.baseAddress),
        length: bytes.count,
        payloadURL: root.appendingPathComponent("resident.bin"),
        metadataURL: root.appendingPathComponent("resident.json"),
        metadata: metadata
      )
    }
    return (root, sealed)
  }

  private func exactArtifacts(
    scanCount: Int,
    detectorPixels: Int
  ) -> [Metal4DSTEMResidentSummaryRole: Data] {
    var result: [Metal4DSTEMResidentSummaryRole: Data] = [:]
    for (offset, role) in Metal4DSTEMResidentSummaryRole.allCases.enumerated() {
      let count = role == .selectedDiffraction ? detectorPixels : scanCount
      if role.dtype == "uint64" {
        let values = (0..<count).map { UInt64(offset * 100 + $0) }
        result[role] = values.withUnsafeBytes { Data($0) }
      } else {
        let values = (0..<count).map { UInt32(offset * 100 + $0) }
        result[role] = values.withUnsafeBytes { Data($0) }
      }
    }
    return result
  }
}
