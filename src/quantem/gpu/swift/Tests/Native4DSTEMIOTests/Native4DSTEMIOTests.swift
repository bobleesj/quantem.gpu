import Foundation
import XCTest

@testable import Native4DSTEMIO

final class Native4DSTEMIOTests: XCTestCase {
  func testValueRangeAuditRequiresExactSourceAndBadPixelIdentity() throws {
    let audit = Native4DSTEMValueRangeAudit(
      sourceIdentitySHA256: String(repeating: "a", count: 64),
      sourceDtype: "uint16",
      badPixelIndices: [9, 2],
      maximum: 53,
      pixelsAbove255: 0
    )
    XCTAssertEqual(audit.schema, Native4DSTEMValueRangeAudit.currentSchema)
    XCTAssertEqual(
      try audit.sha256(),
      "feea583af9482886a18e26dfd0b398c842057a17092e77e712feddce2280147c"
    )
    XCTAssertTrue(
      audit.provesLosslessUInt8(
        sourceIdentitySHA256: String(repeating: "a", count: 64),
        sourceDtype: "uint16",
        badPixelIndices: [2, 9]
      )
    )
    XCTAssertFalse(
      audit.provesLosslessUInt8(
        sourceIdentitySHA256: String(repeating: "b", count: 64),
        sourceDtype: "uint16",
        badPixelIndices: [2, 9]
      )
    )
    XCTAssertFalse(
      audit.provesLosslessUInt8(
        sourceIdentitySHA256: String(repeating: "a", count: 64),
        sourceDtype: "uint16",
        badPixelIndices: [2]
      )
    )

    var legacyObject = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(audit)) as? [String: Any]
    )
    legacyObject["schema"] = "live4dstem.value-range-audit/v1"
    let legacy = try JSONDecoder().decode(
      Native4DSTEMValueRangeAudit.self,
      from: JSONSerialization.data(withJSONObject: legacyObject)
    )
    XCTAssertTrue(
      legacy.provesLosslessUInt8(
        sourceIdentitySHA256: String(repeating: "a", count: 64),
        sourceDtype: "uint16",
        badPixelIndices: [2, 9]
      )
    )

    var invalidObject = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(audit)) as? [String: Any]
    )
    invalidObject["maximum"] = 300
    let invalid = try JSONDecoder().decode(
      Native4DSTEMValueRangeAudit.self,
      from: JSONSerialization.data(withJSONObject: invalidObject)
    )
    XCTAssertThrowsError(try invalid.validate())
    XCTAssertFalse(
      invalid.provesLosslessUInt8(
        sourceIdentitySHA256: String(repeating: "a", count: 64),
        sourceDtype: "uint16",
        badPixelIndices: [2, 9]
      )
    )
  }

  func testResidentCacheRequiresMatchingSealedAuditForExactUInt16Bin2() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "Metal4DSTEMResidentCacheAuditTests-\(UUID().uuidString)",
        isDirectory: true
      )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let sourceURL = root.appendingPathComponent("source.h5")
    try Data([1, 2, 3, 4]).write(to: sourceURL)
    let sourceIdentity = String(repeating: "a", count: 64)
    let audit = Native4DSTEMValueRangeAudit(
      sourceIdentitySHA256: sourceIdentity,
      sourceDtype: "uint16",
      badPixelIndices: [2, 9],
      maximum: 53,
      pixelsAbove255: 0
    )
    let digest = try audit.sha256()
    let metadata = Metal4DSTEMResidentCacheMetadata(
      datasetID: "exact-bin2-fixture",
      sourceIdentitySHA256: sourceIdentity,
      valueRangeAuditSHA256: digest,
      valueRangeAudit: audit,
      sources: [try Metal4DSTEMSourceIdentity(url: sourceURL)],
      sourceScanRows: 2,
      sourceScanColumns: 2,
      sourceDetectorRows: 4,
      sourceDetectorColumns: 4,
      sourceDtype: "uint16",
      outputScanRows: 2,
      outputScanColumns: 2,
      outputDetectorRows: 2,
      outputDetectorColumns: 2,
      outputDtype: "uint16",
      scanRowStart: 0,
      scanRowStop: 2,
      scanColumnStart: 0,
      scanColumnStop: 2,
      scanBin: 1,
      detectorBin: 2,
      badPixelIndices: [2, 9],
      maxCount: 53,
      pixelsAbove255: 0,
      payloadBytes: 32
    )
    XCTAssertNoThrow(
      try Metal4DSTEMResidentCacheIO.validateMetadata(
        metadata,
        requireSealedPayload: false
      )
    )
    var versionOneObject = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(metadata))
        as? [String: Any]
    )
    versionOneObject["formatVersion"] = 1
    let versionOne = try JSONDecoder().decode(
      Metal4DSTEMResidentCacheMetadata.self,
      from: JSONSerialization.data(withJSONObject: versionOneObject)
    )
    XCTAssertThrowsError(
      try Metal4DSTEMResidentCacheIO.validateMetadata(
        versionOne,
        requireSealedPayload: false
      )
    )

    let missingAudit = Metal4DSTEMResidentCacheMetadata(
      datasetID: "exact-bin2-fixture",
      sourceIdentitySHA256: sourceIdentity,
      valueRangeAuditSHA256: digest,
      sources: metadata.sources,
      sourceScanRows: 2,
      sourceScanColumns: 2,
      sourceDetectorRows: 4,
      sourceDetectorColumns: 4,
      sourceDtype: "uint16",
      outputScanRows: 2,
      outputScanColumns: 2,
      outputDetectorRows: 2,
      outputDetectorColumns: 2,
      outputDtype: "uint16",
      scanRowStart: 0,
      scanRowStop: 2,
      scanColumnStart: 0,
      scanColumnStop: 2,
      scanBin: 1,
      detectorBin: 2,
      badPixelIndices: [2, 9],
      maxCount: 53,
      pixelsAbove255: 0,
      payloadBytes: 32
    )
    XCTAssertThrowsError(
      try Metal4DSTEMResidentCacheIO.validateMetadata(
        missingAudit,
        requireSealedPayload: false
      )
    )

    let overflowAudit = Native4DSTEMValueRangeAudit(
      sourceIdentitySHA256: sourceIdentity,
      sourceDtype: "uint16",
      badPixelIndices: [],
      maximum: 16_384,
      pixelsAbove255: 1
    )
    let overflow = Metal4DSTEMResidentCacheMetadata(
      datasetID: "overflow-bin2-fixture",
      sourceIdentitySHA256: sourceIdentity,
      valueRangeAuditSHA256: try overflowAudit.sha256(),
      valueRangeAudit: overflowAudit,
      sources: metadata.sources,
      sourceScanRows: 1,
      sourceScanColumns: 1,
      sourceDetectorRows: 2,
      sourceDetectorColumns: 2,
      sourceDtype: "uint16",
      outputScanRows: 1,
      outputScanColumns: 1,
      outputDetectorRows: 1,
      outputDetectorColumns: 1,
      outputDtype: "uint16",
      scanRowStart: 0,
      scanRowStop: 1,
      scanColumnStart: 0,
      scanColumnStop: 1,
      scanBin: 1,
      detectorBin: 2,
      badPixelIndices: [],
      maxCount: 16_384,
      pixelsAbove255: 1,
      payloadBytes: 4
    )
    XCTAssertThrowsError(
      try Metal4DSTEMResidentCacheIO.validateMetadata(
        overflow,
        requireSealedPayload: false
      )
    )
  }

  func testVeloxEMDFieldOfViewRequiresTwoPositiveAxes() throws {
    let json = Data(
      #"{"Optics":{"FullScanFieldOfView":{"x":{"type":"double","value":"5e-8"},"y":2.5e-8}}}"#.utf8
    )

    let fieldOfView = try XCTUnwrap(
      NativeHDF5Bridge.veloxFieldOfViewNanometer(jsonData: json)
    )

    XCTAssertEqual(fieldOfView.row, 25, accuracy: 1e-12)
    XCTAssertEqual(fieldOfView.column, 50, accuracy: 1e-12)
    XCTAssertNil(
      NativeHDF5Bridge.veloxFieldOfViewNanometer(
        jsonData: Data(#"{"Optics":{"FullScanFieldOfView":{"x":5e-8}}}"#.utf8)
      )
    )
    XCTAssertNil(
      NativeHDF5Bridge.veloxFieldOfViewNanometer(
        jsonData: Data(#"{"Optics":{"FullScanFieldOfView":{"x":5e-8,"y":-1}}}"#.utf8)
      )
    )
  }

  func testResidentCacheRoundTripAndIntegrity() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "Metal4DSTEMResidentCacheTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }

    let sourceURL = root.appendingPathComponent("source.h5")
    try Data([1, 2, 3, 4]).write(to: sourceURL)
    let source = try Metal4DSTEMSourceIdentity(url: sourceURL)
    let payloadURL = root.appendingPathComponent("resident.bin")
    let metadataURL = root.appendingPathComponent("resident.json")
    let words: [UInt16] = [1, 2, 4, 5, 3, 0, 6, 0]
    let payloadBytes = words.count * MemoryLayout<UInt16>.stride
    let metadata = Metal4DSTEMResidentCacheMetadata(
      datasetID: "fixture",
      sourceIdentitySHA256: String(repeating: "a", count: 64),
      sources: [source],
      sourceScanRows: 2,
      sourceScanColumns: 1,
      sourceDetectorRows: 1,
      sourceDetectorColumns: 3,
      sourceDtype: "uint16",
      outputScanRows: 2,
      outputScanColumns: 1,
      outputDetectorRows: 1,
      outputDetectorColumns: 3,
      outputDtype: "uint16",
      scanRowStart: 0,
      scanRowStop: 2,
      scanColumnStart: 0,
      scanColumnStop: 1,
      scanBin: 1,
      detectorBin: 1,
      badPixelIndices: [1],
      maxCount: 6,
      pixelsAbove255: 0,
      payloadBytes: UInt64(payloadBytes)
    )

    let complete = try words.withUnsafeBytes { bytes in
      try Metal4DSTEMResidentCacheIO.write(
        pointer: try XCTUnwrap(bytes.baseAddress),
        length: bytes.count,
        payloadURL: payloadURL,
        metadataURL: metadataURL,
        metadata: metadata
      )
    }
    XCTAssertEqual(complete.payloadSHA256.count, 64)
    XCTAssertEqual(complete.payloadIdentity?.byteCount, UInt64(payloadBytes))
    XCTAssertEqual(
      try FileManager.default.contentsOfDirectory(atPath: root.path).filter {
        $0.hasSuffix(".tmp")
      },
      []
    )
    XCTAssertEqual(try Metal4DSTEMResidentCacheIO.readMetadata(from: metadataURL), complete)
    try Metal4DSTEMResidentCacheIO.validatePayload(at: payloadURL, metadata: complete)
    try Metal4DSTEMResidentCacheIO.validatePayload(
      at: payloadURL,
      metadata: complete,
      verifySHA256: false
    )
    let expectedPayload = words.withUnsafeBytes { Data($0) }
    XCTAssertEqual(try Data(contentsOf: payloadURL), expectedPayload)

    let changedDate = Date(timeIntervalSince1970: 2_000_000_000)
    try FileManager.default.setAttributes(
      [.modificationDate: changedDate],
      ofItemAtPath: payloadURL.path
    )
    XCTAssertThrowsError(
      try Metal4DSTEMResidentCacheIO.validatePayload(
        at: payloadURL,
        metadata: complete,
        verifySHA256: false
      )
    ) { error in
      guard case Metal4DSTEMResidentCacheError.invalidMetadata = error else {
        return XCTFail("Expected identity mismatch, received \(error)")
      }
    }

    var corrupted = try Data(contentsOf: payloadURL)
    corrupted[0] ^= 0xFF
    try corrupted.write(to: payloadURL)
    XCTAssertThrowsError(
      try Metal4DSTEMResidentCacheIO.validatePayload(at: payloadURL, metadata: complete)
    ) { error in
      guard case Metal4DSTEMResidentCacheError.payloadDigestMismatch = error else {
        return XCTFail("Expected digest mismatch, received \(error)")
      }
    }
  }

  func testResidentCacheRejectsShapeOrBinMetadataThatChangesScientificMeaning() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "Metal4DSTEMResidentCacheInvalidMetadataTests-\(UUID().uuidString)",
        isDirectory: true
      )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }

    let sourceURL = root.appendingPathComponent("source.h5")
    try Data([1, 2, 3, 4]).write(to: sourceURL)
    let metadata = Metal4DSTEMResidentCacheMetadata(
      datasetID: "fixture",
      sourceIdentitySHA256: String(repeating: "a", count: 64),
      sources: [try Metal4DSTEMSourceIdentity(url: sourceURL)],
      sourceScanRows: 2,
      sourceScanColumns: 1,
      sourceDetectorRows: 1,
      sourceDetectorColumns: 2,
      sourceDtype: "uint16",
      outputScanRows: 1,
      outputScanColumns: 1,
      outputDetectorRows: 1,
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
      payloadBytes: 4
    )
    let values: [UInt32] = [0]
    XCTAssertThrowsError(
      try values.withUnsafeBytes { bytes in
        try Metal4DSTEMResidentCacheIO.write(
          pointer: try XCTUnwrap(bytes.baseAddress),
          length: bytes.count,
          payloadURL: root.appendingPathComponent("resident.bin"),
          metadataURL: root.appendingPathComponent("resident.json"),
          metadata: metadata
        )
      }
    ) { error in
      guard case Metal4DSTEMResidentCacheError.invalidMetadata = error else {
        return XCTFail("Expected invalid metadata, received \(error)")
      }
    }
    XCTAssertFalse(
      FileManager.default.fileExists(atPath: root.appendingPathComponent("resident.bin").path)
    )
  }

  func testNativeCatalogAndQH5Index() throws {
    let fixture = try copiedFixture()
    let catalog = try Native4DSTEMCatalogBuilder(
      cacheDirectory: fixture.cache
    ).prepare(input: fixture.master)

    let dataset = try XCTUnwrap(catalog.datasets.first)
    XCTAssertEqual(dataset.scanRows, 1)
    XCTAssertEqual(dataset.scanCols, 1)
    XCTAssertEqual(dataset.detectorRows, 64)
    XCTAssertEqual(dataset.detectorCols, 64)
    XCTAssertEqual(dataset.sourceDtype, "uint16")
    XCTAssertEqual(dataset.badPixelIndices, [196])
    XCTAssertEqual(dataset.acquisitionDate, "2026-08-15T10:00:00Z")
    XCTAssertEqual(dataset.metadata?["entry@experiment"], "native-fixture")
    XCTAssertEqual(dataset.kPixelSizeRow ?? 0, 0.7499998593750473, accuracy: 1e-12)
    XCTAssertNil(dataset.scanPixelSizeRowNanometer)
    XCTAssertNil(dataset.scanPixelSizeColNanometer)

    let indexURL = URL(fileURLWithPath: try XCTUnwrap(dataset.indexFiles.first))
    let index = try Data(contentsOf: indexURL)
    XCTAssertEqual(String(decoding: index.prefix(8), as: UTF8.self), "QH5IDX01")
    let jsonLength = Int(readLE32(index, at: 8))
    let wordCount = Int(readLE32(index, at: 12))
    let metadata = try JSONDecoder().decode(
      NativeQH5IndexMetadata.self,
      from: index.subdata(in: 16..<(16 + jsonLength))
    )
    XCTAssertEqual(metadata.nFrames, 1)
    XCTAssertEqual(metadata.blockElems, 4096)
    XCTAssertEqual(metadata.nBlocksPerFrame, 1)
    XCTAssertEqual(wordCount, 2)
  }

  func testCatalogRoundTripPreservesExplicitSpatialCalibration() throws {
    let dataset = Native4DSTEMDataset(
      id: "calibrated",
      label: "calibrated.emd",
      masterPath: "/data/calibrated.emd",
      dataFiles: ["/data/calibrated.emd"],
      indexFiles: [],
      scanRows: 128,
      scanCols: 256,
      detectorRows: 192,
      detectorCols: 192,
      sourceDtype: "uint16",
      sourceBytes: 1,
      badPixelIndices: [],
      scanPixelSizeRowNanometer: 0.08,
      scanPixelSizeColNanometer: 0.1,
      kPixelSizeRow: nil,
      kPixelSizeCol: nil,
      kPixelUnit: nil,
      acquisitionDate: nil,
      metadata: nil
    )
    let catalog = Native4DSTEMCatalog(input: dataset.masterPath!, datasets: [dataset])

    let restored = try JSONDecoder().decode(
      Native4DSTEMCatalog.self,
      from: JSONEncoder().encode(catalog)
    ).datasets[0]

    XCTAssertEqual(restored.scanPixelSizeRowNanometer, 0.08)
    XCTAssertEqual(restored.scanPixelSizeColNanometer, 0.1)
  }

  func testCatalogOnlyDoesNotCreateIndex() throws {
    let fixture = try copiedFixture()
    let catalog = try Native4DSTEMCatalogBuilder(
      cacheDirectory: fixture.cache
    ).prepare(input: fixture.master, mode: .catalogOnly)
    let indexPath = try XCTUnwrap(catalog.datasets.first?.indexFiles.first)
    XCTAssertFalse(FileManager.default.fileExists(atPath: indexPath))
  }

  func testUInt8PartialBitshuffleBlockIndex() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("Native4DSTEMIOU8Tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("fixture_u8_data_000001.h5")
    try FileManager.default.copyItem(
      at: Bundle.module.url(
        forResource: "fixture_u8_data_000001",
        withExtension: "h5",
        subdirectory: "Fixtures"
      )!,
      to: source
    )

    let catalog = try Native4DSTEMCatalogBuilder(
      cacheDirectory: root.appendingPathComponent("cache", isDirectory: true)
    ).prepare(input: source)
    let dataset = try XCTUnwrap(catalog.datasets.first)
    XCTAssertEqual(dataset.detectorRows, 192)
    XCTAssertEqual(dataset.detectorCols, 192)
    XCTAssertEqual(dataset.sourceDtype, "uint8")

    let indexURL = URL(fileURLWithPath: try XCTUnwrap(dataset.indexFiles.first))
    let index = try Data(contentsOf: indexURL)
    let jsonLength = Int(readLE32(index, at: 8))
    let metadata = try JSONDecoder().decode(
      NativeQH5IndexMetadata.self,
      from: index.subdata(in: 16..<(16 + jsonLength))
    )
    XCTAssertEqual(metadata.blockElems, 8192)
    XCTAssertEqual(metadata.nBlocksPerFrame, 5)
    XCTAssertEqual(readLE32(index, at: 12), 10)
  }

  func testVeloxEMDCalibrationAndExactScalarCache() throws {
    let source = try XCTUnwrap(
      Bundle.module.url(
        forResource: "fixture_velox",
        withExtension: "emd",
        subdirectory: "Fixtures"
      )
    )
    let cache = FileManager.default.temporaryDirectory
      .appendingPathComponent("Native4DSTEMIOVeloxTests-\(UUID().uuidString)", isDirectory: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: cache) }
    let builder = Native4DSTEMCatalogBuilder(cacheDirectory: cache)

    let discovery = try builder.prepare(input: source, mode: .catalogOnly)
    let discovered = try XCTUnwrap(discovery.datasets.first)
    XCTAssertEqual(discovered.scanRows, 2)
    XCTAssertEqual(discovered.scanCols, 3)
    XCTAssertEqual(discovered.detectorRows, 1)
    XCTAssertEqual(discovered.detectorCols, 1)
    XCTAssertEqual(discovered.sourceDtype, "uint16")
    XCTAssertEqual(discovered.sourceScanCalibration?.rowSamplingAngstrom, 1)
    XCTAssertEqual(discovered.sourceScanCalibration?.columnSamplingAngstrom, 2)
    XCTAssertEqual(discovered.sourceScanCalibration?.origin, .sourceMetadata)
    XCTAssertTrue(
      discovered.sourceScanCalibration?.evidence.contains("FullScanFieldOfView") == true
    )
    XCTAssertNil(discovered.sourceIdentitySHA256)
    XCTAssertFalse(
      FileManager.default.fileExists(atPath: try XCTUnwrap(discovered.scalarImageRawPath))
    )

    let indexed = try builder.prepare(input: source).datasets[0]
    XCTAssertEqual(indexed.id, discovered.id)
    XCTAssertEqual(indexed.sourceIdentitySHA256?.count, 64)
    XCTAssertEqual(indexed.orderedMemberSHA256?.count, 1)
    let raw = try Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(indexed.scalarImageRawPath)))
    XCTAssertEqual(
      raw,
      Data([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0])
    )
  }

  func testFolderMasterAndShardResolveToSameDataset() throws {
    let fixture = try copiedFixture()
    let builder = Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
    let folder = try builder.prepare(input: fixture.root, mode: .catalogOnly)
    let master = try builder.prepare(input: fixture.master, mode: .catalogOnly)
    let shard = try builder.prepare(input: fixture.data, mode: .catalogOnly)
    XCTAssertEqual(folder.datasets.map(\.id), master.datasets.map(\.id))
    XCTAssertEqual(shard.datasets.map(\.id), master.datasets.map(\.id))
  }

  func testMultipleInputsPreserveOrderAndDeduplicateDatasets() throws {
    let fixture = try copiedFixture()
    let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache).prepare(
      inputs: [fixture.master, fixture.data],
      mode: .catalogOnly
    )

    XCTAssertEqual(catalog.datasets.count, 1)
    XCTAssertEqual(catalog.input, "\(fixture.master.path) | \(fixture.data.path)")
  }

  func testDatasetIdentityIgnoresMetadataOnlyStatusChanges() throws {
    let fixture = try copiedFixture()
    let before = try nativeDatasetSignature(for: [fixture.master, fixture.data])
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o600],
      ofItemAtPath: fixture.master.path
    )
    let after = try nativeDatasetSignature(for: [fixture.master, fixture.data])
    XCTAssertEqual(after, before)
  }

  func testConcurrentNativeInspection() throws {
    let fixture = try copiedFixture()
    let queue = DispatchQueue(label: "native-hdf5-test", attributes: .concurrent)
    let group = DispatchGroup()
    let failures = NSLock()
    nonisolated(unsafe) var errors: [Error] = []
    for _ in 0..<8 {
      group.enter()
      queue.async {
        defer { group.leave() }
        do {
          _ = try Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
            .prepare(input: fixture.master, mode: .catalogOnly)
        } catch {
          failures.lock()
          errors.append(error)
          failures.unlock()
        }
      }
    }
    XCTAssertEqual(group.wait(timeout: .now() + 5), .success)
    XCTAssertTrue(errors.isEmpty, "Concurrent inspection failed: \(errors)")
  }

  func testConcurrentIndexedPreparationUsesOneValidCache() throws {
    let fixture = try copiedFixture()
    let queue = DispatchQueue(label: "native-hdf5-index-test", attributes: .concurrent)
    let group = DispatchGroup()
    let resultLock = NSLock()
    nonisolated(unsafe) var datasetIDs: [String] = []
    nonisolated(unsafe) var errors: [Error] = []
    for _ in 0..<8 {
      group.enter()
      queue.async {
        defer { group.leave() }
        do {
          let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
            .prepare(input: fixture.master)
          resultLock.lock()
          datasetIDs.append(contentsOf: catalog.datasets.map(\.id))
          resultLock.unlock()
        } catch {
          resultLock.lock()
          errors.append(error)
          resultLock.unlock()
        }
      }
    }
    XCTAssertEqual(group.wait(timeout: .now() + 10), .success)
    XCTAssertTrue(errors.isEmpty, "Concurrent index preparation failed: \(errors)")
    XCTAssertEqual(datasetIDs.count, 8)
    XCTAssertEqual(Set(datasetIDs).count, 1)

    let signatureDirectory = try XCTUnwrap(
      FileManager.default.contentsOfDirectory(
        at: fixture.cache,
        includingPropertiesForKeys: nil
      ).first
    )
    let indexURL = try XCTUnwrap(
      FileManager.default.contentsOfDirectory(
        at: signatureDirectory,
        includingPropertiesForKeys: nil
      ).first { $0.pathExtension == "qh5idx" }
    )
    XCTAssertEqual(
      String(decoding: try Data(contentsOf: indexURL).prefix(8), as: UTF8.self),
      "QH5IDX01"
    )
  }

  func testIndexedMultiShardPreparationWritesEveryShard() throws {
    let fixture = try copiedFixture()
    let source = try XCTUnwrap(
      Bundle.module.url(
        forResource: "fixture_data_000001",
        withExtension: "h5",
        subdirectory: "Fixtures"
      )
    )
    try FileManager.default.removeItem(at: fixture.data)
    let names = (1...9).map { String(format: "fixture_data_%06d.h5", $0) }
    for name in names {
      try FileManager.default.copyItem(
        at: source,
        to: fixture.root.appendingPathComponent(name)
      )
    }

    XCTAssertThrowsError(
      try Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
        .prepare(input: fixture.root.appendingPathComponent(names[0]))
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("expects 1 scan positions"))
    }
    let signatureDirectory = try XCTUnwrap(
      FileManager.default.contentsOfDirectory(
        at: fixture.cache,
        includingPropertiesForKeys: nil
      ).first
    )
    let indexFiles = try FileManager.default.contentsOfDirectory(
      at: signatureDirectory,
      includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "qh5idx" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
    XCTAssertEqual(
      indexFiles.map(\.lastPathComponent),
      names.map { URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent + ".qh5idx" }
    )
    for path in indexFiles {
      XCTAssertEqual(
        String(decoding: try Data(contentsOf: path).prefix(8), as: UTF8.self),
        "QH5IDX01"
      )
    }
  }

  func testIndexedFrameWindowPlanBoundsNative18GiBWithoutChangingCoverage() throws {
    let decodedRowBytes = UInt64(512 * 192 * 192 * 2)
    let maximumWindowBytes = decodedRowBytes * 7
    let plan = try Native4DSTEMFrameWindowPlan(
      scanRows: 512,
      scanColumns: 512,
      detectorRows: 192,
      detectorColumns: 192,
      sourceBytesPerValue: 2,
      maximumDecodedBytes: maximumWindowBytes
    )

    XCTAssertEqual(plan.decodedBytesPerFrame, 192 * 192 * 2)
    XCTAssertEqual(plan.logicalDecodedBytes, 19_327_352_832)
    XCTAssertEqual(plan.frameRanges.first, 0..<(7 * 512))
    XCTAssertEqual(plan.frameRanges.last?.upperBound, 512 * 512)
    XCTAssertEqual(plan.frameRanges.reduce(0) { $0 + $1.count }, 512 * 512)
    XCTAssertTrue(
      plan.frameRanges.allSatisfy {
        UInt64($0.count) * plan.decodedBytesPerFrame <= maximumWindowBytes
      }
    )
    XCTAssertTrue(
      plan.frameRanges.dropLast().allSatisfy {
        $0.lowerBound.isMultiple(of: 512) && $0.upperBound.isMultiple(of: 512)
      }
    )
    for (previous, next) in zip(plan.frameRanges, plan.frameRanges.dropFirst()) {
      XCTAssertEqual(previous.upperBound, next.lowerBound)
    }
    XCTAssertThrowsError(
      try Native4DSTEMFrameWindowPlan(
        scanRows: 512,
        scanColumns: 512,
        detectorRows: 192,
        detectorColumns: 192,
        sourceBytesPerValue: 2,
        maximumDecodedBytes: decodedRowBytes - 1
      )
    )
  }

  func testIndexedSourceOpensPreparedFixtureAndResolvesRowColumnFrame() throws {
    let fixture = try copiedFixture()
    let dataset = try XCTUnwrap(
      Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
        .prepare(input: fixture.master).datasets.first
    )

    let source = try Native4DSTEMIndexedSource.open(dataset: dataset)
    XCTAssertEqual(source.logicalFrameCount, 1)
    XCTAssertEqual(source.decodedBytesPerFrame, 64 * 64 * 2)
    XCTAssertEqual(source.logicalDecodedBytes, 64 * 64 * 2)
    XCTAssertEqual(source.shards.count, 1)
    XCTAssertEqual(source.shards[0].index.metadataWords.count, 2)

    let windows = try source.windows(
      maximumDecodedBytes: source.decodedBytesPerFrame,
      alignToScanRows: true
    )
    XCTAssertEqual(windows.count, 1)
    XCTAssertEqual(windows[0].globalFrameRange, 0..<1)
    XCTAssertEqual(windows[0].decodedBytes, 64 * 64 * 2)
    XCTAssertEqual(windows[0].slices.count, 1)
    XCTAssertEqual(windows[0].slices[0].globalFrameRange, 0..<1)
    XCTAssertEqual(windows[0].slices[0].shardFrameRange, 0..<1)
    XCTAssertEqual(windows[0].slices[0].metadataWordRange, 0..<2)
    XCTAssertEqual(try source.frameWindow(scanRow: 0, scanColumn: 0), windows[0])
    XCTAssertThrowsError(try source.frameWindow(scanRow: 0, scanColumn: 1))
  }

  func testIndexedSourceRejectsStaleIndexAndIncompleteLogicalCoverage() throws {
    do {
      let fixture = try copiedFixture()
      let dataset = try XCTUnwrap(
        Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
          .prepare(input: fixture.master).datasets.first
      )
      try FileManager.default.setAttributes(
        [.modificationDate: Date(timeIntervalSince1970: 2_000_000_000)],
        ofItemAtPath: fixture.data.path
      )
      XCTAssertThrowsError(try Native4DSTEMIndexedSource.open(dataset: dataset)) { error in
        XCTAssertTrue(error.localizedDescription.contains("is stale"))
      }
    }

    do {
      let fixture = try copiedFixture()
      let dataset = try XCTUnwrap(
        Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
          .prepare(input: fixture.master).datasets.first
      )
      let incomplete = Native4DSTEMDataset(
        id: dataset.id,
        label: dataset.label,
        masterPath: dataset.masterPath,
        dataFiles: dataset.dataFiles,
        indexFiles: dataset.indexFiles,
        scanRows: 1,
        scanCols: 2,
        detectorRows: dataset.detectorRows,
        detectorCols: dataset.detectorCols,
        sourceDtype: dataset.sourceDtype,
        sourceBytes: dataset.sourceBytes,
        badPixelIndices: dataset.badPixelIndices,
        scanPixelSizeRowNanometer: dataset.scanPixelSizeRowNanometer,
        scanPixelSizeColNanometer: dataset.scanPixelSizeColNanometer,
        kPixelSizeRow: dataset.kPixelSizeRow,
        kPixelSizeCol: dataset.kPixelSizeCol,
        kPixelUnit: dataset.kPixelUnit,
        acquisitionDate: dataset.acquisitionDate,
        metadata: dataset.metadata,
        schemaIdentity: dataset.schemaIdentity,
        sourceIdentitySHA256: dataset.sourceIdentitySHA256,
        masterSHA256: dataset.masterSHA256,
        orderedMemberSHA256: dataset.orderedMemberSHA256,
        sourceScanCalibration: dataset.sourceScanCalibration,
        scalarImageRawPath: dataset.scalarImageRawPath
      )
      XCTAssertThrowsError(try Native4DSTEMIndexedSource.open(dataset: incomplete)) { error in
        XCTAssertTrue(error.localizedDescription.contains("indexes cover 1 frames; expected 2"))
      }
    }
  }

  func testNativeQH5IndexRejectsTrailingBytes() throws {
    let fixture = try copiedFixture()
    let dataset = try XCTUnwrap(
      Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
        .prepare(input: fixture.master).datasets.first
    )
    let indexURL = URL(fileURLWithPath: try XCTUnwrap(dataset.indexFiles.first))
    var index = try Data(contentsOf: indexURL)
    index.append(0)
    try index.write(to: indexURL, options: .atomic)

    XCTAssertThrowsError(
      try NativeQH5Index.open(sourceURL: fixture.data, indexURL: indexURL)
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("Truncated or trailing"))
    }
  }

  func testNativeQH5IndexRejectsIncompatibleBlockGeometry() throws {
    let fixture = try copiedFixture()
    let dataset = try XCTUnwrap(
      Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
        .prepare(input: fixture.master).datasets.first
    )
    let indexURL = URL(fileURLWithPath: try XCTUnwrap(dataset.indexFiles.first))
    var index = try Data(contentsOf: indexURL)
    let valid = Data("\"nBlocksPerFrame\":1".utf8)
    let invalid = Data("\"nBlocksPerFrame\":2".utf8)
    let range = try XCTUnwrap(index.range(of: valid))
    index.replaceSubrange(range, with: invalid)
    try index.write(to: indexURL, options: .atomic)

    XCTAssertThrowsError(
      try NativeQH5Index.open(sourceURL: fixture.data, indexURL: indexURL)
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("block geometry"))
    }
  }

  func testIndexedSourceRejectsRepeatedSourceOrIndexPaths() throws {
    let fixture = try copiedFixture()
    let dataset = try XCTUnwrap(
      Native4DSTEMCatalogBuilder(cacheDirectory: fixture.cache)
        .prepare(input: fixture.master).datasets.first
    )
    let repeated = Native4DSTEMDataset(
      id: dataset.id,
      label: dataset.label,
      masterPath: dataset.masterPath,
      dataFiles: dataset.dataFiles + dataset.dataFiles,
      indexFiles: dataset.indexFiles + dataset.indexFiles,
      scanRows: 1,
      scanCols: 2,
      detectorRows: dataset.detectorRows,
      detectorCols: dataset.detectorCols,
      sourceDtype: dataset.sourceDtype,
      sourceBytes: dataset.sourceBytes * 2,
      badPixelIndices: dataset.badPixelIndices,
      scanPixelSizeRowNanometer: dataset.scanPixelSizeRowNanometer,
      scanPixelSizeColNanometer: dataset.scanPixelSizeColNanometer,
      kPixelSizeRow: dataset.kPixelSizeRow,
      kPixelSizeCol: dataset.kPixelSizeCol,
      kPixelUnit: dataset.kPixelUnit,
      acquisitionDate: dataset.acquisitionDate,
      metadata: dataset.metadata,
      schemaIdentity: dataset.schemaIdentity,
      sourceIdentitySHA256: dataset.sourceIdentitySHA256,
      masterSHA256: dataset.masterSHA256,
      orderedMemberSHA256: dataset.orderedMemberSHA256,
      sourceScanCalibration: dataset.sourceScanCalibration,
      scalarImageRawPath: dataset.scalarImageRawPath
    )

    XCTAssertThrowsError(try Native4DSTEMIndexedSource.open(dataset: repeated)) { error in
      XCTAssertTrue(error.localizedDescription.contains("repeats a source"))
    }
  }

  private func copiedFixture() throws -> (
    root: URL,
    master: URL,
    data: URL,
    cache: URL
  ) {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("Native4DSTEMIOTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let master = root.appendingPathComponent("fixture_master.h5")
    let data = root.appendingPathComponent("fixture_data_000001.h5")
    try FileManager.default.copyItem(
      at: Bundle.module.url(
        forResource: "fixture_master",
        withExtension: "h5",
        subdirectory: "Fixtures"
      )!,
      to: master
    )
    try FileManager.default.copyItem(
      at: Bundle.module.url(
        forResource: "fixture_data_000001",
        withExtension: "h5",
        subdirectory: "Fixtures"
      )!,
      to: data
    )
    return (root, master, data, root.appendingPathComponent("cache", isDirectory: true))
  }

  private func readLE32(_ data: Data, at offset: Int) -> UInt32 {
    data.withUnsafeBytes {
      $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
    }
  }
}
