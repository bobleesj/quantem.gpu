import Foundation
import XCTest

@testable import Native4DSTEMIO

final class Native4DSTEMIOTests: XCTestCase {
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

    let indexURL = URL(fileURLWithPath: try XCTUnwrap(dataset.indexFiles.first))
    let index = try Data(contentsOf: indexURL)
    XCTAssertEqual(String(decoding: index.prefix(8), as: UTF8.self), "QH5IDX01")
    let jsonLength = Int(readLE32(index, at: 8))
    let wordCount = Int(readLE32(index, at: 12))
    let metadata = try JSONDecoder().decode(
      QH5IndexMetadata.self,
      from: index.subdata(in: 16..<(16 + jsonLength))
    )
    XCTAssertEqual(metadata.nFrames, 1)
    XCTAssertEqual(metadata.blockElems, 4096)
    XCTAssertEqual(metadata.nBlocksPerFrame, 1)
    XCTAssertEqual(wordCount, 2)
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
      QH5IndexMetadata.self,
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
