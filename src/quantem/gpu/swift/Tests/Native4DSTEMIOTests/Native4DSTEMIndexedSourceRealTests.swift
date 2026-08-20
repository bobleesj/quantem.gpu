import Foundation
import XCTest

@testable import Native4DSTEMIO

final class Native4DSTEMIndexedSourceRealTests: XCTestCase {
  func testOptInFullNativeIndexCoverageUsesBoundedWindows() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let sourcePath = environment["QUANTEM_GPU_NATIVE_INDEXED_SOURCE_DIR"],
      let indexPath = environment["QUANTEM_GPU_NATIVE_INDEXED_INDEX_DIR"]
    else {
      throw XCTSkip(
        "Set QUANTEM_GPU_NATIVE_INDEXED_SOURCE_DIR and "
          + "QUANTEM_GPU_NATIVE_INDEXED_INDEX_DIR for the real indexed-source gate."
      )
    }
    let sourceDirectory = URL(fileURLWithPath: sourcePath, isDirectory: true)
    let indexDirectory = URL(fileURLWithPath: indexPath, isDirectory: true)
    let sourceFiles = try FileManager.default.contentsOfDirectory(
      at: sourceDirectory,
      includingPropertiesForKeys: [.fileSizeKey]
    ).filter {
      $0.lastPathComponent.contains("_data_") && $0.pathExtension == "h5"
    }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    let indexFiles = try FileManager.default.contentsOfDirectory(
      at: indexDirectory,
      includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "qh5idx" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
    XCTAssertEqual(sourceFiles.count, 27)
    XCTAssertEqual(indexFiles.count, sourceFiles.count)
    let compressedBytes = try sourceFiles.reduce(0) { total, url in
      let bytes = try XCTUnwrap(
        url.resourceValues(forKeys: [.fileSizeKey]).fileSize
      )
      let result = total.addingReportingOverflow(bytes)
      guard !result.overflow else {
        throw Native4DSTEMIOError.invalidData("Fixture compressed bytes overflow Int")
      }
      return result.partialValue
    }
    let dataset = Native4DSTEMDataset(
      id: "frozen-full-native-index-gate",
      label: "frozen full-native source",
      masterPath: sourceDirectory.appendingPathComponent("BTO_18_master.h5").path,
      dataFiles: sourceFiles.map(\.path),
      indexFiles: indexFiles.map(\.path),
      scanRows: 512,
      scanCols: 512,
      detectorRows: 192,
      detectorCols: 192,
      sourceDtype: "uint16",
      sourceBytes: compressedBytes,
      badPixelIndices: [],
      kPixelSizeRow: nil,
      kPixelSizeCol: nil,
      kPixelUnit: nil,
      acquisitionDate: nil,
      metadata: nil,
      sourceIdentitySHA256:
        "9f0ddb932c631b63cb573c38d747fa41941ee585c5389d33bdafb4add962b768"
    )

    let opened = try Native4DSTEMIndexedSource.open(dataset: dataset)
    XCTAssertEqual(opened.shards.count, 27)
    XCTAssertEqual(opened.logicalFrameCount, 512 * 512)
    XCTAssertEqual(opened.sourceBytesPerValue, 2)
    XCTAssertEqual(opened.decodedBytesPerFrame, 192 * 192 * 2)
    XCTAssertEqual(opened.logicalDecodedBytes, 19_327_352_832)

    let fourRowBytes = UInt64(4 * 512 * 192 * 192 * 2)
    let windows = try opened.windows(maximumDecodedBytes: fourRowBytes)
    XCTAssertEqual(windows.count, 128)
    XCTAssertEqual(windows.first?.globalFrameRange, 0..<(4 * 512))
    XCTAssertEqual(windows.last?.globalFrameRange, (508 * 512)..<(512 * 512))
    XCTAssertEqual(windows.reduce(0) { $0 + $1.globalFrameRange.count }, 512 * 512)
    XCTAssertTrue(windows.allSatisfy { $0.decodedBytes <= fourRowBytes })
    for (previous, next) in zip(windows, windows.dropFirst()) {
      XCTAssertEqual(previous.globalFrameRange.upperBound, next.globalFrameRange.lowerBound)
    }
    XCTAssertEqual(
      try opened.frameWindow(scanRow: 0, scanColumn: 0).globalFrameRange,
      0..<1
    )
    XCTAssertEqual(
      try opened.frameWindow(scanRow: 511, scanColumn: 511).globalFrameRange,
      (512 * 512 - 1)..<(512 * 512)
    )
  }
}
