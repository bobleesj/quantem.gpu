import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO
@testable import Native4DSTEMIO

final class Native4DSTEMIndexedSourceRealTests: XCTestCase {
  func testOptInFullNativeIndexCoverageUsesBoundedWindows() throws {
    let dataset = try fullNativeDatasetFromEnvironment()
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

  func testOptInFullNativeStreamingProductsMatchFrozenExactTotal() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["QUANTEM_GPU_NATIVE_STREAMING_REAL"] == "1" else {
      throw XCTSkip("Set QUANTEM_GPU_NATIVE_STREAMING_REAL=1 for full native streaming.")
    }
    guard let totalPath = environment["QUANTEM_GPU_NATIVE_TOTAL_REFERENCE"] else {
      throw XCTSkip(
        "Set QUANTEM_GPU_NATIVE_TOTAL_REFERENCE to an independent exact total.u64 artifact."
      )
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      XCTFail("The explicit full-native streaming gate cannot see a Metal device.")
      return
    }
    let dataset = try fullNativeDatasetFromEnvironment()
    let source = try Native4DSTEMIndexedSource.open(dataset: dataset)
    let detectorPixels = dataset.detectorRows * dataset.detectorCols
    let bands = try Metal4DSTEMDetectorBands(
      detectorRows: dataset.detectorRows,
      detectorColumns: dataset.detectorCols,
      membership: [UInt8](repeating: 7, count: detectorPixels)
    )
    let windowRows = Int(environment["QUANTEM_GPU_NATIVE_STREAMING_ROWS"] ?? "8") ?? 0
    guard (1...dataset.scanRows).contains(windowRows) else {
      XCTFail("QUANTEM_GPU_NATIVE_STREAMING_ROWS must be in 1...\(dataset.scanRows).")
      return
    }
    let windowBytes =
      UInt64(windowRows)
      * UInt64(dataset.scanCols)
      * UInt64(dataset.detectorRows)
      * UInt64(dataset.detectorCols)
      * UInt64(MemoryLayout<UInt16>.stride)
    let plan = try Metal4DSTEMIndexedLoadPlan(
      source: source,
      maximumDecodedWindowBytes: windowBytes,
      detectorBands: bands
    )
    let loader = try Metal4DSTEMIndexedLoader(device: device)
    let result = try loader.loadExactProducts(source: source, plan: plan)

    XCTAssertEqual(result.products.band1, result.products.total)
    XCTAssertEqual(result.products.band2, result.products.total)
    XCTAssertEqual(result.products.band4, result.products.total)
    XCTAssertEqual(
      result.products.detectorSum.reduce(UInt64(0), +),
      result.products.total.reduce(UInt64(0), +)
    )
    XCTAssertEqual(result.sourceAudit.maximumSourceCount, 53)
    XCTAssertEqual(result.sourceAudit.pixelsAbove255, 0)
    XCTAssertEqual(result.provenance.sourceScanRows, 512)
    XCTAssertEqual(result.provenance.workingScanRows, 512)
    XCTAssertEqual(result.provenance.sourceDetectorRows, 192)
    XCTAssertEqual(result.provenance.workingDetectorRows, 192)
    XCTAssertEqual(result.provenance.sourceDtype, .uint16)
    XCTAssertEqual(result.provenance.workingDtype, .uint16)
    XCTAssertEqual(result.provenance.scanBin, 1)
    XCTAssertEqual(result.provenance.detectorBin, 1)
    XCTAssertEqual(result.provenance.scanRowStart, 0)
    XCTAssertEqual(result.provenance.scanRowStop, 512)
    XCTAssertEqual(result.provenance.scanColumnStart, 0)
    XCTAssertEqual(result.provenance.scanColumnStop, 512)
    XCTAssertEqual(result.provenance.detectorBandsSHA256, plan.detectorBandsSHA256)
    XCTAssertEqual(result.provenance.detectorBandsSHA256.count, 64)
    XCTAssertEqual(result.provenance.scanCalibration, plan.scanCalibration)
    XCTAssertEqual(
      result.provenance.scanSamplingRowNanometer,
      plan.scanSamplingRowNanometer
    )
    XCTAssertEqual(
      result.provenance.scanSamplingColumnNanometer,
      plan.scanSamplingColumnNanometer
    )
    XCTAssertEqual(result.provenance.detectorSamplingRow, plan.detectorSamplingRow)
    XCTAssertEqual(
      result.provenance.detectorSamplingColumn,
      plan.detectorSamplingColumn
    )
    XCTAssertEqual(result.provenance.detectorSamplingUnit, plan.detectorSamplingUnit)

    let totalData = littleEndianData(result.products.total)
    let reference = try Data(contentsOf: URL(fileURLWithPath: totalPath))
    XCTAssertEqual(totalData, reference)
    let selected = try loader.diffractionPattern(
      source: source,
      scanRow: 256,
      scanColumn: 256
    )
    XCTAssertEqual(
      selected.values.reduce(UInt64(0)) { $0 + UInt64($1) },
      result.products.total[256 * 512 + 256]
    )
    let digest = SHA256.hash(data: totalData)
      .map { String(format: "%02x", $0) }
      .joined()
    print(
      "QUANTEM_GPU_NATIVE_STREAMING_METRICS "
        + "wall=\(result.metrics.wallSeconds) "
        + "gpu=\(result.metrics.gpuSeconds) "
        + "map=\(result.metrics.sourceMappingSeconds) "
        + "windows=\(result.metrics.windowCount) "
        + "slices=\(result.metrics.sliceCount) "
        + "scratch=\(result.metrics.maximumDecodedSliceBytes) "
        + "total_sha256=\(digest)"
    )
  }

  private func fullNativeDatasetFromEnvironment() throws -> Native4DSTEMDataset {
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
    let datasetURL = indexDirectory.appendingPathComponent("dataset.json")
    let dataset = try JSONDecoder().decode(
      Native4DSTEMDataset.self,
      from: Data(contentsOf: datasetURL)
    )
    XCTAssertEqual(dataset.dataFiles.count, 27)
    XCTAssertEqual(dataset.indexFiles.count, dataset.dataFiles.count)
    XCTAssertEqual(
      dataset.masterPath,
      sourceDirectory.appendingPathComponent("BTO_18_master.h5").path
    )
    XCTAssertEqual(dataset.scanRows, 512)
    XCTAssertEqual(dataset.scanCols, 512)
    XCTAssertEqual(dataset.detectorRows, 192)
    XCTAssertEqual(dataset.detectorCols, 192)
    XCTAssertEqual(dataset.sourceDtype, "uint16")
    XCTAssertEqual(
      dataset.sourceIdentitySHA256,
      "9f0ddb932c631b63cb573c38d747fa41941ee585c5389d33bdafb4add962b768"
    )
    return dataset
  }

  private func littleEndianData(_ values: [UInt64]) -> Data {
    let copy = values.map(\.littleEndian)
    return copy.withUnsafeBytes { Data($0) }
  }
}
