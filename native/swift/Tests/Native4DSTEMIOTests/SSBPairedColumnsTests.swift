import Foundation
import Metal
import Native4DSTEMIO
import XCTest

@testable import Metal4DSTEMStreamingIO

final class SSBPairedColumnsTests: XCTestCase {
  func testOriginalCountsForEveryScanPosition() throws {
    let env = ProcessInfo.processInfo.environment
    guard let input = env["SSB_COLUMN_SOURCE"], let fixture = env["SSB_COLUMN_REFERENCE"] else {
      throw XCTSkip(
        "Set SSB_COLUMN_SOURCE and SSB_COLUMN_REFERENCE for the independent full-scan count comparison."
      )
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let cache = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: cache) }
    let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: cache).prepare(
      input: URL(fileURLWithPath: input))
    let dataset = try XCTUnwrap(catalog.datasets.first)
    let source = try MetalPairedRuntimeTANSResidentSource.load(
      source: Native4DSTEMIndexedSource.open(dataset: dataset), device: device)
    defer { source.releaseResidentStorage() }
    let directory = URL(fileURLWithPath: fixture)
    let pixels = try JSONDecoder().decode(
      [Int].self, from: Data(contentsOf: directory.appendingPathComponent("pixels.json")))
    let reference = try Data(contentsOf: directory.appendingPathComponent("columns.u32"))
    let bytes = pixels.count * dataset.scanRows * dataset.scanCols * 4
    XCTAssertEqual(reference.count, bytes)
    let output = try XCTUnwrap(device.makeBuffer(length: bytes, options: .storageModeShared))
    try source.readDetectorColumns(
      pixels: pixels, into: output, scanRows: dataset.scanRows, scanColumns: dataset.scanCols)
    let actual = Data(bytes: output.contents(), count: bytes)
    XCTAssertEqual(
      actual, reference,
      "Every source scan, requested order, duplicates and UInt32 counts must match.")
    XCTAssertThrowsError(
      try source.readDetectorColumns(pixels: [-1], into: output, scanRows: 512, scanColumns: 512))
    XCTAssertThrowsError(
      try source.readDetectorColumns(pixels: pixels, into: output, scanRows: 128, scanColumns: 128))
    source.releaseResidentStorage()
    XCTAssertThrowsError(
      try source.readDetectorColumns(pixels: pixels, into: output, scanRows: 512, scanColumns: 512))
  }
}
