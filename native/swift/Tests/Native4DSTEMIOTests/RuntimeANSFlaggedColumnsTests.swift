import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO
@testable import Native4DSTEMIO

final class RuntimeANSFlaggedColumnsTests: XCTestCase {
  func testStoredColumnsPreserveFlagsCountsOrderPaddingAndReopening() throws {
    let previous = ProcessInfo.processInfo.environment["QUANTEM_METAL_PARALLEL_COLUMNS"]
    defer {
      if let previous {
        setenv("QUANTEM_METAL_PARALLEL_COLUMNS", previous, 1)
      } else {
        unsetenv("QUANTEM_METAL_PARALLEL_COLUMNS")
      }
    }
    for parallel in ["0", "1"] {
      setenv("QUANTEM_METAL_PARALLEL_COLUMNS", parallel, 1)
      let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: root) }
      for name in ["fixture_master", "fixture_data_000001"] {
        try FileManager.default.copyItem(
          at: XCTUnwrap(
            Bundle.module.url(forResource: name, withExtension: "h5", subdirectory: "Fixtures")),
          to: root.appendingPathComponent(name + ".h5"))
      }
      let catalog = try Native4DSTEMCatalogBuilder(
        cacheDirectory: root.appendingPathComponent("cache")
      )
      .prepare(input: root.appendingPathComponent("fixture_master.h5"))
      let dataset = try XCTUnwrap(catalog.datasets.first)
      let source = try MetalRuntimeANSResidentSource.load(
        source: Native4DSTEMIndexedSource.open(dataset: dataset), device: device,
        includeSpatialIndex: true)
      defer { source.releaseResidentStorage() }
      let saved = root.appendingPathComponent("original.qem")
      try source.saveSnapshot(to: saved)
      let before = try Data(contentsOf: saved)
      let reopened = try MetalRuntimeANSResidentSource.load(
        snapshot: NativeANSSnapshot(url: saved), device: device)
      defer { reopened.releaseResidentStorage() }
      for resident in [source, reopened] {
        let validity = resident.detectorValidityMask
        let flagged = try XCTUnwrap(validity.indices.first { validity[$0] == 0 })
        let valid = try XCTUnwrap(validity.indices.first { validity[$0] != 0 })
        let pixels = [flagged, valid, flagged]
        let rows = dataset.scanRows + 1
        let columns = dataset.scanCols + 2
        let output = try XCTUnwrap(
          device.makeBuffer(length: pixels.count * rows * columns * 4, options: .storageModeShared))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        var reference = [[UInt32]]()
        for scan in 0..<(dataset.scanRows * dataset.scanCols) {
          reference.append(
            try resident.extractRawDiffraction(
              scanRow: scan / dataset.scanCols, scanColumn: scan % dataset.scanCols))
        }
        XCTAssertTrue(
          reference.contains { $0[flagged] != 0 }, "Fixture must exercise stored flagged counts.")
        for includeFlagged in [false, true, false] {
          let command = try XCTUnwrap(queue.makeCommandBuffer())
          try resident.encodeDetectorColumns(
            pixels: pixels, into: output, commands: command,
            scanRows: rows, scanColumns: columns, includeFlaggedPixels: includeFlagged)
          command.commit()
          command.waitUntilCompleted()
          XCTAssertNil(command.error)
          try resident.validateDetectorColumnDecoding()
          let actual = output.contents().assumingMemoryBound(to: UInt32.self)
          for (slot, pixel) in pixels.enumerated() {
            for row in 0..<rows {
              for column in 0..<columns {
                let measured = row < dataset.scanRows && column < dataset.scanCols
                let expected: UInt32 =
                  measured && (includeFlagged || validity[pixel] != 0)
                  ? reference[row * dataset.scanCols + column][pixel] : 0
                XCTAssertEqual(actual[slot * rows * columns + row * columns + column], expected)
              }
            }
          }
          XCTAssertEqual(resident.detectorValidityMask, validity)
        }
      }
      XCTAssertEqual(try Data(contentsOf: saved), before)
    }
  }
}
