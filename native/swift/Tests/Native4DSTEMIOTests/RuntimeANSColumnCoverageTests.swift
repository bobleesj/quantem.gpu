import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO
@testable import Native4DSTEMIO

final class RuntimeANSColumnCoverageTests: XCTestCase {
  func testColumnsCrossPartialChunksAndPreserveAllCountModels() throws {
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
      let scanRows = 129
      let scanColumns = 137
      let pixels = 64
      let scans = scanRows * scanColumns
      let raw = try XCTUnwrap(
        device.makeBuffer(length: scans * pixels * 2, options: .storageModeShared))
      let values = raw.contents().assumingMemoryBound(to: UInt16.self)
      for scan in 0..<scans {
        for pixel in 0..<pixels {
          let hash = UInt32(truncatingIfNeeded: scan &* 1_664_525 &+ pixel &* 1_013_904_223)
          let value: UInt16
          switch pixel % 8 {
          case 0: value = 0
          case 1: value = 7
          case 2: value = scan % 127 == 0 ? 3 : 0
          case 3: value = UInt16((hash >> 13) & 7)
          case 4: value = UInt16(truncatingIfNeeded: hash >> 8)
          case 5: value = scan % 41 == 0 ? 60000 : UInt16((hash >> 17) & 15)
          case 6: value = UInt16((hash >> 11) & 255)
          default: value = UInt16((hash >> 19) & 31)
          }
          values[scan * pixels + pixel] = value
        }
      }
      let dataset = Native4DSTEMDataset(
        id: "column-fixture", label: "Synthetic count models",
        masterPath: nil, dataFiles: [], indexFiles: [], scanRows: scanRows, scanCols: scanColumns,
        detectorRows: 8, detectorCols: 8, sourceDtype: "uint16", sourceBytes: raw.length,
        badPixelIndices: [1, 5], kPixelSizeRow: nil, kPixelSizeCol: nil, kPixelUnit: nil,
        acquisitionDate: nil, metadata: nil)
      let encoder = try RuntimeANSEncoder(
        device: device, pixels: pixels, bytesPerValue: 2,
        allocatedBefore: UInt64(device.currentAllocatedSize), maximumAdditionalBytes: nil)
      // More than 32 restart blocks, a partial block, then a nonzero-offset partial chunk.
      let split = 33 * 512 + 1
      try encoder.append(dense: raw, firstScan: 0, scanCount: split)
      let tail = try XCTUnwrap(
        device.makeBuffer(
          bytes: values.advanced(by: split * pixels),
          length: (scans - split) * pixels * 2, options: .storageModeShared))
      try encoder.append(dense: tail, firstScan: split, scanCount: scans - split)
      let built = try encoder.finish()
      let queue = try XCTUnwrap(device.makeCommandQueue())
      let source = try MetalRuntimeANSResidentSource(
        dataset: dataset, identity: "synthetic",
        built: built, totalSeconds: 0, device: device)
      defer { source.releaseResidentStorage() }
      var expectedSums = [UInt64](repeating: 0, count: pixels)
      for scan in 0..<scans {
        for pixel in 0..<pixels where !dataset.badPixelIndices.contains(pixel) {
          expectedSums[pixel] += UInt64(values[scan * pixels + pixel])
        }
      }
      XCTAssertEqual(try source.detectorColumnSums(), expectedSums)
      XCTAssertNil(source.detectorColumnsPipeline)
      XCTAssertNil(source.detectorColumnsFlatPipeline)
      for selected in [Array(0..<32), Array(32..<64), [63, 5, 1, 63, 0]] {
        for padding in [0, 3] {
          let rows = scanRows + padding
          let columns = scanColumns + padding
          let output = try XCTUnwrap(
            device.makeBuffer(
              length: selected.count * rows * columns * 4,
              options: .storageModeShared))
          for includeFlagged in [false, true] {
            let commands = try XCTUnwrap(queue.makeCommandBuffer())
            try source.encodeDetectorColumns(
              pixels: selected, into: output, commands: commands,
              scanRows: rows, scanColumns: columns, includeFlaggedPixels: includeFlagged)
            XCTAssertEqual(source.detectorColumnChunks != nil, parallel == "1")
            if selected == Array(0..<32) && padding == 0 {
              XCTAssertNil(source.detectorColumnsPipeline)
              XCTAssertNotNil(source.detectorColumnsFlatPipeline)
            }
            commands.commit()
            commands.waitUntilCompleted()
            XCTAssertNil(commands.error)
            try source.validateDetectorColumnDecoding()
            let actual = output.contents().assumingMemoryBound(to: UInt32.self)
            // Compare contiguous integer arrays, without a per-pixel XCTest overhead.
            for (slot, pixel) in selected.enumerated() {
              var expected = [UInt32](repeating: 0, count: rows * columns)
              if includeFlagged || !dataset.badPixelIndices.contains(pixel) {
                for row in 0..<scanRows {
                  for column in 0..<scanColumns {
                    expected[row * columns + column] = UInt32(
                      values[(row * scanColumns + column) * pixels + pixel])
                  }
                }
              }
              XCTAssertEqual(
                Array(
                  UnsafeBufferPointer(
                    start: actual.advanced(by: slot * rows * columns),
                    count: rows * columns)), expected, "pixel \(pixel), padding \(padding)")
            }
          }
        }
      }
      XCTAssertEqual(try source.detectorColumnSums(), expectedSums)
    }
  }
}
