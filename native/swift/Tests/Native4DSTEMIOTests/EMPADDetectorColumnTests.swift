import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO
import XCTest

/// Detector-column extraction preserves the original IEEE words, selection order,
/// duplicates and all scan positions. The oracle indexes the synthetic file bytes.
final class EMPADDetectorColumnTests: XCTestCase {
  func testSignedFloatColumnsAndPaddingMatchOriginalBytes() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let raw = directory.appendingPathComponent("scan_x3_y2.raw")
    var values = [Float](repeating: 0, count: 6 * 130 * 128)
    for frame in 0..<6 {
      for pixel in 0..<(128 * 128) {
        values[frame * 130 * 128 + pixel] = Float((pixel + frame) % 17 - 8) * 0.125
      }
      values[frame * 130 * 128] = Float(bitPattern: 0x8000_0000)
      values[frame * 130 * 128 + 1] = Float(bitPattern: 0x7fc0_1234)
    }
    try values.withUnsafeBytes { Data($0) }.write(to: raw)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let source = try NativeEMPADSource.open(raw)
    let backgroundURL = directory.appendingPathComponent("background_x2_y2.raw")
    let backgroundValues = [Float](repeating: 0.25, count: 4 * 130 * 128)
    try backgroundValues.withUnsafeBytes { Data($0) }.write(to: backgroundURL)
    let background = try MetalEMPADBackground.load(
      NativeEMPADSource.open(backgroundURL, scanShape: (row: 2, col: 2)),
      device: device, memoryBudgetBytes: 256 << 20)
    for corrected in [false, true] {
      let resident = try MetalEMPADResidentSource.load(
        source, device: device, memoryBudgetBytes: 256 << 20,
        subtracting: corrected ? background : nil)
      let selected = [16383, 0, 900, 1, 900]
      let output = try XCTUnwrap(
        device.makeBuffer(length: selected.count * 4 * 4 * 4, options: .storageModeShared))
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      try resident.encodeDetectorColumns(
        pixels: selected, into: output, command: command, scanRows: 4, scanColumns: 4)
      command.commit()
      command.waitUntilCompleted()
      XCTAssertNil(command.error)
      let actual = output.contents().assumingMemoryBound(to: UInt32.self)
      for (slot, pixel) in selected.enumerated() {
        for row in 0..<4 {
          for column in 0..<4 {
            let stored = row < 2 && column < 3 ? values[(row * 3 + column) * 130 * 128 + pixel] : 0
            let expected = row < 2 && column < 3 && corrected ? stored - 0.25 : stored
            if corrected && expected.isNaN {
              XCTAssertTrue(Float(bitPattern: actual[slot * 16 + row * 4 + column]).isNaN)
            } else {
              XCTAssertEqual(actual[slot * 16 + row * 4 + column], expected.bitPattern)
            }
          }
        }
      }
      let rejected = try XCTUnwrap(queue.makeCommandBuffer())
      XCTAssertThrowsError(
        try resident.encodeDetectorColumns(pixels: [-1], into: output, command: rejected))
      XCTAssertThrowsError(
        try resident.encodeDetectorColumns(pixels: [16384], into: output, command: rejected))
      XCTAssertThrowsError(
        try resident.encodeDetectorColumns(
          pixels: [0], into: output, command: rejected, scanRows: 1))
    }
  }
  func testRecordedSearchGridRequiresMatchingFilenameAndLength() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let raw = directory.appendingPathComponent("scan_x3_y2.raw")
    try Data(count: 6 * 130 * 128 * 4).write(to: raw)
    let xml = directory.appendingPathComponent("acquisition.xml")
    let search =
      "<scan_parameters mode=\"search\"><scan_resolution_y>2</scan_resolution_y><scan_resolution_x>3</scan_resolution_x></scan_parameters>"
    let header =
      "<root><raw_file filename=\"scan_x3_y2.raw\"/><scan_parameters mode=\"acquire\"><scan_resolution_y>2</scan_resolution_y><scan_resolution_x>2</scan_resolution_x></scan_parameters>"
    let calibration =
      "<iom_measurements><full_scan_field_of_view><x>1e-8</x><y>1e-8</y><scale_factor>1</scale_factor></full_scan_field_of_view></iom_measurements>"
    try (header + search + calibration + "</root>").write(
      to: xml, atomically: true, encoding: .utf8)
    let source = try NativeEMPADSource.open(xml)
    XCTAssertEqual(source.scanRows, 2)
    XCTAssertEqual(source.scanColumns, 3)
    XCTAssertNil(
      source.scanCalibration, "Do not infer calibration from an unspecified active scan mode")
    try (header + calibration + "</root>").write(to: xml, atomically: true, encoding: .utf8)
    XCTAssertThrowsError(try NativeEMPADSource.open(xml))
    try (header + search + calibration + "</root>").write(
      to: xml, atomically: true, encoding: .utf8)
    try Data(count: 4 * 130 * 128 * 4).write(to: raw)
    XCTAssertThrowsError(try NativeEMPADSource.open(xml))
  }

}
