import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import XCTest

@testable import Native4DSTEMIO

final class NativeCameraSourceTests: XCTestCase {
  func testCameraIdentityUsesRecordedModelNotDM4Extension() {
    var generic = ["camera_model": "Other camera"]
    NativeDM4Source.identifyCamera(in: &generic)
    XCTAssertEqual(generic["sourceFormat"], "DigitalMicrograph DM4")
    var recorded = ["dm4.ImageTags.Acquisition.Device.Source Model": "K3"]
    NativeDM4Source.identifyCamera(in: &recorded)
    XCTAssertEqual(recorded["sourceFormat"], "K3 DM4")
    XCTAssertEqual(recorded["camera_model"], "K3")
    XCTAssertEqual(recorded["sourceFormatVersion"], "digitalmicrograph/native-counts-v1")
  }

  func testSnapshotChecksEncodedBytesBeforeConsumption() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".qem")
    defer { try? FileManager.default.removeItem(at: url) }
    var body = Data(repeating: 0, count: 58)
    body.replaceSubrange(24..<28, with: [UInt8](repeating: 253, count: 4))
    let checksum = SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined()
    let arrays = [(0, 0), (0, 5), (24, 4), (32, 0), (32, 3), (56, 2)].map {
      ["offset": $0.0, "count": $0.1]
    }
    let header: [String: Any] = [
      "version": 1, "profile": "runtime-column-rans-spatial-v2",
      "interval": 512, "shape": [1, 2, 2, 2], "dtype": "uint8", "valid": "f0",
      "chunks": [["first": 0, "scans": 2, "arrays": arrays]], "bytes": body.count,
      "sha256": [checksum],
      "metadata": ["scan_sampling_A": [2.5, 2.5], "detector_sampling_inv_A": [0.025, 0.025]],
      "container": "quantem.qem", "container_version": 1,
      "codec": "runtime-column-rans-spatial-v2",
      "scientific_metadata": [
        "schema": "quantem.scientific-metadata/1",
        "source_metadata": [String: Any](),
        "source_metadata_coverage": "unknown",
        "axes": [
          ["name": "scan_row", "size": 1], ["name": "scan_column", "size": 2],
          ["name": "detector_row", "size": 2], ["name": "detector_column", "size": 2],
        ],
      ],
    ]
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var file = Data(NativeQEMMetadata.magic)
    for size in [json.count, json.count + 56] {
      var value = UInt64(size).littleEndian
      withUnsafeBytes(of: &value) { file.append(contentsOf: $0) }
    }
    file.append(contentsOf: SHA256.hash(data: json))
    file.append(json)
    file.append(body)
    try file.write(to: url)
    let snapshot = try NativeANSSnapshot(url: url)
    XCTAssertEqual(snapshot.shape, [1, 2, 2, 2])
    XCTAssertEqual(snapshot.dataset.sourceScanCalibration?.rowSamplingAngstrom, 2.5)
    XCTAssertEqual(try snapshot.verifiedMapping().count, file.count)
    file[file.count - 1] ^= 1
    try file.write(to: url)
    XCTAssertThrowsError(try NativeANSSnapshot(url: url).verifiedMapping())
  }

  func testDM4KeepsNativeAxesCountsAndCalibration() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("camera.dm4")
    let shape = [2, 3, 64, 65]
    let pixels = shape[2] * shape[3]
    let counts = Data((0..<6 * pixels).map { UInt8(truncatingIfNeeded: $0) })
    func integer(_ value: UInt64, _ count: Int) -> Data {
      Data((0..<count).reversed().map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) })
    }
    func entry(_ name: String, _ kind: UInt8, _ bytes: Data) -> Data {
      let label = name.data(using: .isoLatin1)!
      return Data([kind]) + integer(UInt64(label.count), 2) + label
        + integer(UInt64(bytes.count), 8) + bytes
    }
    func group(_ name: String, _ entries: [Data]) -> Data {
      entry(name, 20, Data([1, 1]) + integer(UInt64(entries.count), 8) + entries.reduce(Data(), +))
    }
    func tag(_ name: String, _ types: [UInt64], _ bytes: Data) -> Data {
      entry(
        name, 21,
        Data("%%%%".utf8) + integer(UInt64(types.count), 8)
          + types.reduce(Data()) { $0 + integer($1, 8) } + bytes)
    }
    func scalar(_ name: String, _ type: UInt64, _ value: UInt32) -> Data {
      tag(name, [type], Data((0..<4).map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) }))
    }
    let calibration = (0..<4).map { axis in
      let unit = axis < 2 ? "1/nm" : "nm"
      let bytes = unit.data(using: .utf16LittleEndian)!
      return group(
        String(axis + 1),
        [
          scalar("Scale", 6, Float(0.25).bitPattern),
          scalar("Origin", 6, 0), tag("Units", [20, 4, UInt64(bytes.count / 2)], bytes),
        ])
    }
    let image = group(
      "1",
      [
        scalar("Field of View (µm)", 6, Float(0.25).bitPattern),
        group(
          "ImageData",
          [
            group("Calibrations", [group("Dimension", calibration)]), scalar("DataType", 5, 6),
            group(
              "Dimensions",
              shape.reversed().enumerated().map {
                scalar(String($0.offset + 1), 5, UInt32($0.element))
              }),
            tag("Data", [20, 10, UInt64(counts.count)], counts),
          ]),
      ])
    let body = Data([1, 1]) + integer(1, 8) + group("ImageList", [image])
    try (integer(4, 4) + integer(UInt64(body.count), 8) + integer(1, 4) + body).write(to: url)
    let alias = directory.appendingPathComponent("linked.dm4")
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: url)
    let source = try NativeDM4Source(url: alias)
    XCTAssertEqual(source.shape, shape)
    XCTAssertEqual(source.dataset.metadata?["sourceFormat"], "DigitalMicrograph DM4")
    XCTAssertEqual(source.dataset.metadata?["dm4.Field of View (µm)"], "0.25")
    XCTAssertEqual(source.dataset.sourceDtype, "uint8")
    XCTAssertEqual(source.dataset.sourceScanCalibration?.rowSamplingAngstrom, 2.5)
    XCTAssertEqual(source.dataset.kPixelSizeRow, 0.025)
    let original = try Data(contentsOf: url)
    XCTAssertEqual(source.dataset.sourceBytes, original.count)
    XCTAssertEqual(
      original.subdata(in: source.dataOffset..<source.dataOffset + source.payloadBytes), counts)
    try source.assertUnchanged()
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let resident = try MetalRuntimeANSResidentSource.load(camera: source, device: device)
    defer { resident.releaseResidentStorage() }
    let saved = directory.appendingPathComponent("camera.compressed.qem")
    try resident.saveSnapshot(to: saved)
    let snapshot = try NativeANSSnapshot(url: saved)
    XCTAssertEqual(snapshot.dataset.sourceBytes, snapshot.dataStart + snapshot.bodyBytes)
    XCTAssertEqual(snapshot.dataset.sourceScanCalibration?.rowSamplingAngstrom, 2.5)
    XCTAssertEqual(snapshot.dataset.metadata?["dm4.Field of View (µm)"], "0.25")
    let reopened = try MetalRuntimeANSResidentSource.load(snapshot: snapshot, device: device)
    defer { reopened.releaseResidentStorage() }
    for scan in 0..<6 {
      XCTAssertEqual(
        try reopened.extractRawDiffraction(scanRow: scan / 3, scanColumn: scan % 3),
        (scan * pixels..<scan * pixels + pixels).map { UInt32(UInt8(truncatingIfNeeded: $0)) })
    }
    let preserved = try Data(contentsOf: saved)
    XCTAssertThrowsError(try resident.saveSnapshot(to: saved))
    XCTAssertEqual(try Data(contentsOf: saved), preserved)
    let cancelled = directory.appendingPathComponent("cancelled.qem")
    XCTAssertThrowsError(try resident.saveSnapshot(to: cancelled, shouldCancel: { true }))
    XCTAssertFalse(FileManager.default.fileExists(atPath: cancelled.path))
  }
}
