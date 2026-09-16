import Foundation
import CryptoKit
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Real acquisition workflow: original counts -> compressed copy -> exact products.
@main
struct CameraCheck {
  static func require(_ condition: Bool, _ message: String) throws {
    guard condition else {
      throw NSError(domain: "CameraCheck", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
  }
  static func checkCameraMetadataPlacement() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let body = Data(repeating: 0, count: 58)
    let arrays = [(0, 0), (0, 5), (24, 4), (32, 0), (32, 3), (56, 2)]
      .map { ["offset": $0.0, "count": $0.1] }
    let cases: [([String: Any], String?)] = [
      (["source_kind": "digitalmicrograph", "camera_model": "K3"], "K3 DM4"),
      (["source_metadata": ["dm4.ImageTags.Acquisition.Device.Source Model": "K3"]], "K3 DM4"),
      (["source_kind": "digitalmicrograph", "camera_model": "Other camera"], "DigitalMicrograph DM4"),
      (["source_kind": "digitalmicrograph"], "DigitalMicrograph DM4"),
      (["source_kind": "hdf5", "camera_model": "K3"], nil)
    ]
    for (index, item) in cases.enumerated() {
      let header: [String: Any] = ["version": 1, "profile": "runtime-column-rans-spatial-v2",
        "interval": 512, "shape": [1, 2, 2, 2], "dtype": "uint8", "valid": "f0",
        "chunks": [["first": 0, "scans": 2, "arrays": arrays]], "bytes": body.count,
        "sha256": [SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined()],
        "metadata": item.0]
      let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
      var file = Data("QGPUSTRM".utf8)
      for length in [json.count, json.count + 56] {
        var value = UInt64(length).littleEndian
        withUnsafeBytes(of: &value) { file.append(contentsOf: $0) }
      }
      file.append(contentsOf: SHA256.hash(data: json)); file.append(json); file.append(body)
      let url = directory.appendingPathComponent("case-\(index).ans")
      try file.write(to: url)
      let restored = try NativeANSSnapshot(url: url)
      try require(restored.dataset.metadata?["sourceFormat"] == item.1,
        "Wrong camera classification for metadata placement \(index)")
    }
    print("PASS camera identity: native/CUDA metadata, generic DM4 and non-DM4 sources")
  }
  static func main() throws {
    setbuf(stdout, nil)
    guard CommandLine.arguments.count == 3 || CommandLine.arguments.count == 4,
      let device = MTLCreateSystemDefaultDevice()
    else {
      fatalError("Usage: k3-camera-check original.dm4 existing.ans [new-copy.ans]; Metal required")
    }
    try checkCameraMetadataPlacement()
    let original = try NativeDM4Source(url: URL(fileURLWithPath: CommandLine.arguments[1]))
    let itemBytes = original.dataset.sourceDtype == "uint8" ? 1 : 2
    try require(original.dataset.sourceBytes == original.fileBytes, "Original source size differs")
    print("source_format=\(original.dataset.metadata?["sourceFormat"] ?? "unknown") camera=\(original.dataset.metadata?["camera_model"] ?? "unknown")")
    var snapshotURL = URL(fileURLWithPath: CommandLine.arguments[2])
    if CommandLine.arguments.count == 4 {
      snapshotURL = URL(fileURLWithPath: CommandLine.arguments[3])
      let started = CFAbsoluteTimeGetCurrent()
      let source = try MetalRuntimeANSResidentSource.load(camera: original, device: device)
      print("original_load_seconds=\(CFAbsoluteTimeGetCurrent() - started)")
      let saveStarted = CFAbsoluteTimeGetCurrent()
      try source.saveSnapshot(to: snapshotURL)
      print("save_seconds=\(CFAbsoluteTimeGetCurrent() - saveStarted)")
      source.releaseResidentStorage()
    }
    let snapshot = try NativeANSSnapshot(url: snapshotURL)
    try require(snapshot.dataset.sourceBytes == snapshot.dataStart + snapshot.bodyBytes,
      "Compressed source size differs")
    try require(snapshot.shape == original.shape, "Shape differs")
    try require(snapshot.dtype == original.dataset.sourceDtype, "Dtype differs")
    for key in ["sourceFormat", "sourceFormatVersion", "camera_model", "camera_id", "acquisition_processing"] {
      try require(snapshot.dataset.metadata?[key] == original.dataset.metadata?[key],
        "Reopened camera metadata differs: \(key)")
    }
    try require(snapshot.dataset.acquisitionDate == original.dataset.acquisitionDate,
      "Acquisition date differs")
    try require(
      snapshot.dataset.sourceScanCalibration?.rowSamplingAngstrom
        == original.dataset.sourceScanCalibration?.rowSamplingAngstrom, "Row calibration differs")
    try require(
      snapshot.dataset.sourceScanCalibration?.columnSamplingAngstrom
        == original.dataset.sourceScanCalibration?.columnSamplingAngstrom,
      "Column calibration differs")
    try require(
      snapshot.dataset.kPixelSizeRow == original.dataset.kPixelSizeRow,
      "Detector row calibration differs")
    try require(
      snapshot.dataset.kPixelSizeCol == original.dataset.kPixelSizeCol,
      "Detector column calibration differs")
    if CommandLine.arguments.count == 4
      || ProcessInfo.processInfo.environment["K3_VERIFY_NATIVE_METADATA"] == "1"
    {
      guard let storedMetadata = snapshot.metadata["source_metadata"] as? [String: String] else {
        throw NSError(
          domain: "CameraCheck", code: 2,
          userInfo: [NSLocalizedDescriptionKey: "Missing original metadata"])
      }
      for (key, value) in original.dataset.metadata ?? [:] {
        try require(storedMetadata[key] == value, "Lost original metadata: \(key)")
      }
    }
    let reopenTrials = min(20, max(0, Int(ProcessInfo.processInfo.environment["K3_REOPEN_TRIALS"] ?? "0") ?? 0))
    for trial in 0..<reopenTrials {
      let start = CFAbsoluteTimeGetCurrent()
      let inspected = try NativeANSSnapshot(url: snapshotURL)
      let inspectedAt = CFAbsoluteTimeGetCurrent()
      let resident = try MetalRuntimeANSResidentSource.load(snapshot: inspected, device: device)
      print("trial=\(trial) header_seconds=\(inspectedAt - start) header_to_resident_seconds=\(CFAbsoluteTimeGetCurrent() - start)")
      resident.releaseResidentStorage()
    }
    let started = CFAbsoluteTimeGetCurrent()
    let testOriginal = ProcessInfo.processInfo.environment["K3_VERIFY_ORIGINAL"] == "1"
    let source = try testOriginal
      ? MetalRuntimeANSResidentSource.load(camera: original, device: device)
      : MetalRuntimeANSResidentSource.load(snapshot: snapshot, device: device)
    defer { source.releaseResidentStorage() }
    print(
      "source=\(testOriginal ? "original" : "snapshot") load_seconds=\(CFAbsoluteTimeGetCurrent() - started) resident_bytes=\(source.residentBytes)"
    )
    let series = try MetalRuntimeANSSeries(sources: [source])
    defer { series.release() }
    let shape = original.shape
    let pixels = shape[2] * shape[3]
    let scans = shape[0] * shape[1]
    let positions = Array(
      Set([0, 1, 511, 512, scans / 2, scans - 2, scans - 1].filter { $0 < scans })
    ).sorted()
    let file = try FileHandle(forReadingFrom: original.url)
    defer { try? file.close() }
    var references = [[UInt32]]()
    for scan in positions {
      try file.seek(
        toOffset: UInt64(original.dataOffset + scan * pixels * itemBytes))
      let bytes = try file.read(upToCount: pixels * itemBytes)!
      let reference: [UInt32] = bytes.withUnsafeBytes { values in
        (0..<pixels).map {
          itemBytes == 1
            ? UInt32(values[$0])
            : UInt32(values.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self).littleEndian)
        }
      }
      let actual = try source.extractRawDiffraction(
        scanRow: scan / shape[1], scanColumn: scan % shape[1])
      try require(actual == reference, "Diffraction counts differ at scan \(scan)")
      references.append(reference)
    }
    for (name, inner, outer) in [("BF", 0.0, 126.0), ("ADF", 180.0, 360.0), ("DF", 126.0, 1222.0)] {
      for shift in [0.0, 0.31, 7.7, -2.1] {
        let mask: [UInt8] = (0..<pixels).map { pixel in
          let row = Double(pixel / shape[3]) - Double(shape[2] - 1) / 2 - shift
          let col = Double(pixel % shape[3]) - Double(shape[3] - 1) / 2 + shift
          return row * row + col * col >= inner * inner && row * row + col * col <= outer * outer
            ? 1 : 0
        }
        let result = try series.updatePriorityVirtualDetectorBuffer(mask: mask, priorityIndex: 0)
        let actual = result.buffer.contents().assumingMemoryBound(to: UInt32.self)
        for (index, scan) in positions.enumerated() {
          var expected: UInt64 = 0
          for pixel in 0..<pixels where mask[pixel] != 0 && snapshot.valid[pixel] != 0 {
            expected += UInt64(references[index][pixel])
          }
          try require(
            UInt64(actual[scan]) == expected, "\(name) mismatch at scan \(scan), shift \(shift)")
        }
      }
      print(
        "PASS \(name): exact raw-reference sums at \(positions.count) scan positions, four translated masks"
      )
    }
    print(
      "PASS native DM4/snapshot shape, dtype, calibration, selected raw counts and detector products"
    )
  }
}
