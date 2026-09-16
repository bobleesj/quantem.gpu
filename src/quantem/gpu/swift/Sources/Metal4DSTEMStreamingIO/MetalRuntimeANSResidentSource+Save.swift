import CryptoKit
import Darwin
import Foundation
import Metal
import Native4DSTEMIO

extension MetalRuntimeANSResidentSource {
  /// Save unchanged encoded counts and indexes as a portable runtime snapshot.
  ///
  /// The destination is published atomically and never replaces an existing file.
  /// Call from a worker while the source remains exclusively owned and live.
  /// Example: `try source.saveSnapshot(to: destination)`.
  public func saveSnapshot(
    to destination: URL,
    calibrationOverrides: NativeQEMCalibration.Overrides? = nil,
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws {
    try requireLive()
    let overrides =
      try calibrationOverrides ?? NativeQEMCalibration.read(metadata: dataset.metadata ?? [:])
    try NativeQEMCalibration.validate(overrides)
    guard destination.pathExtension.lowercased() == "qem" || overrides.isEmpty else {
      throw Self.invalid(
        "Save as .qem to preserve calibration overrides; legacy .ans files do not carry them.")
    }
    guard !chunks.isEmpty, chunks.allSatisfy({ $0.spatial.count == 3 }) else {
      throw Self.invalid(
        "Reload the original with spatial indexes before saving a compressed copy.")
    }
    let manager = FileManager.default
    guard !manager.fileExists(atPath: destination.path) else {
      throw Self.invalid(
        "\(destination.lastPathComponent) already exists. Choose another name; the existing copy was kept."
      )
    }
    let directory = destination.deletingLastPathComponent()
    let bodyURL = directory.appendingPathComponent(".\(UUID().uuidString).body")
    let outputURL = directory.appendingPathComponent(".\(UUID().uuidString).partial")
    defer {
      try? manager.removeItem(at: bodyURL)
      try? manager.removeItem(at: outputURL)
    }
    guard manager.createFile(atPath: bodyURL.path, contents: nil),
      manager.createFile(atPath: outputURL.path, contents: nil)
    else {
      throw Self.invalid("Cannot write in \(directory.path). Choose a writable destination.")
    }
    let body = try FileHandle(forUpdating: bodyURL)
    defer { try? body.close() }
    let blockBytes = 64 << 20
    var digest = SHA256()
    var blockCount = 0
    var checksums = [String]()
    func append(_ bytes: Data) throws {
      var offset = 0
      while offset < bytes.count {
        if shouldCancel() {
          throw Self.invalid("Saving cancelled; no incomplete copy was published.")
        }
        let count = min(blockBytes - blockCount, bytes.count - offset)
        let part = bytes.subdata(in: offset..<offset + count)
        try body.write(contentsOf: part)
        digest.update(data: part)
        offset += count
        blockCount += count
        if blockCount == blockBytes {
          checksums.append(digest.finalize().map { String(format: "%02x", $0) }.joined())
          digest = SHA256()
          blockCount = 0
        }
      }
    }
    var table = [[String: Any]]()
    var cursor = 0
    for (number, chunk) in chunks.enumerated() {
      var spans = [[String: Int]]()
      let buffers = [chunk.payload, chunk.offsets, chunk.models] + chunk.spatial
      var readable = buffers
      if buffers.contains(where: { $0.storageMode == .private }) {
        guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
        else {
          throw Self.invalid("Cannot read compressed storage for saving.")
        }
        for (index, buffer) in buffers.enumerated() where buffer.storageMode == .private {
          readable[index] = try Self.sharedBuffer(
            device: device, bytes: buffer.length, label: "compressed file staging")
          blit.copy(
            from: buffer, sourceOffset: 0, to: readable[index], destinationOffset: 0,
            size: buffer.length)
        }
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
          throw Self.invalid("Compressed storage transfer failed; retry saving.")
        }
      }
      let blocks = (chunk.scanCount + 511) / 512
      let streams = blocks * shape[2] * shape[3]
      let fields =
        ((shape[2] + 7) / 8) * ((shape[3] + 7) / 8)
        + ((shape[2] + 31) / 32) * ((shape[3] + 31) / 32)
      let fieldStreams = blocks * fields
      let lengths = [
        Int(readable[1].contents().assumingMemoryBound(to: UInt32.self)[streams]),
        (streams + 1) * 4, streams,
        Int(readable[4].contents().assumingMemoryBound(to: UInt64.self)[fieldStreams]) * 4,
        (fieldStreams + 1) * 8, fieldStreams,
      ]
      for (index, pair) in zip(readable, [1, 4, 1, 4, 8, 1]).enumerated() {
        let (buffer, itemBytes) = pair
        let length = lengths[index]
        guard length <= buffer.length else {
          throw Self.invalid("Incomplete resident storage; reload the original before saving.")
        }
        let aligned = (cursor + 7) & ~7
        try append(Data(repeating: 0, count: aligned - cursor))
        spans.append(["offset": aligned, "count": length / itemBytes])
        try append(Data(bytesNoCopy: buffer.contents(), count: length, deallocator: .none))
        cursor = aligned + length
      }
      table.append(["first": chunk.firstScan, "scans": chunk.scanCount, "arrays": spans])
      progress(number + 1, chunks.count)
    }
    if blockCount > 0 {
      checksums.append(digest.finalize().map { String(format: "%02x", $0) }.joined())
    }
    var packedValid = [UInt8](repeating: 0, count: (validPixels.count + 7) / 8)
    for (pixel, valid) in validPixels.enumerated() where valid != 0 {
      packedValid[pixel / 8] |= 1 << (7 - pixel % 8)
    }
    var metadata: [String: Any] = [
      "source_path": dataset.masterPath ?? "", "source_shape": shape,
      "file_counts_exact": true, "lossless_exact": true, "median_correction_applied": false,
      "source_metadata": dataset.metadata ?? [:],
      "axis_order": ["scan_row", "scan_col", "detector_row", "detector_col"],
    ]
    if let date = dataset.acquisitionDate { metadata["acquisition_date"] = date }
    if let scan = dataset.sourceScanCalibration {
      metadata["scan_sampling_A"] = [scan.rowSamplingAngstrom, scan.columnSamplingAngstrom]
    }
    if let row = dataset.kPixelSizeRow, let col = dataset.kPixelSizeCol,
      dataset.kPixelUnit == "1/angstrom"
    {
      metadata["detector_sampling_inv_A"] = [row, col]
    }
    if let row = dataset.kPixelSizeRow, let col = dataset.kPixelSizeCol,
      let unit = dataset.kPixelUnit
    {
      metadata["detector_sampling"] = [row, col]
      metadata["detector_sampling_unit"] = unit
    }
    if let text = dataset.metadata?["electron_microscope/electron_source/accelerating_voltage"],
      let voltage = Double(text)
    {
      metadata["voltage_kV"] =
        dataset.metadata?["electron_microscope/electron_source/accelerating_voltage@units"] == "V"
        ? voltage / 1000 : voltage
    }
    for name in ["camera_model", "camera_id", "acquisition_processing"] {
      if let value = dataset.metadata?[name] { metadata[name] = value }
    }
    let origins = (1...4).reversed().compactMap { axis in
      dataset.metadata?["dm4.ImageData.Calibrations.Dimension.\(axis).Origin"].flatMap(Double.init)
    }
    if origins.count == 4 { metadata["pixel_origin"] = origins }
    var header: [String: Any] = [
      "version": 1, "profile": "runtime-column-rans-spatial-v2", "interval": 512,
      "shape": shape, "dtype": logicalDtype == .uint8 ? "uint8" : "uint16",
      "valid": packedValid.map { String(format: "%02x", $0) }.joined(),
      "chunks": table, "bytes": cursor, "sha256": checksums, "metadata": metadata,
    ]
    let qem = destination.pathExtension.lowercased() == "qem"
    if qem {
      header["container"] = NativeQEMMetadata.container
      header["container_version"] = 1
      header["codec"] = "runtime-column-rans-spatial-v2"
      header["scientific_metadata"] = try NativeQEMCalibration.applying(
        overrides,
        to: NativeQEMMetadata.acquisition(dataset))
    }
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    guard json.count <= 16 << 20 else {
      throw Self.invalid("Source metadata exceeds the compressed header limit.")
    }
    var prefix = qem ? NativeQEMMetadata.magic : Data("QGPUSTRM".utf8)
    for size in [json.count, 56 + json.count] {
      var value = UInt64(size).littleEndian
      withUnsafeBytes(of: &value) { prefix.append(contentsOf: $0) }
    }
    prefix.append(contentsOf: SHA256.hash(data: json))
    prefix.append(json)
    let output = try FileHandle(forWritingTo: outputURL)
    defer { try? output.close() }
    try output.write(contentsOf: prefix)
    try body.seek(toOffset: 0)
    while let bytes = try body.read(upToCount: blockBytes), !bytes.isEmpty {
      if shouldCancel() {
        throw Self.invalid("Saving cancelled; no incomplete copy was published.")
      }
      try output.write(contentsOf: bytes)
    }
    try output.synchronize()
    if shouldCancel() { throw Self.invalid("Saving cancelled; no incomplete copy was published.") }
    guard link(outputURL.path, destination.path) == 0 else {
      throw Self.invalid(
        "Cannot publish \(destination.lastPathComponent): \(String(cString: strerror(errno))). Choose another destination."
      )
    }
  }
}
