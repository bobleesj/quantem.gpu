import CryptoKit
import Foundation

/// Versioned, checksummed native-count ANS snapshot shared with the CUDA loader.
public struct NativeANSSnapshot {
  public struct ArraySpan {
    public let offset: Int
    public let count: Int
    public let itemBytes: Int
    public var bytes: Int { count * itemBytes }
  }
  public struct Chunk {
    public let first: Int
    public let scans: Int
    public let arrays: [ArraySpan]
  }
  public let metadata: [String: Any]
  public let scientificMetadata: [String: Any]
  public let url: URL
  public let shape: [Int]
  public let dtype: String
  public let chunks: [Chunk]
  public let valid: [UInt8]
  public let bodyBytes: Int
  public let dataStart: Int
  public let checksums: [String]
  public let identity: String
  public let dataset: Native4DSTEMDataset
  public static let blockBytes = 64 << 20

  public static func matches(_ url: URL) -> Bool {
    guard let file = try? FileHandle(forReadingFrom: url) else { return false }
    defer { try? file.close() }
    return (try? file.read(upToCount: 8)) == NativeQEMMetadata.magic
  }

  public init(url: URL) throws {
    self.url = url
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    let prefix = try file.read(upToCount: 56) ?? Data()
    guard prefix.count == 56, prefix.prefix(8) == NativeQEMMetadata.magic else {
      throw Self.invalid("Choose a complete .qem data file.")
    }
    let rawLength = prefix.withUnsafeBytes {
      $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self).littleEndian
    }
    let rawStart = prefix.withUnsafeBytes {
      $0.loadUnaligned(fromByteOffset: 16, as: UInt64.self).littleEndian
    }
    guard rawLength <= 16 << 20, rawStart <= 56 + (16 << 20) else {
      throw Self.invalid("Invalid ANS header size.")
    }
    let length = Int(rawLength)
    dataStart = Int(rawStart)
    guard length > 0, length <= 16 << 20, dataStart == 56 + length else {
      throw Self.invalid("Invalid ANS header size; recopy the snapshot.")
    }
    let blob = try file.read(upToCount: length) ?? Data()
    let digest = SHA256.hash(data: blob)
    guard blob.count == length, Data(digest) == prefix.suffix(32),
      let header = try JSONSerialization.jsonObject(with: blob) as? [String: Any],
      header["version"] as? Int == 1,
      header["profile"] as? String == "runtime-column-rans-spatial-v2",
      header["interval"] as? Int == 512,
      let shape = header["shape"] as? [Int], shape.count == 4,
      shape.allSatisfy({ $0 > 0 && $0 < 1 << 24 }),
      let dtype = header["dtype"] as? String, ["uint8", "uint16"].contains(dtype),
      let validHex = header["valid"] as? String,
      let bodyBytes = header["bytes"] as? Int, bodyBytes > 0,
      let checksums = header["sha256"] as? [String],
      let entries = header["chunks"] as? [[String: Any]]
    else { throw Self.invalid("Invalid ANS header or checksum; recopy the snapshot.") }
    try NativeQEMMetadata.validate(header, shape: shape)
    scientificMetadata = header["scientific_metadata"] as? [String: Any] ?? [:]
    self.shape = shape
    self.dtype = dtype
    self.bodyBytes = bodyBytes
    self.checksums = checksums
    identity = digest.map { String(format: "%02x", $0) }.joined()
    let pixels = shape[2] * shape[3]
    guard pixels < Int(UInt32.max), shape[0] * shape[1] < Int(UInt32.max),
      shape[0] * shape[1] <= Int.max / pixels / (dtype == "uint8" ? 1 : 2)
    else {
      throw Self.invalid("ANS geometry exceeds addressable native counts.")
    }
    let leaves = ((shape[2] + 7) / 8) * ((shape[3] + 7) / 8)
    let fields = leaves + ((shape[2] + 31) / 32) * ((shape[3] + 31) / 32)
    guard validHex.count == ((pixels + 7) / 8) * 2 else {
      throw Self.invalid("Invalid ANS validity mask.")
    }
    let hex = Array(validHex.utf8)
    var validity = [UInt8]()
    validity.reserveCapacity(pixels)
    for index in stride(from: 0, to: hex.count, by: 2) {
      guard let byte = UInt8(String(decoding: hex[index..<index + 2], as: UTF8.self), radix: 16)
      else {
        throw Self.invalid("Invalid ANS validity encoding.")
      }
      for bit in 0..<8 where validity.count < pixels { validity.append((byte >> (7 - bit)) & 1) }
    }
    valid = validity
    var chunks = [Chunk]()
    var cursor = 0
    var first = 0
    let sizes = [1, 4, 1, 4, 8, 1]
    for entry in entries {
      guard entry["first"] as? Int == first, let scans = entry["scans"] as? Int,
        scans > 0, scans <= shape[0] * shape[1] - first,
        let arrays = entry["arrays"] as? [[String: Any]], arrays.count == 6
      else {
        throw Self.invalid("Invalid ANS chunk coverage.")
      }
      let blocks = (scans + 511) / 512
      guard UInt64(blocks) * UInt64(pixels) < (1 << 32) / UInt64(2 * min(scans, 512) + 4) else {
        throw Self.invalid("ANS chunk offsets exceed uint32.")
      }
      let expected = [
        -1, blocks * pixels + 1, blocks * pixels, -1, blocks * fields + 1, blocks * fields,
      ]
      var spans = [ArraySpan]()
      for index in arrays.indices {
        cursor = (cursor + 7) & ~7
        guard let count = arrays[index]["count"] as? Int, count >= 0,
          count <= bodyBytes / sizes[index], arrays[index]["offset"] as? Int == cursor,
          expected[index] < 0 || count == expected[index],
          count * sizes[index] <= bodyBytes - cursor
        else { throw Self.invalid("Invalid ANS array bounds.") }
        spans.append(ArraySpan(offset: cursor, count: count, itemBytes: sizes[index]))
        cursor += count * sizes[index]
      }
      chunks.append(Chunk(first: first, scans: scans, arrays: spans))
      first += scans
    }
    let fileBytes = try file.seekToEnd()
    guard first == shape[0] * shape[1], cursor == bodyBytes,
      fileBytes == UInt64(dataStart + bodyBytes),
      checksums.count == (bodyBytes + Self.blockBytes - 1) / Self.blockBytes
    else {
      throw Self.invalid("Incomplete ANS snapshot; recopy the complete file.")
    }
    self.chunks = chunks
    var metadata = header["metadata"] as? [String: Any] ?? [:]
    if scientificMetadata["schema"] as? String == NativeQEMMetadataUnits.schema {
      for field in [
        "scan_sampling_A", "detector_sampling", "detector_sampling_inv_A",
        "detector_sampling_unit", "voltage_kV",
      ] { metadata.removeValue(forKey: field) }
      metadata.merge(
        try NativeQEMMetadataUnits.recordedMetadata(scientificMetadata),
        uniquingKeysWith: { _, publicValue in publicValue })
    }
    self.metadata = metadata
    let scan = metadata["scan_sampling_A"] as? [Double]
    let detector =
      metadata["detector_sampling"] as? [Double]
      ?? metadata["detector_sampling_inv_A"] as? [Double]
    let detectorUnit = metadata["detector_sampling_unit"] as? String ?? "1/angstrom"
    let calibration =
      scan?.count == 2
      ? Native4DSTEMScanCalibration(
        rowSamplingAngstrom: scan![0], columnSamplingAngstrom: scan![1],
        origin: .sourceMetadata, evidence: "ANS snapshot original acquisition calibration") : nil
    var nativeMetadata = metadata["source_metadata"] as? [String: String] ?? [:]
    if scientificMetadata["schema"] as? String == NativeQEMMetadataUnits.schema {
      nativeMetadata = try NativeQEMMetadataUnits.microscopeMetadata(scientificMetadata)
    }
    if let original = metadata["original_source_identity_sha256"] as? String {
      guard original.count == 64, original.allSatisfy(\.isHexDigit) else {
        throw Self.invalid("Invalid original acquisition identity; re-export the source.")
      }
      // Keep container identity separate for cache safety when calibration changes.
      nativeMetadata["originalSourceIdentity"] = original
    }
    nativeMetadata[NativeQEMCalibration.metadataKey] = String(
      decoding: try NativeQEMCalibration.encoded(
        NativeQEMCalibration.read(scientific: scientificMetadata)), as: UTF8.self)
    // CUDA/Python snapshots store normalized camera fields beside source_metadata;
    // native snapshots retain them inside it. Both describe the same acquisition.
    for key in ["camera_model", "camera_id", "acquisition_processing"]
    where nativeMetadata[key] == nil {
      nativeMetadata[key] = metadata[key] as? String
    }
    if nativeMetadata["sourceKind"] == "digitalmicrograph"
      || metadata["source_kind"] as? String == "digitalmicrograph"
      || nativeMetadata.keys.contains(where: { $0.hasPrefix("dm4.") })
    {
      NativeDM4Source.identifyCamera(in: &nativeMetadata)
    }
    nativeMetadata.merge(
      ["storageSchema": "runtime-column-rans-spatial-v2", "sourceKind": "ans-snapshot"],
      uniquingKeysWith: { _, new in new })
    if let voltage = metadata["voltage_kV"] as? Double, voltage.isFinite, voltage > 0 {
      nativeMetadata["electron_microscope/electron_source/accelerating_voltage"] = String(voltage)
      nativeMetadata["electron_microscope/electron_source/accelerating_voltage@units"] = "kV"
    }
    dataset = Native4DSTEMDataset(
      id: identity, label: url.lastPathComponent, masterPath: url.path,
      dataFiles: [url.path], indexFiles: [], scanRows: shape[0], scanCols: shape[1],
      detectorRows: shape[2], detectorCols: shape[3], sourceDtype: dtype,
      sourceBytes: Int(fileBytes),
      badPixelIndices: validity.indices.filter { validity[$0] == 0 },
      kPixelSizeRow: detector?.first, kPixelSizeCol: detector?.last,
      kPixelUnit: detector == nil ? nil : detectorUnit,
      acquisitionDate: metadata["acquisition_date"] as? String,
      metadata: nativeMetadata,
      schemaIdentity: "quantem.qem/v1", sourceIdentitySHA256: identity,
      sourceScanCalibration: calibration)
  }

  /// Authenticate every compressed byte before it can be consumed by a GPU kernel.
  public func verifiedMapping() throws -> Data {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    guard data.count == dataStart + bodyBytes else {
      throw Self.invalid("ANS file changed during loading.")
    }
    let expected = checksums
    let start = dataStart
    let length = bodyBytes
    let workers =
      ProcessInfo.processInfo.environment["QGPU_K3_SERIAL_HASH"] == "1" ? 1 : min(8, expected.count)
    try data.withUnsafeBytes { bytes in
      // The immutable mapping outlives synchronous workers. Only failure writes
      // need locking; independent checksum blocks never share mutable data.
      nonisolated(unsafe) let pointer = bytes.baseAddress!
      nonisolated(unsafe) var failedBlock: Int?
      let lock = NSLock()
      DispatchQueue.concurrentPerform(iterations: workers) { worker in
        for number in stride(from: worker, to: expected.count, by: workers) {
          let offset = number * Self.blockBytes
          let count = min(Self.blockBytes, length - offset)
          let view = Data(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: pointer.advanced(by: start + offset)),
            count: count, deallocator: .none)
          let digest = SHA256.hash(data: view).map { String(format: "%02x", $0) }.joined()
          if digest != expected[number] {
            lock.lock()
            failedBlock = number
            lock.unlock()
          }
        }
      }
      if let failedBlock {
        throw Self.invalid("ANS checksum mismatch in block \(failedBlock); recopy the file.")
      }
    }
    return data
  }

  static func invalid(_ message: String) -> NSError {
    NSError(domain: "NativeANSSnapshot", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
  }
}
