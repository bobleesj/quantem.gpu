import CryptoKit
import Foundation

/// Header-only selection of one calibrated native-count DigitalMicrograph image.
/// The original detector payload is never materialized by the metadata reader.
public struct NativeDM4Source {
  public let url: URL
  public let dataset: Native4DSTEMDataset
  public let shape: [Int]
  public let dataOffset: Int
  public let payloadBytes: Int
  public let fileBytes: Int
  public let modificationDate: Date?

  public init(url: URL) throws {
    let url = url.resolvingSymlinksInPath()
    self.url = url
    let parser = try DM4Tags(url: url)
    try parser.parse()
    var candidates = [(String, [Int])]()
    for key in parser.numbers.keys where key.hasSuffix(".ImageData.DataType") {
      let root = String(key.dropLast(".DataType".count))
      let prefix = root + ".Dimensions."
      let axes = parser.numbers.filter { $0.key.hasPrefix(prefix) }
        .sorted {
          (Int($0.key.dropFirst(prefix.count)) ?? -1) < (Int($1.key.dropFirst(prefix.count)) ?? -1)
        }
      if axes.count == 4 {
        guard
          axes.allSatisfy({
            $0.value.isFinite && $0.value > 0 && $0.value < Double(1 << 20)
              && $0.value.rounded() == $0.value
          })
        else {
          throw DM4Tags.invalid("Invalid DM4 dimensions.")
        }
        candidates.append((root, axes.map { Int($0.value) }.reversed()))
      }
    }
    guard candidates.count == 1 else {
      throw DM4Tags.invalid(
        "Choose a DM4 containing one calibrated 4D diffraction image; found \(candidates.count).")
    }
    let (root, shape) = candidates[0]
    guard shape.allSatisfy({ $0 > 0 && $0 < 1 << 20 }),
      let payload = parser.payloads[root + ".Data"],
      let dataType = parser.numbers[root + ".DataType"], [6.0, 10.0].contains(dataType),
      parser.littleEndian
    else {
      throw DM4Tags.invalid("DM4 Metal loading requires little-endian native uint8/uint16 counts.")
    }
    let bytes = dataType == 6 ? 1 : 2
    var count = 1
    for dimension in shape {
      let product = count.multipliedReportingOverflow(by: dimension)
      guard !product.overflow else {
        throw DM4Tags.invalid("DM4 dimensions exceed addressable counts.")
      }
      count = product.partialValue
    }
    guard count <= Int.max / bytes, payload.1 == count * bytes,
      payload.0 <= parser.data.count - payload.1
    else { throw DM4Tags.invalid("The DM4 detector payload is incomplete.") }
    self.shape = shape
    dataOffset = payload.0
    payloadBytes = payload.1
    fileBytes = parser.data.count
    modificationDate = try url.resourceValues(forKeys: [.contentModificationDateKey])
      .contentModificationDate
    let calibrationPrefix = root + ".Calibrations.Dimension."
    let scales = parser.numbers.filter {
      $0.key.hasPrefix(calibrationPrefix) && $0.key.hasSuffix(".Scale")
    }
    .sorted { $0.key < $1.key }.map(\.value).reversed()
    let units = parser.strings.filter {
      $0.key.hasPrefix(calibrationPrefix) && $0.key.hasSuffix(".Units")
    }
    .sorted { $0.key < $1.key }.map(\.value).reversed()
    let sampling = Array(scales)
    let axisUnits = Array(units)
    guard sampling.count == 4, axisUnits.count == 4,
      axisUnits[2] == "1/nm", axisUnits[3] == "1/nm"
    else {
      throw DM4Tags.invalid(
        "DM4 requires two calibrated reciprocal-nanometer detector axes; re-export with native axis calibration."
      )
    }
    let factors: [String: Double] = ["nm": 10, "um": 10000, "µm": 10000, "Å": 1, "A": 1]
    let scan: Native4DSTEMScanCalibration?
    if let row = factors[axisUnits[0]], let col = factors[axisUnits[1]] {
      scan = Native4DSTEMScanCalibration(
        rowSamplingAngstrom: sampling[0] * row,
        columnSamplingAngstrom: sampling[1] * col, origin: .sourceMetadata,
        evidence: "DigitalMicrograph calibrated scan axes")
    } else {
      scan = nil
    }
    let identityData = try JSONSerialization.data(
      withJSONObject: [
        "path": url.standardizedFileURL.path, "bytes": fileBytes, "offset": dataOffset,
        "shape": shape, "sampling": sampling, "units": axisUnits,
        "modified": modificationDate?.timeIntervalSince1970 ?? 0,
      ], options: [.sortedKeys])
    let identity = SHA256.hash(data: identityData).map { String(format: "%02x", $0) }.joined()
    var metadata = [
      "storageSchema": "digitalmicrograph/native-counts-v1", "sourceKind": "digitalmicrograph",
      "axisOrder": "scan_row,scan_col,detector_row,detector_col",
    ]
    let imageRoot = String(root.dropLast(".ImageData".count))
    let prefix = imageRoot + "."
    for (key, value) in parser.retained where key.hasPrefix(prefix) {
      metadata["dm4." + String(key.dropFirst(prefix.count))] = value
    }
    metadata["camera_model"] =
      parser.strings[imageRoot + ".ImageTags.Acquisition.Device.Source Model"]
    metadata["camera_id"] = parser.strings[imageRoot + ".ImageTags.Acquisition.Device.Source ID"]
    metadata["acquisition_processing"] =
      parser.strings[imageRoot + ".ImageTags.Acquisition.Parameters.High Level.Processing"]
    metadata["median_correction_applied"] = "false"
    if let voltage = parser.numbers[imageRoot + ".ImageTags.Microscope Info.Voltage"],
      voltage.isFinite, voltage > 0
    {
      metadata["electron_microscope/electron_source/accelerating_voltage"] = String(voltage)
      metadata["electron_microscope/electron_source/accelerating_voltage@units"] = "V"
    }
    Self.identifyCamera(in: &metadata)
    dataset = Native4DSTEMDataset(
      id: identity, label: url.lastPathComponent,
      masterPath: url.path, dataFiles: [url.path], indexFiles: [],
      scanRows: shape[0], scanCols: shape[1], detectorRows: shape[2], detectorCols: shape[3],
      sourceDtype: bytes == 1 ? "uint8" : "uint16", sourceBytes: fileBytes, badPixelIndices: [],
      kPixelSizeRow: sampling[2] / 10, kPixelSizeCol: sampling[3] / 10,
      kPixelUnit: "1/angstrom",
      acquisitionDate: parser.strings[imageRoot + ".ImageTags.SI.Acquisition.Date"],
      metadata: metadata,
      schemaIdentity: "DM4", sourceIdentitySHA256: identity, sourceScanCalibration: scan)
  }

  public func assertUnchanged() throws {
    let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
    guard values.fileSize == fileBytes, values.contentModificationDate == modificationDate else {
      throw DM4Tags.invalid("DM4 changed during loading; reopen the completed acquisition.")
    }
  }

  /// Classify the detector from recorded device tags, never the file name or size.
  /// Also used when restoring a DigitalMicrograph acquisition from a snapshot.
  static func identifyCamera(in metadata: inout [String: String]) {
    for (field, tag) in [
      ("camera_model", "ImageTags.Acquisition.Device.Source Model"),
      ("camera_id", "ImageTags.Acquisition.Device.Source ID"),
      ("acquisition_processing", "ImageTags.Acquisition.Parameters.High Level.Processing"),
    ] where metadata[field] == nil {
      metadata[field] = metadata["dm4." + tag]
    }
    let model = metadata["camera_model"]?.trimmingCharacters(in: .whitespacesAndNewlines)
    metadata["sourceFormat"] =
      model?.uppercased() == "K3"
      ? "K3 DM4" : "DigitalMicrograph DM4"
    metadata["sourceFormatVersion"] = "digitalmicrograph/native-counts-v1"
  }
}

private final class DM4Tags {
  let data: Data
  var cursor = 0
  var littleEndian = false
  var numbers = [String: Double]()
  var strings = [String: String]()
  var payloads = [String: (Int, Int)]()
  var retained = [String: String]()
  init(url: URL) throws { data = try Data(contentsOf: url, options: .alwaysMapped) }
  static func invalid(_ message: String) -> NSError {
    NSError(domain: "NativeDM4Source", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
  }
  func bytes(_ count: Int) throws -> Data {
    guard count >= 0, cursor <= data.count - count else {
      throw Self.invalid("Truncated DM4 tags.")
    }
    defer { cursor += count }
    return data.subdata(in: cursor..<cursor + count)
  }
  func integer(_ count: Int) throws -> UInt64 {
    try bytes(count).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
  }
  func parse() throws {
    guard try integer(4) == 4 else { throw Self.invalid("This reader requires a DM4 file.") }
    _ = try integer(8)
    littleEndian = try integer(4) == 1
    try group([], limit: data.count, depth: 0)
  }
  func group(_ path: [String], limit: Int, depth: Int) throws {
    guard depth < 64 else { throw Self.invalid("DM4 tags are nested too deeply.") }
    _ = try bytes(2)
    let count = try integer(8)
    guard count <= 1_000_000 else { throw Self.invalid("Invalid DM4 tag count.") }
    for index in 0..<Int(count) {
      let kind = try integer(1)
      let nameBytes = try integer(2)
      let labelBytes = try bytes(Int(nameBytes))
      let label =
        String(data: labelBytes, encoding: .utf8)
        ?? String(data: labelBytes, encoding: .isoLatin1) ?? ""
      let name = label.isEmpty ? String(index + 1) : label
      let size = try integer(8)
      guard size <= UInt64(max(0, limit - cursor)) else {
        throw Self.invalid("DM4 tag exceeds its container.")
      }
      let end = cursor + Int(size)
      let keys = path + [name]
      let key = keys.joined(separator: ".")
      if kind == 20 {
        try group(keys, limit: end, depth: depth + 1)
      } else if kind == 21 {
        guard try bytes(4) == Data("%%%%".utf8) else { throw Self.invalid("Invalid DM4 data tag.") }
        let infoCount = try integer(8)
        guard infoCount > 0, infoCount < 4096 else {
          throw Self.invalid("Invalid DM4 type descriptor.")
        }
        var info = [UInt64]()
        for _ in 0..<infoCount { info.append(try integer(8)) }
        if key.hasSuffix(".ImageData.Data") {
          payloads[key] = (cursor, end - cursor)
        } else if info.count == 1 {
          let sizes: [UInt64: Int] = [
            2: 2, 3: 4, 4: 2, 5: 4, 6: 4, 7: 8, 8: 1, 9: 1, 10: 1, 11: 8, 12: 8,
          ]
          if let size = sizes[info[0]], cursor + size <= end {
            let raw = try bytes(size)
            var value: UInt64 = 0
            for byte in (littleEndian ? Array(raw.reversed()) : Array(raw)) {
              value = (value << 8) | UInt64(byte)
            }
            numbers[key] =
              info[0] == 6
              ? Double(Float(bitPattern: UInt32(value)))
              : info[0] == 7
                ? Double(bitPattern: value)
                : info[0] == 2
                  ? Double(Int16(bitPattern: UInt16(value)))
                  : info[0] == 3 ? Double(Int32(bitPattern: UInt32(value))) : Double(value)
            retained[key] = String(numbers[key]!)
          }
        } else if info.first == 20, info.count == 3, info[1] == 4 {
          strings[key] =
            String(
              data: try bytes(end - cursor),
              encoding: littleEndian ? .utf16LittleEndian : .utf16BigEndian) ?? ""
          retained[key] = strings[key]
        } else {
          // Preserve unknown typed metadata exactly, without interpreting camera flags.
          let raw = try bytes(end - cursor)
          let record: [String: Any] = ["dm_types": info, "bytes_base64": raw.base64EncodedString()]
          retained[key] = String(
            decoding: try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
            as: UTF8.self)
        }
      }
      guard cursor <= end else { throw Self.invalid("DM4 tag overruns its declared size.") }
      cursor = end
    }
  }
}
