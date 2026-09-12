import Darwin
import CNativeHDF5
import Foundation
import CryptoKit

/// Recorded float32 frames from EMPAD-G1, EMPAD-G2 XML/RAW, or EMD 1 datacubes.
///
/// Open an XML acquisition directly, or supply the measured scan shape for a
/// headerless RAW file. Reading a frame preserves its IEEE-754 bits, including
/// signed, fractional and non-finite measurements. No detector correction,
/// transpose or integer conversion is applied. Calibration is parsed separately.
///
/// Example: `try NativeEMPADSource.open(URL(fileURLWithPath: "scan.xml"))`.
public struct NativeEMPADSource: Sendable {
  public static let detectorRows = 128
  public static let detectorColumns = 128
  public static let frameBytes = 130 * 128 * 4
  public let rawURL: URL
  public let metadataURL: URL?
  public let scanRows: Int
  public let scanColumns: Int
  /// Physical sampling supplied by the acquisition, independent of count decoding.
  public let scanCalibration: Native4DSTEMScanCalibration?
  /// Legacy EMPAD XML diffraction sampling, in inverse nanometers per pixel.
  public let diffractionSamplingInverseNanometers: Double?
  public let acquisitionDate: String?
  /// Identified on-disk schema, independent of packed resident encoding.
  public let formatIdentifier: String
  public let formatName: String
  public let microscopeMetadata: [String: String]
  public let recordBytes: Int
  private var dataOffset: UInt64 = 0
  private let rawIdentity: NativeFileIdentity
  private let metadataIdentity: NativeFileIdentity?
  public var frameCount: Int { scanRows * scanColumns }
  public var sourceBytes: Int { frameCount * recordBytes }

  /// Resolve XML/RAW input and verify the complete acquisition length.
  ///
  /// An explicit `(row, col)` shape is required for a RAW file without XML or
  /// an unambiguous `scan_xN_yN.raw` filename. Never infer a square scan from
  /// the file length. Conflicting shape metadata fails rather than reordering
  /// measurements. G1 stores a 256-word footer; G2 must explicitly declare
  /// 128×128 float32 records in XML. Encoded integer acquisition words are not
  /// float exports. EMD supports contiguous little-endian float32 storage.
  public static func open(
    _ input: URL, scanShape: (row: Int, col: Int)? = nil
  ) throws -> NativeEMPADSource {
    let source = input.standardizedFileURL
    if ["h5", "hdf5", "emd"].contains(source.pathExtension.lowercased()) {
      return try openEMD(source, scanShape: scanShape)
    }
    guard ["xml", "raw"].contains(source.pathExtension.lowercased()) else {
      throw EMPADError("Open an EMPAD .xml or .raw file; got \(source.lastPathComponent).")
    }
    var raw = source
    var metadataURL: URL?
    var metadataShape: (row: Int, col: Int)?
    var metadataIdentity: NativeFileIdentity?
    var metadata: EMPADXML?
    var xml =
      source.pathExtension.lowercased() == "xml"
      ? source : source.deletingPathExtension().appendingPathExtension("xml")
    if source.pathExtension.lowercased() == "raw",
      !FileManager.default.fileExists(atPath: xml.path) {
      // Scope software names XML after the acquisition and RAW after its shape.
      // Only metadata explicitly naming this sibling may supply its layout.
      let siblings = try FileManager.default.contentsOfDirectory(
        at: source.deletingLastPathComponent(), includingPropertiesForKeys: nil)
      let matches = siblings.filter { candidate in
        guard candidate.pathExtension.lowercased() == "xml",
          let parsed = try? EMPADXML.read(candidate) else { return false }
        return parsed.filename.replacingOccurrences(of: "\\", with: "/")
          .split(separator: "/").last.map(String.init) == source.lastPathComponent
      }
      guard matches.count <= 1 else {
        throw EMPADError("Multiple XML files name this RAW acquisition. Open the intended XML file.")
      }
      if let match = matches.first { xml = match }
    }
    if FileManager.default.fileExists(atPath: xml.path) {
      metadataIdentity = try nativeFileIdentity(for: xml)
      let parsed = try EMPADXML.read(xml)
      metadata = parsed
      guard metadataIdentity == (try nativeFileIdentity(for: xml)) else {
        throw EMPADError("EMPAD XML changed while being read. Reopen the acquisition.")
      }
      // Acquisition metadata often contains an absolute path from the scope.
      // Resolve only a sibling basename; XML cannot redirect arbitrary reads.
      let basename =
        parsed.filename.replacingOccurrences(of: "\\", with: "/")
        .split(separator: "/").last.map(String.init) ?? ""
      guard !basename.isEmpty, basename != ".", basename != "..",
        URL(fileURLWithPath: basename).pathExtension.lowercased() == "raw"
      else { throw EMPADError("EMPAD XML must name a sibling .raw file.") }
      raw = xml.deletingLastPathComponent().appendingPathComponent(basename)
      if source.pathExtension.lowercased() == "raw",
        raw.resolvingSymlinksInPath() != source.resolvingSymlinksInPath() {
        throw EMPADError(
          "\(xml.lastPathComponent) names a different RAW file. Open the XML acquisition instead.")
      }
      metadataURL = xml
      metadataShape = parsed.shape
    } else if source.pathExtension.lowercased() == "xml" {
      throw EMPADError(
        "EMPAD XML is missing: \(source.lastPathComponent). Restore its metadata file.")
    }
    let namedShape = shapeFromFilename(raw)
    if let metadataShape, let namedShape,
      metadataShape.row != namedShape.row || metadataShape.col != namedShape.col
    {
      throw EMPADError(
        "EMPAD XML and RAW filename disagree about scan shape. Correct the acquisition metadata.")
    }
    let documentedShape = metadataShape ?? namedShape
    if let scanShape, let documentedShape,
      scanShape.row != documentedShape.row || scanShape.col != documentedShape.col
    {
      throw EMPADError(
        "Requested EMPAD scan shape disagrees with acquisition metadata. Use its original (row, col) shape."
      )
    }
    guard let shape = scanShape ?? documentedShape, shape.row > 0, shape.col > 0 else {
      throw NativeEMPADSourceError.missingScanShape(rawURL: raw)
    }
    let (frames, frameOverflow) = shape.row.multipliedReportingOverflow(by: shape.col)
    let recordBytes = metadata?.isGeneration2 == true ? 128 * 128 * 4 : frameBytes
    let (bytes, byteOverflow) = frames.multipliedReportingOverflow(by: recordBytes)
    guard !frameOverflow, !byteOverflow else {
      throw EMPADError("EMPAD scan shape exceeds addressable file size.")
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: raw.path)
    guard attributes[.type] as? FileAttributeType == .typeRegular,
      let size = attributes[.size] as? NSNumber, size.uint64Value == UInt64(bytes)
    else {
      throw EMPADError(
        "EMPAD RAW length does not match \(shape.row)×\(shape.col) records of \(recordBytes) bytes. Restore the matching XML/RAW acquisition; detector layout is never inferred from file length."
      )
    }
    let rawIdentity = try nativeFileIdentity(for: raw)
    guard rawIdentity.bytes == UInt64(bytes) else {
      throw EMPADError("EMPAD RAW changed while being opened. Reopen the acquisition.")
    }
    return NativeEMPADSource(
      rawURL: raw, metadataURL: metadataURL,
      scanRows: shape.row, scanColumns: shape.col,
      scanCalibration: metadata?.scanCalibration(rows: shape.row, columns: shape.col),
      diffractionSamplingInverseNanometers: metadata?.diffractionSampling,
      acquisitionDate: metadata?.acquisitionDate,
      formatIdentifier: metadata?.isGeneration2 == true ? "empad-g2-float32-xml/v1" : "empad-g1-float32-xml/v1",
      formatName: metadata?.isGeneration2 == true ? "EMPAD-G2 · XML/RAW float32" : "EMPAD-G1 · XML/RAW float32",
      microscopeMetadata: metadata?.microscopeMetadata ?? [:], recordBytes: recordBytes,
      rawIdentity: rawIdentity, metadataIdentity: metadataIdentity)
  }

  /// Recognize the supported EMD schema from metadata rather than filenames.
  public static func isEMDFloatAcquisition(_ url: URL) -> Bool {
    guard ["h5", "hdf5", "emd"].contains(url.pathExtension.lowercased()) else { return false }
    return (try? openEMD(url, scanShape: nil)) != nil
  }

  private static func openEMD(_ url: URL, scanShape: (row: Int, col: Int)?) throws -> NativeEMPADSource {
    let identity = try nativeFileIdentity(for: url)
    var info = qh5_emd_float_info()
    var error: UnsafeMutablePointer<CChar>?
    let status = url.path.withCString { qh5_inspect_emd_float($0, &info, &error) }
    defer { qh5_free_error(error) }
    guard status == 0 else { throw EMPADError(error.map { String(cString: $0) } ?? "Unsupported EMD datacube.") }
    guard let rows = Int(exactly: info.rows), let columns = Int(exactly: info.columns),
      Int(exactly: info.bytes) != nil, info.offset <= identity.bytes,
      info.bytes <= identity.bytes - info.offset,
      identity == (try nativeFileIdentity(for: url)) else {
      throw EMPADError("EMD storage is incomplete or changed during inspection. Restore the complete file.")
    }
    if let scanShape, scanShape.row != rows || scanShape.col != columns {
      throw EMPADError("Requested scan shape disagrees with the recorded EMD dimensions.")
    }
    var metadata: [String: String] = [:]
    for (key, value, unit) in [
      ("electron_source/accelerating_voltage", info.voltage, "V"),
      ("illumination_system/semi_convergence_angle", info.semiangle_mrad, "mrad"),
      ("imaging_system/camera_length", info.camera_meters, "m"),
      ("imaging_system/reciprocal_pixel_size_y", info.angle_mrad, "mrad"),
      ("imaging_system/reciprocal_pixel_size_x", info.angle_mrad, "mrad")
    ] where value > 0 { metadata["electron_microscope/" + key] = "\(value) \(unit)" }
    metadata["sourceDataset"] = "/datacube_root/datacube/data"
    metadata["sourceAxisOrder"] = "Recorded EMD axes 0,1,2,3; no transpose"
    let calibration: Native4DSTEMScanCalibration? = info.scan_angstrom > 0
      ? Native4DSTEMScanCalibration(rowSamplingAngstrom: info.scan_angstrom,
          columnSamplingAngstrom: info.scan_angstrom, origin: .sourceMetadata,
          evidence: "EMD calibration R_pixel_size with R_pixel_units=A") : nil
    return NativeEMPADSource(rawURL: url, metadataURL: url, scanRows: rows, scanColumns: columns,
      scanCalibration: calibration, diffractionSamplingInverseNanometers: nil, acquisitionDate: nil,
      formatIdentifier: "emd1-contiguous-float32/v1", formatName: "EMD 1 · HDF5 float32",
      microscopeMetadata: metadata, recordBytes: 65536, dataOffset: info.offset,
      rawIdentity: identity, metadataIdentity: identity)
  }

  /// Reject source replacement or modification since opening this acquisition.
  /// Call before publishing a resident assembled from multiple read windows.
  /// This is a filesystem snapshot check, not a content checksum or file lock.
  public func validateUnchanged() throws {
    guard rawIdentity == (try nativeFileIdentity(for: rawURL)) else {
      throw EMPADError(
        "EMPAD RAW changed during loading. Reopen the acquisition; no partial resident is valid.")
    }
    if let metadataURL {
      guard metadataIdentity == (try nativeFileIdentity(for: metadataURL)) else {
        throw EMPADError(
          "EMPAD XML changed during loading. Reopen the acquisition with its current scan shape.")
      }
    }
  }

  /// Read selected diffraction frames in request order, retaining duplicates.
  ///
  /// Footer words are not detector pixels. This method returns exactly all
  /// 128×128 measured pixels per requested frame, with no binning or crop.
  /// Example: `try source.readFrames([0, 3, 0])`.
  public func readFrames(_ indices: [Int]) throws -> [Float] {
    guard indices.allSatisfy({ (0..<frameCount).contains($0) }) else {
      throw EMPADError("EMPAD frame selection is outside 0..<\(frameCount).")
    }
    let (bytes, overflow) = indices.count.multipliedReportingOverflow(by: 16384 * 4)
    guard !overflow else { throw EMPADError("EMPAD selection exceeds addressable memory.") }
    var values = [Float](repeating: 0, count: bytes / 4)
    try values.withUnsafeMutableBytes { try readFrames(indices, into: $0) }
    return values
  }

  package func sourceSnapshot() throws -> Data {
    try validateUnchanged()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return try encoder.encode(rawIdentity)
  }

  /// Fingerprint the file identities captured at open, including XML metadata.
  /// Used to invalidate local correction choices when either file changes.
  /// This is a file-snapshot fingerprint, not a content checksum.
  /// Example: `try source.snapshotFingerprint()`.
  public func snapshotFingerprint() throws -> String {
    try validateUnchanged()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    var digest = SHA256()
    digest.update(data: try encoder.encode(rawIdentity))
    if let metadataIdentity { digest.update(data: try encoder.encode(metadataIdentity)) }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }

  // Backend-only destination form avoids a second staging allocation/copy.
  // It has the same ordered selection and source snapshot checks as readFrames.
  package func readFrames(_ indices: [Int], into output: UnsafeMutableRawBufferPointer) throws {
    try validateUnchanged()
    guard indices.allSatisfy({ (0..<frameCount).contains($0) }) else {
      throw EMPADError("EMPAD frame selection is outside 0..<\(frameCount).")
    }
    let handle = try FileHandle(forReadingFrom: rawURL)
    defer { try? handle.close() }
    let pixels = Self.detectorRows * Self.detectorColumns
    let (count, overflow) = indices.count.multipliedReportingOverflow(by: pixels)
    guard !overflow else { throw EMPADError("EMPAD selection exceeds addressable memory.") }
    guard output.count / 4 >= count else {
      throw EMPADError("EMPAD frame destination is too small.")
    }
    let footer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 64)
    defer { footer.deallocate() }
    var first = 0
    while first < indices.count {
      // Coalesce only consecutive requests. Arbitrary order and duplicates
      // remain exact; each read covers at most 64 records with 1 KiB scratch.
      var frames = 1
      while frames < 64, first + frames < indices.count,
        indices[first + frames] == indices[first] + frames
      { frames += 1 }
      try handle.seek(toOffset: dataOffset + UInt64(indices[first] * recordBytes))
      // Scatter the original record directly into the destination. Footer
      // slots may share scratch because their contents are never consumed.
      // This avoids allocating and copying a second RAW-sized staging window.
      var vectors: [iovec] = []
      vectors.reserveCapacity(frames * 2)
      for frame in 0..<frames {
        vectors.append(
          iovec(
            iov_base: output.baseAddress!.advanced(by: (first + frame) * pixels * 4),
            iov_len: pixels * 4))
        if recordBytes > pixels * 4 {
          vectors.append(iovec(iov_base: footer, iov_len: recordBytes - pixels * 4))
        }
      }
      var next = 0
      while next < vectors.count {
        let received = vectors.withUnsafeBufferPointer {
          Darwin.readv(
            handle.fileDescriptor, $0.baseAddress!.advanced(by: next), Int32($0.count - next))
        }
        if received < 0 && errno == EINTR { continue }
        guard received > 0 else {
          throw EMPADError(
            "EMPAD RAW ended or failed during frame \(indices[first]). Restore the complete acquisition."
          )
        }
        var remaining = received
        while remaining > 0 {
          if remaining >= vectors[next].iov_len {
            remaining -= vectors[next].iov_len
            next += 1
          } else {
            vectors[next].iov_base = vectors[next].iov_base!.advanced(by: remaining)
            vectors[next].iov_len -= remaining
            remaining = 0
          }
        }
      }
      #if _endian(big)
        for pixel in (first * pixels)..<((first + frames) * pixels) {
          let word = output.loadUnaligned(fromByteOffset: pixel * 4, as: UInt32.self)
          output.storeBytes(
            of: UInt32(littleEndian: word), toByteOffset: pixel * 4, as: UInt32.self)
        }
      #endif
      first += frames
    }
    try validateUnchanged()
  }

  private static func shapeFromFilename(_ url: URL) -> (row: Int, col: Int)? {
    let name = url.lastPathComponent
    guard
      let expression = try? NSRegularExpression(
        pattern: "^scan_x([0-9]+)_y([0-9]+)\\.raw$", options: .caseInsensitive),
      let match = expression.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)),
      let colRange = Range(match.range(at: 1), in: name),
      let rowRange = Range(match.range(at: 2), in: name),
      let col = Int(name[colRange]), let row = Int(name[rowRange])
    else { return nil }
    return (row, col)
  }
}

/// Missing metadata is recoverable without treating malformed data as a scan.
public enum NativeEMPADSourceError: LocalizedError {
  case missingScanShape(rawURL: URL)

  public var errorDescription: String? {
    "EMPAD scan shape is missing. Open the matching XML or provide scanShape: (row: ..., col: ...)."
  }
}

private struct EMPADError: LocalizedError {
  let errorDescription: String?
  init(_ message: String) { errorDescription = message }
}

private final class EMPADXML: NSObject, XMLParserDelegate {
  var filename = ""
  var acquisitionDate: String?
  var isGeneration2: Bool { fields["sensor/type"] == "EMPAD2" }
  var microscopeMetadata: [String: String] {
    var result: [String: String] = [:]
    let root = "electron_microscope/"
    for (source, target, unit) in [
      (isGeneration2 ? "iom_measurements/ColumnSourceHighVoltage" : "iom_measurements/high_voltage", "electron_source/accelerating_voltage", "V"),
      (isGeneration2 ? "iom_measurements/ColumnOpticsGetCameraLengthNominalCameraLength" : "iom_measurements/nominal_camera_length", "imaging_system/camera_length", "m"),
      (isGeneration2 ? "scan/exposure_time" : "exposure_time", "scan_controller/regular_scan/dwell_time", isGeneration2 ? "s" : "ms")
    ] {
      if let value = positive(source) { result[root + target] = "\(value) \(unit)" }
    }
    if isGeneration2, let angle = positive("iom_measurements/calibrated_diffraction_angle") {
      for axis in ["y", "x"] { result[root + "imaging_system/reciprocal_pixel_size_" + axis] = "\(angle) rad" }
    }
    return result
  }
  var diffractionSampling: Double? {
    guard !isGeneration2 else { return nil }
    guard let value = positive("iom_measurements/calibrated_pixelsize") else { return nil }
    let sampling = value * 1e9
    return sampling.isFinite ? sampling : nil
  }

  func scanCalibration(rows: Int, columns: Int) -> Native4DSTEMScanCalibration? {
    let modern = "iom_measurements/full_scan_field_of_view/"
    let legacy = "iom_measurements/optics.get_full_scan_field_of_view"
    let row: Double
    let column: Double
    let evidence: String
    if let x = positive(modern + "x"), let y = positive(modern + "y"),
      x == y, let factor = positive(modern + "scale_factor")
    {
      // EMPAD 1.2 records the same maximum-axis FOV in x and y, including
      // its instrument scale factor. Sampling is isotropic even for rectangles.
      row = x / factor / Double(max(rows, columns)) * 1e10
      column = row
      evidence = "EMPAD XML full_scan_field_of_view / scale_factor / maximum scan dimension"
    } else if let text = fields[legacy],
      let data = text.data(using: .utf8),
      let fov = try? JSONDecoder().decode([Double].self, from: data),
      fov.count == 2, fov.allSatisfy({ $0.isFinite && $0 > 0 })
    {
      row = fov[0] / Double(rows) * 1e10
      column = fov[1] / Double(columns) * 1e10
      evidence = "EMPAD XML optics.get_full_scan_field_of_view (meters, row/column)"
    } else {
      return nil
    }
    let calibration = Native4DSTEMScanCalibration(
      rowSamplingAngstrom: row,
      columnSamplingAngstrom: column, origin: .sourceMetadata, evidence: evidence)
    return calibration.isValid ? calibration : nil
  }

  private func positive(_ key: String) -> Double? {
    guard let value = Double(fields[key] ?? ""), value.isFinite, value > 0 else { return nil }
    return value
  }
  var shape: (row: Int, col: Int)? {
    if isGeneration2 { return Self.pair(fields["scan/shape"]) }
    if let row = Int(fields["pix_y"] ?? ""), let col = Int(fields["pix_x"] ?? "") {
      return (row, col)
    }
    if let row = Int(fields["acquire/scan_resolution_y"] ?? ""),
      let col = Int(fields["acquire/scan_resolution_x"] ?? "")
    {
      return (row, col)
    }
    return nil
  }
  private static func pair(_ value: String?) -> (row: Int, col: Int)? {
    guard let value else { return nil }
    let parts = value.trimmingCharacters(in: CharacterSet(charactersIn: "() "))
      .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    guard parts.count == 2, let row = Int(parts[0]), let col = Int(parts[1]), row > 0, col > 0 else { return nil }
    return (row, col)
  }
  private var fields: [String: String] = [:]
  private var stack: [String] = []
  private var scanMode = ""
  private var content = ""

  static func read(_ url: URL) throws -> EMPADXML {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let bytes = try handle.read(upToCount: 4 * 1024 * 1024 + 1) ?? Data()
    guard bytes.count <= 4 * 1024 * 1024,
      let text = String(data: bytes, encoding: .utf8),
      !text.uppercased().contains("<!DOCTYPE"), !text.uppercased().contains("<!ENTITY")
    else {
      throw EMPADError(
        "EMPAD XML must be UTF-8 metadata without external entities (at most 4 MiB).")
    }
    let result = EMPADXML()
    let parser = XMLParser(data: bytes)
    parser.shouldResolveExternalEntities = false
    parser.delegate = result
    guard parser.parse(), !result.filename.isEmpty else {
      throw EMPADError(
        "Could not read EMPAD XML raw_file metadata. Open the original acquisition XML.")
    }
    if result.fields["sensor/type"] != nil || result.fields["rawfile/filename"] != nil {
      guard result.isGeneration2, result.fields["rawfile/datatype"] == "float32",
        result.fields["scan/type"] == "scan", result.shape != nil,
        let detector = pair(result.fields["sensor/shape"]), detector.row == 128, detector.col == 128
      else { throw EMPADError("Unsupported EMPAD-G2 record layout. Open a raster XML/RAW export declaring 128×128 float32 detector data.") }
    }
    for keys in [("pix_y", "pix_x"), ("acquire/scan_resolution_y", "acquire/scan_resolution_x")] {
      if result.fields[keys.0] != nil || result.fields[keys.1] != nil {
        guard let row = Int(result.fields[keys.0] ?? ""),
          let col = Int(result.fields[keys.1] ?? ""), row > 0, col > 0
        else {
          throw EMPADError(
            "EMPAD XML has incomplete or invalid scan dimensions. Restore both positive row and column dimensions."
          )
        }
      }
    }
    if let row = Int(result.fields["pix_y"] ?? ""),
      let col = Int(result.fields["pix_x"] ?? ""),
      let acquiredRow = Int(result.fields["acquire/scan_resolution_y"] ?? ""),
      let acquiredCol = Int(result.fields["acquire/scan_resolution_x"] ?? ""),
      row != acquiredRow || col != acquiredCol
    {
      throw EMPADError(
        "EMPAD XML contains conflicting scan dimensions. Correct the acquisition metadata.")
    }
    if let type = result.fields["type"], type != "scan" {
      throw EMPADError(
        "EMPAD acquisition type '\(type)' is not a 2D scan. Open a raster scan acquisition.")
    }
    return result
  }

  func parser(
    _ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
    qualifiedName: String?, attributes: [String: String]
  ) {
    stack.append(name)
    content = ""
    if stack.count == 2, name == "raw_file" {
      let candidate = attributes["filename"] ?? ""
      if !filename.isEmpty, filename != candidate { parser.abortParsing() }
      filename = candidate
    }
    if stack.count == 2, name == "scan_parameters" { scanMode = attributes["mode"] ?? "acquire" }
    if stack.count == 2, name == "timestamp" { acquisitionDate = attributes["isoformat"] }
  }

  func parser(_ parser: XMLParser, foundCharacters string: String) { content += string }

  func parser(
    _ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?
  ) {
    let value = content.trimmingCharacters(in: .whitespacesAndNewlines)
    if stack.count == 2 { setField(name, value: value, parser: parser) }
    if stack.count == 3, stack[1] == "scan_parameters" {
      setField(scanMode + "/" + name, value: value, parser: parser)
    }
    if stack.count >= 3, stack[1] == "iom_measurements" {
      setField(stack.dropFirst().joined(separator: "/"), value: value, parser: parser)
    }
    if stack.count == 3, ["scan", "sensor", "rawfile"].contains(stack[1]) {
      setField(stack[1] + "/" + name, value: value, parser: parser)
      if stack[1] == "rawfile", name == "filename" { filename = value }
    }
    if name == "scan_parameters" { scanMode = "" }
    _ = stack.popLast()
    content = ""
  }

  private func setField(_ key: String, value: String, parser: XMLParser) {
    guard
      [
        "pix_x", "pix_y", "type", "acquire/scan_resolution_x", "acquire/scan_resolution_y",
        "exposure_time", "scan/type", "scan/shape", "scan/exposure_time", "sensor/type", "sensor/shape",
        "rawfile/filename", "rawfile/datatype",
        "iom_measurements/high_voltage", "iom_measurements/nominal_camera_length",
        "iom_measurements/ColumnSourceHighVoltage", "iom_measurements/ColumnOpticsGetCameraLengthNominalCameraLength",
        "iom_measurements/calibrated_diffraction_angle",
        "iom_measurements/calibrated_pixelsize",
        "iom_measurements/optics.get_full_scan_field_of_view",
        "iom_measurements/full_scan_field_of_view/x",
        "iom_measurements/full_scan_field_of_view/y",
        "iom_measurements/full_scan_field_of_view/scale_factor",
      ].contains(
        key)
    else { return }
    if let previous = fields[key], previous != value { parser.abortParsing() }
    fields[key] = value
  }
}
