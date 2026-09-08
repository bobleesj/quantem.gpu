import Foundation

/// Original EMPAD float32 frames, with their 256-word frame footer kept separate.
///
/// Open an XML acquisition directly, or supply the measured scan shape for a
/// headerless RAW file. Reading a frame preserves its IEEE-754 bits, including
/// signed, fractional and non-finite measurements. No detector correction,
/// transpose, integer conversion or physical calibration is applied.
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
  private let rawIdentity: NativeFileIdentity
  private let metadataIdentity: NativeFileIdentity?
  public var frameCount: Int { scanRows * scanColumns }
  public var sourceBytes: Int { frameCount * Self.frameBytes }

  /// Resolve XML/RAW input and verify the complete acquisition length.
  ///
  /// An explicit `(row, col)` shape is required for a RAW file without XML or
  /// an unambiguous `scan_xN_yN.raw` filename. Never infer a square scan from
  /// the file length. Conflicting shape metadata fails rather than reordering
  /// measurements. Only conventional EMPAD float32 exports are supported;
  /// uncalibrated EMPAD-G2 acquisition words are a different format.
  public static func open(
    _ input: URL, scanShape: (row: Int, col: Int)? = nil
  ) throws -> NativeEMPADSource {
    let source = input.standardizedFileURL
    guard ["xml", "raw"].contains(source.pathExtension.lowercased()) else {
      throw EMPADError("Open an EMPAD .xml or .raw file; got \(source.lastPathComponent).")
    }
    var raw = source
    var metadataURL: URL?
    var metadataShape: (row: Int, col: Int)?
    var metadataIdentity: NativeFileIdentity?
    let xml =
      source.pathExtension.lowercased() == "xml"
      ? source : source.deletingPathExtension().appendingPathExtension("xml")
    if FileManager.default.fileExists(atPath: xml.path) {
      metadataIdentity = try nativeFileIdentity(for: xml)
      let metadata = try EMPADXML.read(xml)
      guard metadataIdentity == (try nativeFileIdentity(for: xml)) else {
        throw EMPADError("EMPAD XML changed while being read. Reopen the acquisition.")
      }
      // Acquisition metadata often contains an absolute path from the scope.
      // Resolve only a sibling basename; XML cannot redirect arbitrary reads.
      let basename =
        metadata.filename.replacingOccurrences(of: "\\", with: "/")
        .split(separator: "/").last.map(String.init) ?? ""
      guard !basename.isEmpty, basename != ".", basename != "..",
        URL(fileURLWithPath: basename).pathExtension.lowercased() == "raw"
      else { throw EMPADError("EMPAD XML must name a sibling .raw file.") }
      raw = xml.deletingLastPathComponent().appendingPathComponent(basename)
      if source.pathExtension.lowercased() == "raw", raw != source {
        throw EMPADError(
          "\(xml.lastPathComponent) names a different RAW file. Open the XML acquisition instead.")
      }
      metadataURL = xml
      metadataShape = metadata.shape
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
      throw EMPADError(
        "EMPAD scan shape is missing. Open the matching XML or provide scanShape: (row: ..., col: ...)."
      )
    }
    let (frames, frameOverflow) = shape.row.multipliedReportingOverflow(by: shape.col)
    let (bytes, byteOverflow) = frames.multipliedReportingOverflow(by: frameBytes)
    guard !frameOverflow, !byteOverflow else {
      throw EMPADError("EMPAD scan shape exceeds addressable file size.")
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: raw.path)
    guard attributes[.type] as? FileAttributeType == .typeRegular,
      let size = attributes[.size] as? NSNumber, size.uint64Value == UInt64(bytes)
    else {
      throw EMPADError(
        "EMPAD RAW length does not match \(shape.row)×\(shape.col) frames of 130×128 float32 words. Restore the complete RAW file or correct its scan shape."
      )
    }
    let rawIdentity = try nativeFileIdentity(for: raw)
    guard rawIdentity.bytes == UInt64(bytes) else {
      throw EMPADError("EMPAD RAW changed while being opened. Reopen the acquisition.")
    }
    return NativeEMPADSource(
      rawURL: raw, metadataURL: metadataURL,
      scanRows: shape.row, scanColumns: shape.col,
      rawIdentity: rawIdentity, metadataIdentity: metadataIdentity)
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
    try validateUnchanged()
    guard indices.allSatisfy({ (0..<frameCount).contains($0) }) else {
      throw EMPADError("EMPAD frame selection is outside 0..<\(frameCount).")
    }
    let handle = try FileHandle(forReadingFrom: rawURL)
    defer { try? handle.close() }
    var values: [Float] = []
    for index in indices {
      try handle.seek(toOffset: UInt64(index * Self.frameBytes))
      let count = Self.detectorRows * Self.detectorColumns
      guard let bytes = try handle.read(upToCount: count * 4), bytes.count == count * 4 else {
        throw EMPADError("EMPAD RAW ended during frame \(index). Restore the complete acquisition.")
      }
      bytes.withUnsafeBytes { buffer in
        for offset in stride(from: 0, to: bytes.count, by: 4) {
          let word = buffer.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
          values.append(Float(bitPattern: UInt32(littleEndian: word)))
        }
      }
    }
    try validateUnchanged()
    return values
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

private struct EMPADError: LocalizedError {
  let errorDescription: String?
  init(_ message: String) { errorDescription = message }
}

private final class EMPADXML: NSObject, XMLParserDelegate {
  var filename = ""
  var shape: (row: Int, col: Int)? {
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
    if name == "scan_parameters" { scanMode = "" }
    _ = stack.popLast()
    content = ""
  }

  private func setField(_ key: String, value: String, parser: XMLParser) {
    guard
      ["pix_x", "pix_y", "type", "acquire/scan_resolution_x", "acquire/scan_resolution_y"].contains(
        key)
    else { return }
    if let previous = fields[key], previous != value { parser.abortParsing() }
    fields[key] = value
  }
}
