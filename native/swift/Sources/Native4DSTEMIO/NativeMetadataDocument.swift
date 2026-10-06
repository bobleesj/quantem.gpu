import CryptoKit
import Foundation

/// An original metadata document, retained independently of normalized quantities.
/// Example: `let document = try NativeMetadataDocument.read(url)`.
public struct NativeMetadataDocument: Codable, Equatable, Hashable, Sendable {
  public static let metadataKey = "qem_source_documents"
  public static let maximumBytes = 4 << 20
  public let filename: String
  public let mediaType: String
  public let content: String
  public let sha256: String

  public static func read(_ url: URL) throws -> Self {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let bytes = try handle.read(upToCount: maximumBytes + 1) ?? Data()
    guard bytes.count <= maximumBytes, let text = String(data: bytes, encoding: .utf8),
      ["xml", "json"].contains(url.pathExtension.lowercased())
    else { throw invalid("Choose a UTF-8 XML or JSON metadata file no larger than 4 MiB.") }
    let result = Self(
      filename: url.lastPathComponent,
      mediaType: url.pathExtension.lowercased() == "xml" ? "application/xml" : "application/json",
      content: text, sha256: digest(bytes))
    try result.validate()
    return result
  }

  public func validate() throws {
    let bytes = Data(content.utf8)
    guard !filename.isEmpty, !filename.contains("/"), !filename.contains("\\"),
      bytes.count <= Self.maximumBytes, Self.digest(bytes) == sha256
    else {
      throw Self.invalid(
        "Metadata document is damaged or oversized; attach the original file again.")
    }
    switch mediaType {
    case "application/xml":
      guard !content.uppercased().contains("<!DOCTYPE"),
        !content.uppercased().contains("<!ENTITY")
      else { throw Self.invalid("XML external entities and document types are not supported.") }
      let parser = XMLParser(data: bytes)
      parser.shouldResolveExternalEntities = false
      guard parser.parse() else {
        throw Self.invalid("XML metadata is malformed; choose the original file.")
      }
    case "application/json":
      guard try JSONSerialization.jsonObject(with: bytes) is [String: Any] else {
        throw Self.invalid("JSON metadata must be an object with named fields.")
      }
    default: throw Self.invalid("Unsupported metadata media type: \(mediaType).")
    }
  }

  public static func encoded(_ documents: [Self]) throws -> Data {
    guard documents.count <= 16,
      documents.reduce(0, { $0 + $1.content.utf8.count }) <= maximumBytes
    else {
      throw invalid("Attachments exceed 4 MiB total; remove an attachment before adding another.")
    }
    for document in documents { try document.validate() }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return try encoder.encode(documents)
  }

  public static func read(metadata: [String: String]) throws -> [Self] {
    if let text = metadata[metadataKey] {
      let documents = try JSONDecoder().decode([Self].self, from: Data(text.utf8))
      _ = try encoded(documents)
      return documents
    }
    if let text = metadata[NativeQEMMetadataUnits.metadataKey],
      let scientific = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
    {
      return try read(scientific: scientific)
    }
    return []
  }

  public static func read(scientific: [String: Any]) throws -> [Self] {
    guard let raw = scientific["source_documents"] else { return [] }
    let documents = try JSONDecoder().decode(
      [Self].self,
      from: JSONSerialization.data(withJSONObject: raw))
    _ = try encoded(documents)
    return documents
  }

  /// Add original attachments without changing source quantities or user overrides.
  public static func adding(_ documents: [Self], to scientific: [String: Any]) throws -> [String:
    Any]
  {
    var result = scientific
    var combined = try read(scientific: scientific)
    for document in documents where !combined.contains(where: { $0.sha256 == document.sha256 }) {
      combined.append(document)
    }
    if !combined.isEmpty {
      result["source_documents"] = try JSONSerialization.jsonObject(with: encoded(combined))
    }
    return result
  }

  private static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  private static func invalid(_ message: String) -> Native4DSTEMIOError { .invalidData(message) }
}
