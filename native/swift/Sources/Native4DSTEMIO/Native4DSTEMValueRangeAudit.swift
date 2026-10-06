import CryptoKit
import Foundation

public struct Native4DSTEMValueRangeAudit: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.value-range-audit/v1"
  static let compatibleSchemas = [
    currentSchema,
    "live4dstem.value-range-audit/v1",
  ]

  public let schema: String
  public let sourceIdentitySHA256: String
  public let sourceDtype: String
  public let badPixelIndices: [Int]
  public let maximum: UInt32
  public let pixelsAbove255: UInt64

  public init(
    sourceIdentitySHA256: String,
    sourceDtype: String,
    badPixelIndices: [Int],
    maximum: UInt32,
    pixelsAbove255: UInt64
  ) {
    schema = Self.currentSchema
    self.sourceIdentitySHA256 = sourceIdentitySHA256
    self.sourceDtype = sourceDtype
    self.badPixelIndices = badPixelIndices.sorted()
    self.maximum = maximum
    self.pixelsAbove255 = pixelsAbove255
  }

  public func provesLosslessUInt8(
    sourceIdentitySHA256: String,
    sourceDtype: String,
    badPixelIndices: [Int]
  ) -> Bool {
    (try? validate()) != nil
      && self.sourceIdentitySHA256 == sourceIdentitySHA256
      && self.sourceDtype == sourceDtype
      && self.badPixelIndices == badPixelIndices.sorted()
      && pixelsAbove255 == 0
      && maximum <= 255
  }

  /// Canonical identity for provenance that depends on this exact audit.
  public func sha256() throws -> String {
    try validate()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return SHA256.hash(data: try encoder.encode(self))
      .map { String(format: "%02x", $0) }
      .joined()
  }

  /// Fail closed when decoded audit fields cannot describe the stated source.
  public func validate() throws {
    let sortedBadPixels = badPixelIndices.sorted()
    guard Self.compatibleSchemas.contains(schema),
      Self.isSHA256(sourceIdentitySHA256),
      sourceDtype == "uint8" || sourceDtype == "uint16",
      badPixelIndices == sortedBadPixels,
      sortedBadPixels.allSatisfy({ $0 >= 0 }),
      Set(sortedBadPixels).count == sortedBadPixels.count,
      maximum <= UInt32(UInt16.max),
      (maximum > UInt32(UInt8.max)) == (pixelsAbove255 > 0),
      sourceDtype != "uint8"
        || (maximum <= UInt32(UInt8.max) && pixelsAbove255 == 0)
    else { throw Native4DSTEMValueRangeAuditError.invalidFields }
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64
      && value.utf8.allSatisfy {
        (48...57).contains($0) || (97...102).contains($0)
      }
  }
}

public enum Native4DSTEMValueRangeAuditError: LocalizedError, Equatable {
  case invalidFields

  public var errorDescription: String? {
    "The value-range audit has an invalid schema, source identity, dtype, "
      + "maximum, above-255 count, or bad-pixel list. Recompute it from the source."
  }
}

public enum Native4DSTEMValueRangeAuditIO {
  public static func read(from url: URL) throws -> Native4DSTEMValueRangeAudit {
    let audit = try JSONDecoder().decode(
      Native4DSTEMValueRangeAudit.self,
      from: Data(contentsOf: url)
    )
    try audit.validate()
    return audit
  }

  public static func write(
    _ audit: Native4DSTEMValueRangeAudit,
    to url: URL
  ) throws {
    try audit.validate()
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    try JSONEncoder().encode(audit).write(to: url, options: .atomic)
  }
}
