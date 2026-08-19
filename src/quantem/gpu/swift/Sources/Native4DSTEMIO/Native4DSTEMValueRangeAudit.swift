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
    Self.compatibleSchemas.contains(schema)
      && self.sourceIdentitySHA256 == sourceIdentitySHA256
      && self.sourceDtype == sourceDtype
      && self.badPixelIndices == badPixelIndices.sorted()
      && pixelsAbove255 == 0
      && maximum <= 255
  }
}

public enum Native4DSTEMValueRangeAuditIO {
  public static func read(from url: URL) throws -> Native4DSTEMValueRangeAudit {
    try JSONDecoder().decode(
      Native4DSTEMValueRangeAudit.self,
      from: Data(contentsOf: url)
    )
  }

  public static func write(
    _ audit: Native4DSTEMValueRangeAudit,
    to url: URL
  ) throws {
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    try JSONEncoder().encode(audit).write(to: url, options: .atomic)
  }
}
