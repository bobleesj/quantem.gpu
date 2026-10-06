import CNativeHDF5
import Foundation

/// A copied scientific scalar plane, never an RGB/contrast-rendered image.
public struct NativeScientificImage: Sendable {
  public enum ScalarType: UInt32, Sendable {
    case uint32 = 1
    case float32 = 2
  }
  public let name: String
  public let rows: Int
  public let columns: Int
  public let scalarType: ScalarType
  public let values: Data

  public init(name: String, rows: Int, columns: Int, scalarType: ScalarType, values: Data) {
    self.name = name
    self.rows = rows
    self.columns = columns
    self.scalarType = scalarType
    self.values = values
  }
}

/// Write numerical 2D products and a UTF-8 JSON provenance record to one HDF5.
public enum NativeScientificExport {
  public static func write(images: [NativeScientificImage], metadata: Data, to destination: URL)
    throws
  {
    guard !images.isEmpty, metadata.count < 16 * 1024 * 1024,
      Set(images.map(\.name)).count == images.count,
      let json = String(data: metadata, encoding: .utf8),
      (try JSONSerialization.jsonObject(with: metadata)) is [String: Any]
    else {
      throw Native4DSTEMIOError.invalidData("Export requires images and a JSON metadata object")
    }
    let temporary = destination.deletingLastPathComponent().appendingPathComponent(
      ".scientific-export-\(UUID().uuidString).h5")
    defer { try? FileManager.default.removeItem(at: temporary) }
    for (index, image) in images.enumerated() {
      let (count, overflow) = image.rows.multipliedReportingOverflow(by: image.columns)
      let (bytes, byteOverflow) = count.multipliedReportingOverflow(by: 4)
      guard image.rows > 0, image.columns > 0, !overflow, !byteOverflow,
        image.values.count == bytes, !image.name.isEmpty, image.name.utf8.count < 240
      else {
        throw Native4DSTEMIOError.invalidData(
          "Image \(image.name) has inconsistent dimensions or byte count")
      }
      var error: UnsafeMutablePointer<CChar>?
      let status = image.values.withUnsafeBytes { values in
        qh5_export_scientific_image(
          temporary.path, image.name, values.baseAddress,
          UInt64(image.rows), UInt64(image.columns), image.scalarType.rawValue,
          json, index == 0 ? 1 : 0, &error)
      }
      defer { free(error) }
      guard status == 0 else {
        throw Native4DSTEMIOError.invalidData(
          error.map { String(cString: $0) } ?? "HDF5 export failed")
      }
    }
    // The caller must choose a new destination; never overwrite source data.
    try FileManager.default.moveItem(at: temporary, to: destination)
  }

  public static func metadata(at source: URL) throws -> Data {
    guard let raw = qh5_read_scientific_metadata(source.path) else {
      throw Native4DSTEMIOError.invalidData("This file has no saved scientific metadata")
    }
    defer { free(raw) }
    let data = Data(String(cString: raw).utf8)
    guard (try JSONSerialization.jsonObject(with: data)) is [String: Any] else {
      throw Native4DSTEMIOError.invalidData("Saved metadata is not a JSON object")
    }
    return data
  }
}
