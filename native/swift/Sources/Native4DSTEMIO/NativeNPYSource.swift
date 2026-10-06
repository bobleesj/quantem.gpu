import CryptoKit
import Foundation

/// Header-only NumPy count-array reader, without Python or payload materialization.
///
/// Example: `try NativeNPYSource(url: url)` for a C-order uint16 array with shape
/// `(scan_row, scan_column, detector_row, detector_column)`. NumPy supplies no
/// microscope calibration, so physical sampling remains explicitly unknown.
public struct NativeNPYSource: NativeCountArray {
  public let url: URL
  public let dataset: Native4DSTEMDataset
  public let shape: [Int]
  public let dataOffset: Int
  public let fileBytes: Int
  private let modificationDate: Date?

  public init(url: URL) throws {
    try self.init(url: url, measurementDtype: nil)
  }

  // Shared header parsing for the separate float measurement resident.
  package init(url: URL, measurementDtype: String?) throws {
    self.url = url.resolvingSymlinksInPath()
    let values = try self.url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
    guard let size = values.fileSize else {
      throw Self.invalid("Cannot read NumPy file size; finish copying the file.")
    }
    fileBytes = size
    modificationDate = values.contentModificationDate
    let file = try FileHandle(forReadingFrom: self.url)
    defer { try? file.close() }
    let prefix = try file.read(upToCount: 8) ?? Data()
    guard prefix.count == 8, prefix.prefix(6) == Data([0x93, 78, 85, 77, 80, 89]),
      [1, 2, 3].contains(prefix[6]), prefix[7] == 0
    else {
      throw Self.invalid(
        "Choose a NumPy .npy file (versions 1–3); zipped .npz files are not supported.")
    }
    let lengthBytes = prefix[6] == 1 ? 2 : 4
    let lengthData = try file.read(upToCount: lengthBytes) ?? Data()
    guard lengthData.count == lengthBytes else {
      throw Self.invalid("Incomplete NumPy header; finish copying the file.")
    }
    let length = lengthData.enumerated().reduce(0) { $0 | Int($1.element) << ($1.offset * 8) }
    guard length > 0, length <= 1 << 20, size >= 8 + lengthBytes + length else {
      throw Self.invalid("Incomplete or oversized NumPy header; re-export a plain numeric array.")
    }
    let headerData = try file.read(upToCount: length) ?? Data()
    guard headerData.count == length, let header = String(data: headerData, encoding: .utf8) else {
      throw Self.invalid("Cannot read NumPy header; re-export a plain numeric array.")
    }
    func field(_ pattern: String) throws -> String {
      let regex = try NSRegularExpression(pattern: pattern)
      let matches = regex.matches(in: header, range: NSRange(header.startIndex..., in: header))
      guard matches.count == 1, let range = Range(matches[0].range(at: 1), in: header) else {
        throw Self.invalid("Unsupported NumPy header; export one 4D numeric array with numpy.save.")
      }
      return String(header[range])
    }
    let dtype = try field("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]")
    let order = try field("['\"]fortran_order['\"]\\s*:\\s*(True|False)")
    guard order == "False" else {
      throw Self.invalid(
        "NumPy array is Fortran-order. Save numpy.ascontiguousarray(data) before converting.")
    }
    let isFloat = ["<f4", "=f4"].contains(dtype)
    guard
      measurementDtype == "float32"
        ? isFloat
        : ["|u1", "<u1", "=u1", "<u2", "=u2"].contains(dtype)
    else {
      throw Self.invalid(
        "NumPy dtype \(dtype) is not supported by this lossless count encoder. Use native little-endian uint8/uint16 counts; do not cast calibrated or floating-point measurements."
      )
    }
    let dimensions = try field("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)")
      .split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    let parsed = dimensions.compactMap(Int.init)
    guard parsed.count == 4, dimensions.count == 4, parsed.allSatisfy({ $0 > 0 && $0 < 1 << 20 })
    else {
      throw Self.invalid(
        "NumPy shape must be (scan_row, scan_column, detector_row, detector_column); got (\(dimensions.joined(separator: ", "))). Reshape with the known scan dimensions, not a guessed square scan."
      )
    }
    shape = parsed
    dataOffset = 8 + lengthBytes + length
    let itemBytes = isFloat ? 4 : dtype.hasSuffix("1") ? 1 : 2
    var bytes = itemBytes
    for dimension in shape {
      let product = bytes.multipliedReportingOverflow(by: dimension)
      guard !product.overflow else {
        throw Self.invalid("NumPy shape exceeds the addressable file size.")
      }
      bytes = product.partialValue
    }
    guard bytes == size - dataOffset else {
      throw Self.invalid(
        "NumPy payload length does not match its shape; finish copying or re-export the array.")
    }
    let identityData = try JSONSerialization.data(
      withJSONObject: [
        "path": self.url.path, "bytes": size, "header": header,
        "modified": modificationDate?.timeIntervalSince1970 ?? 0,
      ], options: [.sortedKeys])
    let identity = SHA256.hash(data: identityData).map { String(format: "%02x", $0) }.joined()
    dataset = Native4DSTEMDataset(
      id: identity, label: self.url.lastPathComponent,
      masterPath: self.url.path, dataFiles: [self.url.path], indexFiles: [],
      scanRows: shape[0], scanCols: shape[1], detectorRows: shape[2], detectorCols: shape[3],
      sourceDtype: isFloat ? "float32" : itemBytes == 1 ? "uint8" : "uint16", sourceBytes: size,
      badPixelIndices: [], kPixelSizeRow: nil, kPixelSizeCol: nil, kPixelUnit: nil,
      acquisitionDate: nil,
      metadata: [
        "sourceKind": "numpy", "sourceFormat": "NumPy",
        "sourceFormatVersion": "\(prefix[6]).\(prefix[7])", "numpy.header": header,
        "axisOrder": "scan_row,scan_col,detector_row,detector_col",
        "calibration_status": "not_recorded", "median_correction_applied": "false",
      ],
      schemaIdentity: "NPY", sourceIdentitySHA256: identity)
  }

  public func assertUnchanged() throws {
    let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
    guard values.fileSize == fileBytes, values.contentModificationDate == modificationDate else {
      throw Self.invalid("NumPy file changed during conversion; retry with a completed copy.")
    }
  }

  private static func invalid(_ message: String) -> Native4DSTEMIOError { .invalidData(message) }
}
