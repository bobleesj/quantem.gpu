import Foundation

public enum Native4DSTEMCatalogMode: Sendable, Equatable {
  case catalogOnly
  case indexed
}

public enum Native4DSTEMCalibrationOrigin: String, Codable, Equatable, Sendable {
  case sourceMetadata = "source_metadata"
}

public struct Native4DSTEMScanCalibration: Codable, Sendable {
  public let rowSamplingAngstrom: Double
  public let columnSamplingAngstrom: Double
  public let origin: Native4DSTEMCalibrationOrigin
  public let evidence: String

  public init(
    rowSamplingAngstrom: Double,
    columnSamplingAngstrom: Double,
    origin: Native4DSTEMCalibrationOrigin,
    evidence: String
  ) {
    self.rowSamplingAngstrom = rowSamplingAngstrom
    self.columnSamplingAngstrom = columnSamplingAngstrom
    self.origin = origin
    self.evidence = evidence
  }
}

public struct Native4DSTEMCatalog: Codable, Sendable {
  public let version: Int
  public let input: String
  public let datasets: [Native4DSTEMDataset]

  public init(version: Int = 1, input: String, datasets: [Native4DSTEMDataset]) {
    self.version = version
    self.input = input
    self.datasets = datasets
  }

  public static func merging(
    _ catalogs: [Native4DSTEMCatalog],
    inputs: [URL]
  ) throws -> Native4DSTEMCatalog {
    var seen = Set<String>()
    let datasets = catalogs.flatMap(\.datasets).filter { seen.insert($0.id).inserted }
    guard !datasets.isEmpty else { throw Native4DSTEMIOError.noDatasets }
    return Native4DSTEMCatalog(
      version: catalogs.map(\.version).max() ?? 1,
      input: inputs.map(\.path).joined(separator: " | "),
      datasets: datasets
    )
  }
}

public struct Native4DSTEMDataset: Codable, Identifiable, Sendable {
  public let id: String
  public let label: String
  public let masterPath: String?
  public let dataFiles: [String]
  public let indexFiles: [String]
  public let scanRows: Int
  public let scanCols: Int
  public let detectorRows: Int
  public let detectorCols: Int
  public let sourceDtype: String
  public let sourceBytes: Int
  public let badPixelIndices: [Int]
  public let kPixelSizeRow: Double?
  public let kPixelSizeCol: Double?
  public let kPixelUnit: String?
  public let acquisitionDate: String?
  public let metadata: [String: String]?
  public let schemaIdentity: String?
  public let sourceIdentitySHA256: String?
  public let masterSHA256: String?
  public let orderedMemberSHA256: [String]?
  public let sourceScanCalibration: Native4DSTEMScanCalibration?
  public let scalarImageRawPath: String?

  public init(
    id: String,
    label: String,
    masterPath: String?,
    dataFiles: [String],
    indexFiles: [String],
    scanRows: Int,
    scanCols: Int,
    detectorRows: Int,
    detectorCols: Int,
    sourceDtype: String,
    sourceBytes: Int,
    badPixelIndices: [Int],
    kPixelSizeRow: Double?,
    kPixelSizeCol: Double?,
    kPixelUnit: String?,
    acquisitionDate: String?,
    metadata: [String: String]?,
    schemaIdentity: String? = nil,
    sourceIdentitySHA256: String? = nil,
    masterSHA256: String? = nil,
    orderedMemberSHA256: [String]? = nil,
    sourceScanCalibration: Native4DSTEMScanCalibration? = nil,
    scalarImageRawPath: String? = nil
  ) {
    self.id = id
    self.label = label
    self.masterPath = masterPath
    self.dataFiles = dataFiles
    self.indexFiles = indexFiles
    self.scanRows = scanRows
    self.scanCols = scanCols
    self.detectorRows = detectorRows
    self.detectorCols = detectorCols
    self.sourceDtype = sourceDtype
    self.sourceBytes = sourceBytes
    self.badPixelIndices = badPixelIndices
    self.kPixelSizeRow = kPixelSizeRow
    self.kPixelSizeCol = kPixelSizeCol
    self.kPixelUnit = kPixelUnit
    self.acquisitionDate = acquisitionDate
    self.metadata = metadata
    self.schemaIdentity = schemaIdentity
    self.sourceIdentitySHA256 = sourceIdentitySHA256
    self.masterSHA256 = masterSHA256
    self.orderedMemberSHA256 = orderedMemberSHA256
    self.sourceScanCalibration = sourceScanCalibration
    self.scalarImageRawPath = scalarImageRawPath
  }
}

public enum Native4DSTEMIOError: LocalizedError, Sendable {
  case noDatasets
  case invalidData(String)
  case hdf5(String)

  public var errorDescription: String? {
    switch self {
    case .noDatasets:
      return "No recognized datasets were found. Choose a folder containing ARINA/Samsung *_master.h5 or Velox .emd files."
    case .invalidData(let message), .hdf5(let message):
      return message
    }
  }
}
