import Foundation

/// Validation failures for a native Metal 4D-STEM load selection.
public enum Metal4DSTEMLoadPlanError: LocalizedError, Equatable {
  case invalidSourceShape
  case invalidScanRegion
  case invalidScanBin(Int)
  case invalidDetectorBin(Int)
  case invalidSourceBytesPerValue(Int)

  public var errorDescription: String? {
    switch self {
    case .invalidSourceShape:
      "The source scan and detector dimensions must all be positive."
    case .invalidScanRegion:
      "The scan region must be nonempty and contained within the source scan."
    case .invalidScanBin(let factor):
      "Scan bin \(factor) is unsupported. Choose 1, 2, 4, 8, or 16."
    case .invalidDetectorBin(let factor):
      "Detector bin \(factor) is unsupported. Choose 1, 2, or 4."
    case .invalidSourceBytesPerValue(let bytes):
      "Source values must occupy 1 or 2 bytes, not \(bytes)."
    }
  }
}

/// A half-open real-space scan selection using `(row, column)` coordinates.
public struct Metal4DSTEMScanRegion: Codable, Equatable, Hashable, Sendable {
  public let rowStart: Int
  public let rowStop: Int
  public let columnStart: Int
  public let columnStop: Int

  public init(
    rowStart: Int,
    rowStop: Int,
    columnStart: Int,
    columnStop: Int,
    sourceRows: Int,
    sourceColumns: Int
  ) throws {
    guard sourceRows > 0, sourceColumns > 0,
      rowStart >= 0, rowStop > rowStart, rowStop <= sourceRows,
      columnStart >= 0, columnStop > columnStart, columnStop <= sourceColumns
    else { throw Metal4DSTEMLoadPlanError.invalidScanRegion }
    self.rowStart = rowStart
    self.rowStop = rowStop
    self.columnStart = columnStart
    self.columnStop = columnStop
  }

  public static func full(sourceRows: Int, sourceColumns: Int) throws -> Self {
    try Self(
      rowStart: 0,
      rowStop: sourceRows,
      columnStart: 0,
      columnStop: sourceColumns,
      sourceRows: sourceRows,
      sourceColumns: sourceColumns
    )
  }

  public var rows: Int { rowStop - rowStart }
  public var columns: Int { columnStop - columnStart }
  public var scanPositions: Int { rows * columns }
}

/// Exact storage and geometry for a native Metal 4D-STEM browse load.
///
/// Cropping selects only source scan positions inside ``scanRegion``. A
/// `scanBin` larger than one stores the exact integer sum of each neighboring
/// scan block as `uint32`; incomplete edge bins retain every acquired source
/// position. No detector pixels are cropped or binned by this plan.
public struct Metal4DSTEMLoadPlan: Equatable, Hashable, Sendable {
  public static let supportedScanBins = [1, 2, 4, 8, 16]
  public static let supportedDetectorBins = [1, 2, 4]

  public let sourceScanRows: Int
  public let sourceScanColumns: Int
  public let detectorRows: Int
  public let detectorColumns: Int
  public let sourceBytesPerValue: Int
  public let scanRegion: Metal4DSTEMScanRegion
  public let scanBin: Int
  public let detectorBin: Int

  public init(
    sourceScanRows: Int,
    sourceScanColumns: Int,
    detectorRows: Int,
    detectorColumns: Int,
    sourceBytesPerValue: Int,
    scanRegion: Metal4DSTEMScanRegion,
    scanBin: Int = 1,
    detectorBin: Int = 1
  ) throws {
    guard sourceScanRows > 0, sourceScanColumns > 0,
      detectorRows > 0, detectorColumns > 0,
      sourceScanRows <= Int.max / sourceScanColumns,
      detectorRows <= Int.max / detectorColumns
    else { throw Metal4DSTEMLoadPlanError.invalidSourceShape }
    guard sourceBytesPerValue == 1 || sourceBytesPerValue == 2 else {
      throw Metal4DSTEMLoadPlanError.invalidSourceBytesPerValue(sourceBytesPerValue)
    }
    guard Self.supportedScanBins.contains(scanBin) else {
      throw Metal4DSTEMLoadPlanError.invalidScanBin(scanBin)
    }
    guard Self.supportedDetectorBins.contains(detectorBin) else {
      throw Metal4DSTEMLoadPlanError.invalidDetectorBin(detectorBin)
    }
    guard scanRegion.rowStart >= 0,
      scanRegion.rowStop > scanRegion.rowStart,
      scanRegion.rowStop <= sourceScanRows,
      scanRegion.columnStart >= 0,
      scanRegion.columnStop > scanRegion.columnStart,
      scanRegion.columnStop <= sourceScanColumns
    else { throw Metal4DSTEMLoadPlanError.invalidScanRegion }
    self.sourceScanRows = sourceScanRows
    self.sourceScanColumns = sourceScanColumns
    self.detectorRows = detectorRows
    self.detectorColumns = detectorColumns
    self.sourceBytesPerValue = sourceBytesPerValue
    self.scanRegion = scanRegion
    self.scanBin = scanBin
    self.detectorBin = detectorBin
  }

  public var outputScanRows: Int {
    (scanRegion.rows + scanBin - 1) / scanBin
  }

  public var outputScanColumns: Int {
    (scanRegion.columns + scanBin - 1) / scanBin
  }

  public var outputScanPositions: Int {
    outputScanRows * outputScanColumns
  }

  public var detectorPixels: Int { detectorRows * detectorColumns }

  public var outputDetectorRows: Int {
    (detectorRows + detectorBin - 1) / detectorBin
  }

  public var outputDetectorColumns: Int {
    (detectorColumns + detectorBin - 1) / detectorBin
  }

  public var outputDetectorPixels: Int {
    outputDetectorRows * outputDetectorColumns
  }

  public var residentBytesPerValue: Int {
    scanBin == 1 && detectorBin == 1
      ? sourceBytesPerValue : MemoryLayout<UInt32>.stride
  }

  public var residentVolumeBytes: UInt64 {
    UInt64(outputScanPositions) * UInt64(outputDetectorPixels)
      * UInt64(residentBytesPerValue)
  }

  public var isFullNative: Bool {
    scanBin == 1 && detectorBin == 1
      && scanRegion.rowStart == 0 && scanRegion.rowStop == sourceScanRows
      && scanRegion.columnStart == 0 && scanRegion.columnStop == sourceScanColumns
  }

  public var provenanceLabel: String {
    let region =
      "rows \(scanRegion.rowStart)..<\(scanRegion.rowStop), columns "
      + "\(scanRegion.columnStart)..<\(scanRegion.columnStop)"
    if scanBin == 1 && detectorBin == 1 {
      return isFullNative ? "full native scan" : "native scan crop · \(region)"
    }
    var reductions: [String] = []
    if scanBin > 1 { reductions.append("scan-sum bin \(scanBin)×\(scanBin)") }
    if detectorBin > 1 {
      reductions.append("detector-sum bin \(detectorBin)×\(detectorBin)")
    }
    return reductions.joined(separator: " · ") + " · \(region)"
  }

  /// Number of source scan positions summed into one output scan position.
  public func sourceContributionCount(outputRow: Int, outputColumn: Int) -> Int {
    guard outputRow >= 0, outputRow < outputScanRows,
      outputColumn >= 0, outputColumn < outputScanColumns
    else { return 0 }
    let localRowStart = outputRow * scanBin
    let localColumnStart = outputColumn * scanBin
    let rows = min(scanBin, scanRegion.rows - localRowStart)
    let columns = min(scanBin, scanRegion.columns - localColumnStart)
    return rows * columns
  }

  /// Number of source detector pixels summed into one output detector pixel.
  public func detectorContributionCount(outputRow: Int, outputColumn: Int) -> Int {
    guard outputRow >= 0, outputRow < outputDetectorRows,
      outputColumn >= 0, outputColumn < outputDetectorColumns
    else { return 0 }
    let sourceRowStart = outputRow * detectorBin
    let sourceColumnStart = outputColumn * detectorBin
    let rows = min(detectorBin, detectorRows - sourceRowStart)
    let columns = min(detectorBin, detectorColumns - sourceColumnStart)
    return rows * columns
  }
}
