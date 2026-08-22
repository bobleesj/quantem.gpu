import Foundation

/// Validation failures for a native Metal 4D-STEM load selection.
public enum Metal4DSTEMLoadPlanError: LocalizedError, Equatable {
  case invalidSourceShape
  case invalidScanRegion
  case invalidScanBin(Int)
  case invalidDetectorBin(Int)
  case invalidSourceBytesPerValue(Int)
  case invalidStreamingDepth(Int)
  case insufficientStreamingScratchBudget

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
    case .invalidStreamingDepth(let depth):
      "Streaming depth \(depth) is unsupported. Choose a positive value."
    case .insufficientStreamingScratchBudget:
      "The streaming scratch budget cannot hold one selected source scan row."
    }
  }
}

/// Validation failures while proving exact integer accumulator widths.
public enum Metal4DSTEMExactAccumulatorBoundsError: LocalizedError, Equatable {
  case arithmeticOverflow

  public var errorDescription: String? {
    switch self {
    case .arithmeticOverflow:
      "The exact detector accumulator bound exceeds UInt64. Use the general 64-bit "
        + "reduction path for this load plan."
    }
  }
}

/// Conservative exact bounds for fused detector sums and moments.
///
/// The bounds cover every source detector count accumulated into one output
/// scan position, including scan summation and incomplete detector edge bins.
/// Detector row and column moments use the output detector coordinates after
/// detector binning. Consumers may use 32-bit fused accumulators only when
/// ``fitsUInt32Accumulators`` is true; otherwise they must use the general
/// 64-bit reduction path.
public struct Metal4DSTEMExactAccumulatorBounds: Equatable, Hashable, Sendable {
  public let maxSourceCount: UInt32
  public let maximumScanContributions: Int
  public let maximumDetectorSum: UInt64
  public let maximumDetectorRowMoment: UInt64
  public let maximumDetectorColumnMoment: UInt64

  public var fitsUInt32Accumulators: Bool {
    let limit = UInt64(UInt32.max)
    return maximumDetectorSum <= limit
      && maximumDetectorRowMoment <= limit
      && maximumDetectorColumnMoment <= limit
  }
}

/// Conservative exact bound for one working 4D-STEM detector value.
///
/// The contribution count includes both scan-space and detector-space exact
/// summation. Consumers may store the working volume as `uint16` only when
/// ``fitsUInt16`` is true, or as `uint32` only when ``fitsUInt32`` is true.
public struct Metal4DSTEMExactOutputSampleBounds: Equatable, Hashable, Sendable {
  public let maxSourceCount: UInt32
  public let maximumScanContributions: Int
  public let maximumDetectorContributions: Int
  public let maximumTotalContributions: Int
  public let maximumOutputCount: UInt64

  public var fitsUInt16: Bool {
    maximumOutputCount <= UInt64(UInt16.max)
  }

  public var fitsUInt32: Bool {
    maximumOutputCount <= UInt64(UInt32.max)
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
/// `scanBin` larger than one forms the exact integer sum of each neighboring
/// scan block; incomplete edge bins retain every acquired source position. A
/// `detectorBin` larger than one forms exact integer detector-pixel sums;
/// incomplete detector edge bins retain every acquired pixel. The plan budgets
/// conservative `uint32` storage. A loader may use a narrower exact integer
/// representation only after proving the selected source and contribution
/// bounds fit it, and must report that output dtype in provenance.
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

  /// Prove conservative integer bounds for fused detector accumulators.
  ///
  /// - Parameter maxSourceCount: Maximum value of one source detector pixel,
  ///   established by an exact source audit.
  /// - Returns: Bounds for the detector sum and output-coordinate moments.
  /// - Throws: ``Metal4DSTEMExactAccumulatorBoundsError/arithmeticOverflow``
  ///   when the bound cannot be represented by `UInt64`.
  public func exactAccumulatorBounds(
    maxSourceCount: UInt32
  ) throws -> Metal4DSTEMExactAccumulatorBounds {
    let scanRows = min(scanBin, scanRegion.rows)
    let scanColumns = min(scanBin, scanRegion.columns)
    let maximumScanContributions = scanRows * scanColumns

    let factors = [
      UInt64(maxSourceCount),
      UInt64(maximumScanContributions),
      UInt64(detectorRows),
      UInt64(detectorColumns),
    ]
    var maximumDetectorSum = UInt64(1)
    for factor in factors {
      let product = maximumDetectorSum.multipliedReportingOverflow(by: factor)
      guard !product.overflow else {
        throw Metal4DSTEMExactAccumulatorBoundsError.arithmeticOverflow
      }
      maximumDetectorSum = product.partialValue
    }

    func momentBound(coordinateMaximum: Int) throws -> UInt64 {
      let product = maximumDetectorSum.multipliedReportingOverflow(
        by: UInt64(coordinateMaximum)
      )
      guard !product.overflow else {
        throw Metal4DSTEMExactAccumulatorBoundsError.arithmeticOverflow
      }
      return product.partialValue
    }

    return Metal4DSTEMExactAccumulatorBounds(
      maxSourceCount: maxSourceCount,
      maximumScanContributions: maximumScanContributions,
      maximumDetectorSum: maximumDetectorSum,
      maximumDetectorRowMoment: try momentBound(
        coordinateMaximum: outputDetectorRows - 1
      ),
      maximumDetectorColumnMoment: try momentBound(
        coordinateMaximum: outputDetectorColumns - 1
      )
    )
  }

  /// Prove a conservative bound for every exact working detector value.
  ///
  /// - Parameter maxSourceCount: Maximum value of one source detector pixel,
  ///   established by an exact source audit.
  /// - Returns: The largest possible scan-and-detector sum for this plan.
  /// - Throws: ``Metal4DSTEMExactAccumulatorBoundsError/arithmeticOverflow``
  ///   when the contribution count or bound cannot be represented exactly.
  public func exactOutputSampleBounds(
    maxSourceCount: UInt32
  ) throws -> Metal4DSTEMExactOutputSampleBounds {
    let maximumScanContributions =
      min(scanBin, scanRegion.rows) * min(scanBin, scanRegion.columns)
    let maximumDetectorContributions =
      min(detectorBin, detectorRows) * min(detectorBin, detectorColumns)
    let contributionProduct = maximumScanContributions.multipliedReportingOverflow(
      by: maximumDetectorContributions
    )
    guard !contributionProduct.overflow else {
      throw Metal4DSTEMExactAccumulatorBoundsError.arithmeticOverflow
    }
    let totalContributions = contributionProduct.partialValue
    let bound = UInt64(maxSourceCount).multipliedReportingOverflow(
      by: UInt64(totalContributions)
    )
    guard !bound.overflow else {
      throw Metal4DSTEMExactAccumulatorBoundsError.arithmeticOverflow
    }
    return Metal4DSTEMExactOutputSampleBounds(
      maxSourceCount: maxSourceCount,
      maximumScanContributions: maximumScanContributions,
      maximumDetectorContributions: maximumDetectorContributions,
      maximumTotalContributions: totalContributions,
      maximumOutputCount: bound.partialValue
    )
  }
}

/// Reusable streaming geometry for selected 4D-STEM loads.
///
/// The plan keeps total scratch allocation within `scratchBudgetBytes`, aligns
/// non-final batches to the scan bin so exact scan sums keep stable destination
/// offsets, and reports every derived size needed for transparent provenance.
public struct Metal4DSTEMStreamingPlan: Equatable, Hashable, Sendable {
  /// Balanced decode/layout overlap for Macs with ample unified memory.
  public static let recommendedDepth = 4

  /// Hardware-memory-aware decode/layout depth.
  ///
  /// Physical 8 GB M2 measurements favor depth 2: depth 1 loses overlap while
  /// depths 4 and 8 increase paging and tail latency. Larger-memory Macs retain
  /// depth 4 for greater overlap without the constrained-memory penalty.
  public static func recommendedDepth(physicalMemoryBytes: UInt64) -> Int {
    physicalMemoryBytes <= (UInt64(16) << 30) ? 2 : recommendedDepth
  }

  public let depth: Int
  public let rowsPerBatch: Int
  public let framesPerBuffer: Int
  public let bytesPerBuffer: UInt64
  public let totalScratchBytes: UInt64
  public let batchCount: Int

  public init(
    loadPlan: Metal4DSTEMLoadPlan,
    scratchBudgetBytes: UInt64,
    preferredDepth: Int,
    stagingBytesPerValue: Int? = nil
  ) throws {
    guard preferredDepth > 0 else {
      throw Metal4DSTEMLoadPlanError.invalidStreamingDepth(preferredDepth)
    }
    let stagingBytes = stagingBytesPerValue ?? loadPlan.sourceBytesPerValue
    guard stagingBytes == 1 || stagingBytes == 2 else {
      throw Metal4DSTEMLoadPlanError.invalidSourceBytesPerValue(stagingBytes)
    }
    guard
      let detectorBytes = UInt64(exactly: loadPlan.detectorPixels)?
        .multipliedReportingOverflow(by: UInt64(stagingBytes)),
      !detectorBytes.overflow,
      let rowBytesProduct = UInt64(exactly: loadPlan.scanRegion.columns)?
        .multipliedReportingOverflow(by: detectorBytes.partialValue),
      !rowBytesProduct.overflow,
      rowBytesProduct.partialValue > 0
    else { throw Metal4DSTEMLoadPlanError.invalidSourceShape }

    let rowBytes = rowBytesProduct.partialValue
    let maxDepth = min(preferredDepth, loadPlan.scanRegion.rows)
    for candidateDepth in stride(from: maxDepth, through: 1, by: -1) {
      let rowsCapacity = Int(scratchBudgetBytes / UInt64(candidateDepth) / rowBytes)
      guard rowsCapacity > 0 else { continue }

      let rows: Int
      if rowsCapacity >= loadPlan.scanRegion.rows {
        rows = loadPlan.scanRegion.rows
      } else {
        let aligned = rowsCapacity - (rowsCapacity % max(1, loadPlan.scanBin))
        guard aligned > 0 else { continue }
        rows = aligned
      }

      let frames = rows * loadPlan.scanRegion.columns
      let bufferBytes = UInt64(rows) * rowBytes
      let batches = (loadPlan.scanRegion.rows + rows - 1) / rows
      let activeDepth = min(candidateDepth, batches)
      self.depth = activeDepth
      self.rowsPerBatch = rows
      self.framesPerBuffer = frames
      self.bytesPerBuffer = bufferBytes
      self.totalScratchBytes = bufferBytes * UInt64(activeDepth)
      self.batchCount = batches
      return
    }

    throw Metal4DSTEMLoadPlanError.insufficientStreamingScratchBudget
  }
}
