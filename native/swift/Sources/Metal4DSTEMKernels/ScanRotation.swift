import Foundation
import Metal

/// Exact counterclockwise quarter turns in the displayed scan frame.
public enum ScanQuarterTurn: UInt32, Sendable {
  case identity = 0
  case counterclockwise90 = 1
  case halfTurn = 2
  case clockwise90 = 3
}

/// Scan shape produced by a quarter turn.
public struct RotatedScanShape: Equatable, Sendable {
  public let rows: Int
  public let columns: Int

  public init(rows: Int, columns: Int) {
    self.rows = rows
    self.columns = columns
  }
}

/// Errors raised before an exact scan rotation is encoded.
public enum MetalScanRotationError: LocalizedError {
  case invalidGeometry(scanRows: Int, scanColumns: Int, wordsPerScan: Int)
  case bufferTooSmall(name: String, expectedBytes: Int, actualBytes: Int)
  case aliasedBuffers
  case commandEncoderUnavailable

  public var errorDescription: String? {
    switch self {
    case .invalidGeometry(let scanRows, let scanColumns, let wordsPerScan):
      "Scan rotation requires positive rows, columns, and words per scan; "
        + "got (\(scanRows), \(scanColumns), \(wordsPerScan))."
    case .bufferTooSmall(let name, let expectedBytes, let actualBytes):
      "\(name) needs at least \(expectedBytes) bytes for this scan rotation; "
        + "got \(actualBytes)."
    case .aliasedBuffers:
      "Scan rotation source and destination must be different Metal buffers."
    case .commandEncoderUnavailable:
      "Metal could not create a compute encoder for scan rotation."
    }
  }
}

private struct ScanQuarterTurnParameters {
  var sourceRows: UInt32
  var sourceColumns: UInt32
  var wordsPerScan: UInt32
  var quarterTurns: UInt32
  var outputRows: UInt32
  var outputColumns: UInt32
  var totalWords: UInt32
  var padding: UInt32 = 0
}

/// Reusable Metal encoder for exact, GPU-resident scan-plane quarter turns.
public final class MetalScanRotator: @unchecked Sendable {
  private let pipeline: MTLComputePipelineState

  public init(device: MTLDevice) throws {
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let function = try library.function(
      named: Metal4DSTEMKernels.rotateScanWordsQuarterTurnFunction
    )
    pipeline = try device.makeComputePipelineState(function: function)
  }

  /// Encode a scan-plane quarter turn without changing detector payload words.
  ///
  /// Source and destination use scan-major 32-bit words. Packed integer counts
  /// and floating-point bit patterns therefore pass through without conversion.
  /// Reuse the destination buffer and this rotator during interaction.
  ///
  /// ```swift
  /// let rotator = try MetalScanRotator(device: device)
  /// let shape = try rotator.encode(
  ///   source: source,
  ///   destination: destination,
  ///   scanRows: 256,
  ///   scanColumns: 256,
  ///   wordsPerScan: 1152,
  ///   quarterTurn: .clockwise90,
  ///   commandBuffer: commandBuffer
  /// )
  /// ```
  @discardableResult
  public func encode(
    source: MTLBuffer,
    destination: MTLBuffer,
    scanRows: Int,
    scanColumns: Int,
    wordsPerScan: Int,
    quarterTurn: ScanQuarterTurn,
    commandBuffer: MTLCommandBuffer
  ) throws -> RotatedScanShape {
    guard scanRows > 0, scanColumns > 0, wordsPerScan > 0 else {
      throw MetalScanRotationError.invalidGeometry(
        scanRows: scanRows,
        scanColumns: scanColumns,
        wordsPerScan: wordsPerScan
      )
    }
    guard source !== destination else {
      throw MetalScanRotationError.aliasedBuffers
    }
    let (scanCount, scanOverflow) = scanRows.multipliedReportingOverflow(
      by: scanColumns
    )
    let (totalWords, wordOverflow) = scanCount.multipliedReportingOverflow(
      by: wordsPerScan
    )
    let (byteCount, byteOverflow) = totalWords.multipliedReportingOverflow(
      by: MemoryLayout<UInt32>.stride
    )
    guard !scanOverflow, !wordOverflow, !byteOverflow else {
      throw MetalScanRotationError.invalidGeometry(
        scanRows: scanRows,
        scanColumns: scanColumns,
        wordsPerScan: wordsPerScan
      )
    }
    guard source.length >= byteCount else {
      throw MetalScanRotationError.bufferTooSmall(
        name: "source",
        expectedBytes: byteCount,
        actualBytes: source.length
      )
    }
    guard destination.length >= byteCount else {
      throw MetalScanRotationError.bufferTooSmall(
        name: "destination",
        expectedBytes: byteCount,
        actualBytes: destination.length
      )
    }
    guard totalWords <= Int(UInt32.max) else {
      throw MetalScanRotationError.invalidGeometry(
        scanRows: scanRows,
        scanColumns: scanColumns,
        wordsPerScan: wordsPerScan
      )
    }
    let oddTurn = quarterTurn.rawValue % 2 == 1
    let outputRows = oddTurn ? scanColumns : scanRows
    let outputColumns = oddTurn ? scanRows : scanColumns
    var parameters = ScanQuarterTurnParameters(
      sourceRows: UInt32(scanRows),
      sourceColumns: UInt32(scanColumns),
      wordsPerScan: UInt32(wordsPerScan),
      quarterTurns: quarterTurn.rawValue,
      outputRows: UInt32(outputRows),
      outputColumns: UInt32(outputColumns),
      totalWords: UInt32(totalWords)
    )
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw MetalScanRotationError.commandEncoderUnavailable
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    encoder.setBytes(
      &parameters,
      length: MemoryLayout<ScanQuarterTurnParameters>.stride,
      index: 2
    )
    let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    encoder.dispatchThreads(
      MTLSize(width: totalWords, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
    )
    encoder.endEncoding()
    return RotatedScanShape(rows: outputRows, columns: outputColumns)
  }
}

extension MTLLibrary {
  fileprivate func function(named name: String) throws -> MTLFunction {
    guard let function = makeFunction(name: name) else {
      throw Metal4DSTEMKernelsError.missingResource("Metal function \(name)")
    }
    return function
  }
}
