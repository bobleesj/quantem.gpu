import Foundation
@preconcurrency import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

/// Original non-finite scan positions. Coordinates and donor indices are source-bound,
/// zero-based and row-major; interpolation never replaces the encoded measurements.
public struct EMPADScanValidity: Codable, Sendable {
  public let scanRows: Int
  public let scanColumns: Int
  public let invalidFrames: [Int]
  public var interpolatedFrames: [Int] {
    let invalid = Set(invalidFrames)
    return invalidFrames.filter { i in
      let r = i / scanColumns
      let c = i % scanColumns
      return r > 0 && r < scanRows - 1 && c > 0 && c < scanColumns - 1
        && neighbors(of: i).allSatisfy { !invalid.contains($0) }
    }
  }
  public func neighbors(of frame: Int) -> [Int] {
    [frame - scanColumns, frame + scanColumns, frame - 1, frame + 1]
  }
  public var metadata: [String: String] {
    guard !invalidFrames.isEmpty else { return [:] }
    let record: [String: Any] = [
      "schema": "quantem.gpu.scan-validity/v1", "scan_shape": [scanRows, scanColumns],
      "invalid_frames": invalidFrames, "coordinate_order": "row,column; zero-based",
      "direct_ptycho_method": "four-neighbor-dp-mean/v1",
      "interpolatable_frames": interpolatedFrames,
      "donors": interpolatedFrames.map {
        ["frame": $0, "neighbors": neighbors(of: $0)] as [String: Any]
      },
      "originals_preserved": true,
    ]
    let json = try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    return [
      "scanValidity": String(decoding: json, as: UTF8.self),
      "invalidScanPositions": invalidFrames.map { "(\($0 / scanColumns), \($0 % scanColumns))" }
        .joined(separator: ", "),
      "invalidScanCount": String(invalidFrames.count),
    ]
  }
}

extension MetalEMPADResidentSource {
  func validityPipeline(_ name: String) throws -> MTLComputePipelineState {
    let constants = MTLFunctionConstantValues()
    var decoded = true
    var pixels = UInt32(source.detectorPixelCount)
    var columns = UInt32(source.detectorShape.column)
    constants.setConstantValue(&decoded, type: .bool, index: 0)
    constants.setConstantValue(&decoded, type: .bool, index: 1)
    constants.setConstantValue(&pixels, type: .uint, index: 20)
    constants.setConstantValue(&columns, type: .uint, index: 21)
    let library = try Metal4DSTEMKernels.makeEMPADLibrary(device: device)
    return try device.makeComputePipelineState(
      function: library.makeFunction(name: name, constantValues: constants))
  }

  /// Audit bounded GPU windows during loading, including when restoring original QEM words.
  func detectInvalidScans(shouldCancel: () -> Bool) throws {
    let pipeline = try validityPipeline("empad_scan_validity")
    guard let queue = device.makeCommandQueue(),
      let flags = device.makeBuffer(length: source.frameCount * 4, options: .storageModeShared)
    else { throw Native4DSTEMIOError.invalidData("Free memory for the scan-validity mask.") }
    for chunk in chunks {
      if shouldCancel() { throw CancellationError() }
      guard let command = queue.makeCommandBuffer() else {
        throw Native4DSTEMIOError.invalidData("Metal queue unavailable.")
      }
      let workspace = try ans.workspace(
        frames: chunk.frameCount, budget: memoryBudgetBytes, command: command)
      try ans.encode(chunk, into: workspace, command: command)
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Native4DSTEMIOError.invalidData("Metal encoder unavailable.")
      }
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(workspace.words, offset: 0, index: 0)
      encoder.setBuffer(workspace.descriptors, offset: 0, index: 1)
      encoder.setBuffer(flags, offset: 0, index: 2)
      var first = UInt32(chunk.firstFrame)
      encoder.setBytes(&first, length: 4, index: 3)
      encoder.dispatchThreadgroups(
        MTLSize(width: chunk.frameCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      if let error = command.error { throw error }
    }
    invalidScanMask = flags
    let values = flags.contents().assumingMemoryBound(to: UInt32.self)
    scanValidity = EMPADScanValidity(
      scanRows: source.scanRows, scanColumns: source.scanColumns,
      invalidFrames: (0..<source.frameCount).filter { values[$0] != 0 })
  }

  /// Fail before fitting if the requested scan region includes an unrepairable gap.
  public func requireInterpolatableScans(rows: Range<Int>? = nil, columns: Range<Int>? = nil) throws
  {
    let rows = rows ?? 0..<source.scanRows
    let columns = columns ?? 0..<source.scanColumns
    let repaired = Set(scanValidity.interpolatedFrames)
    if scanValidity.invalidFrames.contains(where: {
      rows.contains($0 / source.scanColumns) && columns.contains($0 % source.scanColumns)
        && !repaired.contains($0)
    }) {
      throw Native4DSTEMIOError.invalidData(
        "Missing scan positions include a boundary or adjacent gap. Select a valid region; only isolated positions with four valid neighbors can be interpolated."
      )
    }
  }

  /// Fill only isolated missing scan positions in a transient detector-column buffer.
  /// Each detector pixel is averaged across four original neighboring scan positions.
  /// No detector-space smoothing, cascading fill, or original resident mutation occurs.
  func interpolateScanColumns(
    _ output: MTLBuffer, count: Int, rows: Int, columns: Int,
    command: MTLCommandBuffer
  ) throws {
    let frames = scanValidity.interpolatedFrames
    guard !frames.isEmpty else { return }
    if interpolationPipeline == nil {
      interpolationPipeline = try validityPipeline("empad_interpolate_scan_columns")
    }
    guard let pipeline = interpolationPipeline, let encoder = command.makeComputeCommandEncoder()
    else {
      throw Native4DSTEMIOError.invalidData("Metal interpolation encoder unavailable.")
    }
    let indices = frames.map {
      UInt32(($0 / source.scanColumns) * columns + $0 % source.scanColumns)
    }
    var dimensions = SIMD4<UInt32>(
      UInt32(rows * columns), UInt32(columns), UInt32(frames.count), UInt32(count))
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(output, offset: 0, index: 0)
    // A buffer avoids Metal's small setBytes limit on scans with many isolated gaps.
    let indexBuffer = indices.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
    }
    guard let indexBuffer else {
      encoder.endEncoding()
      throw Native4DSTEMIOError.invalidData("Free memory for interpolation indices.")
    }
    encoder.setBuffer(indexBuffer, offset: 0, index: 1)
    encoder.setBytes(&dimensions, length: MemoryLayout.size(ofValue: dimensions), index: 2)
    encoder.dispatchThreads(
      MTLSize(width: count * frames.count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(128, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    encoder.endEncoding()
  }
}
