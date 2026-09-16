import Foundation
import Metal
import Metal4DSTEMKernels

extension MetalRuntimeANSResidentSource {
  /// Average original counts over a rectangle or circle of scan positions.
  ///
  /// Bounds are half-open (row, column) ranges. Circles require square bounds
  /// and include pixel centers on the circumference with equal weight. Every
  /// detector pixel is retained, matching point diffraction, including pixels
  /// excluded from virtual-detector products. Sums are exact UInt64; division
  /// rounds once to Float32. No dense scan-by-detector allocation is made.
  /// Serialize with other resident queries and release.
  ///
  /// Example: `try source.meanDiffractionPattern(rows: 10..<20,
  /// columns: 30..<40, shape: .circle)`.
  public func meanDiffractionPattern(
    rows: Range<Int>? = nil, columns: Range<Int>? = nil,
    shape selectionShape: MetalScanRegionShape = .rectangle
  ) throws -> MetalCompactH5MeanDiffraction {
    try requireLive()
    let rows = rows ?? 0..<shape[0]
    let columns = columns ?? 0..<shape[1]
    guard !rows.isEmpty, !columns.isEmpty,
      rows.lowerBound >= 0, rows.upperBound <= shape[0],
      columns.lowerBound >= 0, columns.upperBound <= shape[1],
      selectionShape != .circle || rows.count == columns.count
    else {
      throw Self.invalid(
        "Choose a nonempty scan region inside the image; circle bounds must be square.")
    }
    let started = CFAbsoluteTimeGetCurrent()
    if regionMeanPipeline == nil {
      let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
      guard let function = library.makeFunction(name: "streamed_counts_region_total") else {
        throw Self.invalid("Missing region diffraction kernel. Rebuild the backend resources.")
      }
      regionMeanPipeline = try device.makeComputePipelineState(function: function)
    }
    let pixels = shape[2] * shape[3]
    let output = try Self.sharedBuffer(
      device: device, bytes: pixels * 8, label: "Region diffraction sums")
    guard let pipeline = regionMeanPipeline, let failure, let decodingTable,
      let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else {
      throw Self.invalid("Cannot prepare region diffraction. Close inactive datasets and retry.")
    }
    memset(output.contents(), 0, output.length)
    memset(failure.contents(), 0, 4)
    encoder.setComputePipelineState(pipeline)
    var dispatches = 0
    for chunk in chunks {
      let first = rows.lowerBound * shape[1] + columns.lowerBound
      let stop = (rows.upperBound - 1) * shape[1] + columns.upperBound
      guard chunk.firstScan < stop, chunk.firstScan + chunk.scanCount > first else { continue }
      for (index, buffer) in [
        chunk.payload, chunk.offsets, chunk.models, decodingTable, failure, output,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      var parameters = [
        chunk.scanCount, pixels, interval, chunk.firstScan, shape[1],
        rows.lowerBound, rows.upperBound, columns.lowerBound, columns.upperBound,
        Int(selectionShape.rawValue),
      ].map(UInt64.init)
      encoder.setBytes(&parameters, length: parameters.count * 8, index: 6)
      encoder.dispatchThreads(
        MTLSize(width: pixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.memoryBarrier(scope: .buffers)
      dispatches += 1
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try checkFailure(command)
    let sums = Array(
      UnsafeBufferPointer(
        start: output.contents().assumingMemoryBound(to: UInt64.self), count: pixels))
    let count = selectionShape.sampleCount(rowCount: rows.count, columnCount: columns.count)
    return MetalCompactH5MeanDiffraction(
      detectorSum: sums,
      mean: sums.map { Float(Double($0) / Double(count)) },
      wallMilliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1_000,
      gpuMilliseconds: max(0, command.gpuEndTime - command.gpuStartTime) * 1_000,
      dispatchCount: dispatches, readbackBytes: UInt64(output.length))
  }
}
