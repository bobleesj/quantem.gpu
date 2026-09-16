import Foundation
import Metal
import Metal4DSTEMKernels

extension MetalRuntimeANSResidentSource {
  private func prepareDetectorColumnPipelines() throws {
    try requireLive()
    if detectorColumnsPipeline != nil { return }
    let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw Self.invalid("Missing detector-column kernel: \(name). Rebuild the backend resources.")
      }
      return try device.makeComputePipelineState(function: function)
    }
    detectorTotalsPipeline = try pipeline("streamed_counts_detector_total")
    detectorColumnsValidity = validPixels.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
    }
    guard detectorColumnsValidity != nil else { throw Self.invalid("Cannot allocate detector validity.") }
    detectorColumnsPipeline = try pipeline("streamed_counts_detector_columns")
  }

  /// Sum original counts over every scan position, retaining UInt64 precision.
  /// Example: `let sums = try source.detectorColumnSums()`.
  /// Serialize with other queries and release, as for resident diffraction.
  public func detectorColumnSums(shouldCancel: () -> Bool = { false }) throws -> [UInt64] {
    try prepareDetectorColumnPipelines()
    let pixels = shape[2] * shape[3]
    guard let output = device.makeBuffer(length: pixels * 8, options: .storageModeShared),
      let failure, let decodingTable, let detectorColumnsValidity, let detectorTotalsPipeline
    else { throw Self.invalid("Cannot allocate detector totals. Close inactive datasets and retry.") }
    memset(output.contents(), 0, output.length)
    memset(failure.contents(), 0, 4)
    for chunk in chunks {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
      else { throw Self.invalid("Cannot create detector-total command.") }
      encoder.setComputePipelineState(detectorTotalsPipeline)
      for (index, buffer) in [chunk.payload, chunk.offsets, chunk.models, decodingTable, failure, output].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      var parameters = [UInt64(chunk.scanCount), UInt64(pixels), UInt64(interval)]
      encoder.setBytes(&parameters, length: parameters.count * 8, index: 6)
      encoder.setBuffer(detectorColumnsValidity, offset: 0, index: 7)
      encoder.dispatchThreads(MTLSize(width: pixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
      if let error = command.error { throw error }
      try validateDetectorColumnDecoding()
    }
    return Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt64.self), count: pixels))
  }

  /// Extract exact plane-major UInt32 counts, optionally zero-extending the scan.
  /// No detector or measured scan samples are binned or dropped. The caller
  /// commits and waits, then calls `validateDetectorColumnDecoding()`.
  /// Example: `try source.encodeDetectorColumns(pixels: [0], into: output,
  /// commands: command, scanRows: 128, scanColumns: 128)`.
  public func encodeDetectorColumns(pixels: [Int], into output: MTLBuffer,
    commands: MTLCommandBuffer, scanRows: Int, scanColumns: Int) throws {
    try prepareDetectorColumnPipelines()
    guard !pixels.isEmpty, pixels.count <= 32, scanRows >= shape[0], scanColumns >= shape[1],
      scanRows <= 512, scanColumns <= 512,
      pixels.allSatisfy({ 0..<shape[2] * shape[3] ~= $0 }),
      output.length >= pixels.count * scanRows * scanColumns * 4,
      output.device.registryID == device.registryID,
      commands.commandQueue.device.registryID == device.registryID,
      let failure, let decodingTable, let detectorColumnsValidity, let detectorColumnsPipeline
    else { throw Self.invalid("Detector columns need 1...32 valid pixels and same-device full-scan output (at most 512 × 512).") }
    guard let clear = commands.makeBlitCommandEncoder() else { throw Self.invalid("Cannot initialize scan padding.") }
    clear.fill(buffer: output, range: 0..<pixels.count * scanRows * scanColumns * 4, value: 0)
    clear.endEncoding()
    guard let encoder = commands.makeComputeCommandEncoder() else { throw Self.invalid("Cannot encode detector columns.") }
    encoder.setComputePipelineState(detectorColumnsPipeline)
    var selected = pixels.map(UInt32.init)
    encoder.setBytes(&selected, length: selected.count * 4, index: 7)
    encoder.setBuffer(detectorColumnsValidity, offset: 0, index: 8)
    for chunk in chunks {
      for (index, buffer) in [chunk.payload, chunk.offsets, chunk.models, decodingTable, failure, output].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      var parameters = [chunk.scanCount, shape[2] * shape[3], interval, pixels.count,
        chunk.firstScan, shape[1], scanRows, scanColumns].map(UInt32.init)
      encoder.setBytes(&parameters, length: parameters.count * 4, index: 6)
      let jobs = ((chunk.scanCount + interval - 1) / interval) * pixels.count
      encoder.dispatchThreads(MTLSize(width: jobs, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    encoder.endEncoding()
  }

  /// Reject a failed entropy decode before publishing derived scientific data.
  public func validateDetectorColumnDecoding() throws {
    try requireLive()
    guard failure?.contents().load(as: UInt32.self) == 0 else {
      throw Self.invalid("Compressed counts failed validation. Reopen the original acquisition.")
    }
  }
}
