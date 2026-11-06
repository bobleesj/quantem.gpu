import Foundation
import Metal
import Metal4DSTEMKernels

extension MetalRuntimeANSResidentSource {
  private func prepareDetectorColumnValidity() throws {
    try requireLive()
    if detectorColumnsValidity != nil { return }
    detectorColumnsValidity = validPixels.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
    }
    guard detectorColumnsValidity != nil else {
      throw Self.invalid("Cannot allocate detector validity. Close inactive datasets and retry.")
    }
  }

  private func prepareDetectorColumnPipeline(nativeWidth: Bool) throws -> MTLComputePipelineState {
    try prepareDetectorColumnValidity()
    if let cached = nativeWidth ? detectorColumnsFlatPipeline : detectorColumnsPipeline {
      return cached
    }
    // Compile only the layout and scheduling path requested by this resident.
    // Independent chunks share the bounded output; their table is at most 4 KiB.
    let prepared = detectorColumnsPipeline != nil || detectorColumnsFlatPipeline != nil
    let parallel =
      detectorColumnChunks != nil
      || (!prepared
        && ProcessInfo.processInfo.environment["QUANTEM_METAL_PARALLEL_COLUMNS"] != "0"
        && (2...128).contains(chunks.count))
    let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
    let constants = MTLFunctionConstantValues()
    var width = nativeWidth
    constants.setConstantValue(&width, type: .bool, index: 21)
    let function = try library.makeFunction(
      name: parallel
        ? "streamed_counts_detector_columns_parallel" : "streamed_counts_detector_columns",
      constantValues: constants)
    let pipeline = try device.makeComputePipelineState(function: function)
    if parallel && detectorColumnChunks == nil {
      let arguments = function.makeArgumentEncoder(bufferIndex: 0)
      guard arguments.encodedLength == 32 else {
        throw Self.invalid(
          "Detector-column argument layout is incompatible. Rebuild the backend resources.")
      }
      guard
        let table = device.makeBuffer(
          length: chunks.count * arguments.encodedLength,
          options: .storageModeShared)
      else {
        throw Self.invalid(
          "Cannot allocate bounded detector-column metadata. Close inactive datasets and retry.")
      }
      for (index, chunk) in chunks.enumerated() {
        arguments.setArgumentBuffer(table, offset: index * arguments.encodedLength)
        arguments.setBuffer(chunk.payload, offset: 0, index: 0)
        arguments.setBuffer(chunk.offsets, offset: 0, index: 1)
        arguments.setBuffer(chunk.models, offset: 0, index: 2)
        arguments.constantData(at: 3).storeBytes(of: UInt32(chunk.scanCount), as: UInt32.self)
        arguments.constantData(at: 4).storeBytes(of: UInt32(chunk.firstScan), as: UInt32.self)
      }
      detectorColumnChunks = table
    }
    // Publish the pipeline only after all of its resources are prepared.
    if nativeWidth {
      detectorColumnsFlatPipeline = pipeline
    } else {
      detectorColumnsPipeline = pipeline
    }
    return pipeline
  }

  /// Sum original counts over every scan position, retaining UInt64 precision.
  /// Example: `let sums = try source.detectorColumnSums()`.
  /// Serialize with other queries and release, as for resident diffraction.
  public func detectorColumnSums(shouldCancel: () -> Bool = { false }) throws -> [UInt64] {
    try prepareDetectorColumnValidity()
    if detectorTotalsPipeline == nil {
      let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
      guard let function = library.makeFunction(name: "streamed_counts_detector_total") else {
        throw Self.invalid("Missing detector-total kernel. Rebuild the backend resources.")
      }
      detectorTotalsPipeline = try device.makeComputePipelineState(function: function)
    }
    let pixels = shape[2] * shape[3]
    guard let output = device.makeBuffer(length: pixels * 8, options: .storageModeShared),
      let failure, let decodingTable, let detectorColumnsValidity, let detectorTotalsPipeline
    else {
      throw Self.invalid("Cannot allocate detector totals. Close inactive datasets and retry.")
    }
    memset(output.contents(), 0, output.length)
    memset(failure.contents(), 0, 4)
    for chunk in chunks {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      guard let command = queue.makeCommandBuffer(),
        let encoder = command.makeComputeCommandEncoder()
      else { throw Self.invalid("Cannot create detector-total command.") }
      encoder.setComputePipelineState(detectorTotalsPipeline)
      for (index, buffer) in [
        chunk.payload, chunk.offsets, chunk.models, decodingTable, failure, output,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      var parameters = [UInt64(chunk.scanCount), UInt64(pixels), UInt64(interval)]
      encoder.setBytes(&parameters, length: parameters.count * 8, index: 6)
      encoder.setBuffer(detectorColumnsValidity, offset: 0, index: 7)
      encoder.dispatchThreads(
        MTLSize(width: pixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      if let error = command.error { throw error }
      try validateDetectorColumnDecoding()
    }
    return Array(
      UnsafeBufferPointer(
        start: output.contents().assumingMemoryBound(to: UInt64.self), count: pixels))
  }

  /// Extract exact plane-major UInt32 counts, optionally zero-extending the scan.
  /// Flagged detector pixels are zero by default. Set `includeFlaggedPixels`
  /// to read their stored counts when replacing them in an exact moment basis.
  /// This never modifies the resident or its validity mask.
  /// No detector or measured scan samples are binned or dropped. The caller
  /// commits and waits, then calls `validateDetectorColumnDecoding()`.
  /// Example: `try source.encodeDetectorColumns(pixels: [0], into: output,
  /// commands: command, scanRows: 128, scanColumns: 128)`.
  public func encodeDetectorColumns(
    pixels: [Int], into output: MTLBuffer,
    commands: MTLCommandBuffer, scanRows: Int, scanColumns: Int,
    includeFlaggedPixels: Bool = false
  ) throws {
    let pipeline = try prepareDetectorColumnPipeline(nativeWidth: scanColumns == shape[1])
    guard !pixels.isEmpty, pixels.count <= 32, scanRows >= shape[0], scanColumns >= shape[1],
      scanRows <= 512, scanColumns <= 512,
      pixels.allSatisfy({ 0..<shape[2] * shape[3] ~= $0 }),
      output.length >= pixels.count * scanRows * scanColumns * 4,
      output.device.registryID == device.registryID,
      commands.commandQueue.device.registryID == device.registryID,
      let failure, let decodingTable, let detectorColumnsValidity
    else {
      throw Self.invalid(
        "Detector columns need 1...32 valid pixels and same-device full-scan output (at most 512 × 512)."
      )
    }
    guard let clear = commands.makeBlitCommandEncoder() else {
      throw Self.invalid("Cannot initialize scan padding.")
    }
    clear.fill(buffer: output, range: 0..<pixels.count * scanRows * scanColumns * 4, value: 0)
    clear.endEncoding()
    guard let encoder = commands.makeComputeCommandEncoder() else {
      throw Self.invalid("Cannot encode detector columns.")
    }
    encoder.setComputePipelineState(pipeline)
    var selected = pixels.map(UInt32.init)
    encoder.setBytes(&selected, length: selected.count * 4, index: 7)
    encoder.setBuffer(detectorColumnsValidity, offset: 0, index: 8)
    if let table = detectorColumnChunks {
      encoder.setBuffer(table, offset: 0, index: 0)
      encoder.setBuffer(decodingTable, offset: 0, index: 3)
      encoder.setBuffer(failure, offset: 0, index: 4)
      encoder.setBuffer(output, offset: 0, index: 5)
      encoder.useResources(chunks.flatMap { [$0.payload, $0.offsets, $0.models] }, usage: .read)
      var parameters = [
        0, shape[2] * shape[3], interval, pixels.count, 0, shape[1],
        scanRows, scanColumns, includeFlaggedPixels ? 1 : 0,
      ].map(UInt32.init)
      encoder.setBytes(&parameters, length: parameters.count * 4, index: 6)
      let maximumScans = chunks.map(\.scanCount).max() ?? 0
      let blocks = (maximumScans + interval - 1) / interval
      encoder.dispatchThreadgroups(
        MTLSize(width: (blocks + 31) / 32, height: pixels.count, depth: chunks.count),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      encoder.endEncoding()
      return
    }
    for chunk in chunks {
      for (index, buffer) in [
        chunk.payload, chunk.offsets, chunk.models, decodingTable, failure, output,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      var parameters = [
        chunk.scanCount, shape[2] * shape[3], interval, pixels.count,
        chunk.firstScan, shape[1], scanRows, scanColumns, includeFlaggedPixels ? 1 : 0,
      ].map(UInt32.init)
      encoder.setBytes(&parameters, length: parameters.count * 4, index: 6)
      let blocks = (chunk.scanCount + interval - 1) / interval
      encoder.dispatchThreadgroups(
        MTLSize(width: (blocks + 31) / 32, height: pixels.count, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
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
