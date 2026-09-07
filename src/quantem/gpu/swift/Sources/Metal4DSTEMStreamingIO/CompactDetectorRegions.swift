import Foundation
import Metal

/// Optional exact sums over whole detector blocks. Source counts are untouched.
struct CompactDetectorRegions {
  // Additional resident summaries are a diagnostic tradeoff, not the default.
  // Normal interactions use the original compact counts without this cache.
  static var enabled: Bool {
    OriginalPackingDiagnostics.enabled("DETECTOR_REGIONS")
  }

  let blockSide: Int
  let shards: [CompactResidentShard]
  let entries: MTLBuffer
  let intermediate: MTLBuffer
  let bytes: UInt64

  /// Replace only complete, equal-coefficient blocks; every boundary stays raw.
  func decompose(_ raw: inout [CompactDetectorEntry], rows: Int, columns: Int) -> Int {
    guard raw.count >= blockSide * blockSide else { return 0 }
    var coefficients = [Int32](repeating: 0, count: rows * columns)
    for entry in raw { coefficients[Int(entry.pixel)] = entry.coefficient }
    var blocks: [CompactDetectorEntry] = []
    for row in stride(from: 0, to: rows, by: blockSide) {
      for col in stride(from: 0, to: columns, by: blockSide) {
        let coefficient = coefficients[row * columns + col]
        guard coefficient != 0 else { continue }
        var complete = true
        for r in row..<(row + blockSide) {
          for c in col..<(col + blockSide) where coefficients[r * columns + c] != coefficient {
            complete = false
          }
        }
        guard complete else { continue }
        blocks.append(
          CompactDetectorEntry(
            pixel: UInt32((row / blockSide) * (columns / blockSide) + col / blockSide),
            coefficient: coefficient))
        for r in row..<(row + blockSide) {
          for c in col..<(col + blockSide) { coefficients[r * columns + c] = 0 }
        }
      }
    }
    guard !blocks.isEmpty else { return 0 }
    raw = coefficients.indices.compactMap { pixel in
      coefficients[pixel] == 0
        ? nil
        : CompactDetectorEntry(pixel: UInt32(pixel), coefficient: coefficients[pixel])
    }
    blocks.withUnsafeBytes {
      entries.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
    }
    return blocks.count
  }

  struct Build {
    let auxiliary: CompactDetectorRegions?
    let scratchBytes: UInt64
    let peakBytes: UInt64
    let gpuMilliseconds: Double
    let status: String
  }

  /// Build one bounded shard at a time, before original-resident publication.
  static func build(
    shards: [CompactResidentShard], metadata: MetalCompactH5Metadata,
    headerWords: Int, headerEncoding: UInt32, payloadLayout: UInt32,
    device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary,
    availableBytes: UInt64, shouldCancel: () -> Bool
  ) throws -> Build {
    #if QGPU_PACKING_DIAGNOSTICS
      let started = ContinuousClock.now
    #endif
    let blockSide = 8
    let blockPixels = blockSide * blockSide
    let maximumWidth = 22  // 64 * 65535 fits exactly in 22 bits.
    var gpuMilliseconds = 0.0
    var peakBytes: UInt64 = 0
    func result(_ status: String, auxiliary: CompactDetectorRegions? = nil, scratch: UInt64 = 0)
      -> Build
    {
      #if QGPU_PACKING_DIAGNOSTICS
        let duration = started.duration(to: .now)
        let wall =
          Double(duration.components.seconds) * 1_000
          + Double(duration.components.attoseconds) / 1e15
        let record: [String: Any] = [
          "status": status, "block_size": blockSide, "sum_dtype": "uint32",
          "maximum_supported_sum_width": maximumWidth, "resident_bytes": auxiliary?.bytes ?? 0,
          "scratch_bytes": scratch, "peak_additional_bytes": peakBytes,
          "gpu_ms": gpuMilliseconds, "wall_ms": wall,
          "verified_values": auxiliary == nil
            ? 0
            : UInt64(metadata.scanCount) * UInt64(metadata.detectorPixelCount / blockPixels),
        ]
        if let json = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
          let text = String(data: json, encoding: .utf8)
        {
          fputs("ORIGINAL_DETECTOR_REGIONS \(text)\n", stderr)
        }
      #endif
      return Build(
        auxiliary: auxiliary, scratchBytes: scratch, peakBytes: peakBytes,
        gpuMilliseconds: gpuMilliseconds, status: status)
    }
    guard metadata.detectorRows.isMultiple(of: blockSide),
      metadata.detectorColumns.isMultiple(of: blockSide),
      metadata.scansPerShard > 0, metadata.scansPerShard <= 4096,
      metadata.scansPerShard.isMultiple(of: 32),
      metadata.scanCount == metadata.scansPerShard * shards.count,
      [UInt32(1), 2].contains(headerEncoding), payloadLayout == 0,
      metadata.excludedDetectorPixels.isEmpty
    else { return result("unsupportedGeometry") }
    let blocks = metadata.detectorPixelCount / blockPixels
    let frames = metadata.scansPerShard
    let cells = UInt64(blocks) * UInt64(frames / 32)
    let valuesBytes = UInt64(blocks) * UInt64(frames) * 4
    let descriptorBytes = cells * 4
    let worstPayloadBytes = max(4, cells * UInt64(maximumWidth) * 4)
    let mapBytes = UInt64(metadata.scanCount) * 4
    let entriesBytes = UInt64(blocks) * 8
    guard blocks > 0, UInt32(exactly: cells * UInt64(maximumWidth)) != nil,
      cells * UInt64(maximumWidth) < 1 << 27,
      [valuesBytes, descriptorBytes, worstPayloadBytes, mapBytes, entriesBytes]
        .allSatisfy({ $0 <= UInt64(device.maxBufferLength) && $0 <= UInt64(Int.max) })
    else { return result("unsupportedGeometry") }
    // Conservative page rounding before allocation; actual allocatedSize is
    // checked as well. This does not consume the caller's reserved headroom.
    func page(_ bytes: UInt64) -> UInt64 { (bytes + 16_383) / 16_384 * 16_384 }
    let worstRetained =
      UInt64(shards.count) * (page(descriptorBytes) + page(worstPayloadBytes))
      + page(mapBytes) + page(entriesBytes)
    let worstScratch = page(valuesBytes) + page(4) * 2
    guard worstRetained <= availableBytes, worstScratch <= availableBytes - worstRetained else {
      return result("budgetSkipped")
    }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    func buffer(_ bytes: UInt64, shared: Bool = false) -> MTLBuffer? {
      device.makeBuffer(
        length: Int(bytes), options: shared ? .storageModeShared : .storageModePrivate)
    }
    peakBytes = worstScratch + page(mapBytes) + page(entriesBytes)
    guard let sums = buffer(valuesBytes), let count = buffer(4, shared: true),
      let status = buffer(4, shared: true), let entries = buffer(entriesBytes, shared: true),
      let intermediate = buffer(mapBytes)
    else { return result("allocationSkipped", scratch: worstScratch) }
    let scratch = UInt64(sums.allocatedSize + count.allocatedSize + status.allocatedSize)
    var retained = UInt64(entries.allocatedSize + intermediate.allocatedSize)
    peakBytes = max(peakBytes, retained + scratch)
    guard peakBytes <= availableBytes else {
      return result("allocationOrBudgetSkipped", scratch: scratch)
    }
    let pipelines = try ["sum", "widths", "prefix", "pack", "verify"].map { stage in
      guard let function = library.makeFunction(name: "compact_detector_regions_" + stage) else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Exact aggregate kernel \(stage) is missing")
      }
      return try device.makeComputePipelineState(function: function)
    }
    guard
      pipelines.allSatisfy({
        $0.threadExecutionWidth == 32 && $0.maxTotalThreadsPerThreadgroup >= 256
      })
    else { return result("unsupportedDevice", scratch: scratch) }
    var output: [CompactResidentShard] = []
    var parameters: [UInt32] = [
      UInt32(frames), UInt32(frames / 32),
      UInt32(metadata.detectorColumns), UInt32(blocks), UInt32(headerWords), headerEncoding, 0,
      UInt32(blockSide), UInt32(maximumWidth),
    ]
    func encode(_ stage: Int, _ command: MTLCommandBuffer, _ buffers: [MTLBuffer], _ threads: Int)
      throws
    {
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Cannot encode exact aggregate preparation")
      }
      encoder.setComputePipelineState(pipelines[stage])
      for (index, buffer) in buffers.enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      parameters.withUnsafeBytes {
        encoder.setBytes($0.baseAddress!, length: $0.count, index: buffers.count)
      }
      encoder.dispatchThreads(
        MTLSize(width: threads, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      encoder.endEncoding()
    }
    func complete(_ command: MTLCommandBuffer) throws {
      command.commit()
      command.waitUntilCompleted()
      gpuMilliseconds += (command.gpuEndTime - command.gpuStartTime) * 1_000
      guard command.status == .completed else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Exact aggregate preparation failed: \(command.error?.localizedDescription ?? "command did not complete")"
        )
      }
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    }
    for shard in shards {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      let completed: CompactResidentShard? = try autoreleasepool {
        guard
          retained + scratch + page(descriptorBytes) + page(worstPayloadBytes) <= availableBytes,
          let descriptors = buffer(descriptorBytes), let prepare = queue.makeCommandBuffer()
        else { return nil }
        peakBytes = max(peakBytes, retained + scratch + UInt64(descriptors.allocatedSize))
        guard peakBytes <= availableBytes else { return nil }
        status.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        count.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        try encode(0, prepare, [shard.payload, shard.descriptors, sums], Int(valuesBytes / 4))
        try encode(1, prepare, [sums, descriptors], Int(cells))
        try encode(2, prepare, [descriptors, count, status], 1)
        try complete(prepare)
        let words = count.contents().load(as: UInt32.self)
        guard status.contents().load(as: UInt32.self) == 0,
          UInt64(words) <= cells * UInt64(maximumWidth),
          words < 1 << 27
        else {
          throw Metal4DSTEMStreamingIOError.invalidRequest(
            "Exact aggregate width/prefix verification failed")
        }
        let payloadBytes = max(4, UInt64(words) * 4)
        guard
          retained + scratch + UInt64(descriptors.allocatedSize) + page(payloadBytes)
            <= availableBytes,
          let payload = buffer(payloadBytes)
        else { return nil }
        let nextRetained = retained + UInt64(descriptors.allocatedSize + payload.allocatedSize)
        peakBytes = max(peakBytes, nextRetained + scratch)
        guard nextRetained + scratch <= availableBytes else { return nil }
        guard let pack = queue.makeCommandBuffer() else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "Cannot create exact aggregate packing command")
        }
        parameters[6] = words
        try encode(3, pack, [sums, descriptors, payload], Int(cells))
        try encode(4, pack, [sums, descriptors, payload, status], Int(valuesBytes / 4))
        try complete(pack)
        guard status.contents().load(as: UInt32.self) == 0 else {
          throw Metal4DSTEMStreamingIOError.invalidRequest(
            "Exact aggregate every-value verification failed")
        }
        retained = nextRetained
        return CompactResidentShard(payload: payload, descriptors: descriptors)
      }
      guard let completed else { return result("allocationOrBudgetSkipped", scratch: scratch) }
      output.append(completed)
    }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    return result(
      "built",
      auxiliary: CompactDetectorRegions(
        blockSide: blockSide, shards: output, entries: entries,
        intermediate: intermediate, bytes: retained), scratch: scratch)
  }
}
