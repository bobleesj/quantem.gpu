import Foundation
import Metal
import Metal4DSTEMKernels

/// Exact per-scan DPC moments together with the exact sum of stored counts over
/// every scan position of every detector pixel, both produced by one decode of
/// the resident payload.
public struct MetalRuntimeANSExactMomentsAndTotals {
  public let moments: MetalCompactH5ExactDPCMoments
  public let detectorTotals: [UInt64]
}

extension MetalRuntimeANSResidentSource {
  /// Exact total and detector row/column moments for every scan position.
  ///
  /// This is the same exact quantity the prepared compact-H5 moments carry: for
  /// each scan position the sum of all stored counts and the count-weighted
  /// detector row and column indices. It is derived from the resident encoded
  /// streams alone. Nothing is written into the `.qem` payload, no dense copy of
  /// the acquisition is built, and the saved payload, offsets, models and
  /// spatial indexes are reused exactly as stored.
  ///
  /// Work proceeds one bounded batch of stored chunks at a time so a caller can
  /// keep the acquisition interactive, report progress, and cancel between
  /// batches. Cancelling leaves the returned basis unpublished; the caller keeps
  /// whatever representation it had before.
  ///
  /// Example: `try source.exactDPCMoments(shouldCancel: { task.isCancelled })`.
  public func exactDPCMoments(
    maximumAdditionalBytes: UInt64? = nil,
    groupsPerBlock: Int = 0,
    shouldCancel: () -> Bool = { false },
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> MetalCompactH5ExactDPCMoments {
    try deriveExactDependencies(
      maximumAdditionalBytes: maximumAdditionalBytes,
      groupsPerBlock: groupsPerBlock,
      shouldCancel: shouldCancel,
      progress: progress,
      includeDetectorTotals: false
    ).moments
  }

  /// Exact per-scan DPC moments and the exact per-pixel detector totals, derived
  /// together from a single decode of the stored counts.
  ///
  /// `detectorTotals` is the same quantity `detectorColumnSums()` returns, so a
  /// caller that needs both no longer decodes the acquisition twice: the stored
  /// payload is read once and both products fall out of that one pass. The
  /// moments are bit-for-bit the values `exactDPCMoments()` returns.
  ///
  /// Deriving the totals alongside the moments costs a little more than the
  /// moments alone, so a caller that only wants the totals should keep using
  /// `detectorColumnSums()`.
  ///
  /// Example: `let both = try source.exactDPCMomentsAndDetectorTotals()`.
  public func exactDPCMomentsAndDetectorTotals(
    maximumAdditionalBytes: UInt64? = nil,
    groupsPerBlock: Int = 0,
    shouldCancel: () -> Bool = { false },
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> MetalRuntimeANSExactMomentsAndTotals {
    let derived = try deriveExactDependencies(
      maximumAdditionalBytes: maximumAdditionalBytes,
      groupsPerBlock: groupsPerBlock,
      shouldCancel: shouldCancel,
      progress: progress,
      includeDetectorTotals: true)
    guard let totals = derived.totals else {
      throw Self.invalid("Metal did not return the exact detector totals.")
    }
    return MetalRuntimeANSExactMomentsAndTotals(moments: derived.moments, detectorTotals: totals)
  }

  /// Shared derivation. `includeDetectorTotals` selects the kernel that also
  /// publishes per-pixel totals, which is the only difference between the two
  /// public entry points.
  private func deriveExactDependencies(
    maximumAdditionalBytes: UInt64?,
    groupsPerBlock: Int,
    shouldCancel: () -> Bool,
    progress: ((Int, Int) -> Void)?,
    includeDetectorTotals: Bool
  ) throws -> (moments: MetalCompactH5ExactDPCMoments, totals: [UInt64]?) {
    try requireLive()
    let scanCount = shape[0] * shape[1]
    let pixels = shape[2] * shape[3]
    guard scanCount > 0, pixels > 0, !chunks.isEmpty else {
      throw Self.invalid("The runtime ANS resident has no stored counts to measure.")
    }
    guard interval <= Self.exactMomentScans else {
      throw Self.invalid(
        "Stored packets of \(interval) scans exceed the \(Self.exactMomentScans)-scan moment basis."
      )
    }
    guard
      let outputBytes = Self.exactMomentBytes(scanCount: scanCount),
      outputBytes <= UInt64(device.recommendedMaxWorkingSetSize) / 4
    else {
      throw Self.invalid(
        "The exact DPC basis for \(scanCount) scan positions exceeds the resident memory budget.")
    }
    let totalsBytes = includeDetectorTotals ? UInt64(pixels) * 8 : 0
    let validityBytes = includeDetectorTotals ? UInt64(pixels) : 0
    // Account for the entire simultaneous allocation before creating buffers.
    // Each packet in a batch owns a slice until the ordered combine finishes.
    let stripeWidth = 32 * Self.exactMomentStreams
    let stripesPerPacket = (pixels + stripeWidth - 1) / stripeWidth
    var maximumUnits = 1
    for chunk in chunks {
      let blocks = (chunk.scanCount + interval - 1) / interval
      let groups =
        groupsPerBlock > 0
        ? groupsPerBlock : Self.exactMomentGroups(blocks: blocks, stripes: stripesPerPacket)
      let units = blocks.multipliedReportingOverflow(by: groups)
      guard !units.overflow else {
        throw Self.invalid("The requested DPC moment group count exceeds the memory budget.")
      }
      maximumUnits = max(maximumUnits, units.partialValue)
    }
    let batchSize = min(16, chunks.count)
    var partialStride = maximumUnits
    for factor in [interval, Self.exactMomentWordsPerScan, MemoryLayout<UInt32>.stride] {
      let size = partialStride.multipliedReportingOverflow(by: factor)
      guard !size.overflow else {
        throw Self.invalid("The exact DPC moment partials exceed the memory budget.")
      }
      partialStride = size.partialValue
    }
    let partialAllocation = partialStride.multipliedReportingOverflow(by: batchSize)
    guard !partialAllocation.overflow, partialAllocation.partialValue <= device.maxBufferLength
    else {
      throw Self.invalid("The exact DPC moment partials exceed Metal's buffer limit.")
    }
    let partialBytes = partialAllocation.partialValue
    var extraBytes = UInt64(partialBytes)
    for bytes in [outputBytes, totalsBytes, validityBytes, 16] {
      let sum = extraBytes.addingReportingOverflow(bytes)
      guard !sum.overflow else {
        throw Self.invalid("The exact DPC moment buffers exceed the memory budget.")
      }
      extraBytes = sum.partialValue
    }
    if let budget = maximumAdditionalBytes,
      extraBytes > budget || UInt64(device.currentAllocatedSize) > budget - extraBytes
    {
      throw Self.invalid(
        "Exact DPC moments need \(extraBytes / (1 << 20)) MB beyond the current resident; "
          + "release another dataset to derive them.")
    }
    guard
      let output = device.makeBuffer(length: Int(outputBytes), options: .storageModeShared)
    else {
      throw Self.invalid("Metal could not allocate the exact DPC basis for this acquisition.")
    }
    output.label = "runtime ANS exact DPC moments"
    memset(output.contents(), 0, output.length)
    var totals: MTLBuffer?
    var validity: MTLBuffer?
    if includeDetectorTotals {
      guard
        let totalsBuffer = device.makeBuffer(
          length: pixels * 8, options: .storageModeShared),
        let validityBuffer = device.makeBuffer(length: pixels, options: .storageModeShared)
      else {
        throw Self.invalid("Metal could not allocate the exact detector totals.")
      }
      totalsBuffer.label = "runtime ANS exact detector totals"
      validityBuffer.label = "runtime ANS exact moment detector validity"
      memset(totalsBuffer.contents(), 0, totalsBuffer.length)
      validPixels.withUnsafeBytes {
        validityBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
      }
      totals = totalsBuffer
      validity = validityBuffer
    }
    guard let diagnostics = device.makeBuffer(length: 16, options: .storageModeShared) else {
      throw Self.invalid("Metal could not allocate the exact DPC moments diagnostics.")
    }
    diagnostics.label = "runtime ANS exact DPC moments diagnostics"
    let narrow = Self.exactMomentsFitUInt32(
      rows: shape[2], columns: shape[3], maximumValue: Self.exactMomentValueCeiling(logicalDtype))
    var pipeline = includeDetectorTotals ? exactMomentTotalsPipeline : exactMomentsPipeline
    let cachedNarrow =
      includeDetectorTotals ? exactMomentTotalsPipelineNarrow : exactMomentsPipelineNarrow
    if pipeline == nil || cachedNarrow != narrow {
      let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
      let name =
        "streamed_counts_exact_moments" + (includeDetectorTotals ? "_totals" : "")
        + (narrow ? "_narrow" : "")
      guard let function = library.makeFunction(name: name) else {
        throw Self.invalid(
          "Missing exact runtime ANS moments kernel. Rebuild the backend resources.")
      }
      pipeline = try device.makeComputePipelineState(function: function)
      if includeDetectorTotals {
        exactMomentTotalsPipeline = pipeline
        exactMomentTotalsPipelineNarrow = narrow
      } else {
        exactMomentsPipeline = pipeline
        exactMomentsPipelineNarrow = narrow
      }
    }
    guard let pipeline, let decodingTable else {
      throw Self.invalid("Metal could not prepare the exact runtime ANS moments kernel.")
    }
    if exactMomentCombinePipeline == nil {
      let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
      guard
        let function = library.makeFunction(name: "streamed_counts_exact_moments_combine")
      else {
        throw Self.invalid(
          "Missing exact runtime ANS moment combine kernel. Rebuild the backend resources.")
      }
      exactMomentCombinePipeline = try device.makeComputePipelineState(function: function)
    }
    guard let combinePipeline = exactMomentCombinePipeline else {
      throw Self.invalid("Metal could not prepare the exact runtime ANS moment combine kernel.")
    }
    // Every packet in a command buffer holds its own slice so the two passes meet
    // at a single barrier per buffer instead of one per packet: a barrier per
    // packet drains the pipeline and measured slower than the atomics it removed.
    guard
      let partials = device.makeBuffer(
        length: partialBytes, options: .storageModePrivate)
    else {
      throw Self.invalid("Metal could not allocate the exact DPC moment partials.")
    }
    partials.label = "runtime ANS exact DPC moment partials"
    let words = diagnostics.contents().bindMemory(to: UInt32.self, capacity: 4)
    let simdGroups = Self.exactMomentSIMDGroups
    var completed = 0
    if ProcessInfo.processInfo.environment["QGPU_RUNTIME_ANS_MOMENT_DEBUG"] == "1" {
      fputs(
        "MOMENT_DEBUG chunks=\(chunks.count) interval=\(interval) scans=\(scanCount) "
          + "pixels=\(pixels) stripes=\(stripesPerPacket) maximumUnits=\(maximumUnits) "
          + "groupsPerBlock=\(groupsPerBlock) simdGroups=\(simdGroups)\n", stderr)
      for (index, chunk) in chunks.enumerated() where index < 3 || index == chunks.count - 1 {
        let blocks = (chunk.scanCount + interval - 1) / interval
        let groups =
          groupsPerBlock > 0
          ? groupsPerBlock : Self.exactMomentGroups(blocks: blocks, stripes: stripesPerPacket)
        fputs(
          "MOMENT_DEBUG chunk=\(index) first=\(chunk.firstScan) scans=\(chunk.scanCount) "
            + "blocks=\(blocks) groups=\(groups) units=\(blocks * groups)\n", stderr)
      }
    }
    while completed < chunks.count {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      let stop = min(chunks.count, completed + batchSize)
      guard let command = queue.makeCommandBuffer() else {
        throw Self.invalid("Metal could not encode exact DPC moments.")
      }
      words[0] = 0
      words[1] = UInt32.max
      words[2] = UInt32.max
      words[3] = UInt32.max
      // The slots accumulate, so the batch starts from zero. A blit encoder
      // always runs before the compute encoders of the same command buffer.
      guard let clear = command.makeBlitCommandEncoder() else {
        throw Self.invalid("Metal could not clear the exact DPC moment partials.")
      }
      clear.fill(
        buffer: partials,
        range: 0..<((stop - completed) * partialStride),
        value: 0)
      clear.endEncoding()
      guard let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
        throw Self.invalid("Metal could not encode exact DPC moments.")
      }
      command.label = "Exact DPC moments \(completed)..<\(stop)"
      // The decode and the fold cannot share an encoder: under a concurrent
      // encoder a memory barrier makes earlier writes visible without ordering
      // the dispatches, and a fold that starts early reads unwritten partials.
      // Two encoders in one command buffer run in creation order and keep the
      // parallelism inside each pass.
      // Each packet decodes into its own stripe partials and is then folded by
      // the combine pass, so the packets stay independent but the shared slice
      // has to be re-read before the next packet overwrites it.
      var pending: [(chunk: Chunk, blocks: Int, groups: Int)] = []
      for chunk in chunks[completed..<stop] {
        let blocks = (chunk.scanCount + interval - 1) / interval
        let stripes = stripesPerPacket
        let groups =
          groupsPerBlock > 0
          ? groupsPerBlock : Self.exactMomentGroups(blocks: blocks, stripes: stripes)
        let units = blocks * groups
        var parameters: [UInt64] = [
          UInt64(chunk.scanCount), UInt64(pixels), UInt64(interval), UInt64(chunk.firstScan),
          UInt64(shape[3]), 0, UInt64(groups), UInt64(simdGroups),
        ]
        encoder.setComputePipelineState(pipeline)
        for (slot, buffer) in [
          chunk.payload, chunk.offsets, chunk.models, decodingTable, diagnostics, output,
        ].enumerated() {
          encoder.setBuffer(buffer, offset: 0, index: slot)
        }
        encoder.setBytes(&parameters, length: parameters.count * 8, index: 6)
        if let totals, let validity {
          encoder.setBuffer(totals, offset: 0, index: 7)
          encoder.setBuffer(validity, offset: 0, index: 8)
        }
        let sliceOffset = pending.count * partialStride
        encoder.setBuffer(partials, offset: sliceOffset, index: 9)
        encoder.dispatchThreadgroups(
          MTLSize(width: (units + simdGroups - 1) / simdGroups, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32 * simdGroups, height: 1, depth: 1))
        pending.append((chunk, blocks, groups))
      }
      encoder.endEncoding()
      guard !pending.isEmpty else { continue }
      guard let fold = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
        throw Self.invalid("Metal could not encode the exact DPC moment fold.")
      }
      for (index, entry) in pending.enumerated() {
        var combine: [UInt64] = [
          UInt64(entry.chunk.scanCount), UInt64(interval), UInt64(entry.groups),
          UInt64(entry.chunk.firstScan), 0,
        ]
        fold.setComputePipelineState(combinePipeline)
        fold.setBuffer(partials, offset: index * partialStride, index: 0)
        fold.setBuffer(output, offset: 0, index: 1)
        fold.setBytes(&combine, length: combine.count * 8, index: 2)
        fold.dispatchThreads(
          MTLSize(width: entry.chunk.scanCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(
            width: min(256, max(1, entry.chunk.scanCount)), height: 1, depth: 1))
      }
      fold.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed, command.error == nil else {
        throw Self.invalid(
          "Metal could not complete exact DPC moments: "
            + (command.error?.localizedDescription ?? "the command did not complete")
            + ". Close another dataset and try again.")
      }
      if words[0] != 0 { throw Self.momentFailure(words) }
      completed = stop
      progress?(completed, chunks.count)
    }
    let basis = output.contents().bindMemory(to: UInt32.self, capacity: scanCount * 6)
    var total = [UInt64](repeating: 0, count: scanCount)
    var row = [UInt64](repeating: 0, count: scanCount)
    var column = [UInt64](repeating: 0, count: scanCount)
    for scan in 0..<scanCount {
      let base = scan * 6
      total[scan] = UInt64(basis[base]) | (UInt64(basis[base + 1]) << 32)
      row[scan] = UInt64(basis[base + 2]) | (UInt64(basis[base + 3]) << 32)
      column[scan] = UInt64(basis[base + 4]) | (UInt64(basis[base + 5]) << 32)
    }
    let moments = MetalCompactH5ExactDPCMoments(
      total: total,
      detectorRowMoment: row,
      detectorColumnMoment: column,
      sourceIdentitySHA256: sourceIdentitySHA256)
    guard let totals else { return (moments, nil) }
    let sums = Array(
      UnsafeBufferPointer(
        start: totals.contents().assumingMemoryBound(to: UInt64.self), count: pixels))
    return (moments, sums)
  }

  private static func momentFailure(_ words: UnsafeMutablePointer<UInt32>) -> Error {
    let reason =
      switch words[3] {
      case 1: "the stored stream is invalid"
      case 2: "the stored stream has unconsumed bytes"
      case 3: "the stored stream ended in an invalid entropy state"
      default: "an unknown decode fault"
      }
    return invalid(
      "Exact DPC moments could not read stream \(words[1]) of model \(words[2]): \(reason).")
  }

  /// Exact DPC basis storage for one scan grid, in bytes, without overflow.
  static func exactMomentBytes(scanCount: Int) -> UInt64? {
    guard scanCount > 0 else { return nil }
    let words = UInt64(scanCount).multipliedReportingOverflow(
      by: UInt64(exactMomentWordsPerScan) * UInt64(MemoryLayout<UInt32>.stride))
    return words.overflow ? nil : words.partialValue
  }

  /// Stripe units per stored packet. Narrow detectors hold few stripes, so aim
  /// for a fixed dispatch size instead: enough units to fill the GPU, few enough
  /// that per-unit setup never dominates.
  static func exactMomentGroups(blocks: Int, stripes: Int) -> Int {
    max(1, min(stripes, exactMomentTargetUnits / max(1, blocks)))
  }

  /// SIMD groups per threadgroup. Decoding a stored stream is a serial chain, so
  /// this kernel needs many resident threads rather than wide arithmetic, and a
  /// wider threadgroup reaches that with less dispatch cost.
  static var exactMomentSIMDGroups: Int {
    let raw = ProcessInfo.processInfo.environment["QGPU_RUNTIME_ANS_MOMENT_SIMD"] ?? ""
    guard let value = Int(raw), [1, 2, 4, 8].contains(value) else { return 4 }
    return value
  }

  static let exactMomentTargetUnits = 512

  /// Whether every lane partial stays inside 32 bits, which lets the kernel use
  /// 32-bit products and 16-bit split reductions instead of 64-bit arithmetic.
  /// Each lane sums at most `exactMomentStreams` stored counts, so the bound is
  /// streams x largest stored count x largest row or column index.
  static func exactMomentsFitUInt32(rows: Int, columns: Int, maximumValue: UInt64) -> Bool {
    let weight = UInt64(max(max(rows, columns) - 1, 0))
    let digits = UInt64(exactMomentStreams).multipliedReportingOverflow(by: maximumValue)
    guard !digits.overflow else { return false }
    let bound = digits.partialValue.multipliedReportingOverflow(by: weight)
    return !bound.overflow && bound.partialValue <= UInt64(UInt32.max)
  }

  /// Largest stored count the resident can decode for its logical dtype.
  static func exactMomentValueCeiling(_ dtype: Metal4DSTEMIntegerDType) -> UInt64 {
    switch dtype {
    case .uint8: 255
    case .uint16: 65535
    case .uint32: UInt64(UInt32.max)
    }
  }

  static let exactMomentWordsPerScan = 6
  static let exactMomentStreams = 4
  /// Scans per stored packet, matching the runtime ANS stream interval.
  static let exactMomentScans = 512
}
