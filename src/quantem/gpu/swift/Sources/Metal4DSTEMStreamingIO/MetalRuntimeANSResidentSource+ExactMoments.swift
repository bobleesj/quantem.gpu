import Foundation
import Metal
import Metal4DSTEMKernels

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
    try requireLive()
    let scanCount = shape[0] * shape[1]
    let pixels = shape[2] * shape[3]
    guard scanCount > 0, pixels > 0, !chunks.isEmpty else {
      throw Self.invalid("The runtime ANS resident has no stored counts to measure.")
    }
    guard interval <= Self.exactMomentScans else {
      throw Self.invalid(
        "Stored packets of \(interval) scans exceed the \(Self.exactMomentScans)-scan moment basis.")
    }
    guard
      let outputBytes = Self.exactMomentBytes(scanCount: scanCount),
      outputBytes <= UInt64(device.recommendedMaxWorkingSetSize) / 4
    else {
      throw Self.invalid(
        "The exact DPC basis for \(scanCount) scan positions exceeds the resident memory budget.")
    }
    if let budget = maximumAdditionalBytes,
      UInt64(device.currentAllocatedSize) + outputBytes > budget
    {
      throw Self.invalid(
        "Exact DPC moments need \(outputBytes / (1 << 20)) MB beyond the current resident; "
          + "release another dataset to derive them.")
    }
    guard
      let output = device.makeBuffer(length: Int(outputBytes), options: .storageModeShared)
    else {
      throw Self.invalid("Metal could not allocate the exact DPC basis for this acquisition.")
    }
    output.label = "runtime ANS exact DPC moments"
    memset(output.contents(), 0, output.length)
    guard let diagnostics = device.makeBuffer(length: 16, options: .storageModeShared) else {
      throw Self.invalid("Metal could not allocate the exact DPC moments diagnostics.")
    }
    diagnostics.label = "runtime ANS exact DPC moments diagnostics"
    let narrow = Self.exactMomentsFitUInt32(
      rows: shape[2], columns: shape[3], maximumValue: Self.exactMomentValueCeiling(logicalDtype))
    if exactMomentsPipeline == nil || exactMomentsPipelineNarrow != narrow {
      let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
      let name = narrow ? "streamed_counts_exact_moments_narrow" : "streamed_counts_exact_moments"
      guard let function = library.makeFunction(name: name) else {
        throw Self.invalid("Missing exact runtime ANS moments kernel. Rebuild the backend resources.")
      }
      exactMomentsPipeline = try device.makeComputePipelineState(function: function)
      exactMomentsPipelineNarrow = narrow
    }
    guard let pipeline = exactMomentsPipeline, let decodingTable else {
      throw Self.invalid("Metal could not prepare the exact runtime ANS moments kernel.")
    }
    let words = diagnostics.contents().bindMemory(to: UInt32.self, capacity: 4)
    let simdGroups = Self.exactMomentSIMDGroups
    let batchSize = 16
    var completed = 0
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
      guard let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
        throw Self.invalid("Metal could not encode exact DPC moments.")
      }
      command.label = "Exact DPC moments \(completed)..<\(stop)"
      // Stored packets own disjoint scan ranges, so their dispatches share this
      // encoder without a barrier between them.
      for chunk in chunks[completed..<stop] {
        let blocks = (chunk.scanCount + interval - 1) / interval
        let stripeWidth = 32 * Self.exactMomentStreams
        let stripes = (pixels + stripeWidth - 1) / stripeWidth
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
        encoder.dispatchThreadgroups(
          MTLSize(width: (units + simdGroups - 1) / simdGroups, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32 * simdGroups, height: 1, depth: 1))
      }
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
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
    return MetalCompactH5ExactDPCMoments(
      total: total,
      detectorRowMoment: row,
      detectorColumnMoment: column,
      sourceIdentitySHA256: sourceIdentitySHA256)
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
