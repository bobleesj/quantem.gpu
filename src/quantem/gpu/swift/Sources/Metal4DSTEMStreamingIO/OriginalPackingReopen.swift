import Foundation
import Metal
import Native4DSTEMIO

extension OriginalHDF5Packing {
  func packBitshufflePlan(
    source: Native4DSTEMIndexedSource, windows: [Native4DSTEMIndexedWindow], frames: Int,
    moments: Data?, packingPlanURL: URL, maximumAdditionalBytes: UInt64?, priorProfile: Profile?,
    validateInputs: () throws -> Void, shouldCancel: () -> Bool,
    progress: (Int, Int) -> Void
  ) throws -> OriginalPackedBuffers? {
    let dataset = source.dataset
    let pixels = dataset.detectorRows * dataset.detectorCols
    guard cpuPlanDecode, let scalarDecode, let bitshuffleValues, let bitshuffleReduce,
      let rangesPipeline, scalarDecode.threadExecutionWidth == 32,
      source.sourceBytesPerValue == 2, pixels.isMultiple(of: 4096),
      moments != nil || (bitshuffleDPC != nil && dataset.detectorCols.isMultiple(of: 32)),
      source.shards.allSatisfy({ Int($0.index.metadata.nBlocksPerFrame) * 4096 == pixels }),
      let identity = dataset.sourceIdentitySHA256
    else { return nil }
    let tiles = frames / 32
    let useZeroTail =
      moments != nil && frames <= 8192 && alignedRepeatFill && !alignedHistoryCopy
      && bitshufflePayloadLayout == 1 && bitshufflePixelsPerThread == 4
      && zeroTailDecode != nil && zeroTailValues != nil
    let checkpoints = (tiles + 31) / 32
    let headerStride = checkpoints + (tiles + 7) / 8
    let headerBytes = pixels * headerStride * 4
    let headerCapacity = ((headerBytes + 8191) / 8192) * 8192
    // After retaining the private header copy, the shared slot is dead until
    // its next CPU plan decode. Reuse it for exact per-scan DPC values, never
    // allocate another GPU staging or resident buffer.
    guard moments != nil || headerCapacity >= frames * 32 else { return nil }
    var residentMomentData = moments ?? Data()
    if moments == nil { residentMomentData.reserveCapacity(source.logicalFrameCount * 32) }
    let sourceFiles =
      (dataset.masterPath.map { [URL(fileURLWithPath: $0)] } ?? []) + source.shards.map(\.sourceURL)
    guard
      let binding = try? OriginalPackingLayoutCache.Binding.capture(
        sourceIdentity: identity, sourceFiles: sourceFiles,
        scanRows: dataset.scanRows, scanColumns: dataset.scanCols,
        detectorRows: dataset.detectorRows, detectorColumns: dataset.detectorCols,
        sourceDtype: dataset.sourceDtype, framesPerWindow: frames, headerWordsPerPixel: headerStride
      ),
      let planReader = OriginalPackingLayoutCache.Reader(url: packingPlanURL, binding: binding)
    else { return nil }
    var profile = priorProfile ?? Profile()
    profile.scalarDecodeThreads = scalarDecodeThreads
    profile.decodePipelineThreadLimit = scalarDecode.maxTotalThreadsPerThreadgroup
    profile.packingPipelineThreadLimit =
      (useZeroTail ? zeroTailValues! : bitshuffleValues).maxTotalThreadsPerThreadgroup
    profile.bitshufflePackingThreads = bitshufflePackingThreads
    profile.bitshufflePixelsPerThread = bitshufflePixelsPerThread
    profile.decodeWindowFrames = frames
    profile.planStatus = "hit"
    profile.reusedDPC = moments != nil
    let stageProfiler = OriginalPackingStageProfiler.makeIfRequested(
      device: device, expectedWindows: windows.count)
    defer { stageProfiler?.reportSummary() }
    let scratchBytes = frames * pixels * 2
    let partialBytes = pixels * checkpoints * 4
    let overlapPlans =
      windows.count > 1
      && OriginalPackingDiagnostics.enabled("PLAN_OVERLAP", byDefault: true)
    let extraHeaderBytes = overlapPlans ? headerCapacity + 1 : 0
    profile.planOverlapHeaderBytes = max(profile.planOverlapHeaderBytes, UInt64(extraHeaderBytes))
    // Includes bounded metadata codec/hash workspace; no dense count window.
    let fixedStaging =
      UInt64(
        scratchBytes + 2 * partialBytes + headerCapacity + 1
          + extraHeaderBytes + pixels * 16 + 8 + source.logicalFrameCount * 32) + (64 << 20)
    var residentBytes: UInt64 = 0
    var peakStaging = fixedStaging
    func admit(inputBytes: UInt64, payloadBytes: UInt64) throws {
      let staging = fixedStaging + inputBytes + payloadBytes + UInt64(headerBytes)
      peakStaging = max(peakStaging, staging)
      if let maximumAdditionalBytes, residentBytes + staging > maximumAdditionalBytes {
        throw CacheMismatch(profile: profile, retryWithoutPlan: true)
      }
    }
    try admit(inputBytes: 0, payloadBytes: 0)
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    let scratch: MTLBuffer
    let errors: MTLBuffer
    let payloadWords: MTLBuffer
    let headerBuffers: [MTLBuffer]
    let partialSums: MTLBuffer
    let partialMaximums: MTLBuffer
    let sums: MTLBuffer
    let widths: MTLBuffer
    let maxima: MTLBuffer
    do {
      scratch = try buffer(scratchBytes, privateStorage: true)
      let firstHeaders = try buffer(headerCapacity + 1)
      headerBuffers =
        overlapPlans ? [firstHeaders, try buffer(headerCapacity + 1)] : [firstHeaders]
      errors = try buffer(4)
      payloadWords = try buffer(4)
      partialSums = try buffer(partialBytes, privateStorage: true)
      partialMaximums = try buffer(partialBytes, privateStorage: true)
      sums = try buffer(pixels * 8)
      widths = try buffer(pixels * 4)
      maxima = try buffer(pixels * 4)
    } catch { throw CacheMismatch(profile: profile, retryWithoutPlan: true) }
    if (alignedRepeatFill || alignedHistoryCopy) && scratch.gpuAddress & 15 != 0 {
      throw Self.invalid("Aligned decompression requires a 16-byte-aligned scratch buffer")
    }
    memset(widths.contents(), 0, widths.length)
    memset(maxima.contents(), 0, maxima.length)
    var detectorSum = [UInt64](repeating: 0, count: pixels)
    var residentShards: [(payload: MTLBuffer, headers: MTLBuffer)] = []
    let readAhead =
      OriginalPackingDiagnostics.enabled("DIRECT_READ", byDefault: true)
      && OriginalPackingDiagnostics.enabled("READ_AHEAD", byDefault: true)
    profile.readAheadEnabled = readAhead
    let reader = readAhead ? CompressedReadAhead(device: device) : nil
    defer { reader?.cancelAndDrain() }
    let orderedSlices = windows.flatMap(\.slices)
    var sliceOrdinal = 0
    var pendingInputBytes: UInt64 = 0
    if let reader, let first = orderedSlices.first {
      let plan = try compressedReadPlan(first, source: source)
      pendingInputBytes = plan.reservedBytes
      try admit(inputBytes: pendingInputBytes, payloadBytes: 0)
      try reader.submit(plan)
    }
    var shape = Shape(
      scans: UInt32(frames), pixels: UInt32(pixels),
      columns: UInt32(dataset.detectorCols), sourceBytes: 2)
    func readDecodedPlan(_ ordinal: Int, into headers: MTLBuffer) throws -> UInt32 {
      // Release the compressed record before returning; only its payload size
      // and decoded header slot survive, never a second count buffer.
      try autoreleasepool {
        let readStarted = CFAbsoluteTimeGetCurrent()
        let cachedPlan = planReader.read(window: ordinal)
        profile.planRead += CFAbsoluteTimeGetCurrent() - readStarted
        guard let cachedPlan, cachedPlan.headerBytes == headerBytes,
          cachedPlan.paddedHeaderBytes == headerCapacity
        else { throw CacheMismatch(profile: profile) }
        profile.planReadBytes += UInt64(cachedPlan.compressed.count + cachedPlan.metadata.count * 4)
        try decodePlan(cachedPlan, into: headers, errors: errors, profile: &profile)
        profile.planWindows += 1
        return cachedPlan.payloadWordCount
      }
    }
    var prefetchedPayloadWords: UInt32?
    for (ordinal, window) in windows.enumerated() {
      try autoreleasepool {
        if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
        let headers = headerBuffers[overlapPlans ? ordinal % 2 : 0]
        let wordCount: UInt32
        if overlapPlans && ordinal > 0 {
          guard let prepared = prefetchedPayloadWords else { throw CacheMismatch(profile: profile) }
          wordCount = prepared
          prefetchedPayloadWords = nil
        } else {
          wordCount = try readDecodedPlan(ordinal, into: headers)
        }
        let payloadBytes = max(4, Int(wordCount) * 4)
        try admit(inputBytes: pendingInputBytes, payloadBytes: UInt64(payloadBytes))
        let payload: MTLBuffer
        let privateHeaders: MTLBuffer
        do {
          payload = try buffer(payloadBytes, privateStorage: true)
          privateHeaders = try buffer(headerBytes, privateStorage: true)
        } catch { throw CacheMismatch(profile: profile, retryWithoutPlan: true) }
        // One error lifetime spans all decoders and consumers in this window.
        // In particular, no later slice may erase an earlier malformed stream.
        memset(errors.contents(), 0, 4)
        payloadWords.contents().storeBytes(of: wordCount, as: UInt32.self)
        let command = try commandBuffer()
        let stageWindow = stageProfiler?.makeWindow(
          ordinal: ordinal,
          sliceFrames: window.slices.map { $0.globalFrameRange.count })
        var retainedInputs: [CompressedReadInput] = []
        var retainedInputBytes: UInt64 = 0
        for (sliceIndex, slice) in window.slices.enumerated() {
          if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
          let input: CompressedReadInput
          if let reader {
            let waitStarted = CFAbsoluteTimeGetCurrent()
            input = try reader.take(shouldCancel: shouldCancel)
            profile.readWait += CFAbsoluteTimeGetCurrent() - waitStarted
            pendingInputBytes = 0
          } else {
            let plan = try compressedReadPlan(slice, source: source)
            try admit(
              inputBytes: retainedInputBytes + plan.reservedBytes,
              payloadBytes: UInt64(payloadBytes))
            input = try Self.readCompressed(plan, device: device, isCancelled: shouldCancel)
          }
          guard input.shardIndex == slice.shardIndex, input.frameRange == slice.globalFrameRange
          else {
            throw Self.invalid("Compressed read-ahead does not match the requested scan slice")
          }
          retainedInputs.append(input)
          retainedInputBytes += input.reservedBytes
          profile.read += input.readSeconds
          profile.copy += input.copySeconds
          profile.readBytes += UInt64(input.compressed.length)
          sliceOrdinal += 1
          if let reader, sliceOrdinal < orderedSlices.count {
            let next = try compressedReadPlan(orderedSlices[sliceOrdinal], source: source)
            pendingInputBytes = next.reservedBytes
            try admit(
              inputBytes: retainedInputBytes + pendingInputBytes, payloadBytes: UInt64(payloadBytes)
            )
            try reader.submit(next)
          }
          try admit(
            inputBytes: retainedInputBytes + pendingInputBytes, payloadBytes: UInt64(payloadBytes))
          profile.maximumConcurrentInputBytes = max(
            profile.maximumConcurrentInputBytes,
            retainedInputBytes + pendingInputBytes)
          profile.additionalReadReserveBytes = max(
            profile.additionalReadReserveBytes, pendingInputBytes)
          let frameOffset = slice.globalFrameRange.lowerBound - window.globalFrameRange.lowerBound
          let count = slice.globalFrameRange.count
          let blocks = pixels / 4096
          guard frameOffset >= 0, count > 0, frameOffset + count <= frames,
            input.metadata.length == count * blocks * 8
          else {
            throw Self.invalid(
              "Original compressed metadata does not cover the complete packing window")
          }
          var zero64: UInt64 = 0
          var zero: UInt32 = 0
          var blockCount = UInt32(blocks)
          var pixelCount = UInt32(pixels)
          var frameCount = UInt32(count)
          guard
            let encoder = stageWindow.map({ $0.computeEncoder(command, stage: sliceIndex) })
              ?? command.makeComputeCommandEncoder()
          else { throw Self.invalid("Cannot encode exact source decode") }
          encoder.setComputePipelineState(useZeroTail ? zeroTailDecode! : scalarDecode)
          encoder.setBuffer(input.compressed, offset: 0, index: 0)
          encoder.setBuffer(input.metadata, offset: 0, index: 1)
          encoder.setBytes(&zero64, length: 8, index: 2)
          encoder.setBytes(&blockCount, length: 4, index: 3)
          encoder.setBytes(&pixelCount, length: 4, index: 4)
          encoder.setBuffer(scratch, offset: frameOffset * pixels * 2, index: 5)
          encoder.setBytes(&zero, length: 4, index: 6)
          encoder.setBuffer(errors, offset: 0, index: 10)
          encoder.setBytes(&frameCount, length: 4, index: 11)
          if useZeroTail {
            // Exact terminal-zero metadata temporarily reuses the sum buffer.
            // Packing consumes it before the summary reduction overwrites it.
            var tailOffset = UInt32(frameOffset * blocks)
            encoder.setBuffer(sums, offset: 0, index: 12)
            encoder.setBytes(&tailOffset, length: 4, index: 13)
            profile.zeroTailSlices += 1
          }
          encoder.dispatchThreads(
            MTLSize(width: count * blocks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: scalarDecodeThreads, height: 1, depth: 1))
          encoder.endEncoding()
          profile.scalarSlices += 1
          if count < 2048 { profile.directBitshuffleShortSlices += 1 }
          if alignedRepeatFill { profile.alignedFillSlices += 1 }
          if alignedHistoryCopy { profile.alignedCopySlices += 1 }
        }
        try encode(
          command, pipeline: rangesPipeline, buffers: [headers, errors, payloadWords],
          shape: &shape, count: pixels,
          sampledEncoder: stageWindow?.computeEncoder(command, stage: window.slices.count))
        guard
          let packing = stageWindow.map({
            $0.computeEncoder(command, stage: window.slices.count + 1)
          })
            ?? command.makeComputeCommandEncoder()
        else { throw Self.invalid("Cannot encode verified bitshuffle packing") }
        packing.setComputePipelineState(useZeroTail ? zeroTailValues! : bitshuffleValues)
        for (index, value) in [scratch, headers, payload, errors].enumerated() {
          packing.setBuffer(value, offset: 0, index: index)
        }
        packing.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 4)
        packing.setBuffer(partialSums, offset: 0, index: 5)
        packing.setBuffer(partialMaximums, offset: 0, index: 6)
        if useZeroTail { packing.setBuffer(sums, offset: 0, index: 7) }
        packing.dispatchThreads(
          MTLSize(width: pixels * checkpoints / bitshufflePixelsPerThread, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(
            width: bitshuffleSIMDGather
              ? bitshufflePackingThreads
              : min(64, bitshuffleValues.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        packing.endEncoding()
        guard
          let reduce = stageWindow.map({
            $0.computeEncoder(command, stage: window.slices.count + 2)
          })
            ?? command.makeComputeCommandEncoder()
        else { throw Self.invalid("Cannot encode fresh source summaries") }
        reduce.setComputePipelineState(bitshuffleReduce)
        for (index, value) in [partialSums, headers, sums, widths, errors].enumerated() {
          reduce.setBuffer(value, offset: 0, index: index)
        }
        reduce.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 5)
        reduce.setBuffer(partialMaximums, offset: 0, index: 6)
        reduce.setBuffer(maxima, offset: 0, index: 7)
        reduce.dispatchThreads(
          MTLSize(width: pixels, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(
            width: min(128, bitshuffleReduce.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        reduce.endEncoding()
        guard
          let blit = stageWindow.map({ $0.blitEncoder(command) })
            ?? command.makeBlitCommandEncoder()
        else { throw Self.invalid("Cannot retain packed headers") }
        blit.copy(
          from: headers, sourceOffset: 0, to: privateHeaders, destinationOffset: 0,
          size: headerBytes)
        blit.endEncoding()
        if moments == nil {
          guard let bitshuffleDPC, let dpc = command.makeComputeCommandEncoder() else {
            throw Self.invalid("Cannot encode exact source-bitshuffle DPC")
          }
          dpc.setComputePipelineState(bitshuffleDPC)
          dpc.setBuffer(scratch, offset: 0, index: 0)
          dpc.setBuffer(headers, offset: 0, index: 1)
          dpc.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 2)
          dpc.setBuffer(errors, offset: 0, index: 3)
          dpc.dispatchThreads(
            MTLSize(width: frames * 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          dpc.endEncoding()
        }
        if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
        let commandStarted = CFAbsoluteTimeGetCurrent()
        // No pending command escapes this scope, even when cancellation or a
        // malformed input is observed. Inputs and scratch cannot be reused early.
        var preparationError: Error?
        let gpuSeconds = try withExtendedLifetime(retainedInputs) {
          stageProfiler?.willCommit(stageWindow)
          defer {
            stageProfiler?.record(
              stageWindow, command: command,
              countErrors: errors.contents().load(as: UInt32.self))
          }
          guard overlapPlans && ordinal + 1 < windows.count else { return try finish(command) }
          command.commit()
          do {
            if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
            // The alternate slot is not bound to this command. CPU plan decode
            // never touches the in-flight error counter or payload-word count.
            prefetchedPayloadWords = try readDecodedPlan(
              ordinal + 1, into: headerBuffers[(ordinal + 1) % 2])
            profile.planOverlapWindows += 1
            if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
          } catch { preparationError = error }
          // Even an invalid next plan or cancellation must drain the submitted
          // command before releasing its inputs, reusing scratch or retrying.
          command.waitUntilCompleted()
          guard command.status == .completed else {
            throw Self.invalid(
              command.error?.localizedDescription ?? "Metal packing command failed")
          }
          return max(0, command.gpuEndTime - command.gpuStartTime)
        }
        profile.directBitshuffleGPU += gpuSeconds
        profile.directBitshuffleWall += CFAbsoluteTimeGetCurrent() - commandStarted
        profile.directBitshuffleWindows += 1
        if bitshuffleSIMDGather { profile.directBitshuffleSIMDGatherWindows += 1 }
        guard errors.contents().load(as: UInt32.self) == 0 else {
          throw CacheMismatch(profile: profile)
        }
        if let mismatch = preparationError as? CacheMismatch {
          // The captured error predates completion. Carry current timing into
          // the whole-load retry rather than dropping this GPU interval.
          throw CacheMismatch(profile: profile, retryWithoutPlan: mismatch.retryWithoutPlan)
        }
        if let preparationError { throw preparationError }
        if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
        if moments == nil {
          residentMomentData.append(Data(bytes: headers.contents(), count: frames * 32))
        }
        let sumWords = sums.contents().assumingMemoryBound(to: UInt64.self)
        for pixel in 0..<pixels { detectorSum[pixel] += sumWords[pixel] }
        residentShards.append((payload, privateHeaders))
        residentBytes += UInt64(payloadBytes + headerBytes)
        let allocated = UInt64(device.currentAllocatedSize)
        if let maximumAdditionalBytes, allocated > allocatedBefore,
          allocated - allocatedBefore > maximumAdditionalBytes
        {
          throw CacheMismatch(profile: profile, retryWithoutPlan: true)
        }
        progress(window.globalFrameRange.upperBound, source.logicalFrameCount)
      }
    }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    try validateInputs()
    _ = try Native4DSTEMIndexedSource.open(dataset: dataset)
    guard binding.isCurrent(sourceFiles: sourceFiles) else {
      throw Self.invalid("Original source changed during loading; reopen its folder and retry")
    }
    let maximumWords = maxima.contents().assumingMemoryBound(to: UInt32.self)
    let maximum = (0..<pixels).reduce(UInt32(0)) { max($0, maximumWords[$1]) }
    let widthWords = widths.contents().assumingMemoryBound(to: UInt32.self)
    let maximumWidths = (0..<pixels).map { UInt8(widthWords[$0]) }
    if maximum <= 255 && maximumWidths.contains(where: { $0 > 8 }) {
      throw CacheMismatch(profile: profile)
    }
    profile.packedPayloadLayout = bitshufflePayloadLayout
    profile.maximumWidthHistogram = maximumWidths.reduce(into: [Int](repeating: 0, count: 17)) {
      $0[Int($1)] += 1
    }
    reportProfile(profile)
    return OriginalPackedBuffers(
      payloadLayout: bitshufflePayloadLayout,
      dataset: dataset, frames: frames, headerStride: headerStride,
      shards: residentShards, moments: residentMomentData, detectorSum: detectorSum,
      maximum: maximum,
      maximumWidths: maximumWidths,
      calibration: measuredDetector(
        detectorSum, rows: dataset.detectorRows, columns: dataset.detectorCols,
        excludedFromEstimate: dataset.badPixelIndices), stagingBytes: peakStaging,
      readSeconds: profile.read, decodeSeconds: profile.decodeGPU,
      decodeAndHeaderSeconds: profile.decodeAndHeadersGPU,
      packingSeconds: profile.productsGPU + profile.packingGPU + profile.planDecodeGPU,
      reusedDPC: moments != nil, combinedDecodePackingSeconds: profile.directBitshuffleGPU)
  }

  func decodePlan(
    _ window: OriginalPackingLayoutCache.Window, into headers: MTLBuffer,
    errors: MTLBuffer, profile: inout Profile
  ) throws {
    if cpuPlanDecode {
      let started = CFAbsoluteTimeGetCurrent()
      let valid = window.decodeHeaders(
        into: UnsafeMutableRawBufferPointer(
          start: headers.contents(), count: headers.length))
      profile.planDecodeCPU += CFAbsoluteTimeGetCurrent() - started
      guard valid else { throw CacheMismatch(profile: profile) }
      return
    }
    guard let planDecode, window.metadata.count.isMultiple(of: 2),
      window.metadata.count / 2 == window.paddedHeaderBytes / 8192,
      window.paddedHeaderBytes <= headers.length, headers.gpuAddress & 15 == 0
    else { throw CacheMismatch(profile: profile) }
    let compressed = try window.compressed.withUnsafeBytes { try copiedBuffer($0) }
    let metadata = try window.metadata.withUnsafeBytes { try copiedBuffer($0) }
    var zero64: UInt64 = 0
    var zero: UInt32 = 0
    var blocks: UInt32 = 1
    var pixels: UInt32 = 4096
    var frames = UInt32(window.metadata.count / 2)
    memset(errors.contents(), 0, 4)
    let command = try commandBuffer()
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Self.invalid("Cannot decode cached packing layout")
    }
    encoder.setComputePipelineState(planDecode)
    encoder.setBuffer(compressed, offset: 0, index: 0)
    encoder.setBuffer(metadata, offset: 0, index: 1)
    encoder.setBytes(&zero64, length: 8, index: 2)
    encoder.setBytes(&blocks, length: 4, index: 3)
    encoder.setBytes(&pixels, length: 4, index: 4)
    encoder.setBuffer(headers, offset: 0, index: 5)
    encoder.setBytes(&zero, length: 4, index: 6)
    encoder.setBuffer(errors, offset: 0, index: 10)
    encoder.setBytes(&frames, length: 4, index: 11)
    encoder.dispatchThreads(
      MTLSize(width: Int(frames), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(128, planDecode.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    encoder.endEncoding()
    profile.planDecodeGPU += try finish(command)
    guard errors.contents().load(as: UInt32.self) == 0 else {
      throw CacheMismatch(profile: profile)
    }
  }

  func reportProfile(_ profile: Profile) {
    // Aggregate timings only: no paths, labels or scientific values are logged.
    guard ProcessInfo.processInfo.environment["QGPU_ORIGINAL_PROFILE"] == "1" else { return }
    guard
      let data = try? JSONSerialization.data(withJSONObject: profile.json, options: [.sortedKeys]),
      let measured = String(data: data, encoding: .utf8)
    else { return }
    fputs("ORIGINAL_PACK_PROFILE \(measured)\n", stderr)
  }

}
