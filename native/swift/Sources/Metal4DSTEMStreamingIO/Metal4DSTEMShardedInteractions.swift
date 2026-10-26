import CMetal4DSTEMInteractions
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels

/// Interactions over one exact sharded 4D-STEM working volume.
///
/// The caller owns scientific detector geometry, interaction scheduling, and
/// publication. This type validates the package's exact shard contract and
/// performs selected-diffraction extraction and custom-detector reduction
/// without concatenating or changing the resident volume's exact uint16 data.
public final class Metal4DSTEMShardedInteractions {
  private struct DetectorWordEntry {
    let word: UInt32
    let coefficients: UInt32
  }

  private struct SharedInteractionPointers: @unchecked Sendable {
    let sources: [UnsafePointer<UInt32>]
    let destination: UnsafeMutablePointer<UInt32>
  }

  private struct SharedDetectorEntries: @unchecked Sendable {
    let pointer: UnsafePointer<QDetectorWordEntry>
    let count: Int
  }

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let extractUInt16ToUInt32: MTLComputePipelineState
  private let fullSumUInt16: MTLComputePipelineState
  private let signedDeltaUInt16: MTLComputePipelineState

  public init(device: MTLDevice) throws {
    guard let queue = device.makeCommandQueue() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not create the sharded-interaction command queue."
      )
    }
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    self.device = device
    self.queue = queue
    extractUInt16ToUInt32 = try Self.pipeline(
      library: library,
      name: Metal4DSTEMKernels.extractU16ToU32Function,
      device: device
    )
    fullSumUInt16 = try Self.pipeline(
      library: library,
      name: Metal4DSTEMKernels.fullSumU16Function,
      device: device
    )
    signedDeltaUInt16 = try Self.pipeline(
      library: library,
      name: Metal4DSTEMKernels.signedDeltaU16Function,
      device: device
    )
  }

  /// Extract one exact diffraction pattern from private or shared shards.
  public func extractDiffraction(
    shards: [MTLBuffer],
    plan: Metal4DSTEMExactBinningShardPlan,
    scanRow: Int,
    scanColumn: Int,
    into output: MTLBuffer,
    shouldCancel: () -> Bool = { false }
  ) throws {
    try validate(shards: shards, plan: plan)
    try validateOutput(
      output,
      requiredBytes: plan.provenance.outputDetectorRows
        * plan.provenance.outputDetectorColumns
        * MemoryLayout<UInt32>.stride,
      label: "selected diffraction"
    )
    guard plan.provenance.outputScanRows > scanRow, scanRow >= 0,
      plan.provenance.outputScanColumns > scanColumn, scanColumn >= 0,
      let shardIndex = plan.shards.firstIndex(where: {
        $0.outputScanRowStart <= scanRow && scanRow < $0.outputScanRowStop
      })
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Selected scan position (row: \(scanRow), column: \(scanColumn)) is outside "
          + "the sharded working shape \(plan.provenance.outputScanRows)x"
          + "\(plan.provenance.outputScanColumns)."
      )
    }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    let shard = plan.shards[shardIndex]
    var localScan = UInt32(
      (scanRow - shard.outputScanRowStart) * plan.provenance.outputScanColumns
        + scanColumn
    )
    var localScanCount = UInt32(shard.outputScanPositionCount)
    var detectorPixels = UInt32(
      plan.provenance.outputDetectorRows * plan.provenance.outputDetectorColumns
    )
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode selected diffraction for the sharded resident volume."
      )
    }
    encoder.setComputePipelineState(extractUInt16ToUInt32)
    encoder.setBuffer(shards[shardIndex], offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBytes(&localScan, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.setBytes(&localScanCount, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&detectorPixels, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.dispatchThreads(
      MTLSize(width: Int(detectorPixels), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error {
      throw Metal4DSTEMStreamingIOError.commandFailed(error.localizedDescription)
    }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
  }

  /// Update one exact custom-detector image across every resident shard.
  @discardableResult
  public func updateVirtualDetector(
    shards: [MTLBuffer],
    plan: Metal4DSTEMExactBinningShardPlan,
    previousMask: [UInt8]?,
    nextMask: [UInt8],
    into output: MTLBuffer,
    shouldCancel: () -> Bool = { false }
  ) throws -> Double {
    try validate(shards: shards, plan: plan)
    let detectorPixels =
      plan.provenance.outputDetectorRows * plan.provenance.outputDetectorColumns
    guard nextMask.count == detectorPixels,
      previousMask == nil || previousMask?.count == detectorPixels
    else {
      let previousCount = previousMask.map { String($0.count) } ?? "none"
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Custom detector masks must contain \(detectorPixels) pixels for the "
          + "working detector; received previous \(previousCount) "
          + "and next \(nextMask.count)."
      )
    }
    let invalidValue =
      nextMask.first(where: { $0 > 1 })
      ?? previousMask?.first(where: { $0 > 1 })
    if let invalidValue {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Custom detector masks must contain only binary 0/1 values; received "
          + "\(invalidValue). Use a binary selection instead of detector weights."
      )
    }
    let outputBytes =
      plan.provenance.outputScanRows
      * plan.provenance.outputScanColumns * MemoryLayout<UInt32>.stride
    try validateOutput(
      output,
      requiredBytes: outputBytes,
      label: "custom virtual detector"
    )
    let entries = Self.wordEntries(previousMask: previousMask, nextMask: nextMask)
    guard !entries.isEmpty else {
      // An unchanged detector is a no-op. An empty rebase replaces any old sum.
      guard previousMask == nil else { return 0 }
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      let started = CFAbsoluteTimeGetCurrent()
      guard let command = queue.makeCommandBuffer(),
        let blit = command.makeBlitCommandEncoder()
      else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not clear the empty custom-detector result."
        )
      }
      blit.fill(buffer: output, range: 0..<outputBytes, value: 0)
      blit.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      if let error = command.error {
        throw Metal4DSTEMStreamingIOError.commandFailed(error.localizedDescription)
      }
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      return (CFAbsoluteTimeGetCurrent() - started) * 1_000
    }
    if previousMask != nil,
      shards.allSatisfy({ $0.storageMode == .shared })
    {
      return try updateVirtualDetectorOnCPU(
        shards: shards,
        plan: plan,
        entries: entries,
        output: output,
        shouldCancel: shouldCancel
      )
    }
    guard
      let entryBuffer = entries.withUnsafeBytes({ bytes in
        bytes.baseAddress.flatMap {
          device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared)
        }
      })
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "custom-detector word entries",
        bytes: UInt64(entries.count * MemoryLayout<DetectorWordEntry>.stride)
      )
    }
    let started = CFAbsoluteTimeGetCurrent()
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    guard let command = queue.makeCommandBuffer() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not create the sharded custom-detector command buffer."
      )
    }
    for (shard, buffer) in zip(plan.shards, shards) {
      var localScanCount = UInt32(shard.outputScanPositionCount)
      var entryCount = UInt32(entries.count)
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not encode custom-detector reduction for shard \(shard.index)."
        )
      }
      encoder.setComputePipelineState(
        previousMask == nil ? fullSumUInt16 : signedDeltaUInt16
      )
      encoder.setBuffer(buffer, offset: 0, index: 0)
      encoder.setBuffer(entryBuffer, offset: 0, index: 1)
      encoder.setBuffer(
        output,
        offset: shard.outputScanPositionStart * MemoryLayout<UInt32>.stride,
        index: 2
      )
      encoder.setBytes(&localScanCount, length: MemoryLayout<UInt32>.stride, index: 3)
      encoder.setBytes(&entryCount, length: MemoryLayout<UInt32>.stride, index: 4)
      encoder.dispatchThreads(
        MTLSize(width: shard.outputScanPositionCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
      )
      encoder.endEncoding()
    }
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error {
      throw Metal4DSTEMStreamingIOError.commandFailed(
        "Sharded custom-detector reduction failed: \(error.localizedDescription)"
      )
    }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    let wallMilliseconds = (CFAbsoluteTimeGetCurrent() - started) * 1_000
    if ProcessInfo.processInfo.environment["QUANTEM_METAL_SHARDED_PROFILE"] == "1" {
      let gpuMilliseconds = max(0, command.gpuEndTime - command.gpuStartTime) * 1_000
      let line = String(
        format:
          "SHARDED_INTERACTION entries=%d shards=%d cpu=0 wall_ms=%.3f gpu_ms=%.3f allocated_bytes=%llu recommended_bytes=%llu\n",
        entries.count,
        shards.count,
        wallMilliseconds,
        gpuMilliseconds,
        device.currentAllocatedSize,
        device.recommendedMaxWorkingSetSize
      )
      FileHandle.standardError.write(Data(line.utf8))
    }
    return wallMilliseconds
  }

  private func updateVirtualDetectorOnCPU(
    shards: [MTLBuffer],
    plan: Metal4DSTEMExactBinningShardPlan,
    entries: [DetectorWordEntry],
    output: MTLBuffer,
    shouldCancel: () -> Bool
  ) throws -> Double {
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    let started = CFAbsoluteTimeGetCurrent()
    let pointers = SharedInteractionPointers(
      sources: shards.map {
        UnsafePointer($0.contents().assumingMemoryBound(to: UInt32.self))
      },
      destination: output.contents().assumingMemoryBound(to: UInt32.self)
    )
    let requestedWorkers = ProcessInfo.processInfo.environment[
      "QUANTEM_SHARDED_CPU_WORKERS"
    ].flatMap(Int.init)
    let shardCount = shards.count
    // Submit one balanced unit per shard. Grand Central Dispatch still limits
    // execution to available cores, while avoiding a long four-shard tail on
    // wide annular detector changes.
    let defaultWorkers = shardCount
    let workerCount = min(shardCount, max(1, requestedWorkers ?? defaultWorkers))
    entries.withUnsafeBytes { entryBytes in
      let sharedEntries = SharedDetectorEntries(
        pointer: entryBytes.baseAddress!.assumingMemoryBound(
          to: QDetectorWordEntry.self),
        count: entries.count
      )
      DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
        for shardIndex in stride(from: worker, to: shardCount, by: workerCount) {
          let shard = plan.shards[shardIndex]
          q_update_virtual_detector_u16(
            pointers.sources[shardIndex],
            pointers.destination.advanced(by: shard.outputScanPositionStart),
            shard.outputScanPositionCount,
            sharedEntries.pointer,
            sharedEntries.count
          )
        }
      }
    }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    let wallMilliseconds = (CFAbsoluteTimeGetCurrent() - started) * 1_000
    if ProcessInfo.processInfo.environment["QUANTEM_METAL_SHARDED_PROFILE"] == "1" {
      let line = String(
        format:
          "SHARDED_INTERACTION entries=%d shards=%d workers=%d cpu=1 wall_ms=%.3f allocated_bytes=%llu recommended_bytes=%llu\n",
        entries.count,
        shards.count,
        workerCount,
        wallMilliseconds,
        device.currentAllocatedSize,
        device.recommendedMaxWorkingSetSize
      )
      FileHandle.standardError.write(Data(line.utf8))
    }
    return wallMilliseconds
  }

  private func validate(
    shards: [MTLBuffer],
    plan: Metal4DSTEMExactBinningShardPlan
  ) throws {
    guard plan.schema == Metal4DSTEMExactBinningShardPlan.currentSchema,
      plan.provenance.outputDtype == .uint16,
      plan.provenance.outputLayout == .detectorWordMajorPackedUInt16,
      shards.count == plan.shards.count,
      Set(shards.map(ObjectIdentifier.init)).count == shards.count
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Sharded interactions require one distinct packed-uint16 buffer for every "
          + "validated shard; received \(shards.count) buffers for \(plan.shards.count) shards."
      )
    }
    for (shard, buffer) in zip(plan.shards, shards) {
      guard buffer.device.registryID == device.registryID,
        UInt64(buffer.length) == shard.payloadBytes,
        buffer.storageMode == .private || buffer.storageMode == .shared
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Resident shard \(shard.index) must belong to the selected Metal device, "
            + "contain exactly \(shard.payloadBytes) bytes, and use private or shared storage."
        )
      }
    }
  }

  private func validateOutput(
    _ output: MTLBuffer,
    requiredBytes: Int,
    label: String
  ) throws {
    guard output.device.registryID == device.registryID,
      output.storageMode == .shared,
      output.length >= requiredBytes
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The \(label) output must be a shared buffer on the selected device with at "
          + "least \(requiredBytes) bytes; received \(output.length) bytes in storage "
          + "mode \(output.storageMode.rawValue)."
      )
    }
  }

  /// Two detector pixels per uint32 word of the uint16 working volume.
  private static func wordEntries(
    previousMask: [UInt8]?,
    nextMask: [UInt8]
  ) -> [DetectorWordEntry] {
    let pixelsPerWord = 2
    return (0..<((nextMask.count + pixelsPerWord - 1) / pixelsPerWord)).compactMap { word in
      var coefficients: UInt32 = 0
      for lane in 0..<pixelsPerWord {
        let pixel = word * pixelsPerWord + lane
        guard pixel < nextMask.count else { continue }
        let before = previousMask?[pixel] ?? 0
        let after = nextMask[pixel]
        guard before != after else { continue }
        coefficients |= UInt32(after == 0 ? 2 : 1) << UInt32(lane * 2)
      }
      return coefficients == 0
        ? nil
        : DetectorWordEntry(word: UInt32(word), coefficients: coefficients)
    }
  }

  private static func pipeline(
    library: MTLLibrary,
    name: String,
    device: MTLDevice
  ) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal4DSTEMKernels is missing the sharded-interaction function \(name)."
      )
    }
    return try device.makeComputePipelineState(function: function)
  }
}
