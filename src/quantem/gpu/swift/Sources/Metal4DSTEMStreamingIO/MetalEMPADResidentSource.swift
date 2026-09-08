import CryptoKit
import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

/// Full EMPAD measurements in lossless, randomly addressable Metal word packing.
///
/// This format preserves float32 bit patterns, not integer counts. It stores
/// each detector row's common XOR prefix and suffix plus the remaining bits.
/// Compression is data-dependent and can be slightly larger than float32 for
/// incompressible rows. It never quantizes or clips measurements.
///
/// Example: `try MetalEMPADResidentSource.load(source, device: device,
/// memoryBudgetBytes: budget)` followed by encoding a selected DP or mask sum.
/// Calls, release and command completion must be serialized by the owner.
public final class MetalEMPADResidentSource {
  public let source: NativeEMPADSource
  public let residentBytes: UInt64
  /// SHA-256 of all original little-endian float32 detector words in scan order.
  public let logicalSHA256: String
  /// Tensor identity binding shape and dtype to `logicalSHA256`, not a file checksum.
  public let sourceIdentitySHA256: String
  /// True only when a checksum-protected source hash matched the file snapshot.
  /// The complete original measurements are still reread and packed.
  public let reusedSourceHash: Bool
  public private(set) var isReleased = false
  private let device: MTLDevice
  private let diffractionPipeline: MTLComputePipelineState
  private let detectorPipeline: MTLComputePipelineState
  private let serialDetector: Bool
  private let detectorThreads: Int
  private let centerOfMassPipeline: MTLComputePipelineState
  private let serialCenterOfMass: Bool
  private let meanPipeline: MTLComputePipelineState
  private let incrementalPipeline: MTLComputePipelineState?
  private let memoryBudgetBytes: UInt64
  private var detectorAccumulation: MTLBuffer?
  private var priorDetectorMask: [UInt8]?
  private var priorDetectorCommand: MTLCommandBuffer?
  private var incrementalUpdates = 0
  private var chunks: [Chunk]

  private struct Chunk {
    let firstFrame: Int
    let frameCount: Int
    let payload: MTLBuffer
    let descriptors: MTLBuffer
  }

  /// One in-flight read-only window, ordered exactly like the source tensor.
  /// The producer waits before submitting another hash, so slow checksumming
  /// cannot accumulate a hidden dense volume of staging buffers.
  // Mutable state is confined to the serial queue. Input ownership is retained
  // there until hashing finishes; the producer and Metal only read its bytes.
  private final class SourceDigest: @unchecked Sendable {
    private let queue = DispatchQueue(label: "quantem.gpu.empad-source-digest", qos: .userInitiated)
    private var digest = SHA256()
    private var seconds = 0.0
    private var input: MTLBuffer?

    func append(_ input: MTLBuffer) {
      queue.sync { self.input = input }
      queue.async {
        let started = CFAbsoluteTimeGetCurrent()
        self.digest.update(bufferPointer: UnsafeRawBufferPointer(
          start: self.input!.contents(), count: self.input!.length))
        self.seconds += CFAbsoluteTimeGetCurrent() - started
        self.input = nil
      }
    }

    func wait() { queue.sync {} }

    func finish() -> (hash: String, seconds: Double) {
      queue.sync {
        (digest.finalize().map { String(format: "%02x", $0) }.joined(), seconds)
      }
    }
  }

  private init(
    source: NativeEMPADSource, device: MTLDevice,
    diffraction: MTLComputePipelineState, detector: MTLComputePipelineState, chunks: [Chunk],
    logicalSHA256: String, centerOfMass: MTLComputePipelineState, mean: MTLComputePipelineState,
    serialDetector: Bool, detectorThreads: Int, serialCenterOfMass: Bool,
    incremental: MTLComputePipelineState?, memoryBudgetBytes: UInt64, reusedSourceHash: Bool
  ) {
    self.source = source
    self.device = device
    self.diffractionPipeline = diffraction
    self.detectorPipeline = detector
    self.serialDetector = serialDetector
    self.detectorThreads = detectorThreads
    centerOfMassPipeline = centerOfMass
    self.serialCenterOfMass = serialCenterOfMass
    meanPipeline = mean
    incrementalPipeline = incremental
    self.memoryBudgetBytes = memoryBudgetBytes
    self.reusedSourceHash = reusedSourceHash
    self.chunks = chunks
    self.logicalSHA256 = logicalSHA256
    var identity = SHA256()
    identity.update(data: Data("quantem.gpu.empad-tensor/v1\0float32-le\0".utf8))
    for dimension in [source.scanRows, source.scanColumns, 128, 128] {
      var word = UInt64(dimension).littleEndian
      withUnsafeBytes(of: &word) { identity.update(bufferPointer: $0) }
    }
    identity.update(data: Data(logicalSHA256.utf8))
    sourceIdentitySHA256 = identity.finalize().map { String(format: "%02x", $0) }.joined()
    residentBytes = chunks.reduce(0) { $0 + UInt64($1.payload.length + $1.descriptors.length) }
  }

  /// Read every original detector pixel and finish packing before returning.
  ///
  /// The budget caps total current Metal allocation plus the next bounded
  /// staging window. It is not a process-RSS or operating-system page-cache cap.
  /// No full dense host or device volume, prepared 2D preview or packed-data sidecar is
  /// used. An optional sourceHashCacheURL stores only a completed SHA-256 bound
  /// to the original path, inode, size, mtime and ctime; it never replaces reads.
  /// Cancellation throws and releases partial residents.
  public static func load(
    _ source: NativeEMPADSource, device: MTLDevice, memoryBudgetBytes: UInt64,
    sourceHashCacheURL: URL? = nil,
    shouldCancel: () -> Bool = { false }
  ) throws -> MetalEMPADResidentSource {
    let started = CFAbsoluteTimeGetCurrent()
    let profile = ProcessInfo.processInfo.environment["QGPU_EMPAD_LOAD_PROFILE"] == "1"
    var readSeconds = 0.0, hashSeconds = 0.0, analyzeSeconds = 0.0, packSeconds = 0.0
    guard source.frameCount <= Int(UInt32.max) else {
      throw failure("EMPAD scan exceeds the supported frame-address range.")
    }
    try checkCancellation(shouldCancel)
    try source.validateUnchanged()
    let snapshot = try source.sourceSnapshot()
    let hashCacheURL = EMPADSourceHashCache.safeURL(sourceHashCacheURL, source: source)
    let cachedHash = EMPADSourceHashCache.read(hashCacheURL, snapshot: snapshot)
    let pipelinedHash = cachedHash == nil && ProcessInfo.processInfo.environment["QGPU_EMPAD_HASH_SERIAL"] != "1"
      ? SourceDigest() : nil
    // Cancellation/error must drain readers before returning to the owner.
    defer { pipelinedHash?.wait() }
    var hashWaitSeconds = 0.0
    let library = try Metal4DSTEMKernels.makeEMPADLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw failure("EMPAD kernel is missing: \(name). Rebuild the backend resources.")
      }
      return try device.makeComputePipelineState(function: function)
    }
    let cooperativePacking = ProcessInfo.processInfo.environment["QGPU_EMPAD_PACK_CONTROL"] != "1"
    let describe = try pipeline(cooperativePacking ? "empad_describe_simd" : "empad_describe")
    let pack = try pipeline(cooperativePacking ? "empad_pack_simd" : "empad_pack")
    let diffraction = try pipeline("empad_diffraction")
    // Retained control for reproducible native A/B/A qualification.
    let serialDetector = ProcessInfo.processInfo.environment["QGPU_EMPAD_SERIAL_DETECTOR"] == "1"
    let detectorThreads = ProcessInfo.processInfo.environment["QGPU_EMPAD_DETECTOR_GROUPS"] == "1" ? 32 : 128
    let detector = try pipeline(serialDetector ? "empad_virtual_image_serial" : "empad_virtual_image")
    guard detector.threadExecutionWidth == 32,
      detector.maxTotalThreadsPerThreadgroup >= detectorThreads else {
      throw failure("EMPAD cooperative reduction requires 32-lane SIMD groups and a 128-thread-capable Metal pipeline.")
    }
    let serialCenterOfMass = ProcessInfo.processInfo.environment["QGPU_EMPAD_COM_CONTROL"] == "1"
    let centerOfMass = try pipeline(serialCenterOfMass ? "empad_center_of_mass" : "empad_center_of_mass_simd")
    let mean = try pipeline("empad_mean_diffraction")
    let incremental = ProcessInfo.processInfo.environment["QGPU_EMPAD_INCREMENTAL"] != "0"
      ? try pipeline("empad_virtual_image_changes") : nil
    if let incremental, incremental.maxTotalThreadsPerThreadgroup < detectorThreads {
      throw failure("EMPAD aperture-change kernel requires a 128-thread-capable Metal pipeline.")
    }
    guard let queue = device.makeCommandQueue() else {
      throw failure("Metal command queue is unavailable.")
    }
    var chunks: [Chunk] = []
    var logicalDigest = SHA256()
    // Two 16 MiB windows overlap first-use hashing at the same staging bound
    // as the single 32 MiB window used when a prior source hash is available.
    let requestedWindow = ProcessInfo.processInfo.environment["QGPU_EMPAD_WINDOW"]
      ?? (pipelinedHash == nil ? "512" : "256")
    let window = requestedWindow == "64" ? 64 : requestedWindow == "256" ? 256 : 512
    var first = 0
    while first < source.frameCount {
      try checkCancellation(shouldCancel)
      var allocatedBytes = UInt64(device.currentAllocatedSize)
      // A nearly full budget may fit the remaining source only after the last
      // hash releases its window. Do not reject a valid source prematurely.
      if memoryBudgetBytes <= allocatedBytes + 16384 * 4 * 2 + 128 * 16 * 2 + UInt64(getpagesize()) * 3 {
        pipelinedHash?.wait()
        allocatedBytes = UInt64(device.currentAllocatedSize)
      }
      let available = memoryBudgetBytes > allocatedBytes ? memoryBudgetBytes - allocatedBytes : 0
      let peakBytesPerFrame: UInt64 = 16384 * 4 * 2 + 128 * 16 * 2
      // The input is page-aligned in size; allow page rounding for the two
      // descriptor buffers and a variable-length packed payload as well.
      let pageHeadroom = UInt64(getpagesize()) * 3
      let windowBudget = available > pageHeadroom ? available - pageHeadroom : 0
      let framesWithinBudget = Int(min(UInt64(window), windowBudget / peakBytesPerFrame))
      let frameCount = min(framesWithinBudget, source.frameCount - first)
      let blocks = frameCount * 128
      let sourceBytes = frameCount * 16384 * 4
      let descriptorBytes = blocks * MemoryLayout<SIMD4<UInt32>>.stride
      // Includes worst-case payload and private descriptor copy before reading.
      let additionalBytes = UInt64(sourceBytes * 2 + descriptorBytes * 2)
      guard frameCount > 0, allocatedBytes <= memoryBudgetBytes,
        additionalBytes <= memoryBudgetBytes - allocatedBytes
      else {
        if profile {
          fputs("EMPAD_BUDGET allocated=\(allocatedBytes) budget=\(memoryBudgetBytes) first=\(first) frames=\(frameCount) additional=\(additionalBytes)\n", stderr)
        }
        throw failure(
          "The full EMPAD resident exceeds the Metal memory budget. Close another dataset or open a smaller acquisition; no pixels were reduced."
        )
      }
      try autoreleasepool {
        let readStarted = CFAbsoluteTimeGetCurrent()
        guard let input = device.makeBuffer(length: sourceBytes, options: .storageModeShared) else {
          throw failure("Metal could not allocate the bounded EMPAD input window.")
        }
        let inputBytes = UnsafeMutableRawBufferPointer(start: input.contents(), count: sourceBytes)
        try source.readFrames(Array(first..<(first + frameCount)), into: inputBytes)
        readSeconds += CFAbsoluteTimeGetCurrent() - readStarted
        let analyzeStarted = CFAbsoluteTimeGetCurrent()
        guard
          let descriptors = device.makeBuffer(length: descriptorBytes, options: .storageModeShared),
          let analysis = queue.makeCommandBuffer(),
          let encoder = analysis.makeComputeCommandEncoder()
        else { throw failure("Metal could not allocate the bounded EMPAD packing window.") }
        encoder.setComputePipelineState(describe)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(descriptors, offset: 0, index: 1)
        if cooperativePacking {
          encoder.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else { dispatch(encoder, pipeline: describe, count: blocks) }
        encoder.endEncoding()
        // Both consumers only read input. An ordered, bounded hash queue also
        // overlaps packing and the following read without changing SHA-256.
        analysis.commit()
        let hashStarted = CFAbsoluteTimeGetCurrent()
        if let pipelinedHash { pipelinedHash.append(input) }
        else if cachedHash == nil { logicalDigest.update(bufferPointer: UnsafeRawBufferPointer(inputBytes)) }
        let hashElapsed = CFAbsoluteTimeGetCurrent() - hashStarted
        if pipelinedHash != nil { hashWaitSeconds += hashElapsed }
        else { hashSeconds += hashElapsed }
        analysis.waitUntilCompleted()
        guard analysis.status == .completed else {
          throw failure("EMPAD row analysis failed: \(String(describing: analysis.error))")
        }
        analyzeSeconds += CFAbsoluteTimeGetCurrent() - analyzeStarted - hashElapsed
        let packStarted = CFAbsoluteTimeGetCurrent()
        try checkCancellation(shouldCancel)
        let entries = descriptors.contents().bindMemory(to: SIMD4<UInt32>.self, capacity: blocks)
        var words: UInt32 = 0
        for block in 0..<blocks {
          entries[block].w = words
          words += entries[block].y * 4
        }
        guard
          let payload = device.makeBuffer(
            length: max(4, Int(words) * 4), options: .storageModePrivate),
          let residentDescriptors = device.makeBuffer(
            length: descriptorBytes, options: .storageModePrivate),
          let command = queue.makeCommandBuffer(),
          let packEncoder = command.makeComputeCommandEncoder()
        else {
          throw failure("Metal could not allocate the exact EMPAD resident. Free memory and retry.")
        }
        packEncoder.setComputePipelineState(pack)
        packEncoder.setBuffer(input, offset: 0, index: 0)
        packEncoder.setBuffer(descriptors, offset: 0, index: 1)
        packEncoder.setBuffer(payload, offset: 0, index: 2)
        if cooperativePacking {
          packEncoder.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else { dispatch(packEncoder, pipeline: pack, count: blocks) }
        packEncoder.endEncoding()
        guard let copy = command.makeBlitCommandEncoder() else {
          throw failure("Metal descriptor upload failed.")
        }
        copy.copy(
          from: descriptors, sourceOffset: 0, to: residentDescriptors,
          destinationOffset: 0, size: descriptorBytes)
        copy.endEncoding()
        try finish(command)
        packSeconds += CFAbsoluteTimeGetCurrent() - packStarted
        guard UInt64(device.currentAllocatedSize) <= memoryBudgetBytes else {
          throw failure("EMPAD packing exceeded the Metal memory budget, including command resources. Free memory and retry; no partial resident was published.")
        }
        try checkCancellation(shouldCancel)
        chunks.append(
          Chunk(
            firstFrame: first, frameCount: frameCount,
            payload: payload, descriptors: residentDescriptors))
      }
      first += frameCount
    }
    let hashFinishStarted = CFAbsoluteTimeGetCurrent()
    let completedHash = pipelinedHash?.finish()
    if let completedHash {
      hashSeconds = completedHash.seconds
      hashWaitSeconds += CFAbsoluteTimeGetCurrent() - hashFinishStarted
    }
    try source.validateUnchanged()
    let logicalHash = cachedHash ?? completedHash?.hash ?? logicalDigest.finalize().map { String(format: "%02x", $0) }.joined()
    if profile {
      fputs(String(format: "EMPAD_LOAD total=%.6f read=%.6f hash=%.6f analyze=%.6f pack=%.6f chunks=%d hash_cached=%d frames_read=%d hash_pipeline=%d hash_wait=%.6f\n",
        CFAbsoluteTimeGetCurrent() - started, readSeconds, hashSeconds, analyzeSeconds, packSeconds, chunks.count, cachedHash == nil ? 0 : 1, first, pipelinedHash == nil ? 0 : 1, hashWaitSeconds), stderr)
    }
    try checkCancellation(shouldCancel)
    if cachedHash == nil { EMPADSourceHashCache.write(hashCacheURL, snapshot: snapshot, hash: logicalHash) }
    return MetalEMPADResidentSource(
      source: source, device: device,
      diffraction: diffraction, detector: detector, chunks: chunks,
      logicalSHA256: logicalHash,
      centerOfMass: centerOfMass, mean: mean, serialDetector: serialDetector,
      detectorThreads: detectorThreads, serialCenterOfMass: serialCenterOfMass,
      incremental: incremental, memoryBudgetBytes: memoryBudgetBytes, reusedSourceHash: cachedHash != nil)
  }

  /// Encode a complete selected float32 DP without reading it back to the CPU.
  /// Output needs 128×128×4 bytes. The caller waits for its command completion
  /// before publishing the frame or reusing the output buffer.
  public func encodeDiffraction(
    scanRow: Int, scanColumn: Int, into output: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    guard !isReleased, (0..<source.scanRows).contains(scanRow),
      (0..<source.scanColumns).contains(scanColumn), output.length >= 16384 * 4,
      command.commandQueue.device.registryID == device.registryID,
      output.device.registryID == device.registryID
    else {
      throw Self.failure(
        "EMPAD diffraction needs a resident source, valid scan coordinates and a same-device 128×128 float32 output."
      )
    }
    let frame = scanRow * source.scanColumns + scanColumn
    guard
      let chunk = chunks.first(where: {
        ($0.firstFrame..<($0.firstFrame + $0.frameCount)).contains(frame)
      }),
      let encoder = command.makeComputeCommandEncoder()
    else { throw Self.failure("EMPAD selected-frame encoding failed.") }
    var local = UInt32(frame - chunk.firstFrame)
    encoder.setComputePipelineState(diffractionPipeline)
    encoder.setBuffer(chunk.payload, offset: 0, index: 0)
    encoder.setBuffer(chunk.descriptors, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)
    encoder.setBytes(&local, length: 4, index: 3)
    Self.dispatch(encoder, pipeline: diffractionPipeline, count: 16384)
    encoder.endEncoding()
  }

  /// Encode mask integration over all scan positions, directly from packing.
  ///
  /// Any nonzero uint8 mask entry selects that detector pixel. Output is
  /// float32 in scan-row-major order. Selected NaNs propagate; unselected
  /// pixels do not participate. Output requires `source.frameCount * 4` bytes.
  public func encodeVirtualImage(
    mask: MTLBuffer, into output: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    guard !isReleased, mask !== output, mask.length >= 16384,
      output.length >= source.frameCount * 4,
      command.commandQueue.device.registryID == device.registryID,
      mask.device.registryID == device.registryID, output.device.registryID == device.registryID
    else {
      throw Self.failure(
        "EMPAD integration needs a resident source, a same-device 128×128 uint8 mask and a separate full-scan float32 output."
      )
    }
    if let incrementalPipeline, mask.storageMode == .shared,
      try encodeChanges(mask: mask, output: output, command: command, pipeline: incrementalPipeline) {
      return
    }
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Self.failure("EMPAD detector encoding failed.")
    }
    encoder.setComputePipelineState(detectorPipeline)
    encoder.setBuffer(mask, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    for chunk in chunks {
      var offset = UInt32(chunk.firstFrame)
      encoder.setBuffer(chunk.payload, offset: 0, index: 0)
      encoder.setBuffer(chunk.descriptors, offset: 0, index: 1)
      encoder.setBytes(&offset, length: 4, index: 4)
      if serialDetector {
        Self.dispatch(encoder, pipeline: detectorPipeline, count: chunk.frameCount)
      } else {
        encoder.dispatchThreadgroups(
          MTLSize(width: chunk.frameCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: detectorThreads, height: 1, depth: 1))
      }
    }
    encoder.endEncoding()
  }

  /// Encode full-detector intensity-weighted `(row, column)` coordinates.
  /// Signed measurements participate unchanged. Zero total or non-finite input
  /// produces NaN coordinates, not an invented detector center.
  public func encodeCenterOfMass(
    intoRow row: MTLBuffer, intoColumn column: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    guard !isReleased, row !== column,
      row.length >= source.frameCount * 4, column.length >= source.frameCount * 4,
      row.device.registryID == device.registryID, column.device.registryID == device.registryID,
      command.commandQueue.device.registryID == device.registryID,
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Self.failure(
        "EMPAD CoM needs a resident and two separate same-device full-scan float32 outputs.")
    }
    encoder.setComputePipelineState(centerOfMassPipeline)
    encoder.setBuffer(row, offset: 0, index: 2)
    encoder.setBuffer(column, offset: 0, index: 3)
    for chunk in chunks {
      var offset = UInt32(chunk.firstFrame)
      encoder.setBuffer(chunk.payload, offset: 0, index: 0)
      encoder.setBuffer(chunk.descriptors, offset: 0, index: 1)
      encoder.setBytes(&offset, length: 4, index: 4)
      if serialCenterOfMass {
        Self.dispatch(encoder, pipeline: centerOfMassPipeline, count: chunk.frameCount)
      } else {
        encoder.dispatchThreadgroups(MTLSize(width: chunk.frameCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      }
    }
    encoder.endEncoding()
  }

  /// Encode the arithmetic mean of every scan position, with compensated sums.
  /// Output is a 128×128 float32 DP. Temporary compensation is only one DP,
  /// retained by the command until completion; no dense 4D array is allocated.
  public func encodeMeanDiffraction(into output: MTLBuffer, command: MTLCommandBuffer) throws {
    guard !isReleased, output.length >= 16384 * 4,
      output.device.registryID == device.registryID,
      command.commandQueue.device.registryID == device.registryID,
      let accumulator = device.makeBuffer(length: 16384 * 8, options: .storageModePrivate),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Self.failure("EMPAD mean DP needs a resident and a same-device 128×128 float32 output.")
    }
    encoder.setComputePipelineState(meanPipeline)
    encoder.setBuffer(accumulator, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    for chunk in chunks {
      var dimensions = SIMD3<UInt32>(
        UInt32(chunk.firstFrame), UInt32(chunk.frameCount), UInt32(source.frameCount))
      encoder.setBuffer(chunk.payload, offset: 0, index: 0)
      encoder.setBuffer(chunk.descriptors, offset: 0, index: 1)
      encoder.setBytes(&dimensions, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 4)
      Self.dispatch(encoder, pipeline: meanPipeline, count: 16384)
      encoder.memoryBarrier(scope: .buffers)
    }
    encoder.endEncoding()
  }

  /// Release after outstanding commands finish. Encoding after release fails.
  public func releaseResidentStorage() {
    chunks.removeAll()
    detectorAccumulation = nil
    priorDetectorMask = nil
    priorDetectorCommand = nil
    isReleased = true
  }

  private func encodeChanges(mask: MTLBuffer, output: MTLBuffer, command: MTLCommandBuffer,
    pipeline: MTLComputePipelineState) throws -> Bool {
    let cacheBytes = source.frameCount * 8
    let allocated = UInt64(device.currentAllocatedSize)
    guard allocated <= memoryBudgetBytes,
      UInt64((detectorAccumulation == nil ? cacheBytes : 0) + 16384 * 8 + 32768)
        <= memoryBudgetBytes - allocated else { return false }
    if detectorAccumulation == nil {
      detectorAccumulation = device.makeBuffer(length: cacheBytes, options: .storageModePrivate)
    }
    guard let accumulated = detectorAccumulation else { return false }
    let current = UnsafeBufferPointer(start: mask.contents().assumingMemoryBound(to: UInt8.self), count: 16384)
      .map { $0 == 0 ? UInt8(0) : UInt8(1) }
    var reset = priorDetectorCommand?.status != .completed || priorDetectorMask == nil || incrementalUpdates >= 64
    var entries: [SIMD2<Int32>] = []
    if !reset, let prior = priorDetectorMask {
      entries = current.indices.compactMap {
        current[$0] == prior[$0] ? nil : SIMD2(Int32($0), Int32(current[$0]) - Int32(prior[$0]))
      }
      reset = entries.count >= current.reduce(0) { $0 + Int($1) }
    }
    if reset {
      entries = current.indices.compactMap { current[$0] == 0 ? nil : SIMD2(Int32($0), 1) }
    }
    let count = entries.count
    if entries.isEmpty { entries.append(.zero) }
    guard let list = entries.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
      let encoder = command.makeComputeCommandEncoder() else {
      throw Self.failure("EMPAD aperture change encoding failed.")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(list, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    encoder.setBuffer(accumulated, offset: 0, index: 5)
    var parameters = SIMD2<UInt32>(UInt32(count), reset ? 1 : 0)
    encoder.setBytes(&parameters, length: 8, index: 6)
    encoder.setBuffer(mask, offset: 0, index: 7)
    for chunk in chunks {
      var offset = UInt32(chunk.firstFrame)
      encoder.setBuffer(chunk.payload, offset: 0, index: 0)
      encoder.setBuffer(chunk.descriptors, offset: 0, index: 1)
      encoder.setBytes(&offset, length: 4, index: 4)
      encoder.dispatchThreadgroups(MTLSize(width: chunk.frameCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: detectorThreads, height: 1, depth: 1))
    }
    encoder.endEncoding()
    priorDetectorMask = current
    priorDetectorCommand = command
    incrementalUpdates = reset ? 0 : incrementalUpdates + 1
    return true
  }

  private static func dispatch(
    _ encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState, count: Int
  ) {
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(32, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
  }

  private static func finish(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if command.status != .completed {
      throw failure(
        "EMPAD Metal command failed: \(command.error?.localizedDescription ?? "unknown GPU error")."
      )
    }
  }

  private static func checkCancellation(_ cancelled: () -> Bool) throws {
    if cancelled() { throw CancellationError() }
  }

  private static func failure(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }
}
