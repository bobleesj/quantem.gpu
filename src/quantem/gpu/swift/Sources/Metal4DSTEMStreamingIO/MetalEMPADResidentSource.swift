import CryptoKit
import Foundation
@preconcurrency import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

/// Exact float32 measurements resident as ANS-coded IEEE bit lanes.
/// Scientific consumers decode bounded GPU windows without a full dense cube.
/// Calls, release and command completion must be serialized by the owner.
public final class MetalEMPADResidentSource {
  public let source: NativeEMPADSource
  public let residentBytes: UInt64
  /// SHA-256 of all original little-endian float32 detector words in scan order.
  public let logicalSHA256: String
  /// Tensor identity binding shape and dtype to `logicalSHA256`, not a file checksum.
  public let sourceIdentitySHA256: String
  /// Uncorrected tensor identity, suitable for source-bound notes and metadata.
  public let originalSourceIdentitySHA256: String
  /// True only when a checksum-protected source hash matched the file snapshot.
  /// The complete original measurements are still reread and ANS encoded.
  public let reusedSourceHash: Bool
  /// Optional calibrated product transform; original encoded measurements remain intact.
  public private(set) var background: MetalEMPADBackground?
  public private(set) var isReleased = false
  let device: MTLDevice
  let ans: MetalFloatANS
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
  var chunks: [Chunk]

  struct Chunk {
    let firstFrame: Int
    let frameCount: Int
    let payload: MTLBuffer
    let offsets: MTLBuffer
    let models: MTLBuffer
  }

  init(
    source: NativeEMPADSource, device: MTLDevice, ans: MetalFloatANS,
    diffraction: MTLComputePipelineState, detector: MTLComputePipelineState, chunks: [Chunk],
    logicalSHA256: String, centerOfMass: MTLComputePipelineState, mean: MTLComputePipelineState,
    serialDetector: Bool, detectorThreads: Int, serialCenterOfMass: Bool,
    incremental: MTLComputePipelineState?, memoryBudgetBytes: UInt64, reusedSourceHash: Bool,
    background: MetalEMPADBackground?
  ) {
    self.source = source
    self.device = device
    self.ans = ans
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
    self.background = background
    self.chunks = chunks
    self.logicalSHA256 = logicalSHA256
    var identity = SHA256()
    identity.update(data: Data("quantem.gpu.empad-tensor/v1\0float32-le\0".utf8))
    for dimension in [
      source.scanRows, source.scanColumns, source.detectorShape.row, source.detectorShape.column,
    ] {
      var word = UInt64(dimension).littleEndian
      withUnsafeBytes(of: &word) { identity.update(bufferPointer: $0) }
    }
    identity.update(data: Data(logicalSHA256.utf8))
    originalSourceIdentitySHA256 = identity.finalize().map { String(format: "%02x", $0) }.joined()
    if let background {
      identity.update(
        data: Data((MetalEMPADBackground.schema + "\0" + background.identitySHA256).utf8))
    }
    sourceIdentitySHA256 = identity.finalize().map { String(format: "%02x", $0) }.joined()
    residentBytes =
      chunks.reduce(0) { $0 + UInt64($1.payload.length + $1.offsets.length + $1.models.length) }
      + UInt64(ans.table.length) + UInt64(background?.values.length ?? 0)
  }

  /// Read every original detector pixel and finish ANS encoding before returning.
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
    subtracting background: MetalEMPADBackground? = nil,
    shouldCancel: () -> Bool = { false }
  ) throws -> MetalEMPADResidentSource {
    if source.hasQEMStorage {
      guard background == nil else {
        throw failure(
          "A QEM file restores its saved background state. Do not apply another dark during loading."
        )
      }
      return try restoreQEM(
        source, device: device, memoryBudgetBytes: memoryBudgetBytes,
        shouldCancel: shouldCancel)
    }
    let started = CFAbsoluteTimeGetCurrent()
    let profile = ProcessInfo.processInfo.environment["QGPU_EMPAD_LOAD_PROFILE"] == "1"
    var readSeconds = 0.0
    var hashSeconds = 0.0
    var encodeSeconds = 0.0
    guard source.frameCount <= Int(UInt32.max) else {
      throw failure("EMPAD scan exceeds the supported frame-address range.")
    }
    try checkCancellation(shouldCancel)
    try source.validateUnchanged()
    if let background {
      guard background.values.device.registryID == device.registryID,
        background.source.rawURL.resolvingSymlinksInPath()
          != source.rawURL.resolvingSymlinksInPath(),
        background.source.formatIdentifier == source.formatIdentifier,
        background.values.length == source.detectorPixelCount * 4,
        background.source.detectorShape == source.detectorShape
      else {
        throw failure(
          "Choose a different dark acquisition with the same reader format and Metal device.")
      }
      try background.source.validateUnchanged()
      let sampleMetadata = NativeMicroscopeMetadata(metadata: source.microscopeMetadata)
      let darkMetadata = NativeMicroscopeMetadata(metadata: background.source.microscopeMetadata)
      if let sample = sampleMetadata.dwellTimeMicroseconds,
        let dark = darkMetadata.dwellTimeMicroseconds,
        abs(sample - dark) > max(sample, dark) * 1e-6
      {
        throw failure(
          "Sample and dark exposure times differ. Choose a matching dark; automatic exposure scaling is not supported."
        )
      }
    }
    let snapshot = try source.sourceSnapshot()
    let hashCacheURL = EMPADSourceHashCache.safeURL(sourceHashCacheURL, source: source)
    let cachedHash = EMPADSourceHashCache.read(hashCacheURL, snapshot: snapshot)
    let library = try Metal4DSTEMKernels.makeEMPADLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let constants = MTLFunctionConstantValues()
      var decoded = true
      constants.setConstantValue(&decoded, type: .bool, index: 0)
      constants.setConstantValue(&decoded, type: .bool, index: 1)
      var pixels = UInt32(source.detectorPixelCount)
      var columns = UInt32(source.detectorShape.column)
      constants.setConstantValue(&pixels, type: .uint, index: 20)
      constants.setConstantValue(&columns, type: .uint, index: 21)
      let function = try library.makeFunction(name: name, constantValues: constants)
      return try device.makeComputePipelineState(function: function)
    }
    let diffraction = try pipeline("empad_diffraction")
    // Retained control for reproducible native A/B/A qualification.
    let serialDetector = ProcessInfo.processInfo.environment["QGPU_EMPAD_SERIAL_DETECTOR"] == "1"
    let detectorThreads =
      ProcessInfo.processInfo.environment["QGPU_EMPAD_DETECTOR_GROUPS"] == "1" ? 32 : 128
    let detector = try pipeline(
      serialDetector ? "empad_virtual_image_serial" : "empad_virtual_image")
    guard detector.threadExecutionWidth == 32,
      detector.maxTotalThreadsPerThreadgroup >= detectorThreads
    else {
      throw failure(
        "EMPAD cooperative reduction requires 32-lane SIMD groups and a 128-thread-capable Metal pipeline."
      )
    }
    let serialCenterOfMass = ProcessInfo.processInfo.environment["QGPU_EMPAD_COM_CONTROL"] == "1"
    let centerOfMass = try pipeline(
      serialCenterOfMass ? "empad_center_of_mass" : "empad_center_of_mass_simd")
    let mean = try pipeline("empad_mean_diffraction")
    let incremental =
      ProcessInfo.processInfo.environment["QGPU_EMPAD_INCREMENTAL"] != "0"
      ? try pipeline("empad_virtual_image_changes") : nil
    if let incremental, incremental.maxTotalThreadsPerThreadgroup < detectorThreads {
      throw failure("EMPAD aperture-change kernel requires a 128-thread-capable Metal pipeline.")
    }
    let ans = try MetalFloatANS(device: device, pixels: source.detectorPixelCount)
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    guard allocatedBefore < memoryBudgetBytes else {
      throw failure("Free memory before loading the ANS resident.")
    }
    let encoder = try RuntimeANSEncoder(
      device: device, pixels: ans.lanes, bytesPerValue: 2,
      allocatedBefore: allocatedBefore, maximumAdditionalBytes: memoryBudgetBytes - allocatedBefore)
    var digest = SHA256()
    // A bounded restart window also fixes scientific reduction order.
    let requested = ProcessInfo.processInfo.environment["QGPU_EMPAD_WINDOW"]
    let window = min(
      requested == "64" ? 64 : requested == "512" ? 512 : 256,
      (32 << 20) / (source.detectorPixelCount * 4))
    var first = 0
    while first < source.frameCount {
      try checkCancellation(shouldCancel)
      let allocated = UInt64(device.currentAllocatedSize)
      let available = memoryBudgetBytes > allocated ? memoryBudgetBytes - allocated : 0
      // Include input, ANS scratch, worst-case output, and stream tables.
      // Shrinking this window changes neither detector nor scan coverage.
      let frameWorkspaceBytes = UInt64(source.detectorPixelCount * 16)
      let boundedFrames =
        available > 1 << 20
        ? Int((available - (1 << 20)) / frameWorkspaceBytes) : 0
      let count = min(window, boundedFrames, source.frameCount - first)
      guard count > 0 else { throw failure("Free memory for a bounded ANS encoding window.") }
      try autoreleasepool {
        let bytes = count * source.detectorPixelCount * 4
        guard UInt64(device.currentAllocatedSize) + UInt64(bytes) < memoryBudgetBytes,
          let input = device.makeBuffer(length: bytes, options: .storageModeShared)
        else { throw failure("Free memory for the bounded ANS input window.") }
        let readStarted = CFAbsoluteTimeGetCurrent()
        try source.readFrames(
          Array(first..<(first + count)),
          into: UnsafeMutableRawBufferPointer(start: input.contents(), count: bytes))
        try checkCancellation(shouldCancel)
        readSeconds += CFAbsoluteTimeGetCurrent() - readStarted
        if cachedHash == nil {
          let hashStarted = CFAbsoluteTimeGetCurrent()
          digest.update(
            bufferPointer: UnsafeRawBufferPointer(start: input.contents(), count: bytes))
          hashSeconds += CFAbsoluteTimeGetCurrent() - hashStarted
        }
        let encodeStarted = CFAbsoluteTimeGetCurrent()
        try encoder.append(dense: input, firstScan: first, scanCount: count)
        try checkCancellation(shouldCancel)
        encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStarted
        if count < window { encoder.releaseScratch() }
      }
      first += count
    }
    let encoded = try encoder.finish()
    let chunks = encoded.chunks.map {
      Chunk(
        firstFrame: $0.firstScan, frameCount: $0.scanCount,
        payload: $0.payload, offsets: $0.offsets, models: $0.models)
    }
    try source.validateUnchanged()
    try background?.source.validateUnchanged()
    try checkCancellation(shouldCancel)
    let logicalHash = cachedHash ?? digest.finalize().map { String(format: "%02x", $0) }.joined()
    if cachedHash == nil {
      EMPADSourceHashCache.write(hashCacheURL, snapshot: snapshot, hash: logicalHash)
    }
    if profile {
      fputs(
        "EMPAD_ANS total=\(CFAbsoluteTimeGetCurrent() - started) read=\(readSeconds) hash=\(hashSeconds) encode=\(encodeSeconds) chunks=\(chunks.count)\n",
        stderr)
    }
    return MetalEMPADResidentSource(
      source: source, device: device, ans: ans,
      diffraction: diffraction, detector: detector, chunks: chunks,
      logicalSHA256: logicalHash,
      centerOfMass: centerOfMass, mean: mean, serialDetector: serialDetector,
      detectorThreads: detectorThreads, serialCenterOfMass: serialCenterOfMass,
      incremental: incremental, memoryBudgetBytes: memoryBudgetBytes,
      reusedSourceHash: cachedHash != nil, background: background)
  }

  /// Encode a complete selected float32 DP without reading it back to the CPU.
  /// Output needs native-detector×4 bytes. The caller waits for its command completion
  /// before publishing the frame or reusing the output buffer.
  public func encodeDiffraction(
    scanRow: Int, scanColumn: Int, into output: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    guard !isReleased, (0..<source.scanRows).contains(scanRow),
      (0..<source.scanColumns).contains(scanColumn), output.length >= source.detectorPixelCount * 4,
      command.commandQueue.device.registryID == device.registryID,
      output.device.registryID == device.registryID
    else {
      throw Self.failure(
        "EMPAD diffraction needs a resident source, valid scan coordinates and a same-device native-detector float32 output."
      )
    }
    let frame = scanRow * source.scanColumns + scanColumn
    guard
      let chunk = chunks.first(where: {
        ($0.firstFrame..<($0.firstFrame + $0.frameCount)).contains(frame)
      })
    else { throw Self.failure("EMPAD selected-frame encoding failed.") }
    let workspace = try ans.workspace(frames: 1, budget: memoryBudgetBytes, command: command)
    try ans.encode(
      chunk, first: frame - chunk.firstFrame, count: 1, into: workspace, command: command)
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Self.failure("EMPAD selected-frame encoding failed.")
    }
    var local: UInt32 = 0
    encoder.setComputePipelineState(diffractionPipeline)
    bindBackground(encoder, fallback: chunk.payload)
    encoder.setBuffer(workspace.words, offset: 0, index: 0)
    encoder.setBuffer(workspace.descriptors, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)
    encoder.setBytes(&local, length: 4, index: 3)
    Self.dispatch(encoder, pipeline: diffractionPipeline, count: source.detectorPixelCount)
    encoder.endEncoding()
  }

  /// Encode mask integration over all scan positions using bounded ANS windows.
  ///
  /// Any nonzero uint8 mask entry selects that detector pixel. Output is
  /// float32 in scan-row-major order. Selected NaNs propagate; unselected
  /// pixels do not participate. Output requires `source.frameCount * 4` bytes.
  public func encodeVirtualImage(
    mask: MTLBuffer, into output: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    guard !isReleased, mask !== output, mask.length >= source.detectorPixelCount,
      output.length >= source.frameCount * 4,
      command.commandQueue.device.registryID == device.registryID,
      mask.device.registryID == device.registryID, output.device.registryID == device.registryID
    else {
      throw Self.failure(
        "EMPAD integration needs a resident source, a same-device native-detector uint8 mask and a separate full-scan float32 output."
      )
    }
    if let incrementalPipeline, mask.storageMode == .shared,
      try encodeChanges(mask: mask, output: output, command: command, pipeline: incrementalPipeline)
    {
      return
    }
    let workspace = try ans.workspace(
      frames: chunks.map(\.frameCount).max() ?? 0, budget: memoryBudgetBytes, command: command)
    for chunk in chunks {
      try ans.encode(chunk, into: workspace, command: command)
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Self.failure("EMPAD consumer encoder unavailable.")
      }
      encoder.setComputePipelineState(detectorPipeline)
      bindBackground(encoder, fallback: output)
      encoder.setBuffer(mask, offset: 0, index: 2)
      encoder.setBuffer(output, offset: 0, index: 3)
      var offset = UInt32(chunk.firstFrame)
      encoder.setBuffer(workspace.words, offset: 0, index: 0)
      encoder.setBuffer(workspace.descriptors, offset: 0, index: 1)
      encoder.setBytes(&offset, length: 4, index: 4)
      if serialDetector {
        Self.dispatch(encoder, pipeline: detectorPipeline, count: chunk.frameCount)
      } else {
        encoder.dispatchThreadgroups(
          MTLSize(width: chunk.frameCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: detectorThreads, height: 1, depth: 1))
      }
      encoder.endEncoding()
    }
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
      command.commandQueue.device.registryID == device.registryID
    else {
      throw Self.failure(
        "EMPAD CoM needs a resident and two separate same-device full-scan float32 outputs.")
    }
    let workspace = try ans.workspace(
      frames: chunks.map(\.frameCount).max() ?? 0, budget: memoryBudgetBytes, command: command)
    for chunk in chunks {
      try ans.encode(chunk, into: workspace, command: command)
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Self.failure("EMPAD consumer encoder unavailable.")
      }
      encoder.setComputePipelineState(centerOfMassPipeline)
      bindBackground(encoder, fallback: row)
      encoder.setBuffer(row, offset: 0, index: 2)
      encoder.setBuffer(column, offset: 0, index: 3)
      var offset = UInt32(chunk.firstFrame)
      encoder.setBuffer(workspace.words, offset: 0, index: 0)
      encoder.setBuffer(workspace.descriptors, offset: 0, index: 1)
      encoder.setBytes(&offset, length: 4, index: 4)
      if serialCenterOfMass {
        Self.dispatch(encoder, pipeline: centerOfMassPipeline, count: chunk.frameCount)
      } else {
        encoder.dispatchThreadgroups(
          MTLSize(width: chunk.frameCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      }
      encoder.endEncoding()
    }
  }

  /// Encode the arithmetic mean of every scan position, with compensated sums.
  /// Output is a native-detector float32 DP. Temporary compensation is only one DP,
  /// retained by the command until completion; no dense 4D array is allocated.
  /// Optional half-open scan `rows` and `columns` restrict the mean to a
  /// rectangle or circle with square bounds. Circle pixel centers inside or on
  /// the boundary count equally. Background correction and signed values are preserved.
  public func encodeMeanDiffraction(
    into output: MTLBuffer, command: MTLCommandBuffer,
    rows: Range<Int>? = nil, columns: Range<Int>? = nil,
    shape: MetalScanRegionShape = .rectangle
  ) throws {
    let selectedRows = rows ?? 0..<source.scanRows
    let selectedColumns = columns ?? 0..<source.scanColumns
    guard !selectedRows.isEmpty, !selectedColumns.isEmpty,
      selectedRows.lowerBound >= 0, selectedRows.upperBound <= source.scanRows,
      selectedColumns.lowerBound >= 0, selectedColumns.upperBound <= source.scanColumns,
      shape != .circle || selectedRows.count == selectedColumns.count
    else {
      throw Self.failure(
        "Mean DP requires a nonempty region inside the loaded scan; circle bounds must be square.")
    }
    guard !isReleased, output.length >= source.detectorPixelCount * 4,
      output.device.registryID == device.registryID,
      command.commandQueue.device.registryID == device.registryID,
      let accumulator = device.makeBuffer(
        length: source.detectorPixelCount * 8, options: .storageModePrivate)
    else {
      throw Self.failure(
        "EMPAD mean DP needs a resident and a same-device native-detector float32 output.")
    }
    if rows != nil || columns != nil,
      ProcessInfo.processInfo.environment["QGPU_FLOAT_ANS_MEAN_CONTROL"] != "1"
    {
      try ans.encodeMean(
        chunks: chunks, rows: selectedRows, columns: selectedColumns,
        scanColumns: source.scanColumns, shape: shape, output: output,
        accumulator: accumulator, background: background?.values, command: command)
      return
    }
    let workspace = try ans.workspace(
      frames: chunks.map(\.frameCount).max() ?? 0, budget: memoryBudgetBytes, command: command)
    for chunk in chunks {
      try ans.encode(chunk, into: workspace, command: command)
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Self.failure("EMPAD consumer encoder unavailable.")
      }
      encoder.setComputePipelineState(meanPipeline)
      bindBackground(encoder, fallback: output)
      encoder.setBuffer(accumulator, offset: 0, index: 2)
      encoder.setBuffer(output, offset: 0, index: 3)
      var region = SIMD4<UInt32>(
        UInt32(selectedRows.lowerBound), UInt32(selectedRows.upperBound),
        UInt32(selectedColumns.lowerBound), UInt32(selectedColumns.upperBound))
      var scanColumns = UInt32(source.scanColumns)
      encoder.setBytes(&region, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 5)
      encoder.setBytes(&scanColumns, length: 4, index: 6)
      var regionShape = shape.rawValue
      encoder.setBytes(&regionShape, length: 4, index: 7)
      var dimensions = SIMD3<UInt32>(
        UInt32(chunk.firstFrame), UInt32(chunk.frameCount),
        UInt32(shape.sampleCount(rowCount: selectedRows.count, columnCount: selectedColumns.count)))
      encoder.setBuffer(workspace.words, offset: 0, index: 0)
      encoder.setBuffer(workspace.descriptors, offset: 0, index: 1)
      encoder.setBytes(&dimensions, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 4)
      Self.dispatch(encoder, pipeline: meanPipeline, count: source.detectorPixelCount)
      encoder.memoryBarrier(scope: .buffers)
      encoder.endEncoding()
    }
  }

  /// Release after outstanding commands finish. Encoding after release fails.
  public func releaseResidentStorage() {
    chunks.removeAll()
    ans.releaseScratch()
    background = nil
    detectorAccumulation = nil
    priorDetectorMask = nil
    priorDetectorCommand = nil
    isReleased = true
  }

  private func encodeChanges(
    mask: MTLBuffer, output: MTLBuffer, command: MTLCommandBuffer,
    pipeline: MTLComputePipelineState
  ) throws -> Bool {
    let cacheBytes = source.frameCount * 8
    let allocated = UInt64(device.currentAllocatedSize)
    guard allocated <= memoryBudgetBytes,
      UInt64((detectorAccumulation == nil ? cacheBytes : 0) + source.detectorPixelCount * 8 + 32768)
        <= memoryBudgetBytes - allocated
    else { return false }
    if detectorAccumulation == nil {
      detectorAccumulation = device.makeBuffer(length: cacheBytes, options: .storageModePrivate)
    }
    guard let accumulated = detectorAccumulation else { return false }
    let current = UnsafeBufferPointer(
      start: mask.contents().assumingMemoryBound(to: UInt8.self), count: source.detectorPixelCount
    )
    .map { $0 == 0 ? UInt8(0) : UInt8(1) }
    var reset =
      priorDetectorCommand?.status != .completed || priorDetectorMask == nil
      || incrementalUpdates >= 64
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
    var changedPixels = [UInt8](repeating: 0, count: source.detectorPixelCount)
    for entry in entries { changedPixels[Int(entry.x)] = 1 }
    if entries.isEmpty { entries.append(.zero) }
    guard
      let list = entries.withUnsafeBytes({
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      }),
      let changed = changedPixels.withUnsafeBytes({
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    else {
      throw Self.failure("EMPAD aperture change encoding failed.")
    }
    let recovery = try ans.recoveryFlag(
      accumulated: accumulated, reset: reset,
      frames: source.frameCount, command: command)
    let fullDecodeControl =
      ProcessInfo.processInfo.environment["QGPU_FLOAT_ANS_FULL_DECODE_CONTROL"] == "1"
    let workspace = try ans.workspace(
      frames: chunks.map(\.frameCount).max() ?? 0, budget: memoryBudgetBytes, command: command)
    var samples: MTLCounterSampleBuffer?
    if !fullDecodeControl,
      ProcessInfo.processInfo.environment["QGPU_FLOAT_ANS_STAGE_PROFILE"] == "1",
      let counter = device.counterSets?.first(where: { $0.name == "timestamp" })
    {
      let descriptor = MTLCounterSampleBufferDescriptor()
      descriptor.counterSet = counter
      descriptor.storageMode = .shared
      descriptor.sampleCount = chunks.count * 4
      samples = try device.makeCounterSampleBuffer(descriptor: descriptor)
    }
    func measuredEncoder(_ index: Int) -> MTLComputeCommandEncoder? {
      guard let samples else { return command.makeComputeCommandEncoder() }
      let pass = MTLComputePassDescriptor()
      pass.sampleBufferAttachments[0].sampleBuffer = samples
      pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = index
      pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = index + 1
      return command.makeComputeCommandEncoder(descriptor: pass)
    }
    let sharedEncoder =
      fullDecodeControl || samples != nil
      ? nil : command.makeComputeCommandEncoder(dispatchType: .serial)
    if !fullDecodeControl && samples == nil && sharedEncoder == nil {
      throw Self.failure("EMPAD consumer encoder unavailable.")
    }
    for (chunkIndex, chunk) in chunks.enumerated() {
      if fullDecodeControl {
        try ans.encode(chunk, into: workspace, command: command)
      } else {
        guard let decoder = sharedEncoder ?? measuredEncoder(chunkIndex * 4) else {
          throw Self.failure("EMPAD decoder encoder unavailable.")
        }
        try ans.encodeSelected(
          chunk, changed: changed, entries: list, count: count, mask: mask, recovery: recovery,
          into: workspace, encoder: decoder, preferParallel: !reset)
        if sharedEncoder != nil {
          decoder.memoryBarrier(resources: [workspace.words, workspace.descriptors])
        } else {
          decoder.endEncoding()
        }
      }
      guard let encoder = sharedEncoder ?? measuredEncoder(chunkIndex * 4 + 2) else {
        throw Self.failure("EMPAD consumer encoder unavailable.")
      }
      encoder.setComputePipelineState(pipeline)
      bindBackground(encoder, fallback: output)
      encoder.setBuffer(list, offset: 0, index: 2)
      encoder.setBuffer(output, offset: 0, index: 3)
      encoder.setBuffer(accumulated, offset: 0, index: 5)
      var parameters = SIMD2<UInt32>(UInt32(count), reset ? 1 : 0)
      encoder.setBytes(&parameters, length: 8, index: 6)
      encoder.setBuffer(mask, offset: 0, index: 7)
      encoder.setBuffer(chunk.payload, offset: 0, index: 10)
      encoder.setBuffer(chunk.offsets, offset: 0, index: 11)
      encoder.setBuffer(chunk.models, offset: 0, index: 12)
      var offset = UInt32(chunk.firstFrame)
      encoder.setBuffer(workspace.words, offset: 0, index: 0)
      encoder.setBuffer(workspace.descriptors, offset: 0, index: 1)
      encoder.setBytes(&offset, length: 4, index: 4)
      encoder.dispatchThreadgroups(
        MTLSize(width: chunk.frameCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: detectorThreads, height: 1, depth: 1))
      if sharedEncoder != nil {
        encoder.memoryBarrier(resources: [workspace.words, workspace.descriptors])
      } else {
        encoder.endEncoding()
      }
    }
    sharedEncoder?.endEncoding()
    if let samples {
      let chunkCount = chunks.count
      let isReset = reset
      command.addCompletedHandler { _ in
        guard let data = try? samples.resolveCounterRange(0..<chunkCount * 4) else { return }
        data.withUnsafeBytes { bytes in
          let times = bytes.bindMemory(to: UInt64.self)
          var decode: UInt64 = 0
          var reduce: UInt64 = 0
          for chunk in 0..<chunkCount {
            decode += times[chunk * 4 + 1] - times[chunk * 4]
            reduce += times[chunk * 4 + 3] - times[chunk * 4 + 2]
          }
          print(
            "FLOAT_ANS_STAGES changed=\(count) reset=\(isReset) decode_ticks=\(decode) reduce_ticks=\(reduce)"
          )
        }
      }
    }
    priorDetectorMask = current
    priorDetectorCommand = command
    incrementalUpdates = reset ? 0 : incrementalUpdates + 1
    return true
  }

  private func bindBackground(_ encoder: MTLComputeCommandEncoder, fallback: MTLBuffer) {
    var corrected: UInt32 = background == nil ? 0 : 1
    encoder.setBuffer(background?.values ?? fallback, offset: 0, index: 8)
    encoder.setBytes(&corrected, length: 4, index: 9)
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
