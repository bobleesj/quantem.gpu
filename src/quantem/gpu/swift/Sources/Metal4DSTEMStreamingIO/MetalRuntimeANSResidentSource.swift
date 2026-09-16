import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

/// Timings for an exact original-HDF5 to in-memory runtime-ANS load.
public struct MetalRuntimeANSLoadMetrics: Sendable {
  public let totalSeconds: Double
  public let fusedDecodeAndEncodeSeconds: Double
  public let prefixSeconds: Double
  public let compactSeconds: Double
  public let residentBytes: UInt64
  public let logicalBytes: UInt64
}

/// Timing and work accounting for one exact series virtual-detector update.
public struct MetalRuntimeANSDetectorMetrics: Sendable {
  public let changedDetectorPixels: Int
  public let gpuMilliseconds: Double
  public let wallMilliseconds: Double
  public let acquisitionCount: Int
  public let submissionCount: Int
}

/// Exact native integer counts retained as chunked fixed-model rANS streams.
///
/// This is an in-memory representation, not a file format. Loading reads the
/// original indexed HDF5 acquisition in bounded windows, decodes it on Metal,
/// and immediately encodes every original count into ANS. It never creates a
/// dense full-volume allocation or a derived ANS file.
public final class MetalRuntimeANSResidentSource: @unchecked Sendable {
  public let dataset: Native4DSTEMDataset
  public let shape: [Int]
  public let logicalDtype: Metal4DSTEMIntegerDType
  public let sourceIdentitySHA256: String
  public let loadMetrics: MetalRuntimeANSLoadMetrics
  public private(set) var isReleased = false

  struct Chunk {
    let firstScan: Int
    let scanCount: Int
    let payload: MTLBuffer
    let offsets: MTLBuffer
    let models: MTLBuffer
    var spatial: [MTLBuffer] = []

    var bytes: UInt64 {
      UInt64(payload.length + offsets.length + models.length + spatial.reduce(0) { $0 + $1.length })
    }
  }

  let device: MTLDevice
  let queue: MTLCommandQueue
  let decodePipeline: MTLComputePipelineState
  let detectorDeltaPipeline: MTLComputePipelineState
  let detectorPacketPipeline: MTLComputePipelineState
  let detectorPacketSIMDs: Int
  var decodingTable: MTLBuffer?
  var chunks: [Chunk]
  var failure: MTLBuffer?
  var diffraction: MTLBuffer?
  let validPixels: [UInt8]
  let interval = 512
  var spatialQuery: RuntimeSpatialQuery?
  var detectorColumnsPipeline: MTLComputePipelineState?
  var detectorTotalsPipeline: MTLComputePipelineState?
  var detectorColumnsValidity: MTLBuffer?

  var usesSpatialIndex: Bool {
    ProcessInfo.processInfo.environment["QGPU_K3_SPATIAL_INDEX"] != "0"
      && !chunks.isEmpty && chunks.allSatisfy { $0.spatial.count == 3 }
  }

  func indexedDetector(mask: [UInt8], output: MTLBuffer) throws -> MetalRuntimeANSDetectorMetrics {
    if spatialQuery == nil {
      spatialQuery = try RuntimeSpatialQuery(device: device, shape: Array(shape[2...]), validity: validPixels)
    }
    return try spatialQuery!.update(mask, source: self, output: output)
  }

  public var residentBytes: UInt64 {
    guard !isReleased else { return 0 }
    let buffers = [decodingTable, failure, diffraction, detectorColumnsValidity]
    let queryBytes = buffers.reduce(UInt64(0)) { $0 + UInt64($1?.length ?? 0) }
    return queryBytes + chunks.reduce(UInt64(0)) { $0 + $1.bytes }
  }

  /// Binary detector-validity mask used by derived products; raw DPs stay untouched.
  public var detectorValidityMask: [UInt8] { validPixels }

  /// Convert one original indexed HDF5 acquisition directly to runtime ANS.
  public static func load(
    source: Native4DSTEMIndexedSource,
    device: MTLDevice,
    includeSpatialIndex: Bool = false,
    maximumAdditionalBytes: UInt64? = nil,
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> MetalRuntimeANSResidentSource {
    guard source.sourceBytesPerValue == 1 || source.sourceBytesPerValue == 2,
      let identity = source.dataset.sourceIdentitySHA256
    else {
      throw invalid(
        "Runtime ANS requires exact indexed uint8 or uint16 counts with a source identity")
    }
    let started = CFAbsoluteTimeGetCurrent()
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    let packing = try OriginalHDF5Packing(device: device)
    let encoder = try RuntimeANSEncoder(
      device: device, source: source, allocatedBefore: allocatedBefore,
      maximumAdditionalBytes: maximumAdditionalBytes)
    let maximumFrames = try RuntimeANSEncoder.maximumWindowFrames(
      source: source, maximumAdditionalBytes: maximumAdditionalBytes)
    let spatial = try includeSpatialIndex ? RuntimeSpatialIndex(device: device,
      shape: [source.dataset.detectorRows, source.dataset.detectorCols],
      validity: (0..<source.dataset.detectorRows * source.dataset.detectorCols).map { source.dataset.badPixelIndices.contains($0) ? 0 : 1 }) : nil
    try packing.forEachExactDecodedWindow(
      source: source, maximumFrames: includeSpatialIndex ? min(512, maximumFrames) : maximumFrames,
      shouldCancel: shouldCancel, progress: progress
    ) { dense, _, range, command in
      try encoder.append(
        dense: dense, firstScan: range.lowerBound, scanCount: range.count,
        afterDecode: command)
      if let spatial {
        try encoder.addSpatialIndex(spatial.build(raw: dense, scans: range.count, itemBytes: source.sourceBytesPerValue))
      }
    }
    guard encoder.readyScans == source.logicalFrameCount else {
      throw invalid("Runtime ANS did not retain the complete scan")
    }
    let built = try encoder.finish()
    return try MetalRuntimeANSResidentSource(
      source: source, identity: identity, built: built,
      totalSeconds: CFAbsoluteTimeGetCurrent() - started, device: device)
  }

  private convenience init(
    source: Native4DSTEMIndexedSource, identity: String,
    built: RuntimeANSEncoder.Output, totalSeconds: Double, device: MTLDevice
  ) throws {
    try self.init(dataset: source.dataset, identity: identity, built: built,
                  totalSeconds: totalSeconds, device: device)
  }

  init(dataset: Native4DSTEMDataset, identity: String,
       built: RuntimeANSEncoder.Output, totalSeconds: Double, device: MTLDevice) throws {
    self.dataset = dataset
    shape = [dataset.scanRows, dataset.scanCols, dataset.detectorRows, dataset.detectorCols]
    logicalDtype = dataset.sourceDtype == "uint8" ? .uint8 : .uint16
    sourceIdentitySHA256 = identity
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      throw Self.invalid("Metal could not create a runtime ANS query queue")
    }
    self.queue = queue
    decodePipeline = built.decodePipeline
    detectorDeltaPipeline = built.detectorDeltaPipeline
    detectorPacketPipeline = built.detectorPacketPipeline
    detectorPacketSIMDs = built.detectorPacketSIMDs
    decodingTable = built.decoding
    chunks = built.chunks
    failure = try Self.sharedBuffer(device: device, bytes: 4, label: "runtime ANS failure")
    let pixels = dataset.detectorRows * dataset.detectorCols
    diffraction = try Self.sharedBuffer(
      device: device, bytes: pixels * 4,
      label: "runtime ANS diffraction")
    var valid = [UInt8](repeating: 1, count: pixels)
    for pixel in dataset.badPixelIndices where valid.indices.contains(pixel) {
      valid[pixel] = 0
    }
    validPixels = valid
    let retained =
      UInt64(built.decoding.length + 4 + pixels * 4)
      + chunks.reduce(0) { $0 + $1.bytes }
    loadMetrics = MetalRuntimeANSLoadMetrics(
      totalSeconds: totalSeconds,
      fusedDecodeAndEncodeSeconds: built.fusedDecodeAndEncodeSeconds,
      prefixSeconds: built.prefixSeconds,
      compactSeconds: built.compactSeconds,
      residentBytes: retained,
      logicalBytes: UInt64(shape.reduce(1, *)) * (logicalDtype == .uint8 ? 1 : 2))
  }

  /// Return one original diffraction pattern exactly, widened only for the API.
  public func extractRawDiffraction(scanRow: Int, scanColumn: Int) throws -> [UInt32] {
    try requireLive()
    guard (0..<shape[0]).contains(scanRow), (0..<shape[1]).contains(scanColumn),
      let diffraction, let failure
    else { throw Self.invalid("Choose an in-bounds scan row and column") }
    let scan = scanRow * shape[1] + scanColumn
    memset(failure.contents(), 0, 4)
    guard let command = queue.makeCommandBuffer()
    else { throw Self.invalid("Metal could not encode a runtime ANS diffraction query") }
    try encodeDiffraction(scan: scan, output: diffraction, command: command)
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed,
      failure.contents().load(as: UInt32.self) == 0
    else {
      throw Self.invalid(
        "Runtime ANS diffraction failed exact validation: "
          + (command.error?.localizedDescription ?? "invalid stream"))
    }
    return diffractionValues(from: diffraction)
  }

  var stableBuffers: [MTLBuffer] {
    guard let decodingTable, let failure else { return [] }
    return [decodingTable, failure]
      + chunks.flatMap { [$0.payload, $0.offsets, $0.models] + $0.spatial }
  }

  func encodeDiffraction(
    scan: Int, output: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Self.invalid("Metal could not encode a runtime ANS diffraction query")
    }
    try encodeDiffraction(scan: scan, output: output, encoder: encoder)
    encoder.endEncoding()
  }

  func encodeDiffraction(
    scan: Int, output: MTLBuffer, encoder: MTLComputeCommandEncoder
  ) throws {
    try requireLive()
    guard
      let chunk = chunks.first(where: {
        $0.firstScan <= scan && scan < $0.firstScan + $0.scanCount
      }), let failure, let decodingTable
    else {
      throw Self.invalid("The selected scan is missing from runtime ANS")
    }
    let local = scan - chunk.firstScan
    let pixels = shape[2] * shape[3]
    let firstStream = local / interval * pixels
    let stopStream = ((local + 1 + interval - 1) / interval) * pixels
    var parameters: [UInt64] = [
      UInt64(chunk.scanCount), UInt64(pixels), UInt64(interval), UInt64(local), 1,
      UInt64(firstStream), UInt64(stopStream), logicalDtype == .uint8 ? 1 : 2,
    ]
    encoder.setComputePipelineState(decodePipeline)
    for (index, buffer) in [chunk.payload, chunk.offsets, chunk.models, decodingTable, failure]
      .enumerated()
    {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBuffer(output, offset: 0, index: 5)
    encoder.setBytes(&parameters, length: parameters.count * 8, index: 6)
    encoder.dispatchThreads(
      MTLSize(width: stopStream - firstStream, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  }

  func diffractionValues(from buffer: MTLBuffer) -> [UInt32] {
    let pixels = shape[2] * shape[3]
    return Array(
      UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: pixels)
    )
  }

  func encodeDetectorDelta(
    selected: MTLBuffer, coefficients: MTLBuffer, changed: Int,
    output: MTLBuffer, encoder: MTLComputeCommandEncoder
  ) throws {
    try requireLive()
    guard changed > 0, let failure, let decodingTable else { return }
    let pixels = shape[2] * shape[3]
    let cameraColumns = usesSpatialIndex && chunks.allSatisfy { $0.scanCount <= interval }
      && ProcessInfo.processInfo.environment["QGPU_K3_PACKET"] != "1"
    let usePacketOwner = changed > 32 && !cameraColumns
    let width = changed <= 32 ? 32 : 128
    encoder.setComputePipelineState(usePacketOwner ? detectorPacketPipeline : detectorDeltaPipeline)
    for chunk in chunks {
      var parameters: [UInt64] = [
        UInt64(chunk.scanCount), UInt64(pixels), UInt64(interval),
        UInt64(changed), UInt64(chunk.firstScan), UInt64(width),
      ]
      for (index, buffer) in [
        chunk.payload, chunk.offsets, chunk.models, decodingTable, failure,
        selected, coefficients, output,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(&parameters, length: parameters.count * 8, index: 8)
      let blocks = (chunk.scanCount + interval - 1) / interval
      let groups =
        usePacketOwner
        ? (blocks + detectorPacketSIMDs - 1) / detectorPacketSIMDs
        : (changed + width - 1) / width
      encoder.dispatchThreadgroups(
        MTLSize(width: usePacketOwner ? groups : blocks * groups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
          width: usePacketOwner ? 32 * detectorPacketSIMDs : width,
          height: 1, depth: 1))
    }
  }

  func checkFailure(_ command: MTLCommandBuffer) throws {
    guard let failure, command.status == .completed,
      failure.contents().load(as: UInt32.self) == 0
    else {
      throw Self.invalid(
        "Runtime ANS diffraction failed exact validation: "
          + (command.error?.localizedDescription ?? "invalid stream"))
    }
  }

  public func releaseResidentStorage() {
    chunks.removeAll(keepingCapacity: false)
    spatialQuery = nil
    decodingTable = nil
    failure = nil
    diffraction = nil
    detectorColumnsValidity = nil
    detectorColumnsPipeline = nil
    detectorTotalsPipeline = nil
    isReleased = true
  }

  func requireLive() throws {
    guard !isReleased, decodingTable != nil, failure != nil, diffraction != nil else {
      throw Self.invalid("The runtime ANS source was released; load it again")
    }
  }

  static func sharedBuffer(device: MTLDevice, bytes: Int, label: String) throws -> MTLBuffer {
    guard bytes > 0, bytes <= device.maxBufferLength,
      let result = device.makeBuffer(length: bytes, options: .storageModeShared)
    else { throw invalid("Metal could not allocate \(bytes) bytes for \(label)") }
    result.label = label
    return result
  }

  static func privateBuffer(device: MTLDevice, bytes: Int, label: String) throws -> MTLBuffer {
    guard bytes > 0, bytes <= device.maxBufferLength,
      let result = device.makeBuffer(length: bytes, options: .storageModePrivate)
    else { throw invalid("Metal could not allocate \(bytes) bytes for \(label)") }
    result.label = label
    return result
  }

  static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }
}

/// Explicit residency and ordered diffraction queries for compatible ANS acquisitions.
@available(macOS 15.0, iOS 18.0, *)
public final class MetalRuntimeANSSeries: @unchecked Sendable {
  public let sources: [MetalRuntimeANSResidentSource]
  public private(set) var isReleased = false
  public var residentBytes: UInt64 { isReleased ? 0 : residency.allocatedSize }

  private let queue: MTLCommandQueue
  private let residency: MTLResidencySet
  private var outputs: [MTLBuffer]
  private var virtualDetectorOutputs: [MTLBuffer]
  private var detectorSelected: [MTLBuffer]
  private var detectorCoefficients: [MTLBuffer]
  private var detectorMasks: [[UInt8]]

  /// Exact interactive product/control storage, separate from encoded counts.
  public var productBytes: UInt64 {
    let outputBytes = virtualDetectorOutputs.reduce(0) { $0 + $1.length }
    let indexBytes = detectorSelected.reduce(0) { $0 + $1.length }
    let coefficientBytes = detectorCoefficients.reduce(0) { $0 + $1.length }
    return UInt64(outputBytes + indexBytes + coefficientBytes)
  }

  public init(sources: [MetalRuntimeANSResidentSource]) throws {
    guard let first = sources.first, !first.isReleased,
      sources.allSatisfy({
        !$0.isReleased && $0.shape == first.shape && $0.logicalDtype == first.logicalDtype
          && $0.device.registryID == first.device.registryID
      }), let queue = first.device.makeCommandQueue()
    else {
      throw MetalRuntimeANSResidentSource.invalid(
        "A runtime ANS series needs compatible live acquisitions on one Metal device")
    }
    self.sources = sources
    self.queue = queue
    let descriptor = MTLResidencySetDescriptor()
    descriptor.label = "Runtime ANS acquisition series"
    var identities = Set<ObjectIdentifier>()
    let buffers = sources.flatMap(\.stableBuffers).filter {
      identities.insert(ObjectIdentifier($0 as AnyObject)).inserted
    }
    descriptor.initialCapacity = buffers.count
    residency = try first.device.makeResidencySet(descriptor: descriptor)
    for buffer in buffers { residency.addAllocation(buffer) }
    residency.commit()
    guard residency.allocationCount == buffers.count else {
      residency.removeAllAllocations()
      residency.commit()
      throw MetalRuntimeANSResidentSource.invalid(
        "Metal residency did not retain every runtime ANS allocation")
    }
    residency.requestResidency()
    let bytes = first.shape[2] * first.shape[3] * 4
    outputs = try sources.indices.map { index in
      try MetalRuntimeANSResidentSource.sharedBuffer(
        device: first.device, bytes: bytes, label: "runtime ANS series DP \(index)")
    }
    let pixels = first.shape[2] * first.shape[3]
    let scans = first.shape[0] * first.shape[1]
    virtualDetectorOutputs = try sources.indices.map { index in
      try MetalRuntimeANSResidentSource.sharedBuffer(
        device: first.device, bytes: scans * 4,
        label: "runtime ANS virtual detector \(index)")
    }
    detectorSelected = try sources.indices.map { index in
      try MetalRuntimeANSResidentSource.sharedBuffer(
        device: first.device, bytes: pixels * 4,
        label: "runtime ANS detector indices \(index)")
    }
    detectorCoefficients = try sources.indices.map { index in
      try MetalRuntimeANSResidentSource.sharedBuffer(
        device: first.device, bytes: pixels * 4,
        label: "runtime ANS detector coefficients \(index)")
    }
    detectorMasks = sources.map { _ in [UInt8](repeating: 0, count: pixels) }
  }

  /// Complete the selected acquisition first and return its exact DP.
  public func extractPriorityRawDiffraction(
    scanRow: Int, scanColumn: Int, priorityIndex: Int
  ) throws -> [UInt32] {
    let output = try updatePriorityDiffractionBuffer(
      scanRow: scanRow, scanColumn: scanColumn, priorityIndex: priorityIndex)
    return sources[priorityIndex].diffractionValues(from: output)
  }

  /// Update and return the series-owned Metal buffer for the selected acquisition.
  /// The caller must not release or mutate this buffer.
  public func updatePriorityDiffractionBuffer(
    scanRow: Int, scanColumn: Int, priorityIndex: Int
  ) throws -> MTLBuffer {
    try requireLive()
    guard sources.indices.contains(priorityIndex),
      (0..<sources[0].shape[0]).contains(scanRow),
      (0..<sources[0].shape[1]).contains(scanColumn),
      let failure = sources[priorityIndex].failure,
      let command = queue.makeCommandBuffer()
    else { throw MetalRuntimeANSResidentSource.invalid("Choose an in-bounds series DP") }
    memset(failure.contents(), 0, 4)
    command.useResidencySet(residency)
    let scan = scanRow * sources[0].shape[1] + scanColumn
    try sources[priorityIndex].encodeDiffraction(
      scan: scan, output: outputs[priorityIndex], command: command)
    command.commit()
    command.waitUntilCompleted()
    try sources[priorityIndex].checkFailure(command)
    return outputs[priorityIndex]
  }

  /// Decode one exact DP from every acquisition in one residency-qualified command.
  public func extractAllRawDiffraction(scanRow: Int, scanColumn: Int) throws -> [[UInt32]] {
    let buffers = try updateDiffractionBuffers(scanRow: scanRow, scanColumn: scanColumn)
    return zip(sources, buffers).map { $0.diffractionValues(from: $1) }
  }

  /// Update every series-owned DP buffer in one residency-qualified command.
  /// Returned buffers remain owned by this series and must not be released or mutated.
  public func updateDiffractionBuffers(scanRow: Int, scanColumn: Int) throws -> [MTLBuffer] {
    try requireLive()
    guard (0..<sources[0].shape[0]).contains(scanRow),
      (0..<sources[0].shape[1]).contains(scanColumn),
      let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent)
    else { throw MetalRuntimeANSResidentSource.invalid("Choose an in-bounds series DP") }
    let scan = scanRow * sources[0].shape[1] + scanColumn
    command.useResidencySet(residency)
    for (source, output) in zip(sources, outputs) {
      guard let failure = source.failure else {
        throw MetalRuntimeANSResidentSource.invalid("A runtime ANS source was released")
      }
      memset(failure.contents(), 0, 4)
      try source.encodeDiffraction(scan: scan, output: output, encoder: encoder)
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    for source in sources { try source.checkFailure(command) }
    return outputs
  }

  /// Update and publish the selected acquisition's virtual image first.
  ///
  /// This advances only that acquisition's exact delta seed. Other sources
  /// remain valid and can be caught up by a later all-series update.
  public func updatePriorityVirtualDetectorBuffer(
    mask: [UInt8], priorityIndex: Int, forceRebase: Bool = false
  ) throws -> (buffer: MTLBuffer, metrics: MetalRuntimeANSDetectorMetrics) {
    try requireLive()
    let pixels = sources[0].shape[2] * sources[0].shape[3]
    guard sources.indices.contains(priorityIndex), mask.count == pixels,
      mask.allSatisfy({ $0 == 0 || $0 == 1 })
    else {
      throw MetalRuntimeANSResidentSource.invalid(
        "Choose a live acquisition and a binary mask matching its detector")
    }
    let wallStarted = CFAbsoluteTimeGetCurrent()
    let source = sources[priorityIndex]
    var next = detectorMasks[priorityIndex]
    var deltaFromCurrent = 0
    var selectedFromZero = 0
    for pixel in 0..<pixels {
      let value = mask[pixel] == 1 && source.validPixels[pixel] == 1 ? UInt8(1) : 0
      next[pixel] = value
      deltaFromCurrent += value == detectorMasks[priorityIndex][pixel] ? 0 : 1
      selectedFromZero += value == 1 ? 1 : 0
    }
    if source.usesSpatialIndex && (forceRebase || min(selectedFromZero, deltaFromCurrent) > 4096) {
      let metrics = try source.indexedDetector(mask: next, output: virtualDetectorOutputs[priorityIndex])
      detectorMasks[priorityIndex] = next
      return (virtualDetectorOutputs[priorityIndex], metrics)
    }
    let rebaseFromZero = forceRebase || selectedFromZero < deltaFromCurrent
    let previous =
      rebaseFromZero
      ? [UInt8](repeating: 0, count: pixels) : detectorMasks[priorityIndex]
    let selected = detectorSelected[priorityIndex].contents().bindMemory(
      to: UInt32.self, capacity: pixels)
    let coefficients = detectorCoefficients[priorityIndex].contents().bindMemory(
      to: Int32.self, capacity: pixels)
    var changed = 0
    for pixel in 0..<pixels {
      let value = next[pixel]
      if value != previous[pixel] {
        selected[changed] = UInt32(pixel)
        coefficients[changed] = value == 1 ? 1 : -1
        changed += 1
      }
    }
    if !rebaseFromZero && changed == 0 {
      return (
        virtualDetectorOutputs[priorityIndex],
        MetalRuntimeANSDetectorMetrics(
          changedDetectorPixels: 0, gpuMilliseconds: 0,
          wallMilliseconds: (CFAbsoluteTimeGetCurrent() - wallStarted) * 1000,
          acquisitionCount: 1, submissionCount: 0)
      )
    }
    guard let failure = source.failure, let command = queue.makeCommandBuffer() else {
      throw MetalRuntimeANSResidentSource.invalid(
        "Metal could not encode a priority detector update")
    }
    memset(failure.contents(), 0, 4)
    command.useResidencySet(residency)
    if rebaseFromZero {
      guard let blit = command.makeBlitCommandEncoder() else {
        throw MetalRuntimeANSResidentSource.invalid("Metal could not reset the priority product")
      }
      let output = virtualDetectorOutputs[priorityIndex]
      blit.fill(buffer: output, range: 0..<output.length, value: 0)
      blit.endEncoding()
    }
    guard let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
      throw MetalRuntimeANSResidentSource.invalid(
        "Metal could not encode priority detector kernels")
    }
    try source.encodeDetectorDelta(
      selected: detectorSelected[priorityIndex],
      coefficients: detectorCoefficients[priorityIndex], changed: changed,
      output: virtualDetectorOutputs[priorityIndex], encoder: encoder)
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try source.checkFailure(command)
    detectorMasks[priorityIndex] = next
    let gpuMilliseconds =
      command.gpuEndTime > command.gpuStartTime
      ? (command.gpuEndTime - command.gpuStartTime) * 1000 : 0
    return (
      virtualDetectorOutputs[priorityIndex],
      MetalRuntimeANSDetectorMetrics(
        changedDetectorPixels: changed, gpuMilliseconds: gpuMilliseconds,
        wallMilliseconds: (CFAbsoluteTimeGetCurrent() - wallStarted) * 1000,
        acquisitionCount: 1, submissionCount: 1)
    )
  }

  /// Update one binary virtual-detector image for every acquisition exactly.
  ///
  /// The returned buffers remain owned by this series. Known invalid detector
  /// pixels are excluded independently for each acquisition. A failed command
  /// leaves the preceding mask state intact so the next update can safely retry.
  public func updateVirtualDetectorBuffers(
    mask: [UInt8], forceRebase: Bool = false
  ) throws -> (buffers: [MTLBuffer], metrics: MetalRuntimeANSDetectorMetrics) {
    try requireLive()
    let pixels = sources[0].shape[2] * sources[0].shape[3]
    guard mask.count == pixels, mask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
      throw MetalRuntimeANSResidentSource.invalid(
        "A runtime ANS detector mask must match the detector and contain only zero or one")
    }
    let wallStarted = CFAbsoluteTimeGetCurrent()
    if sources.allSatisfy({ $0.usesSpatialIndex }) {
      var milliseconds = 0.0, changed = 0, submissions = 0
      for index in sources.indices {
        let (_, metrics) = try updatePriorityVirtualDetectorBuffer(mask: mask, priorityIndex: index, forceRebase: forceRebase)
        milliseconds += metrics.gpuMilliseconds; changed = max(changed, metrics.changedDetectorPixels)
        submissions += metrics.submissionCount
      }
      return (virtualDetectorOutputs, MetalRuntimeANSDetectorMetrics(changedDetectorPixels: changed,
        gpuMilliseconds: milliseconds, wallMilliseconds: (CFAbsoluteTimeGetCurrent() - wallStarted) * 1000,
        acquisitionCount: sources.count, submissionCount: submissions))
    }
    var nextMasks = detectorMasks
    var changedCounts = [Int](repeating: 0, count: sources.count)
    var rebaseFromZero = [Bool](repeating: forceRebase, count: sources.count)
    for index in sources.indices {
      var deltaFromCurrent = 0
      var selectedFromZero = 0
      for pixel in 0..<pixels {
        let next = mask[pixel] == 1 && sources[index].validPixels[pixel] == 1 ? UInt8(1) : 0
        nextMasks[index][pixel] = next
        deltaFromCurrent += next == detectorMasks[index][pixel] ? 0 : 1
        selectedFromZero += next == 1 ? 1 : 0
      }
      rebaseFromZero[index] = forceRebase || selectedFromZero < deltaFromCurrent
      let previous =
        rebaseFromZero[index]
        ? [UInt8](repeating: 0, count: pixels) : detectorMasks[index]
      let selected = detectorSelected[index].contents().bindMemory(
        to: UInt32.self, capacity: pixels)
      let coefficients = detectorCoefficients[index].contents().bindMemory(
        to: Int32.self, capacity: pixels)
      var changed = 0
      for pixel in 0..<pixels {
        let next = nextMasks[index][pixel]
        if next != previous[pixel] {
          selected[changed] = UInt32(pixel)
          coefficients[changed] = next == 1 ? 1 : -1
          changed += 1
        }
      }
      changedCounts[index] = changed
    }
    if !rebaseFromZero.contains(true) && changedCounts.allSatisfy({ $0 == 0 }) {
      return (
        virtualDetectorOutputs,
        MetalRuntimeANSDetectorMetrics(
          changedDetectorPixels: 0, gpuMilliseconds: 0,
          wallMilliseconds: (CFAbsoluteTimeGetCurrent() - wallStarted) * 1000,
          acquisitionCount: sources.count, submissionCount: 0)
      )
    }
    guard let command = queue.makeCommandBuffer() else {
      throw MetalRuntimeANSResidentSource.invalid("Metal could not encode a detector update")
    }
    command.useResidencySet(residency)
    for source in sources {
      guard let failure = source.failure else {
        throw MetalRuntimeANSResidentSource.invalid("A runtime ANS source was released")
      }
      memset(failure.contents(), 0, 4)
    }
    if rebaseFromZero.contains(true) {
      guard let blit = command.makeBlitCommandEncoder() else {
        throw MetalRuntimeANSResidentSource.invalid("Metal could not reset detector products")
      }
      for index in virtualDetectorOutputs.indices where rebaseFromZero[index] {
        let output = virtualDetectorOutputs[index]
        blit.fill(buffer: output, range: 0..<output.length, value: 0)
      }
      blit.endEncoding()
    }
    guard let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
      throw MetalRuntimeANSResidentSource.invalid("Metal could not encode detector kernels")
    }
    for index in sources.indices where changedCounts[index] > 0 {
      try sources[index].encodeDetectorDelta(
        selected: detectorSelected[index], coefficients: detectorCoefficients[index],
        changed: changedCounts[index], output: virtualDetectorOutputs[index], encoder: encoder)
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    for source in sources { try source.checkFailure(command) }
    detectorMasks = nextMasks
    let gpuMilliseconds =
      command.gpuEndTime > command.gpuStartTime
      ? (command.gpuEndTime - command.gpuStartTime) * 1000 : 0
    return (
      virtualDetectorOutputs,
      MetalRuntimeANSDetectorMetrics(
        changedDetectorPixels: changedCounts.max() ?? 0,
        gpuMilliseconds: gpuMilliseconds,
        wallMilliseconds: (CFAbsoluteTimeGetCurrent() - wallStarted) * 1000,
        acquisitionCount: sources.count, submissionCount: 1)
    )
  }

  public func release() {
    guard !isReleased else { return }
    residency.endResidency()
    residency.removeAllAllocations()
    residency.commit()
    outputs.removeAll(keepingCapacity: false)
    virtualDetectorOutputs.removeAll(keepingCapacity: false)
    detectorSelected.removeAll(keepingCapacity: false)
    detectorCoefficients.removeAll(keepingCapacity: false)
    detectorMasks.removeAll(keepingCapacity: false)
    isReleased = true
  }

  private func requireLive() throws {
    guard !isReleased, !sources.isEmpty, sources.allSatisfy({ !$0.isReleased }) else {
      throw MetalRuntimeANSResidentSource.invalid(
        "The runtime ANS series was released; load it again")
    }
  }
}

final class RuntimeANSEncoder {
  struct Output {
    let chunks: [MetalRuntimeANSResidentSource.Chunk]
    let decoding: MTLBuffer
    let fusedDecodeAndEncodeSeconds: Double
    let prefixSeconds: Double
    let compactSeconds: Double
    let decodePipeline: MTLComputePipelineState
    let detectorDeltaPipeline: MTLComputePipelineState
    let detectorPacketPipeline: MTLComputePipelineState
    let detectorPacketSIMDs: Int
  }

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let encodePipeline: MTLComputePipelineState
  private let compactPipeline: MTLComputePipelineState
  private let decodePipeline: MTLComputePipelineState
  private let detectorDeltaPipeline: MTLComputePipelineState
  private let detectorPacketPipeline: MTLComputePipelineState
  private let detectorPacketSIMDs: Int
  private let encoding: MTLBuffer
  private let decoding: MTLBuffer
  private let pixels: Int
  private let bytesPerValue: Int
  private let allocatedBefore: UInt64
  private let maximumAdditionalBytes: UInt64?
  private var chunks: [MetalRuntimeANSResidentSource.Chunk] = []
  private(set) var readyScans = 0
  private var fusedDecodeAndEncodeSeconds = 0.0
  private var prefixSeconds = 0.0
  private var compactSeconds = 0.0
  private var reusableScratch: MTLBuffer?
  private var reusableSizes: MTLBuffer?
  private var reusableStates: MTLBuffer?
  private var reusableModels: MTLBuffer?
  private var reusableOffsets: MTLBuffer?
  private let interval = 512

  convenience init(
    device: MTLDevice, source: Native4DSTEMIndexedSource,
    allocatedBefore: UInt64,
    maximumAdditionalBytes: UInt64?
  ) throws {
    try self.init(device: device, pixels: source.dataset.detectorRows * source.dataset.detectorCols,
                  bytesPerValue: source.sourceBytesPerValue, allocatedBefore: allocatedBefore,
                  maximumAdditionalBytes: maximumAdditionalBytes)
  }

  init(device: MTLDevice, pixels: Int, bytesPerValue: Int, allocatedBefore: UInt64,
       maximumAdditionalBytes: UInt64?, singleFrameQueries: Bool = false) throws {
    self.device = device
    self.allocatedBefore = allocatedBefore
    self.maximumAdditionalBytes = maximumAdditionalBytes
    self.pixels = pixels
    self.bytesPerValue = bytesPerValue
    guard let queue = device.makeCommandQueue() else {
      throw MetalRuntimeANSResidentSource.invalid("Metal could not create a runtime ANS encoder")
    }
    self.queue = queue
    let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
    let requestedPacketSIMDs =
      ProcessInfo.processInfo.environment[
        "QGPU_RUNTIME_ANS_PACKET_OWNER_SIMDS"] ?? "4"
    guard requestedPacketSIMDs == "1" || requestedPacketSIMDs == "4" else {
      throw MetalRuntimeANSResidentSource.invalid(
        "QGPU_RUNTIME_ANS_PACKET_OWNER_SIMDS must be 1 or 4")
    }
    detectorPacketSIMDs = requestedPacketSIMDs == "4" ? 4 : 1
    let detectorPacketName =
      detectorPacketSIMDs == 4
      ? "streamed_counts_detector_packet4" : "streamed_counts_detector_packet"
    // Match camera snapshot queries: stop decoding once the selected frame is reached.
    let decodeName = singleFrameQueries
      && ProcessInfo.processInfo.environment["QGPU_K3_FRAME_PREFIX"] != "0"
      ? "camera_frame" : "streamed_counts_decode_range"
    guard let encode = library.makeFunction(name: "streamed_counts_encode"),
      let compact = library.makeFunction(name: "streamed_counts_compact"),
      let decode = library.makeFunction(name: decodeName),
      let detectorDelta = library.makeFunction(name: "streamed_counts_detector_delta"),
      let detectorPacket = library.makeFunction(name: detectorPacketName)
    else { throw MetalRuntimeANSResidentSource.invalid("Runtime ANS encoder kernels are missing") }
    encodePipeline = try device.makeComputePipelineState(function: encode)
    compactPipeline = try device.makeComputePipelineState(function: compact)
    decodePipeline = try device.makeComputePipelineState(function: decode)
    detectorDeltaPipeline = try device.makeComputePipelineState(function: detectorDelta)
    detectorPacketPipeline = try device.makeComputePipelineState(function: detectorPacket)
    let tables = Self.tables()
    encoding = try Self.upload(tables.encoding, device: device, label: "runtime ANS encoding")
    decoding = try Self.upload(tables.decoding, device: device, label: "runtime ANS decoding")
  }

  func addSpatialIndex(_ buffers: [MTLBuffer]) throws {
    guard !chunks.isEmpty else { throw MetalRuntimeANSResidentSource.invalid("Encode counts before indexing.") }
    chunks[chunks.count - 1].spatial = buffers
  }

  func append(
    dense: MTLBuffer, firstScan: Int, scanCount: Int,
    afterDecode suppliedCommand: MTLCommandBuffer? = nil
  ) throws {
    let streamCount = ((scanCount + interval - 1) / interval) * pixels
    let scratchBytes = (2 * min(scanCount, interval) + 4) * streamCount
    let worstPayloadBytes = scanCount * pixels * bytesPerValue + streamCount * 4
    let allocatedNow = UInt64(device.currentAllocatedSize)
    let alreadyAllocated = allocatedNow > allocatedBefore ? allocatedNow - allocatedBefore : 0
    func allocationNeeded(_ buffer: MTLBuffer?, _ bytes: Int) -> UInt64 {
      buffer?.length ?? 0 >= bytes ? 0 : UInt64(bytes)
    }
    let newStagingBytes =
      allocationNeeded(reusableScratch, scratchBytes)
      + allocationNeeded(reusableSizes, streamCount * 4)
      + allocationNeeded(reusableStates, streamCount * 4)
      + allocationNeeded(reusableModels, streamCount)
      + allocationNeeded(reusableOffsets, (streamCount + 1) * 4)
    if let maximumAdditionalBytes,
      alreadyAllocated + newStagingBytes + UInt64(worstPayloadBytes) > maximumAdditionalBytes
    {
      let required = alreadyAllocated + newStagingBytes + UInt64(worstPayloadBytes)
      throw MetalRuntimeANSResidentSource.invalid(
        "Runtime ANS encoding needs \(required) bytes at this window, but the load budget is \(maximumAdditionalBytes) bytes"
      )
    }
    let reuseStaging =
      ProcessInfo.processInfo.environment[
        "QGPU_RUNTIME_ANS_REUSE_STAGING"] != "0"
    let scratch =
      try reuseStaging
      ? reusablePrivateBuffer(
        &reusableScratch, bytes: scratchBytes, label: "runtime ANS scratch")
      : MetalRuntimeANSResidentSource.privateBuffer(
        device: device, bytes: scratchBytes, label: "runtime ANS scratch")
    let sizes =
      try reuseStaging
      ? reusableSharedBuffer(
        &reusableSizes, bytes: streamCount * 4, label: "runtime ANS sizes")
      : MetalRuntimeANSResidentSource.sharedBuffer(
        device: device, bytes: streamCount * 4, label: "runtime ANS sizes")
    let states =
      try reuseStaging
      ? reusablePrivateBuffer(
        &reusableStates, bytes: streamCount * 4, label: "runtime ANS states")
      : MetalRuntimeANSResidentSource.privateBuffer(
        device: device, bytes: streamCount * 4, label: "runtime ANS states")
    let models =
      try reuseStaging
      ? reusablePrivateBuffer(
        &reusableModels, bytes: streamCount, label: "runtime ANS models")
      : MetalRuntimeANSResidentSource.privateBuffer(
        device: device, bytes: streamCount, label: "runtime ANS models")
    var parameters: [UInt64] = [
      UInt64(scanCount), UInt64(pixels), UInt64(interval), UInt64(streamCount),
      UInt64(bytesPerValue),
    ]
    guard let command = suppliedCommand ?? queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw MetalRuntimeANSResidentSource.invalid("Cannot encode runtime ANS sizes") }
    encoder.setComputePipelineState(encodePipeline)
    for (index, buffer) in [dense, encoding, scratch, sizes, states, models].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&parameters, length: parameters.count * 8, index: 6)
    encoder.dispatchThreads(
      MTLSize(width: streamCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    let fusedStarted = CFAbsoluteTimeGetCurrent()
    try Self.finish(command, message: "Runtime ANS size encoding failed")
    fusedDecodeAndEncodeSeconds += CFAbsoluteTimeGetCurrent() - fusedStarted

    let prefixStarted = CFAbsoluteTimeGetCurrent()
    let offsets =
      try reuseStaging
      ? reusableSharedBuffer(
        &reusableOffsets, bytes: (streamCount + 1) * 4, label: "runtime ANS offsets")
      : MetalRuntimeANSResidentSource.sharedBuffer(
        device: device, bytes: (streamCount + 1) * 4, label: "runtime ANS offsets")
    let sizeWords = sizes.contents().bindMemory(to: UInt32.self, capacity: streamCount)
    let offsetWords = offsets.contents().bindMemory(to: UInt32.self, capacity: streamCount + 1)
    offsetWords[0] = 0
    var total: UInt64 = 0
    for stream in 0..<streamCount {
      total += UInt64(sizeWords[stream])
      guard total <= UInt64(UInt32.max) else {
        throw MetalRuntimeANSResidentSource.invalid("One runtime ANS chunk exceeds 4 GiB")
      }
      offsetWords[stream + 1] = UInt32(total)
    }
    prefixSeconds += CFAbsoluteTimeGetCurrent() - prefixStarted
    let allocatedBeforePayload = UInt64(device.currentAllocatedSize)
    let activeBeforePayload =
      allocatedBeforePayload > allocatedBefore
      ? allocatedBeforePayload - allocatedBefore : 0
    let finalMetadataBytes = (streamCount + 1) * 4 + streamCount
    if let maximumAdditionalBytes,
      activeBeforePayload + total + UInt64(finalMetadataBytes) > maximumAdditionalBytes
    {
      throw MetalRuntimeANSResidentSource.invalid(
        "Runtime ANS payload needs \(activeBeforePayload + total + UInt64(finalMetadataBytes)) bytes at this window, but the load budget is \(maximumAdditionalBytes) bytes"
      )
    }
    let payload = try MetalRuntimeANSResidentSource.privateBuffer(
      device: device, bytes: max(1, Int(total)), label: "runtime ANS payload")
    let residentOffsets = try MetalRuntimeANSResidentSource.privateBuffer(
      device: device, bytes: (streamCount + 1) * 4, label: "runtime ANS offsets")
    let residentModels = try MetalRuntimeANSResidentSource.privateBuffer(
      device: device, bytes: streamCount, label: "runtime ANS models")
    guard let compactCommand = queue.makeCommandBuffer(),
      let compact = compactCommand.makeComputeCommandEncoder()
    else { throw MetalRuntimeANSResidentSource.invalid("Cannot compact runtime ANS payload") }
    compact.setComputePipelineState(compactPipeline)
    for (index, buffer) in [dense, scratch, offsets, states, models, payload].enumerated() {
      compact.setBuffer(buffer, offset: 0, index: index)
    }
    compact.setBytes(&parameters, length: parameters.count * 8, index: 6)
    compact.dispatchThreads(
      MTLSize(width: streamCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    compact.endEncoding()
    guard let blit = compactCommand.makeBlitCommandEncoder() else {
      throw MetalRuntimeANSResidentSource.invalid("Cannot retain runtime ANS metadata")
    }
    blit.copy(
      from: offsets, sourceOffset: 0, to: residentOffsets, destinationOffset: 0,
      size: (streamCount + 1) * 4)
    blit.copy(
      from: models, sourceOffset: 0, to: residentModels, destinationOffset: 0,
      size: streamCount)
    blit.endEncoding()
    let compactStarted = CFAbsoluteTimeGetCurrent()
    try Self.finish(compactCommand, message: "Runtime ANS compaction failed")
    compactSeconds += CFAbsoluteTimeGetCurrent() - compactStarted
    chunks.append(
      MetalRuntimeANSResidentSource.Chunk(
        firstScan: firstScan, scanCount: scanCount, payload: payload,
        offsets: residentOffsets, models: residentModels))
    readyScans += scanCount
  }

  func finish() throws -> Output {
    Output(
      chunks: chunks, decoding: decoding,
      fusedDecodeAndEncodeSeconds: fusedDecodeAndEncodeSeconds,
      prefixSeconds: prefixSeconds, compactSeconds: compactSeconds,
      decodePipeline: decodePipeline,
      detectorDeltaPipeline: detectorDeltaPipeline,
      detectorPacketPipeline: detectorPacketPipeline,
      detectorPacketSIMDs: detectorPacketSIMDs)
  }

  private func reusablePrivateBuffer(
    _ slot: inout MTLBuffer?, bytes: Int, label: String
  ) throws -> MTLBuffer {
    if let slot, slot.length >= bytes { return slot }
    let buffer = try MetalRuntimeANSResidentSource.privateBuffer(
      device: device, bytes: bytes, label: label)
    slot = buffer
    return buffer
  }

  private func reusableSharedBuffer(
    _ slot: inout MTLBuffer?, bytes: Int, label: String
  ) throws -> MTLBuffer {
    if let slot, slot.length >= bytes { return slot }
    let buffer = try MetalRuntimeANSResidentSource.sharedBuffer(
      device: device, bytes: bytes, label: label)
    slot = buffer
    return buffer
  }

  private static func upload<T>(_ values: [T], device: MTLDevice, label: String) throws -> MTLBuffer
  {
    try values.withUnsafeBytes { raw in
      guard let base = raw.baseAddress,
        let buffer = device.makeBuffer(
          bytes: base, length: raw.count, options: .storageModeShared)
      else { throw MetalRuntimeANSResidentSource.invalid("Cannot upload \(label)") }
      buffer.label = label
      return buffer
    }
  }

  static func maximumWindowFrames(
    source: Native4DSTEMIndexedSource, maximumAdditionalBytes: UInt64?
  ) throws -> Int {
    var frames = min(32_768, source.logicalFrameCount)
    guard let maximumAdditionalBytes else { return frames }
    while frames > 512,
      estimatedAdmissionBytes(source: source, frames: frames) > maximumAdditionalBytes
    {
      frames = max(512, frames / 2)
    }
    guard estimatedAdmissionBytes(source: source, frames: frames) <= maximumAdditionalBytes else {
      throw MetalRuntimeANSResidentSource.invalid(
        "Runtime ANS needs more transient memory even at its smallest window; release another resident and retry"
      )
    }
    return frames
  }

  private static func estimatedAdmissionBytes(
    source: Native4DSTEMIndexedSource, frames: Int
  ) -> UInt64 {
    let peak = estimatedPeakBytes(source: source, frames: frames)
    // Metal command objects and compressed-input buffers are driver-owned and
    // not fully reflected by the explicit buffer sum. A progressive seven-source
    // load measured nearly 3 GiB beyond the explicit 16K-window sum, so preserve
    // that bounded allowance. The ratio matters for smaller windows, while the
    // cap prevents the allowance growing without bound for large windows.
    let margin = min(UInt64(3) << 30, peak / 3 * 2)
    return peak.addingReportingOverflow(margin).overflow ? UInt64.max : peak + margin
  }

  private static func estimatedPeakBytes(
    source: Native4DSTEMIndexedSource, frames: Int
  ) -> UInt64 {
    let pixels = source.dataset.detectorRows * source.dataset.detectorCols
    let bytesPerValue = source.sourceBytesPerValue
    let denseBytes = frames * pixels * bytesPerValue
    let decodeScratchBytes = bytesPerValue == 2 ? denseBytes : 0
    let decodeAuxiliaryBytes = pixels + frames * (8 + 32) + 4
    let streams = ((frames + 511) / 512) * pixels
    let encodeScratchBytes = (2 * min(frames, 512) + 4) * streams
    let encodeMetadataBytes = streams * (4 + 4 + 1) + (streams + 1) * 4
    let worstPayloadBytes = denseBytes + streams * 4
    let tableBytes = (64 * 33 + 64 * 1024) * 4
    return UInt64(
      denseBytes + decodeScratchBytes + decodeAuxiliaryBytes + encodeScratchBytes
        + encodeMetadataBytes + worstPayloadBytes + tableBytes)
  }

  static func tables() -> (encoding: [UInt32], decoding: [UInt32]) {
    var encoding = [UInt32](repeating: 0, count: 64 * 33)
    var decoding = [UInt32](repeating: 0, count: 64 * 1024)
    let low = log(0.002)
    let span = log(16_000.0)
    for model in 0..<64 {
      let mean = exp(low + Double(model) / 63.0 * span)
      var probability = [Double](repeating: 0, count: 33)
      for value in 0..<33 {
        probability[value] = exp(Double(value) * log(mean) - lgamma(Double(value + 1)) - mean)
      }
      probability[32] = max(0, 1 - probability[..<32].reduce(0, +))
      let sum = probability.reduce(0, +)
      let allocation = probability.map { $0 / sum * Double(1024 - 33) }
      var frequencies = allocation.map { UInt32(floor($0)) + 1 }
      let remainder = 1024 - frequencies.reduce(0) { $0 + Int($1) }
      let order = allocation.indices.sorted {
        let lhs = allocation[$0] - floor(allocation[$0])
        let rhs = allocation[$1] - floor(allocation[$1])
        return lhs == rhs ? $0 < $1 : lhs > rhs
      }
      for index in order.prefix(remainder) { frequencies[index] += 1 }
      var first = 0
      for symbol in 0..<33 {
        let frequency = Int(frequencies[symbol])
        encoding[model * 33 + symbol] = (UInt32(frequency) << 16) | UInt32(first)
        let code = (UInt32(frequency) << 16) | (UInt32(first) << 6) | UInt32(symbol)
        for slot in first..<(first + frequency) { decoding[model * 1024 + slot] = code }
        first += frequency
      }
    }
    return (encoding, decoding)
  }

  private static func finish(_ command: MTLCommandBuffer, message: String) throws {
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw MetalRuntimeANSResidentSource.invalid(
        message + ": " + (command.error?.localizedDescription ?? "unknown Metal error"))
    }
  }
}

extension OriginalHDF5Packing {
  /// Decode every original count in bounded private windows for a direct GPU consumer.
  func forEachExactDecodedWindow(
    source: Native4DSTEMIndexedSource,
    maximumFrames: Int,
    includeDPCMoments: Bool = false,
    shouldCancel: () -> Bool,
    progress: (Int, Int) -> Void,
    consume: (MTLBuffer, MTLBuffer?, Range<Int>, MTLCommandBuffer) throws -> Void
  ) throws {
    let initialInputs = try Self.inputStamps(source)
    let frames = min(maximumFrames, source.logicalFrameCount)
    let windows = try source.windows(
      maximumDecodedBytes: UInt64(frames) * source.decodedBytesPerFrame,
      alignToScanRows: false)
    let pixels = source.dataset.detectorRows * source.dataset.detectorCols
    let denseBytes = frames * pixels * source.sourceBytesPerValue
    let dense = try buffer(denseBytes, privateStorage: true)
    let scratch =
      source.sourceBytesPerValue == 2
      ? try buffer(denseBytes, privateStorage: true) : nil
    let mask = try buffer(pixels)
    let audit = try buffer(frames * 8)
    let errors = try buffer(4)
    let moments = try buffer(frames * 32)
    memset(mask.contents(), 0, mask.length)
    var profile = Profile()
    for (ordinal, window) in windows.enumerated() {
      try autoreleasepool {
        if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
        memset(audit.contents(), 0, audit.length)
        memset(errors.contents(), 0, errors.length)
        let command = try commandBuffer()
        for slice in window.slices {
          _ = try decodeSlice(
            slice, source: source, firstFrame: window.globalFrameRange.lowerBound,
            dense: dense, mask: mask, audit: audit, scratch: scratch, errors: errors,
            partialDPC: nil, moments: moments,
            commandBufferOverride: command,
            shouldCancel: shouldCancel, profile: &profile)
        }
        if includeDPCMoments {
          guard let encoder = command.makeComputeCommandEncoder() else {
            throw Self.invalid("Cannot encode exact runtime-ANS DPC moments")
          }
          var shape = Shape(
            scans: UInt32(window.globalFrameRange.count), pixels: UInt32(pixels),
            columns: UInt32(source.dataset.detectorCols),
            sourceBytes: UInt32(source.sourceBytesPerValue))
          encoder.setComputePipelineState(momentsPipeline)
          encoder.setBuffer(dense, offset: 0, index: 0)
          encoder.setBuffer(moments, offset: 0, index: 1)
          encoder.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 2)
          encoder.dispatchThreads(
            MTLSize(width: window.globalFrameRange.count * 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          encoder.endEncoding()
        }
        try consume(
          dense, includeDPCMoments ? moments : nil,
          window.globalFrameRange, command)
        guard errors.contents().load(as: UInt32.self) == 0 else {
          throw Self.invalid("Invalid compressed original counts; no runtime ANS was published")
        }
        progress(ordinal + 1, windows.count)
      }
    }
    guard try Self.inputStamps(source) == initialInputs else {
      throw Self.invalid("Original data changed while building runtime ANS; reopen and retry")
    }
  }
}
