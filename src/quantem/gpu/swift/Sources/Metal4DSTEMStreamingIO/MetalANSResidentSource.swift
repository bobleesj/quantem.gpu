import Foundation
import Metal
import Metal4DSTEMKernels

/// Exact native ANS counts with bounded decoding and no dense source expansion.
///
/// This advanced arrays constructor is not a disk-format loader. The caller
/// serializes operations and release, and remains the owner of its host arrays.
/// All streams and the declared uint8/uint16 range are validated on Metal before
/// the resident source is returned. No masks, binning, cropping or dtype fallback
/// are applied. Complete stream validation is distinct from prefix-only DP reads.
public final class MetalANSResidentSource {
  public let shape: [Int]
  public let logicalDtype: Metal4DSTEMIntegerDType
  public let blockFrames: Int
  public let scale: Int
  public private(set) var isReleased = false
  /// Additional buffers owned by the last operation, excluding this source and host results.
  /// This is a buffer-length sum, not measured process or driver peak memory.
  public private(set) var lastOperationScratchBytes = 0

  public let device: MTLDevice
  private var residentCountMeans: (diffraction: MTLBuffer, brightField: MTLBuffer)?
  private let queue: MTLCommandQueue
  private let decodePipeline: MTLComputePipelineState
  private let gatherPipeline: MTLComputePipelineState
  private let reducePipeline: MTLComputePipelineState
  private var tables: [MTLBuffer]
  private var failure: MTLBuffer?
  private var diffraction: MTLBuffer?
  private var request: MTLBuffer?
  private var scanCount: Int { shape[0] * shape[1] }
  private var pixels: Int { shape[2] * shape[3] }
  private var bytesPerValue: Int { logicalDtype == .uint8 ? 1 : 2 }

  /// Bytes in buffers owned by this source; caller arrays and driver overhead are excluded.
  public var residentBytes: Int {
    tables.reduce(0) { $0 + $1.length }
      + (failure?.length ?? 0) + (diffraction?.length ?? 0) + (request?.length ?? 0)
      + (residentCountMeans?.diffraction.length ?? 0)
      + (residentCountMeans?.brightField.length ?? 0)
  }

  public init(
    arrays: MetalANSCountArrays, device: MTLDevice,
    maximumAdditionalBytes: UInt64? = nil
  ) throws {
    shape = arrays.shape
    logicalDtype = arrays.logicalDtype
    blockFrames = arrays.blockFrames
    scale = arrays.scale
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      throw Self.invalid("Metal could not create a count-ANS command queue.")
    }
    self.queue = queue
    let library = try Metal4DSTEMKernels.makeANSCountsLibrary(device: device)
    let validation = try Self.pipeline(library, "ans_counts_validate", device)
    decodePipeline = try Self.pipeline(library, "ans_counts_decode", device)
    gatherPipeline = try Self.pipeline(library, "ans_counts_gather", device)
    reducePipeline = try Self.pipeline(library, "ans_counts_reduce", device)
    let sizes = [
      arrays.payload.count, arrays.offsets.count * 8, arrays.modelIDs.count * 4,
      arrays.contextOffsets.count * 4, arrays.symbols.count * 2,
      arrays.cumulative.count * 2, arrays.frequencies.count * 2, arrays.literal.count,
    ].map { max(4, $0) }
    let outputBytes = arrays.detectorPixelCount * (arrays.logicalDtype == .uint8 ? 1 : 2)
    let stagingBytes = min(1024 * 1024, sizes.max() ?? 4)
    let required = UInt64(sizes.reduce(0, +) + 4 + outputBytes + 8 + stagingBytes)
    try Self.admit(
      sizes: sizes + [4, outputBytes, 8, stagingBytes], additionalBytes: required,
      budget: maximumAdditionalBytes, device: device)
    tables = []
    failure = try Self.buffer(device, bytes: 4, label: "ANS validation status")
    diffraction = try Self.buffer(device, bytes: outputBytes, label: "ANS selected diffraction")
    request = try Self.buffer(device, bytes: 8, label: "ANS selected scan")
    func upload<T>(_ values: [T]) throws -> MTLBuffer {
      let bytes = max(4, values.count * MemoryLayout<T>.stride)
      let staging = try Self.buffer(
        device, bytes: min(stagingBytes, bytes), label: "bounded ANS table staging")
      guard let resident = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
        throw Self.invalid("Metal could not admit exact ANS counts.")
      }
      try values.withUnsafeBytes { raw in
        for offset in stride(from: 0, to: bytes, by: staging.length) {
          let count = min(staging.length, bytes - offset)
          memset(staging.contents(), 0, count)
          if let pointer = raw.baseAddress, offset < raw.count {
            memcpy(staging.contents(), pointer.advanced(by: offset), min(count, raw.count - offset))
          }
          guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
          else {
            throw Self.invalid("Metal could not stage exact ANS counts.")
          }
          blit.copy(
            from: staging, sourceOffset: 0, to: resident, destinationOffset: offset, size: count)
          blit.endEncoding()
          command.commit()
          command.waitUntilCompleted()
          guard command.status == .completed else {
            throw Self.invalid(
              "ANS upload failed: \(command.error?.localizedDescription ?? "unknown error")")
          }
        }
      }
      return resident
    }
    tables = [
      try upload(arrays.payload), try upload(arrays.offsets), try upload(arrays.modelIDs),
      try upload(arrays.contextOffsets), try upload(arrays.symbols),
      try upload(arrays.cumulative), try upload(arrays.frequencies), try upload(arrays.literal),
    ]
    let command = try makeCommand()
    let encoder = try bind(command, pipeline: validation, parameters: parameters())
    encoder.dispatchThreads(
      MTLSize(width: arrays.modelIDs.count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    try complete(command)
  }

  /// Load a self-contained QGANS v1 file into private Metal buffers.
  ///
  /// The reader validates the manifest and section geometry first, then streams
  /// each typed section through a bounded shared staging buffer into a private
  /// resident buffer. It never materializes the logical four-dimensional count
  /// volume or copies a multi-gigabyte section into a Swift array. By default,
  /// every section checksum is verified while it is uploaded; an optional
  /// whole-file SHA-256 can provide external source authentication.
  public convenience init(
    sourceURL: URL,
    device: MTLDevice,
    expectedSHA256: String? = nil,
    verifyChecksums: Bool = true,
    maximumAdditionalBytes: UInt64? = nil
  ) throws {
    let reader = try MetalANSFileReader(
      sourceURL: sourceURL, expectedSHA256: expectedSHA256, verifyChecksums: verifyChecksums)
    try self.init(
      fileReader: reader, device: device, maximumAdditionalBytes: maximumAdditionalBytes)
  }

  private init(
    fileReader: MetalANSFileReader,
    device: MTLDevice,
    maximumAdditionalBytes: UInt64?
  ) throws {
    let index = fileReader.index
    shape = index.shape
    logicalDtype = index.logicalDtype
    blockFrames = index.blockFrames
    scale = index.scale
    self.device = device
    tables = []
    failure = nil
    diffraction = nil
    request = nil
    isReleased = false
    lastOperationScratchBytes = 0
    guard let queue = device.makeCommandQueue() else {
      throw Self.invalid("Metal could not create a count-ANS command queue for QGANS.")
    }
    self.queue = queue
    let library = try Metal4DSTEMKernels.makeANSCountsLibrary(device: device)
    let validation = try Self.pipeline(library, "ans_counts_validate", device)
    decodePipeline = try Self.pipeline(library, "ans_counts_decode", device)
    gatherPipeline = try Self.pipeline(library, "ans_counts_gather", device)
    reducePipeline = try Self.pipeline(library, "ans_counts_reduce", device)
    let sizes = index.sections.map { max(4, $0.byteCount) }
    let outputBytesResult = index.detectorPixelCount.multipliedReportingOverflow(
      by: index.logicalDtype == .uint8 ? 1 : 2)
    guard !outputBytesResult.overflow else {
      throw Self.invalid("QGANS diffraction-buffer size overflows Int.")
    }
    let outputBytes = outputBytesResult.partialValue
    let stagingBytes = min(32 * 1024 * 1024, max(4, sizes.max() ?? 4))
    var required = UInt64(0)
    for size in sizes + [4, outputBytes, 8, stagingBytes] {
      let next = required.addingReportingOverflow(UInt64(size))
      guard !next.overflow else {
        throw Self.invalid("QGANS resident-size arithmetic overflows UInt64.")
      }
      required = next.partialValue
    }
    try Self.admit(
      sizes: sizes + [4, outputBytes, 8, stagingBytes], additionalBytes: required,
      budget: maximumAdditionalBytes, device: device)
    failure = try Self.buffer(device, bytes: 4, label: "ANS validation status")
    diffraction = try Self.buffer(
      device, bytes: outputBytes, label: "ANS selected diffraction")
    request = try Self.buffer(device, bytes: 8, label: "ANS selected scan")
    do {
      tables = try index.sections.map {
        try fileReader.upload(
          section: $0, device: device, queue: queue, stagingBytes: stagingBytes)
      }
    } catch {
      tables.removeAll(keepingCapacity: false)
      failure = nil
      diffraction = nil
      request = nil
      throw error
    }
    let command = try makeCommand()
    let encoder = try bind(command, pipeline: validation, parameters: parameters())
    encoder.dispatchThreads(
      MTLSize(width: index.sections[2].elementCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    try complete(command)
  }

  /// Read one full raw DP exactly, preserving original counts including rare high values.
  ///
  /// The small result is widened exactly to UInt32 for the native interaction API.
  /// Stored logical dtype is unchanged. Repeated calls preserve request order and
  /// duplicates, but this method is not a batched gather API.
  public func extractRawDiffraction(scanRow: Int, scanColumn: Int) throws -> [UInt32] {
    try requireLive()
    guard 0..<shape[0] ~= scanRow, 0..<shape[1] ~= scanColumn,
      let diffraction, let request
    else { throw Self.invalid("Choose an in-bounds scan row and column.") }
    request.contents().storeBytes(of: UInt64(scanRow * shape[1] + scanColumn), as: UInt64.self)
    var values = parameters()
    values[6] = 1
    let command = try makeCommand()
    let encoder = try bind(command, pipeline: gatherPipeline, parameters: values)
    encoder.setBuffer(diffraction, offset: 0, index: 10)
    encoder.setBuffer(request, offset: 0, index: 11)
    encoder.dispatchThreads(
      MTLSize(width: pixels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    try complete(command)
    lastOperationScratchBytes = 0
    if bytesPerValue == 1 {
      return Array(
        UnsafeBufferPointer(
          start: diffraction.contents().assumingMemoryBound(to: UInt8.self), count: pixels
        )
      ).map(UInt32.init)
    }
    return Array(
      UnsafeBufferPointer(
        start: diffraction.contents().assumingMemoryBound(to: UInt16.self), count: pixels
      )
    ).map(UInt32.init)
  }

  /// Sum an explicit binary detector mask for every scan position using uint64 arithmetic.
  ///
  /// Only one bounded decoded block segment exists at a time. This correctness
  /// path does not promise real-time performance: small scratch budgets may cause
  /// repeated prefix decoding within an entropy block. No full dense cube exists.
  /// `maximumDecodedBytes` caps decoded staging only; the binary detector mask
  /// and uint64 scan output are additional. `lastOperationScratchBytes` reports
  /// the sum of all three buffers, separately from `residentBytes`.
  public func sumVirtualDetector(
    mask: [UInt8], maximumDecodedBytes: Int = 16 * 1024 * 1024
  ) throws -> [UInt64] {
    try requireLive()
    guard mask.count == pixels, mask.allSatisfy({ $0 <= 1 }),
      maximumDecodedBytes >= pixels * bytesPerValue
    else { throw Self.invalid("Use a binary detector mask and enough scratch for one exact DP.") }
    let rows = min(blockFrames, scanCount, maximumDecodedBytes / (pixels * bytesPerValue))
    let decodedBytes = rows * pixels * bytesPerValue
    let resultBytes = scanCount.multipliedReportingOverflow(by: 8)
    guard !resultBytes.overflow else { throw Self.invalid("Exact detector output size overflows.") }
    try Self.admit(
      sizes: [decodedBytes, pixels, resultBytes.partialValue],
      additionalBytes: UInt64(decodedBytes + pixels + resultBytes.partialValue),
      budget: nil, device: device)
    let decoded = try Self.buffer(device, bytes: decodedBytes, label: "bounded ANS decoded segment")
    let selected = try Self.buffer(device, bytes: pixels, label: "ANS binary detector mask")
    let result = try Self.buffer(
      device, bytes: resultBytes.partialValue, label: "ANS exact detector image")
    _ = mask.withUnsafeBytes { memcpy(selected.contents(), $0.baseAddress!, $0.count) }
    lastOperationScratchBytes = decodedBytes + pixels + resultBytes.partialValue
    for first in stride(from: 0, to: scanCount, by: blockFrames) {
      let blockRows = min(blockFrames, scanCount - first)
      for offset in stride(from: 0, to: blockRows, by: rows) {
        let count = min(rows, blockRows - offset)
        var values = parameters()
        values[5] = UInt64(first / blockFrames)
        values[6] = UInt64(offset)
        values[7] = UInt64(count)
        let command = try makeCommand()
        let decoder = try bind(command, pipeline: decodePipeline, parameters: values)
        decoder.setBuffer(decoded, offset: 0, index: 10)
        decoder.dispatchThreads(
          MTLSize(width: pixels, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        decoder.endEncoding()
        guard let reduction = command.makeComputeCommandEncoder() else {
          throw Self.invalid("Metal could not encode the exact detector reduction.")
        }
        var reductionParameters = SIMD4<UInt64>(
          UInt64(pixels), UInt64(bytesPerValue), UInt64(first + offset), 0)
        reduction.setComputePipelineState(reducePipeline)
        reduction.setBuffer(decoded, offset: 0, index: 0)
        reduction.setBuffer(selected, offset: 0, index: 1)
        reduction.setBuffer(result, offset: 0, index: 2)
        reduction.setBytes(&reductionParameters, length: 32, index: 3)
        reduction.dispatchThreadgroups(
          MTLSize(width: count, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        reduction.endEncoding()
        try complete(command)
      }
    }
    return Array(
      UnsafeBufferPointer(
        start: result.contents().assumingMemoryBound(to: UInt64.self), count: scanCount))
  }

  public func releaseResidentStorage() {
    residentCountMeans = nil
    tables.removeAll(keepingCapacity: false)
    failure = nil
    diffraction = nil
    request = nil
    isReleased = true
  }

  private func parameters() -> [UInt64] {
    [
      UInt64(scanCount), UInt64(pixels), UInt64(blockFrames), UInt64(scale),
      logicalDtype == .uint8 ? 255 : 65535, 0, 0, 0, UInt64(bytesPerValue),
    ]
  }

  private func requireLive() throws {
    guard !isReleased, tables.count == 8, failure != nil else {
      throw Self.invalid("The ANS source was released; load it before requesting counts.")
    }
  }

  private func makeCommand() throws -> MTLCommandBuffer {
    try requireLive()
    guard let command = queue.makeCommandBuffer(), let failure else {
      throw Self.invalid("Metal could not create an ANS command buffer.")
    }
    failure.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    return command
  }

  private func bind(
    _ command: MTLCommandBuffer, pipeline: MTLComputePipelineState, parameters: [UInt64]
  ) throws -> MTLComputeCommandEncoder {
    guard let encoder = command.makeComputeCommandEncoder(), let failure else {
      throw Self.invalid("Metal could not encode the ANS operation.")
    }
    encoder.setComputePipelineState(pipeline)
    for (index, table) in tables.enumerated() { encoder.setBuffer(table, offset: 0, index: index) }
    encoder.setBuffer(failure, offset: 0, index: 8)
    parameters.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 9) }
    return encoder
  }

  private func complete(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw Self.invalid(
        "ANS GPU command failed: \(command.error?.localizedDescription ?? "unknown error")")
    }
    try checkErrors()
  }

  public func checkErrors() throws {
    guard let failure, failure.contents().load(as: UInt32.self) == 0 else {
      throw Self.invalid(
        "ANS stream termination or native count range is invalid; no result was published.")
    }
  }

  private static func pipeline(
    _ library: MTLLibrary, _ name: String, _ device: MTLDevice
  ) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
      throw invalid("Missing kernel \(name).")
    }
    return try device.makeComputePipelineState(function: function)
  }

  private static func buffer(_ device: MTLDevice, bytes: Int, label: String) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(label: label, bytes: UInt64(bytes))
    }
    return buffer
  }

  private static func admit(
    sizes: [Int], additionalBytes: UInt64, budget: UInt64?, device: MTLDevice
  ) throws {
    let allocated = UInt64(device.currentAllocatedSize)
    let remaining =
      device.recommendedMaxWorkingSetSize > allocated
      ? device.recommendedMaxWorkingSetSize - allocated : 0
    guard sizes.allSatisfy({ $0 > 0 && $0 <= device.maxBufferLength }),
      additionalBytes <= remaining, budget == nil || additionalBytes <= budget!
    else {
      throw invalid(
        "Exact ANS buffers exceed the explicit budget or Metal limits; no fallback applied.")
    }
  }

  private static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }
}

extension MetalANSResidentSource: MetalResidentCounts {
  public var hotPixelIndices: [Int] { [] }
  public var hotPixelCorrection: String { "none" }
  public var itemBytes: Int { bytesPerValue }
  public var readyFrames: Int { isReleased ? 0 : scanCount }
  public var representation: Metal4DSTEMResidentRepresentation { .encoded }
  public func countMeans() throws -> (diffraction: MTLBuffer, brightField: MTLBuffer) {
    try requireLive()
    if let residentCountMeans { return residentCountMeans }
    let result = try ResidentCountMeans.calculate(self)
    residentCountMeans = result
    return result
  }
  /// Decode an owned native-count region across entropy-block boundaries.
  public func encodeRead(_ frames: Range<Int>, into output: MTLBuffer, command: MTLCommandBuffer)
    throws
  {
    try requireLive()
    guard !frames.isEmpty, frames.lowerBound >= 0, frames.upperBound <= scanCount,
      frames.count <= 8192
    else { throw Self.invalid("Read 1...8192 available frames from an open encoded acquisition.") }
    guard output.length >= frames.count * pixels * bytesPerValue,
      output.device.registryID == device.registryID,
      command.commandQueue.device.registryID == device.registryID
    else { throw Self.invalid("Use a same-device destination large enough for the count region.") }
    for block in (frames.lowerBound / blockFrames)...((frames.upperBound - 1) / blockFrames) {
      let first = max(frames.lowerBound, block * blockFrames)
      let stop = min(frames.upperBound, (block + 1) * blockFrames)
      var values = parameters()
      values[5] = UInt64(block)
      values[6] = UInt64(first - block * blockFrames)
      values[7] = UInt64(stop - first)
      let decoder = try bind(command, pipeline: decodePipeline, parameters: values)
      decoder.setBuffer(
        output, offset: (first - frames.lowerBound) * pixels * bytesPerValue, index: 10)
      decoder.dispatchThreads(
        MTLSize(width: pixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      decoder.endEncoding()
    }
    lastOperationScratchBytes = output.length
  }
}
