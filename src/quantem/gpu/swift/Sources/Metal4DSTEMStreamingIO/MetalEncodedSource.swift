import Foundation
import Metal
import MetalCountResources
import Native4DSTEMIO

/// Exact native counts encoded on the GPU, with bounded reads and complete means.
/// Operations on a source are serialized by its caller. No Python runtime is used.
public final class MetalEncodedSource {
  public let shape: [Int]
  public let itemBytes: Int
  public let hotPixelIndices: [Int]
  public let meanDiffraction: MTLBuffer
  public let meanBrightField: MTLBuffer
  public private(set) var readyFrames = 0
  public private(set) var isReleased = false
  public private(set) var sourceReadPasses = 0
  public private(set) var peakAllocatedBytes = 0
  public let device: MTLDevice
  public let queue: MTLCommandQueue
  private let pipelines: [String: MTLComputePipelineState]
  private let encoding, decoding, error, detectorSum: MTLBuffer
  private let valid, bad: MTLBuffer
  private struct Chunk {
    let first, count: Int
    let payload, offsets, models: MTLBuffer
  }
  private var chunks: [Chunk] = []
  private var encodingWorkspace:
    (scratch: MTLBuffer, sizes: MTLBuffer, states: MTLBuffer, totals: MTLBuffer)?
  private let interval = 512
  public var residentBytes: Int {
    chunks.reduce(0) { $0 + $1.payload.length + $1.offsets.length + $1.models.length }
      + [encoding, decoding, error, detectorSum, meanDiffraction, meanBrightField, valid, bad]
      .reduce(0) { $0 + $1.length }
      + (encodingWorkspace.map {
        $0.scratch.length + $0.sizes.length + $0.states.length + $0.totals.length
      } ?? 0)
  }

  /// Allocate an empty source; append consecutive native count regions to fill it.
  public init(shape: [Int], itemBytes: Int = 2, hotPixelIndices: [Int] = [], device: MTLDevice)
    throws
  {
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 }), [1, 2].contains(itemBytes),
      let queue = device.makeCommandQueue()
    else {
      throw Self.invalid("Use a positive 4D uint8/uint16 shape and an available Metal device.")
    }
    self.shape = shape
    self.itemBytes = itemBytes
    self.device = device
    self.queue = queue
    let pixels = shape[2] * shape[3]
    guard Set(hotPixelIndices).count == hotPixelIndices.count,
      hotPixelIndices.allSatisfy({ (0..<pixels).contains($0) })
    else {
      throw Self.invalid("Hot-pixel indices must be unique and inside the detector.")
    }
    self.hotPixelIndices = hotPixelIndices
    let library = try MetalKernelCache.library(device: device, key: "encoded-counts") {
      try ["streamed_counts", "hot_pixels", "resident_utilities", "count_tables"].map {
        try MetalCountResources.source($0)
      }.joined(separator: "\n")
    }
    var compiled: [String: MTLComputePipelineState] = [:]
    for name in [
      "streamed_counts_encode", "streamed_counts_compact", "streamed_counts_decode_range",
      "hot_median", "count_prefix", "count_prefix_totals", "count_prefix_add",
      "count_summary", "count_bright", "count_normalize", "count_verify", "count_tables",
      "streamed_counts_masked_sums", "streamed_counts_group_sums",
    ] {
      compiled[name] = try MetalKernelCache.pipeline(
        device: device, library: library, key: "encoded-counts", function: name)
    }
    pipelines = compiled
    encoding = try Self.buffer(device, 64 * 33 * 4)
    decoding = try Self.buffer(device, 64 * 1024 * 4)
    error = try Self.buffer(device, 4)
    detectorSum = try Self.buffer(device, pixels * 8)
    meanDiffraction = try Self.buffer(device, pixels * 4)
    meanBrightField = try Self.buffer(device, shape[0] * shape[1] * 4)
    valid = try Self.buffer(device, pixels)
    memset(valid.contents(), 1, pixels)
    bad = try Self.buffer(device, max(4, hotPixelIndices.count * 4))
    for (index, pixel) in hotPixelIndices.enumerated() {
      valid.contents().storeBytes(of: UInt8(0), toByteOffset: pixel, as: UInt8.self)
      bad.contents().storeBytes(of: Int32(pixel), toByteOffset: index * 4, as: Int32.self)
    }
    memset(detectorSum.contents(), 0, detectorSum.length)
    memset(error.contents(), 0, error.length)
    let initialize = try command()
    let tableEncoder = try encoder(initialize, "count_tables", [encoding, decoding])
    dispatch(tableEncoder, 64)
    try complete(initialize)
  }

  /// Read original compressed HDF5 once, correcting and encoding bounded regions.
  public static func load(
    source: Native4DSTEMIndexedSource, device: MTLDevice,
    shouldCancel: () -> Bool = { false }
  ) throws -> MetalEncodedSource {
    let d = source.dataset
    guard source.sourceBytesPerValue <= 2 else {
      throw invalid("Encoded counts require native uint8/uint16 input.")
    }
    let result = try MetalEncodedSource(
      shape: [d.scanRows, d.scanCols, d.detectorRows, d.detectorCols],
      itemBytes: source.sourceBytesPerValue, hotPixelIndices: d.badPixelIndices, device: device)
    let verifyCounts = ProcessInfo.processInfo.environment["QUANTEM_GPU_VALIDATE_COUNTS"] == "1"
    try MetalHDF5Reader.read(source: source, device: device, shouldCancel: shouldCancel) {
      raw, frames in
      try result.append(raw, frames: frames.count, verify: verifyCounts)
    }
    result.sourceReadPasses = 1
    return result
  }

  /// Append a contiguous region, applying median correction before exact encoding.
  public func append(_ raw: MTLBuffer, frames: Int, verify: Bool = false) throws {
    let pixels = shape[2] * shape[3]
    guard !isReleased, frames > 0, readyFrames + frames <= shape[0] * shape[1],
      raw.device.registryID == device.registryID, raw.length >= frames * pixels * itemBytes
    else {
      throw Self.invalid("Append an in-bounds consecutive count region on the source device.")
    }
    let streams = ((frames + interval - 1) / interval) * pixels
    guard streams <= Int(UInt32.max) / (2 * min(frames, interval) + 4) else {
      throw Self.invalid(
        "This encoding region exceeds 32-bit stream offsets; append smaller frame regions.")
    }
    let scratchBytes = (2 * min(frames, interval) + 4) * streams
    if encodingWorkspace == nil || encodingWorkspace!.scratch.length < scratchBytes
      || encodingWorkspace!.sizes.length < streams * 4
    {
      encodingWorkspace = (
        try Self.buffer(device, scratchBytes),
        try Self.buffer(device, streams * 4), try Self.buffer(device, streams * 4),
        try Self.buffer(device, ((streams + 255) / 256) * 4)
      )
    }
    let (scratch, sizes, states, totals) = encodingWorkspace!
    let models = try Self.buffer(device, streams)
    let offsets = try Self.buffer(device, (streams + 1) * 4)
    let p = [UInt64(frames), UInt64(pixels), UInt64(interval), UInt64(streams), UInt64(itemBytes)]
    let command = try command()
    if !hotPixelIndices.isEmpty {
      try encode(
        command, "hot_median", [raw, valid, bad],
        [
          UInt64(hotPixelIndices.count), UInt64(shape[2]), UInt64(shape[3]),
          UInt64(frames * hotPixelIndices.count), UInt64(itemBytes),
        ], frames * hotPixelIndices.count)
    }
    try encode(
      command, "streamed_counts_encode", [raw, encoding, scratch, sizes, states, models], p, streams
    )
    let prefix = try encoder(command, "count_prefix", [sizes, offsets, totals])
    var count = UInt32(streams)
    prefix.setBytes(&count, length: 4, index: 3)
    prefix.dispatchThreadgroups(
      MTLSize(width: (streams + 255) / 256, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    prefix.endEncoding()
    let total = try encoder(command, "count_prefix_totals", [totals, offsets])
    var counts = SIMD2<UInt32>(UInt32((streams + 255) / 256), UInt32(streams))
    total.setBytes(&counts, length: 8, index: 2)
    dispatch(total, 1)
    let add = try encoder(command, "count_prefix_add", [offsets, totals])
    add.setBytes(&count, length: 4, index: 2)
    dispatch(add, streams)
    try complete(command)
    let payloadBytes = Int(offsets.contents().load(fromByteOffset: streams * 4, as: UInt32.self))
    let payload = try Self.buffer(device, max(4, payloadBytes))
    let compact = try self.command()
    try encode(
      compact, "streamed_counts_compact", [raw, scratch, offsets, states, models, payload], p,
      streams)
    try encode(
      compact, "count_summary", [raw, detectorSum, meanBrightField],
      [UInt64(frames), UInt64(pixels), UInt64(itemBytes), UInt64(readyFrames)], pixels)
    let bright = try encoder(compact, "count_bright", [raw, meanBrightField])
    var dims = SIMD4<UInt64>(UInt64(frames), UInt64(pixels), UInt64(itemBytes), UInt64(readyFrames))
    bright.setBytes(&dims, length: 32, index: 2)
    bright.dispatchThreadgroups(
      MTLSize(width: frames, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    bright.endEncoding()
    let norm = try encoder(compact, "count_normalize", [detectorSum, meanDiffraction])
    var np = SIMD2<UInt32>(UInt32(pixels), UInt32(shape[0] * shape[1]))
    norm.setBytes(&np, length: 8, index: 2)
    dispatch(norm, pixels)
    try complete(compact)
    chunks.append(
      Chunk(first: readyFrames, count: frames, payload: payload, offsets: offsets, models: models))
    let first = readyFrames
    readyFrames += frames
    peakAllocatedBytes = max(peakAllocatedBytes, device.currentAllocatedSize)
    if readyFrames == shape[0] * shape[1] { encodingWorkspace = nil }
    if verify {
      let decoded = try read(first..<(first + frames))
      let command = try self.command()
      let checker = try encoder(command, "count_verify", [raw, decoded, error])
      var bytes = UInt32(frames * pixels * itemBytes)
      checker.setBytes(&bytes, length: 4, index: 3)
      dispatch(checker, Int(bytes))
      try complete(command)
      try checkErrors()
    }
  }

  /// Read a bounded frame range into an exact native-count buffer.
  public func read(_ frames: Range<Int>) throws -> MTLBuffer {
    let result = try Self.buffer(device, frames.count * shape[2] * shape[3] * itemBytes)
    let command = try self.command()
    try encodeRead(frames, into: result, command: command)
    try complete(command)
    try checkErrors()
    return result
  }

  /// Enqueue reads so downstream native operations can share one command buffer.
  public func encodeRead(_ frames: Range<Int>, into result: MTLBuffer, command: MTLCommandBuffer)
    throws
  {
    guard !isReleased, !frames.isEmpty, frames.lowerBound >= 0, frames.upperBound <= readyFrames
    else {
      throw Self.invalid("Read a nonempty range inside the loaded acquisition.")
    }
    let pixels = shape[2] * shape[3]
    guard result.length >= frames.count * pixels * itemBytes else {
      throw Self.invalid("Read destination is too small.")
    }
    error.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    for chunk in chunks {
      let start = max(frames.lowerBound, chunk.first)
      let stop = min(frames.upperBound, chunk.first + chunk.count)
      if start >= stop { continue }
      let local = start - chunk.first
      let firstStream = (local / interval) * pixels
      let stopStream = ((stop - chunk.first + interval - 1) / interval) * pixels
      let p = [
        chunk.count, pixels, interval, local, stop - start, firstStream, stopStream, itemBytes,
      ].map(UInt64.init)
      let enc = try encoder(
        command, "streamed_counts_decode_range",
        [chunk.payload, chunk.offsets, chunk.models, decoding, error])
      enc.setBuffer(result, offset: (start - frames.lowerBound) * pixels * itemBytes, index: 5)
      p.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 6) }
      dispatch(enc, stopStream - firstStream)
    }
  }

  public func checkErrors() throws {
    if error.contents().load(as: UInt32.self) != 0 {
      throw Self.invalid("Encoded counts failed exact reconstruction.")
    }
  }
  public func releaseResidentStorage() {
    chunks.removeAll()
    encodingWorkspace = nil
    isReleased = true
  }

  private func command() throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer() else {
      throw Self.invalid("Cannot create a Metal command buffer.")
    }
    return command
  }
  private func encoder(_ command: MTLCommandBuffer, _ name: String, _ buffers: [MTLBuffer]) throws
    -> MTLComputeCommandEncoder
  {
    guard let enc = command.makeComputeCommandEncoder(), let pipeline = pipelines[name] else {
      throw Self.invalid("Cannot encode \(name).")
    }
    enc.setComputePipelineState(pipeline)
    for (i, buffer) in buffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: i) }
    return enc
  }
  private func encode(
    _ command: MTLCommandBuffer, _ name: String, _ buffers: [MTLBuffer], _ p: [UInt64], _ count: Int
  ) throws {
    let enc = try encoder(command, name, buffers)
    p.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
    dispatch(enc, count)
  }
  private func dispatch(_ enc: MTLComputeCommandEncoder, _ count: Int) {
    enc.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
  }
  private func complete(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if command.status != .completed {
      throw Self.invalid(command.error?.localizedDescription ?? "Metal command failed.")
    }
  }
  static func buffer(_ device: MTLDevice, _ bytes: Int) throws -> MTLBuffer {
    guard bytes > 0, bytes <= device.maxBufferLength,
      let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
    else {
      throw invalid("Cannot allocate \(bytes) bytes; reduce the requested region.")
    }
    return buffer
  }
  static func invalid(_ text: String) -> Metal4DSTEMStreamingIOError { .invalidRequest(text) }
}

extension MetalEncodedSource {
  /// Enqueue exact per-frame code sums over `count` listed detector pixels (uint32
  /// indices in `listed`). Only those pixels' streams are decoded. `sums` receives one
  /// uint32 per frame of this source starting at `sumsOffset` bytes and must be zeroed
  /// by the caller; keep `count` at or below 65,536 so a frame total fits in 32 bits.
  /// Call `checkErrors()` after the command completes.
  func encodeMaskedSums(
    listed: MTLBuffer, count: Int, into sums: MTLBuffer, sumsOffset: Int,
    command: MTLCommandBuffer
  ) throws {
    let pixels = shape[2] * shape[3]
    guard !isReleased, count > 0, count <= 65_536, count <= pixels, listed.length >= count * 4,
      sums.length >= sumsOffset + readyFrames * 4,
      let pipeline = pipelines["streamed_counts_masked_sums"]
    else {
      throw Self.invalid("Masked sums need 1-65536 listed pixels and a frame-sized output.")
    }
    error.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    for chunk in chunks {
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Self.invalid("Cannot encode masked sums.")
      }
      encoder.setComputePipelineState(pipeline)
      for (index, buffer) in [chunk.payload, chunk.offsets, chunk.models, decoding, error]
        .enumerated()
      {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBuffer(sums, offset: sumsOffset + chunk.first * 4, index: 5)
      encoder.setBuffer(listed, offset: 0, index: 6)
      let p = [UInt64(chunk.count), UInt64(pixels), UInt64(interval), UInt64(count)]
      p.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 7) }
      encoder.dispatchThreadgroups(
        MTLSize(
          width: (count + 255) / 256, height: (chunk.count + interval - 1) / interval, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      encoder.endEncoding()
    }
  }

  /// Enqueue exact per-frame sums for pixel groups. `listed` holds `count` uint32 pixel
  /// indices (a multiple of 32; padding lanes are 0xFFFFFFFF) and `groups` one group
  /// index per 32 listed entries. `sums` holds `groupCount` uint32 per frame of this
  /// source, starting at `sumsOffset` bytes, zeroed by the caller.
  func encodeGroupSums(
    listed: MTLBuffer, count: Int, groups: MTLBuffer, groupCount: Int, into sums: MTLBuffer,
    sumsOffset: Int, command: MTLCommandBuffer
  ) throws {
    let pixels = shape[2] * shape[3]
    guard !isReleased, count > 0, count % 32 == 0, groupCount > 0, listed.length >= count * 4,
      groups.length >= count / 32 * 4, sums.length >= sumsOffset + readyFrames * groupCount * 4,
      let pipeline = pipelines["streamed_counts_group_sums"]
    else {
      throw Self.invalid("Group sums need whole SIMD groups of listed pixels and a sized output.")
    }
    error.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    for chunk in chunks {
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Self.invalid("Cannot encode group sums.")
      }
      encoder.setComputePipelineState(pipeline)
      for (index, buffer) in [chunk.payload, chunk.offsets, chunk.models, decoding, error]
        .enumerated()
      {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBuffer(sums, offset: sumsOffset + chunk.first * groupCount * 4, index: 5)
      encoder.setBuffer(listed, offset: 0, index: 6)
      encoder.setBuffer(groups, offset: 0, index: 7)
      let p = [
        UInt64(chunk.count), UInt64(pixels), UInt64(interval), UInt64(count), UInt64(groupCount),
      ]
      p.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 8) }
      encoder.dispatchThreads(
        MTLSize(width: count, height: (chunk.count + interval - 1) / interval, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      encoder.endEncoding()
    }
  }

  /// Encoded arrays of one appended frame range. The layout is the QEM integer count
  /// stream: `ceil(count / 512) * detectorPixels` streams, uint32 payload offsets and
  /// uint8 model selectors. Buffers are the resident storage itself, not copies.
  struct EncodedChunk {
    let first: Int
    let count: Int
    let payload: MTLBuffer
    let offsets: MTLBuffer
    let models: MTLBuffer
  }

  var encodedChunks: [EncodedChunk] {
    chunks.map {
      EncodedChunk(
        first: $0.first, count: $0.count, payload: $0.payload, offsets: $0.offsets,
        models: $0.models)
    }
  }

  /// Take ownership of verified uint16 streams for the next `frames`, without copying.
  ///
  /// `payload`, `offsets` and `models` hold `payloadBytes`, `(streams + 1) × 4` and
  /// `streams` bytes. Offsets must start at zero, never decrease and stay inside the
  /// payload; models are 0...63 or the reserved 252...255 stream kinds. Every later
  /// decode still checks entropy completion. Code summaries (detector sums and means)
  /// are not recomputed, so only owners that never read them may adopt streams.
  func adoptEncodedChunk(
    frames: Int, payload: MTLBuffer, payloadBytes: Int, offsets: MTLBuffer, models: MTLBuffer
  ) throws {
    let pixels = shape[2] * shape[3]
    guard !isReleased, itemBytes == 2, frames > 0, readyFrames + frames <= shape[0] * shape[1]
    else {
      throw Self.invalid("Adopt uint16 streams for an in-bounds frame range of an open source.")
    }
    let streams = ((frames + interval - 1) / interval) * pixels
    guard offsets.length >= (streams + 1) * 4, models.length >= streams,
      payload.length >= payloadBytes
    else {
      throw Self.invalid("Encoded stream tables do not match the declared frames.")
    }
    let table = offsets.contents().assumingMemoryBound(to: UInt32.self)
    var previous: UInt32 = 0
    for index in 0...streams {
      let value = UInt32(littleEndian: table[index])
      guard index > 0 || value == 0, value >= previous else {
        throw Self.invalid("Encoded stream offsets must start at zero and never decrease.")
      }
      previous = value
    }
    guard Int(previous) <= payloadBytes else {
      throw Self.invalid("Encoded stream offsets exceed the stored payload.")
    }
    for model in UnsafeRawBufferPointer(start: models.contents(), count: streams)
    where !(model < 64 || model >= 252) {
      throw Self.invalid("Unknown encoded stream model \(model).")
    }
    chunks.append(
      Chunk(first: readyFrames, count: frames, payload: payload, offsets: offsets, models: models))
    readyFrames += frames
    peakAllocatedBytes = max(peakAllocatedBytes, device.currentAllocatedSize)
  }
}
