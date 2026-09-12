import CNativeHDF5
import Foundation
import Metal

/// Bounded standard HDF5 writing with Metal bitshuffle/LZ4 compression.
/// The destination appears only after all frames and metadata have been saved.
public final class MetalHDF5Writer {
  public let path: URL
  public let shape: [Int]
  private let temporary: URL
  private let runtime: MetalPrecision
  private var writer: OpaquePointer?
  private var nextFrame = 0
  public private(set) var compressionSeconds = 0.0
  public private(set) var writeSeconds = 0.0
  public private(set) var peakAllocatedBytes = 0

  public init(path: URL, shape: [Int], runtime: MetalPrecision) throws {
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 }), shape[2] * shape[3] % 4096 == 0 else {
      throw MetalPrecision.invalid(
        "Native HDF5 decoding currently requires a detector pixel count divisible by 4096; use a supported acquisition shape."
      )
    }
    guard !FileManager.default.fileExists(atPath: path.path) else {
      throw MetalPrecision.invalid("The output exists; choose a new path to preserve it.")
    }
    self.path = path
    self.shape = shape
    self.runtime = runtime
    temporary = path.deletingLastPathComponent().appendingPathComponent(
      ".\(path.lastPathComponent).\(UUID().uuidString).tmp")
    var error: UnsafeMutablePointer<CChar>?
    let status = shape.map(UInt64.init).withUnsafeBufferPointer { dimensions in
      qh5_chunk_writer_open(temporary.path, dimensions.baseAddress, &writer, &error)
    }
    try Self.check(status, error)
  }
  /// Append native uint16 codes in consecutive frame order; data never stages as a host array.
  public func append(_ values: MTLBuffer, frames: Int) throws {
    guard writer != nil, frames > 0, nextFrame + frames <= shape[0] * shape[1] else {
      throw MetalPrecision.invalid("Append an in-bounds consecutive region to an open writer.")
    }
    let started = Date.timeIntervalSinceReferenceDate
    let frameBytes = shape[2] * shape[3] * 2
    guard frames <= Int(UInt32.max) / frameBytes else {
      throw MetalPrecision.invalid("This compression region exceeds 32-bit indexing; append fewer frames at a time.")
    }
    let blocks = (frameBytes + 8191) / 8192
    let maximum = 9216
    try runtime.validate(values, count: frames * shape[2] * shape[3], bytes: 2)
    let shuffled = try runtime.buffer(frames * frameBytes)
    let compressed = try runtime.buffer(frames * blocks * maximum)
    let sizes = try runtime.buffer(frames * blocks * 4)
    let frameSizes = try runtime.buffer(frames * 4)
    let stride = 12 + blocks * (4 + maximum)
    let packed = try runtime.buffer(frames * stride)
    peakAllocatedBytes = max(peakAllocatedBytes, runtime.device.currentAllocatedSize)
    let command = try runtime.command()
    let shuffle = try runtime.encoder(command, "bshuf_u16_save", [values, shuffled])
    for (index, item) in [0, frames, frameBytes].enumerated() {
      var word = UInt32(item)
      shuffle.setBytes(&word, length: 4, index: index + 2)
    }
    runtime.dispatch(shuffle, count: frames * frameBytes)
    let encode = try runtime.encoder(command, "lz4_rle_save", [shuffled, compressed, sizes])
    for (index, item) in [frames * blocks, frameBytes, blocks, maximum].enumerated() {
      var word = UInt32(item)
      encode.setBytes(&word, length: 4, index: index + 3)
    }
    runtime.dispatch(encode, count: frames * blocks * 32, groupSize: 32, groups: true)
    let pack = try runtime.encoder(
      command, "save_frame_pack", [compressed, sizes, packed, frameSizes])
    var p = SIMD4<UInt32>(UInt32(frames), UInt32(frameBytes), UInt32(blocks), UInt32(maximum))
    pack.setBytes(&p, length: 16, index: 4)
    runtime.dispatch(pack, count: frames * 32, groupSize: 32, groups: true)
    try runtime.complete(command)
    compressionSeconds += Date.timeIntervalSinceReferenceDate - started
    let writing = Date.timeIntervalSinceReferenceDate
    var error: UnsafeMutablePointer<CChar>?
    let status = qh5_chunk_writer_append(
      writer, UInt64(nextFrame), UInt64(frames),
      packed.contents().assumingMemoryBound(to: UInt8.self), UInt64(stride),
      frameSizes.contents().assumingMemoryBound(to: UInt32.self), &error)
    try Self.check(status, error)
    nextFrame += frames
    writeSeconds += Date.timeIntervalSinceReferenceDate - writing
  }
  public func finish(metadata: [String: String]) throws {
    guard writer != nil else { throw MetalPrecision.invalid("The writer is already closed.") }
    let started = Date.timeIntervalSinceReferenceDate
    for (name, value) in metadata {
      var error: UnsafeMutablePointer<CChar>?
      let status = qh5_chunk_writer_attribute(writer, name, value, &error)
      try Self.check(status, error)
    }
    var error: UnsafeMutablePointer<CChar>?
    let status = qh5_chunk_writer_close(writer, &error)
    writer = nil
    try Self.check(status, error)
    // moveItem fails if another writer created the destination while we ran.
    try FileManager.default.moveItem(at: temporary, to: path)
    writeSeconds += Date.timeIntervalSinceReferenceDate - started
  }
  public func cancel() {
    if let writer {
      qh5_chunk_writer_abort(writer)
      self.writer = nil
    }
    if FileManager.default.fileExists(atPath: temporary.path) {
      try? FileManager.default.removeItem(at: temporary)
    }
  }
  deinit { cancel() }
  static func check(_ status: Int32, _ error: UnsafeMutablePointer<CChar>?) throws {
    defer { if let error { qh5_free_error(error) } }
    if status != 0 {
      throw MetalPrecision.invalid(
        error.map { String(cString: $0) } ?? "Native HDF5 operation failed.")
    }
  }
}
