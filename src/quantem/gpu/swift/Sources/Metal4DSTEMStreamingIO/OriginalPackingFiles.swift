import CryptoKit
import Darwin
import Foundation
import Metal
import Native4DSTEMIO

extension OriginalHDF5Packing {
  struct CompressedReadPlan: Sendable {
    let shardIndex: Int
    let frameRange: Range<Int>
    let sourceURL: URL
    let offset: UInt64
    let count: Int
    let words: [UInt32]
    // Includes the normalized CPU metadata while it is copied into Metal.
    var reservedBytes: UInt64 { UInt64(count + words.count * 8) }
  }
  struct CompressedReadInput {
    let shardIndex: Int
    let frameRange: Range<Int>
    let compressed: MTLBuffer
    let metadata: MTLBuffer
    let readSeconds: Double
    let copySeconds: Double
    let reservedBytes: UInt64
  }
  /// One owned input in flight. Only the caller touches scientific state/Profile;
  /// the reader owns its file descriptor and writes only its private input buffer.
  final class CompressedReadAhead: @unchecked Sendable {
    // Two independent reads let the SSD and the GPU overlap on the uint32 path.
    // A single reader leaves the GPU idle while the next compressed shard is read.
    private let queue = DispatchQueue(
      label: "qgpu.original-packing.input", qos: .userInitiated, attributes: .concurrent)
    private let completed = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private let device: MTLDevice
    private let depth: Int
    private var stopped = false
    private var pendingCount = 0
    private var results: [Int: Result<CompressedReadInput, Error>] = [:]

    init(device: MTLDevice, depth: Int = 2) {
      self.device = device
      self.depth = max(1, depth)
    }

    func submit(_ plan: CompressedReadPlan) throws {
      let key = plan.frameRange.lowerBound
      try lock.withLock {
        guard !stopped, pendingCount < depth, results[key] == nil else {
          throw OriginalHDF5Packing.invalid("Compressed read-ahead ownership is invalid")
        }
        pendingCount += 1
      }
      queue.async {
        let result = Result {
          try autoreleasepool {
            try OriginalHDF5Packing.readCompressed(
              plan, device: self.device,
              isCancelled: { self.lock.withLock { self.stopped } })
          }
        }
        self.lock.withLock { self.results[key] = result }
        self.completed.signal()
      }
    }

    func take(expectedFrameStart: Int, shouldCancel: () -> Bool) throws -> CompressedReadInput {
      while true {
        if let value = lock.withLock({ () -> Result<CompressedReadInput, Error>? in
          guard let value = results.removeValue(forKey: expectedFrameStart) else { return nil }
          pendingCount -= 1
          return value
        }) {
          return try value.get()
        }
        _ = completed.wait(timeout: .now() + .milliseconds(5))
        // The supplied callback is not Sendable and remains on its owner thread.
        if shouldCancel() {
          lock.withLock { stopped = true }
          throw Metal4DSTEMStreamingIOError.cancelled
        }
      }
    }

    func cancelAndDrain() {
      lock.withLock { stopped = true }
      queue.sync(flags: .barrier) {}
      lock.withLock {
        results.removeAll(keepingCapacity: false)
        pendingCount = 0
      }
    }
  }
  /// Single-owner ordered hashing/writing, overlapped with the next GPU window.
  /// A slot is released only after all reads from that window have completed.
  final class OutputWriter: @unchecked Sendable {
    private let queue = DispatchQueue(label: "qgpu.original-packing.output", qos: .userInitiated)
    private let lock = NSLock()
    private var failure: Error?
    private let output: FileHandle
    private var raw = SHA256(), low = SHA256()
    private var records: [(UInt64, UInt64, UInt64, UInt64, Data)] = []
    private var moments = Data()
    private var hashing = 0.0, writing = 0.0

    init(output: FileHandle) { self.output = output }
    func drain() { queue.sync {} }
    func check() throws { if let error = lock.withLock({ failure }) { throw error } }

    func submit(
      rawData: Data, lowData: Data?, payload: Data, headers: Data,
      momentData: Data, release: DispatchSemaphore
    ) {
      queue.async {
        defer { release.signal() }
        do {
          try self.check()
          let hashStarted = CFAbsoluteTimeGetCurrent()
          self.raw.update(data: rawData)
          if let lowData { self.low.update(data: lowData) }
          self.hashing += CFAbsoluteTimeGetCurrent() - hashStarted
          let writeStarted = CFAbsoluteTimeGetCurrent()
          let payloadOffset = try self.output.offset()
          try self.output.write(contentsOf: payload)
          let headerOffset = try self.output.offset()
          try self.output.write(contentsOf: headers)
          self.records.append(
            (
              payloadOffset, UInt64(payload.count), headerOffset,
              UInt64(headers.count), Data(SHA256.hash(data: payload))
            ))
          self.moments.append(momentData)
          self.writing += CFAbsoluteTimeGetCurrent() - writeStarted
        } catch { self.lock.withLock { if self.failure == nil { self.failure = error } } }
      }
    }

    func finish() throws -> (
      records: [(UInt64, UInt64, UInt64, UInt64, Data)], moments: Data,
      raw: String, low: String, hashing: Double, writing: Double
    ) {
      try queue.sync {
        try check()
        return (
          records, moments,
          raw.finalize().map { String(format: "%02x", $0) }.joined(),
          low.finalize().map { String(format: "%02x", $0) }.joined(), hashing, writing
        )
      }
    }
  }
  /// Read only the compressed blocks covering the requested scan slice.
  func compressedReadPlan(
    _ slice: Native4DSTEMIndexedSlice,
    source: Native4DSTEMIndexedSource
  ) throws -> CompressedReadPlan {
    let shard = source.shards[slice.shardIndex]
    let range = slice.chunkCompressedByteRange
    var words = Array(shard.index.metadataWords[slice.metadataWordRange])
    guard !words.isEmpty, words.count.isMultiple(of: 2) else {
      throw Self.invalid("Original source block metadata is incomplete")
    }
    var first = UInt64.max
    var last: UInt64 = 0
    for index in stride(from: 0, to: words.count, by: 2) {
      let offset = UInt64(words[index])
      let end = offset + UInt64(words[index + 1])
      guard words[index + 1] > 0, end <= range.upperBound - range.lowerBound else {
        throw Self.invalid("Original compressed block lies outside its indexed chunk")
      }
      first = min(first, offset)
      last = max(last, end)
    }
    for index in stride(from: 0, to: words.count, by: 2) { words[index] -= UInt32(first) }
    let count = Int(last - first)
    guard count > 0, count <= 512 << 20 else {
      throw Self.invalid("Oversized compressed source chunk; reopen the original acquisition")
    }
    let (offset, overflow) = range.lowerBound.addingReportingOverflow(first)
    guard !overflow, offset <= UInt64(Int64.max), UInt64(count) <= UInt64(Int64.max) - offset else {
      throw Self.invalid("Original compressed slice is outside the readable file range")
    }
    return CompressedReadPlan(
      shardIndex: slice.shardIndex, frameRange: slice.globalFrameRange,
      sourceURL: shard.sourceURL, offset: offset, count: count, words: words)
  }

  static func readCompressed(
    _ plan: CompressedReadPlan, device: MTLDevice,
    isCancelled: () -> Bool
  ) throws -> CompressedReadInput {
    let started = CFAbsoluteTimeGetCurrent()
    if isCancelled() { throw Metal4DSTEMStreamingIOError.cancelled }
    let handle = try FileHandle(forReadingFrom: plan.sourceURL)
    defer { try? handle.close() }
    guard plan.count <= device.maxBufferLength,
      let compressed = device.makeBuffer(length: plan.count, options: .storageModeShared)
    else { throw invalid("Cannot allocate bounded compressed read-ahead input") }
    try withExtendedLifetime(compressed) {
      var received = 0
      while received < plan.count {
        if isCancelled() { throw Metal4DSTEMStreamingIOError.cancelled }
        let bytes = Darwin.pread(
          handle.fileDescriptor, compressed.contents().advanced(by: received),
          min(plan.count - received, 8 << 20), off_t(plan.offset + UInt64(received)))
        if bytes < 0 {
          if errno == EINTR { continue }
          throw invalid(
            "Cannot read original compressed counts; reconnect the source disk and reopen the acquisition (POSIX \(errno))"
          )
        }
        guard bytes > 0 else {
          throw invalid("Truncated original compressed counts; reopen an intact acquisition")
        }
        received += bytes
      }
      if isCancelled() { throw Metal4DSTEMStreamingIOError.cancelled }
    }
    let readSeconds = CFAbsoluteTimeGetCurrent() - started
    let copyStarted = CFAbsoluteTimeGetCurrent()
    let metadata = try plan.words.withUnsafeBytes { bytes -> MTLBuffer in
      guard bytes.count <= device.maxBufferLength,
        let value = device.makeBuffer(
          bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      else { throw invalid("Cannot allocate compressed read-ahead metadata") }
      return value
    }
    return CompressedReadInput(
      shardIndex: plan.shardIndex, frameRange: plan.frameRange,
      compressed: compressed, metadata: metadata,
      readSeconds: readSeconds, copySeconds: CFAbsoluteTimeGetCurrent() - copyStarted,
      reservedBytes: plan.reservedBytes)
  }

}
