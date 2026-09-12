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
    // The decoder's metadata offsets are relative to this buffer's file range.
    // This is nonzero only for the diagnostic coalesced-read path.
    let compressedRangeStart: UInt64
    // Physical source accounting is assigned to the first input in a batch so
    // Profile does not double-count a shared compressed arena.
    let readBytes: UInt64
    let coalescedBatchSlices: Int
    let coalescedGapBytes: UInt64
  }

  /// Diagnostic zero-copy owner for one indexed compressed range. The mapping
  /// is retained by the Metal buffer's deallocator until every GPU consumer
  /// has released that input.
  private final class MappedCompressedRange: @unchecked Sendable {
    let address: UnsafeMutableRawPointer
    let length: Int

    init(plan: CompressedReadPlan) throws {
      let descriptor = plan.sourceURL.path.withCString { Darwin.open($0, O_RDONLY) }
      guard descriptor >= 0 else {
        throw invalid(
          "Could not map original compressed counts; reconnect the source disk and reopen the acquisition"
        )
      }
      let page = max(4_096, Int(getpagesize()))
      let alignedOffset = plan.offset - plan.offset % UInt64(page)
      let delta = Int(plan.offset - alignedOffset)
      let mapLength = delta + plan.count
      guard alignedOffset <= UInt64(Int64.max), mapLength > 0 else {
        Darwin.close(descriptor)
        throw invalid("Original compressed mapping range is outside the readable file range")
      }
      guard let mapped = Darwin.mmap(
        nil, mapLength, PROT_READ, MAP_PRIVATE, descriptor, off_t(alignedOffset)),
        mapped != MAP_FAILED
      else {
        Darwin.close(descriptor)
        throw invalid(
          "Could not map original compressed counts; reconnect the source disk and reopen the acquisition"
        )
      }
      Darwin.close(descriptor)
      address = mapped
      length = mapLength
    }

    deinit { Darwin.munmap(address, length) }
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
    private let coalesce: Bool
    private let coalesceBatchSize: Int
    private var stopped = false
    private var pendingCount = 0
    private var results: [Int: Result<CompressedReadInput, Error>] = [:]
    private var pendingBatch: [CompressedReadPlan] = []
    private var descriptors: [String: Int32] = [:]

    init(device: MTLDevice, depth: Int = 2, coalesce: Bool = false, coalesceBatchSize: Int = 2) {
      self.device = device
      self.depth = max(1, depth)
      self.coalesce = coalesce
      self.coalesceBatchSize = min(4, max(1, coalesceBatchSize))
    }

    func submit(_ plan: CompressedReadPlan) throws {
      let key = plan.frameRange.lowerBound
      var batchToLaunch: [CompressedReadPlan]?
      try lock.withLock {
        guard !stopped, pendingCount < depth, results[key] == nil else {
          throw OriginalHDF5Packing.invalid("Compressed read-ahead ownership is invalid")
        }
        pendingCount += 1
        guard coalesce else {
          batchToLaunch = [plan]
          return
        }
        if pendingBatch.isEmpty {
          pendingBatch = [plan]
        } else if canAppend(plan, to: pendingBatch) {
          pendingBatch.append(plan)
          if pendingBatch.count == coalesceBatchSize {
            batchToLaunch = pendingBatch
            pendingBatch.removeAll(keepingCapacity: true)
          }
        } else {
          batchToLaunch = pendingBatch
          pendingBatch = [plan]
        }
      }
      if let batchToLaunch {
        launch(batchToLaunch)
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
        // A final or cross-file partial batch is held until a consumer asks
        // for it. Flush only when the requested slice is in that batch; this
        // preserves coalescing without allowing the ordered consumer to wait
        // forever for a batch that never reaches its size limit.
        var batchToLaunch: [CompressedReadPlan]?
        lock.withLock {
          if let pendingIndex = pendingBatch.firstIndex(where: {
            $0.frameRange.lowerBound == expectedFrameStart
          }), pendingIndex >= 0 {
            batchToLaunch = pendingBatch
            pendingBatch.removeAll(keepingCapacity: true)
          }
        }
        if let batchToLaunch {
          launch(batchToLaunch)
          continue
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
      lock.withLock {
        stopped = true
        pendingBatch.removeAll(keepingCapacity: false)
      }
      queue.sync(flags: .barrier) {}
      let descriptorsToClose = lock.withLock { () -> [Int32] in
        results.removeAll(keepingCapacity: false)
        pendingCount = 0
        let values = Array(descriptors.values)
        descriptors.removeAll(keepingCapacity: false)
        return values
      }
      for descriptor in descriptorsToClose { Darwin.close(descriptor) }
    }

    private func canAppend(
      _ plan: CompressedReadPlan, to batch: [CompressedReadPlan]
    ) -> Bool {
      guard batch.count < coalesceBatchSize, let last = batch.last,
        last.sourceURL == plan.sourceURL
      else { return false }
      let (lastEnd, lastOverflow) = last.offset.addingReportingOverflow(UInt64(last.count))
      let (end, endOverflow) = plan.offset.addingReportingOverflow(UInt64(plan.count))
      guard !lastOverflow, !endOverflow, plan.offset >= lastEnd else { return false }
      let base = batch[0].offset
      return end >= base && end - base <= UInt64(64 << 20)
    }

    private func launch(_ plans: [CompressedReadPlan]) {
      guard !plans.isEmpty else { return }
      queue.async {
        let result: Result<[CompressedReadInput], Error> = Result {
          try autoreleasepool {
            if self.coalesce {
              let descriptor = try self.descriptor(for: plans[0].sourceURL)
              return try OriginalHDF5Packing.readCompressedBatch(
                plans, device: self.device, fileDescriptor: descriptor,
                isCancelled: { self.lock.withLock { self.stopped } })
            }
            return try plans.map { plan in
              return try OriginalHDF5Packing.readCompressed(
                plan, device: self.device,
                isCancelled: { self.lock.withLock { self.stopped } })
            }
          }
        }
        self.lock.withLock {
          switch result {
          case .success(let inputs):
            for input in inputs { self.results[input.frameRange.lowerBound] = .success(input) }
          case .failure(let error):
            for plan in plans {
              self.results[plan.frameRange.lowerBound] = .failure(error)
            }
          }
        }
        for _ in plans { self.completed.signal() }
      }
    }

    private func descriptor(for url: URL) throws -> Int32 {
      let key = url.path
      if let descriptor = lock.withLock({ descriptors[key] }) { return descriptor }
      let descriptor = key.withCString { Darwin.open($0, O_RDONLY | O_CLOEXEC) }
      guard descriptor >= 0 else {
        throw OriginalHDF5Packing.invalid(
          "Could not open original compressed counts; reconnect the source disk and reopen the acquisition"
        )
      }
      return lock.withLock {
        if let existing = descriptors[key] {
          Darwin.close(descriptor)
          return existing
        }
        descriptors[key] = descriptor
        return descriptor
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
    #if QGPU_PACKING_DIAGNOSTICS
      if ProcessInfo.processInfo.environment["QGPU_ORIGINAL_MMAP_INPUT"] == "1" {
        return try readCompressedMapped(plan, device: device, isCancelled: isCancelled)
      }
    #endif
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
          min(plan.count - received, 64 << 20), off_t(plan.offset + UInt64(received)))
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
      reservedBytes: plan.reservedBytes, compressedRangeStart: 0,
      readBytes: UInt64(plan.count), coalescedBatchSlices: 0, coalescedGapBytes: 0)
  }

  /// Read a small ordered group of adjacent ranges into one Metal arena.
  /// Metadata remains per-slice because its offsets are slice-relative; the
  /// decoder receives the arena displacement through buffer 2.
  static func readCompressedBatch(
    _ plans: [CompressedReadPlan], device: MTLDevice, fileDescriptor: Int32,
    isCancelled: () -> Bool
  ) throws -> [CompressedReadInput] {
    guard !plans.isEmpty, plans.count <= 4,
      plans.allSatisfy({ $0.sourceURL == plans[0].sourceURL })
    else { throw invalid("Invalid coalesced original compressed read batch") }
    let base = plans[0].offset
    var end = base
    var payloadBytes: UInt64 = 0
    var previousEnd = base
    for plan in plans {
      guard plan.offset >= previousEnd else {
        throw invalid("Overlapping original compressed read batch")
      }
      let (planEnd, overflow) = plan.offset.addingReportingOverflow(UInt64(plan.count))
      guard !overflow else { throw invalid("Original compressed read overflows") }
      end = max(end, planEnd)
      payloadBytes += UInt64(plan.count)
      previousEnd = planEnd
    }
    let span = end - base
    guard span > 0, span <= UInt64(64 << 20), span <= UInt64(Int.max) else {
      throw invalid("Oversized coalesced original compressed read")
    }
    let started = CFAbsoluteTimeGetCurrent()
    guard !isCancelled() else { throw Metal4DSTEMStreamingIOError.cancelled }
    guard let compressed = device.makeBuffer(
      length: Int(span), options: .storageModeShared)
    else { throw invalid("Cannot allocate coalesced compressed read-ahead input") }
    try withExtendedLifetime(compressed) {
      var received = 0
      while received < Int(span) {
        if isCancelled() { throw Metal4DSTEMStreamingIOError.cancelled }
        let bytes = Darwin.pread(
          fileDescriptor, compressed.contents().advanced(by: received),
          min(Int(span) - received, 64 << 20), off_t(base + UInt64(received)))
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
    }
    let readSeconds = CFAbsoluteTimeGetCurrent() - started
    let gapBytes = span >= payloadBytes ? span - payloadBytes : 0
    return try plans.enumerated().map { index, plan in
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
        readSeconds: index == 0 ? readSeconds : 0,
        copySeconds: CFAbsoluteTimeGetCurrent() - copyStarted,
        reservedBytes: plan.reservedBytes,
        compressedRangeStart: plan.offset - base,
        readBytes: index == 0 ? span : 0,
        coalescedBatchSlices: index == 0 ? plans.count : 0,
        coalescedGapBytes: index == 0 ? gapBytes : 0)
    }
  }

  #if QGPU_PACKING_DIAGNOSTICS
    private static func readCompressedMapped(
      _ plan: CompressedReadPlan, device: MTLDevice,
      isCancelled: () -> Bool
    ) throws -> CompressedReadInput {
      let started = CFAbsoluteTimeGetCurrent()
      guard !isCancelled() else { throw Metal4DSTEMStreamingIOError.cancelled }
      guard plan.count <= device.maxBufferLength else {
        throw invalid("Cannot allocate bounded compressed mapping")
      }
      let mapped = try MappedCompressedRange(plan: plan)
      let page = max(4_096, Int(getpagesize()))
      let delta = Int(plan.offset % UInt64(page))
      guard let compressed = device.makeBuffer(
        bytesNoCopy: mapped.address.advanced(by: delta), length: plan.count,
        options: .storageModeShared, deallocator: { _, _ in _ = mapped }
      ) else { throw invalid("Cannot create mapped compressed Metal input") }
      guard !isCancelled() else { throw Metal4DSTEMStreamingIOError.cancelled }
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
        readSeconds: CFAbsoluteTimeGetCurrent() - started, copySeconds: 0,
        reservedBytes: plan.reservedBytes, compressedRangeStart: 0,
        readBytes: UInt64(plan.count), coalescedBatchSlices: 0, coalescedGapBytes: 0)
    }
  #endif

}
