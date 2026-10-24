import CryptoKit
import Darwin
import Foundation
import Metal
import Native4DSTEMIO

extension MetalRuntimeANSResidentSource {
  /// Save unchanged encoded counts and indexes as a portable `.qem` file.
  ///
  /// Counts are stored as-is; nothing is re-encoded. The destination is published
  /// atomically and never replaces an existing file. Call from a worker while the
  /// source remains exclusively owned and live.
  /// Example: `try source.saveSnapshot(to: destination)`.
  public func saveSnapshot(
    to destination: URL,
    calibrationOverrides: NativeQEMCalibration.Overrides? = nil,
    sourceDocuments: [NativeMetadataDocument] = [],
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws {
    try requireLive()
    let profiling = ProcessInfo.processInfo.environment["QGPU_RUNTIME_ANS_PROFILE"] == "1"
    let saveStarted = profiling ? CFAbsoluteTimeGetCurrent() : 0
    var blitSeconds = 0.0
    let overrides =
      try calibrationOverrides ?? NativeQEMCalibration.read(metadata: dataset.metadata ?? [:])
    try NativeQEMCalibration.validate(overrides)
    guard destination.pathExtension.lowercased() == "qem" else {
      throw Self.invalid(
        "\(Self.self) writes .qem files only; nothing else carries calibration overrides.")
    }
    guard !chunks.isEmpty, chunks.allSatisfy({ $0.spatial.count == 3 }) else {
      throw Self.invalid(
        "Reload the original with spatial indexes before saving a compressed copy.")
    }
    let manager = FileManager.default
    guard !manager.fileExists(atPath: destination.path) else {
      throw Self.invalid(
        "\(destination.lastPathComponent) already exists. Choose another name; the existing copy was kept."
      )
    }
    let directory = destination.deletingLastPathComponent()
    let outputURL = directory.appendingPathComponent(".\(UUID().uuidString).partial")
    defer { try? manager.removeItem(at: outputURL) }
    let blockBytes = 64 << 20

    typealias Readback = (buffers: [MTLBuffer], command: MTLCommandBuffer?)
    func beginReadback(_ buffers: [MTLBuffer], indices: [Int]) throws -> Readback {
      let started = profiling ? CFAbsoluteTimeGetCurrent() : 0
      defer { if profiling { blitSeconds += CFAbsoluteTimeGetCurrent() - started } }
      var readable = buffers
      let privateIndices = indices.filter { buffers[$0].storageMode == .private }
      guard !privateIndices.isEmpty else { return (readable, nil) }
      guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
      else { throw Self.invalid("Cannot read compressed storage for saving.") }
      for index in privateIndices {
        let buffer = buffers[index]
        readable[index] = try Self.sharedBuffer(
          device: device, bytes: buffer.length, label: "compressed file staging")
        blit.copy(
          from: buffer, sourceOffset: 0, to: readable[index], destinationOffset: 0,
          size: buffer.length)
      }
      blit.endEncoding()
      command.commit()
      return (readable, command)
    }
    func finishReadback(_ readback: Readback) throws -> [MTLBuffer] {
      if let command = readback.command {
        let started = profiling ? CFAbsoluteTimeGetCurrent() : 0
        defer { if profiling { blitSeconds += CFAbsoluteTimeGetCurrent() - started } }
        command.waitUntilCompleted()
        guard command.status == .completed else {
          throw Self.invalid("Compressed storage transfer failed; retry saving.")
        }
      }
      return readback.buffers
    }
    func readableBuffers(_ buffers: [MTLBuffer], indices: [Int]) throws -> [MTLBuffer] {
      try finishReadback(beginReadback(buffers, indices: indices))
    }
    var table = [[String: Any]]()
    var cursor = 0
    var chunkLengths = [[Int]]()
    for chunk in chunks {
      try autoreleasepool {
        if shouldCancel() {
          throw Self.invalid("Saving cancelled; no incomplete copy was published.")
        }
        var spans = [[String: Int]]()
        let buffers = [chunk.payload, chunk.offsets, chunk.models] + chunk.spatial
        // Only the offset arrays are needed to plan the exact header size.
        let readable = try readableBuffers(buffers, indices: [1, 4])
        let blocks = (chunk.scanCount + 511) / 512
        let streams = blocks * shape[2] * shape[3]
        let fields =
          ((shape[2] + 7) / 8) * ((shape[3] + 7) / 8)
          + ((shape[2] + 31) / 32) * ((shape[3] + 31) / 32)
        let fieldStreams = blocks * fields
        let lengths = [
          Int(readable[1].contents().assumingMemoryBound(to: UInt32.self)[streams]),
          (streams + 1) * 4, streams,
          Int(readable[4].contents().assumingMemoryBound(to: UInt64.self)[fieldStreams]) * 4,
          (fieldStreams + 1) * 8, fieldStreams,
        ]
        for (index, pair) in zip(readable, [1, 4, 1, 4, 8, 1]).enumerated() {
          let (buffer, itemBytes) = pair
          let length = lengths[index]
          guard length <= buffer.length else {
            throw Self.invalid("Incomplete resident storage; reload the original before saving.")
          }
          let aligned = (cursor + 7) & ~7
          spans.append(["offset": aligned, "count": length / itemBytes])
          cursor = aligned + length
        }
        table.append(["first": chunk.firstScan, "scans": chunk.scanCount, "arrays": spans])
        chunkLengths.append(lengths)
      }
    }
    var packedValid = [UInt8](repeating: 0, count: (validPixels.count + 7) / 8)
    for (pixel, valid) in validPixels.enumerated() where valid != 0 {
      packedValid[pixel / 8] |= 1 << (7 - pixel % 8)
    }
    // The retained block holds provenance only (QEM specification 0.0.4). Calibration, source
    // fields and loader state live once in scientific_metadata; a second copy here is what a
    // reader could take instead of the stored value.
    var metadata: [String: Any] = [
      "source_path": dataset.masterPath ?? "",
      "original_source_identity_sha256": dataset.metadata?["originalSourceIdentity"]
        ?? sourceIdentitySHA256,
      "file_counts_exact": true, "median_correction_applied": false,
    ]
    if let date = dataset.acquisitionDate { metadata["acquisition_date"] = date }
    for name in ["camera_model", "camera_id", "acquisition_processing"] {
      if let value = dataset.metadata?[name] { metadata[name] = value }
    }
    var header: [String: Any] = [
      "version": 1, "profile": "runtime-column-rans-spatial-v2", "interval": 512,
      "shape": shape, "dtype": logicalDtype == .uint8 ? "uint8" : "uint16",
      "valid": packedValid.map { String(format: "%02x", $0) }.joined(),
      "chunks": table, "bytes": cursor,
      "sha256": [String](
        repeating: String(repeating: "0", count: 64), count: (cursor + blockBytes - 1) / blockBytes),
      "metadata": metadata,
    ]
    header["container"] = NativeQEMMetadata.container
    header["container_version"] = 1
    header["codec"] = "runtime-column-rans-spatial-v2"
    header["scientific_metadata"] = try NativeMetadataDocument.adding(
      sourceDocuments,
      to: NativeQEMCalibration.applying(
        overrides,
        to: NativeQEMMetadata.acquisition(dataset)))
    let plannedJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    let planningSeconds = profiling ? CFAbsoluteTimeGetCurrent() - saveStarted : 0
    guard plannedJSON.count <= 16 << 20 else {
      throw Self.invalid("Source metadata exceeds the compressed header limit.")
    }
    guard manager.createFile(atPath: outputURL.path, contents: nil) else {
      throw Self.invalid("Cannot write in \(directory.path). Choose a writable destination.")
    }
    let output = try FileHandle(forWritingTo: outputURL)
    defer { try? output.close() }
    try output.seek(toOffset: UInt64(56 + plannedJSON.count))
    // Keep owned staging blocks alive until both their write and hash finish.
    // At most eight queued blocks plus one producer block use 576 MiB. Tiny
    // files allocate only their body size; the calling save producer can prepare
    // the next block while completed blocks are written and authenticated.
    let bodyBlocks = (cursor + blockBytes - 1) / blockBytes
    let pending = DispatchSemaphore(
      value: min(8, bodyBlocks, ProcessInfo.processInfo.activeProcessorCount))
    let group = DispatchGroup()
    let statusLock = NSLock()
    let descriptor = output.fileDescriptor
    let bodyStart = 56 + plannedJSON.count
    nonisolated(unsafe) var checksums = [String](
      repeating: "", count: (cursor + blockBytes - 1) / blockBytes)
    nonisolated(unsafe) var writeFailure: String?
    nonisolated(unsafe) var hashSeconds = 0.0
    nonisolated(unsafe) var writeSeconds = 0.0
    var blockNumber = 0
    var blockCount = 0
    let stagingCapacity = min(blockBytes, cursor)
    var staging = Data(count: stagingCapacity)
    defer { group.wait() }

    func checkedWriteStatus() throws {
      statusLock.lock()
      let failure = writeFailure
      statusLock.unlock()
      if let failure { throw Self.invalid(failure) }
    }
    func submitBlock() {
      pending.wait()
      if blockCount < staging.count { staging.count = blockCount }
      let bytes = staging
      let number = blockNumber
      blockNumber += 1
      blockCount = 0
      staging = blockNumber < bodyBlocks ? Data(count: stagingCapacity) : Data()
      group.enter()
      DispatchQueue.global(qos: .userInitiated).async {
        defer {
          pending.signal()
          group.leave()
        }
        let hashStarted = profiling ? CFAbsoluteTimeGetCurrent() : 0
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        let hashed = profiling ? CFAbsoluteTimeGetCurrent() : 0
        let failure: String? = bytes.withUnsafeBytes { raw in
          var done = 0
          while done < raw.count {
            let written = pwrite(
              descriptor, raw.baseAddress!.advanced(by: done), raw.count - done,
              off_t(bodyStart + number * blockBytes + done))
            if written < 0 && errno == EINTR { continue }
            if written <= 0 {
              return "Cannot write QEM body: \(String(cString: strerror(errno))). Check free space."
            }
            done += written
          }
          return nil
        }
        let written = profiling ? CFAbsoluteTimeGetCurrent() : 0
        statusLock.lock()
        hashSeconds += hashed - hashStarted
        writeSeconds += written - hashed
        if let failure { writeFailure = writeFailure ?? failure }
        checksums[number] = digest
        statusLock.unlock()
      }
    }
    func append(_ bytes: Data) throws {
      try bytes.withUnsafeBytes { raw in
        var offset = 0
        while offset < raw.count {
          if shouldCancel() {
            throw Self.invalid("Saving cancelled; no incomplete copy was published.")
          }
          try checkedWriteStatus()
          let count = min(blockBytes - blockCount, raw.count - offset)
          _ = staging.withUnsafeMutableBytes { target in
            memcpy(
              target.baseAddress!.advanced(by: blockCount),
              raw.baseAddress!.advanced(by: offset), count)
          }
          offset += count
          blockCount += count
          if blockCount == blockBytes { submitBlock() }
        }
      }
    }
    let producerStarted = profiling ? CFAbsoluteTimeGetCurrent() : 0
    var prefetched: Readback?
    var prefetchedChunks = 0
    // One next chunk may overlap CPU staging and disk workers. Extra shared
    // staging is capped at 128 MiB; larger chunks keep synchronous readback.
    let prefetchByteLimit = 128 << 20
    defer { prefetched?.command?.waitUntilCompleted() }
    var position = 0
    for (number, chunk) in chunks.enumerated() {
      // Private-buffer blits live only through this chunk. append copies each
      // byte into owned blocks before the pool releases its Metal staging.
      try autoreleasepool {
        let buffers = [chunk.payload, chunk.offsets, chunk.models] + chunk.spatial
        let current = try prefetched ?? beginReadback(buffers, indices: Array(buffers.indices))
        prefetched = nil
        let readable = try finishReadback(current)
        if number + 1 < chunks.count {
          let next = chunks[number + 1]
          let nextBuffers = [next.payload, next.offsets, next.models] + next.spatial
          let privateBytes = nextBuffers.reduce(0) {
            $0 + ($1.storageMode == .private ? $1.length : 0)
          }
          if privateBytes > 0 && privateBytes <= prefetchByteLimit {
            prefetched = try autoreleasepool {
              try beginReadback(nextBuffers, indices: Array(nextBuffers.indices))
            }
            prefetchedChunks += 1
          }
        }
        for (buffer, length) in zip(readable, chunkLengths[number]) {
          let aligned = (position + 7) & ~7
          try append(Data(repeating: 0, count: aligned - position))
          try append(Data(bytesNoCopy: buffer.contents(), count: length, deallocator: .none))
          position = aligned + length
        }
        progress(number + 1, chunks.count)
      }
    }
    if blockCount > 0 { submitBlock() }
    let produced = profiling ? CFAbsoluteTimeGetCurrent() : 0
    group.wait()
    let waited = profiling ? CFAbsoluteTimeGetCurrent() : 0
    try checkedWriteStatus()
    header["sha256"] = checksums
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    guard json.count == plannedJSON.count, position == cursor else {
      throw Self.invalid("Compressed header or body size changed while saving; reload and retry.")
    }
    var prefix = NativeQEMMetadata.magic
    for size in [json.count, 56 + json.count] {
      var value = UInt64(size).littleEndian
      withUnsafeBytes(of: &value) { prefix.append(contentsOf: $0) }
    }
    prefix.append(contentsOf: SHA256.hash(data: json))
    prefix.append(json)
    try output.seek(toOffset: 0)
    try output.write(contentsOf: prefix)
    let syncStarted = profiling ? CFAbsoluteTimeGetCurrent() : 0
    try output.synchronize()
    let syncSeconds = profiling ? CFAbsoluteTimeGetCurrent() - syncStarted : 0
    if shouldCancel() { throw Self.invalid("Saving cancelled; no incomplete copy was published.") }
    guard link(outputURL.path, destination.path) == 0 else {
      throw Self.invalid(
        "Cannot publish \(destination.lastPathComponent): \(String(cString: strerror(errno))). Choose another destination."
      )
    }
    if profiling {
      let profile: [String: Any] = [
        "record": "runtime_ans_save_profile", "body_bytes": cursor,
        "planning_seconds": planningSeconds, "private_blit_wall_seconds": blitSeconds,
        "prefetched_private_chunks": prefetchedChunks, "prefetch_byte_limit": prefetchByteLimit,
        "producer_seconds": produced - producerStarted,
        // Worker intervals overlap each other and the producer; these are not wall stages.
        "hash_worker_seconds_sum": hashSeconds, "write_worker_seconds_sum": writeSeconds,
        "final_worker_wait_seconds": waited - produced, "fsync_seconds": syncSeconds,
        "total_seconds": CFAbsoluteTimeGetCurrent() - saveStarted,
      ]
      if let data = try? JSONSerialization.data(withJSONObject: profile, options: [.sortedKeys]),
        let line = String(data: data, encoding: .utf8)
      {
        FileHandle.standardError.write(Data(("QGPU_SAVE_PROFILE " + line + "\n").utf8))
      }
    }
  }
}
