import Darwin
import Foundation
import Metal
import Native4DSTEMIO

extension MetalRuntimeANSResidentSource {
  /// Encode contiguous native counts with the same bounded Metal path as DM4.
  /// Example: `try load(array: NativeNPYSource(url: url), device: device)`.
  public static func load(
    array camera: any NativeCountArray, device: MTLDevice,
    maximumAdditionalBytes: UInt64? = nil,
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> MetalRuntimeANSResidentSource {
    let started = CFAbsoluteTimeGetCurrent()
    let shape = camera.shape
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 && $0 < 1 << 20 }),
      ["uint8", "uint16"].contains(camera.dataset.sourceDtype),
      shape == [
        camera.dataset.scanRows, camera.dataset.scanCols, camera.dataset.detectorRows,
        camera.dataset.detectorCols,
      ],
      camera.dataOffset >= 0
    else {
      throw invalid("Expected a contiguous 4D uint8/uint16 array matching its metadata.")
    }
    let scans = shape[0] * shape[1]
    let pixels = shape[2] * shape[3]
    let itemBytes = camera.dataset.sourceDtype == "uint8" ? 1 : 2
    guard scans < Int(UInt32.max), pixels < Int(UInt32.max),
      scans <= Int.max / pixels / itemBytes,
      UInt64(pixels) * (itemBytes == 1 ? 255 : 65_535) <= UInt64(UInt32.max),
      let identity = camera.dataset.sourceIdentitySHA256
    else {
      throw invalid(
        "This array exceeds native count/product storage or lacks a source identity. Use a supported source reader; large detectors require exact UInt64 detector sums."
      )
    }
    var profileReadSeconds = 0.0
    var profileReadBytes: UInt64 = 0
    var profileReadCalls = 0
    var profileEncodeSeconds = 0.0
    var profileIndexSeconds = 0.0
    var profileWindows = 0
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    let encoder = try RuntimeANSEncoder(
      device: device, pixels: pixels, bytesPerValue: itemBytes,
      allocatedBefore: allocatedBefore,
      maximumAdditionalBytes: maximumAdditionalBytes,
      singleFrameQueries: true)
    let index = try RuntimeSpatialIndex(
      device: device, shape: Array(shape[2...]),
      validity: [UInt8](repeating: 1, count: pixels))
    let windowBytes = min(512, scans) * pixels * itemBytes
    let stagingA = try sharedBuffer(
      device: device, bytes: windowBytes, label: "native count staging")
    let fd = open(camera.url.path, O_RDONLY)
    guard fd >= 0 else {
      throw invalid("Cannot open the detector payload: \(camera.url.lastPathComponent).")
    }
    defer { close(fd) }
    let incomplete = "Incomplete detector payload; finish copying the file."
    let windowStride = 512

    /// Fill one window exactly, as the single-buffer loader always did.
    func readWindow(_ first: Int, into buffer: MTLBuffer) throws {
      let count = min(windowStride, scans - first)
      let bytes = count * pixels * itemBytes
      let target = buffer.contents()
      let offset = camera.dataOffset + first * pixels * itemBytes
      let started = CFAbsoluteTimeGetCurrent()
      var read = 0
      var calls = 0
      while read < bytes {
        let got = pread(fd, target.advanced(by: read), bytes - read, off_t(offset + read))
        guard got > 0 else { throw invalid(incomplete) }
        read += got
        calls += 1
      }
      profileReadSeconds += CFAbsoluteTimeGetCurrent() - started
      profileReadBytes += UInt64(bytes)
      profileReadCalls += calls
    }

    func encodeWindow(_ first: Int, _ raw: MTLBuffer) throws {
      let count = min(windowStride, scans - first)
      let encodeStarted = CFAbsoluteTimeGetCurrent()
      try encoder.append(dense: raw, firstScan: first, scanCount: count)
      profileEncodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStarted
      let indexStarted = CFAbsoluteTimeGetCurrent()
      try encoder.addSpatialIndex(index.build(raw: raw, scans: count, itemBytes: itemBytes))
      profileIndexSeconds += CFAbsoluteTimeGetCurrent() - indexStarted
      profileWindows += 1
      progress(first + count, scans)
    }

    // A second staging buffer lets the CPU pread window N+1 while the GPU
    // encodes and indexes window N, which touches only the first buffer. The
    // extra buffer is admitted against the same bounded budget as the encoder:
    // when it does not fit, the loader keeps the original single-buffer order
    // rather than growing past its limit.
    var stagingB: MTLBuffer?
    let overlapRequested =
      ProcessInfo.processInfo.environment["QGPU_COUNT_READ_OVERLAP"] != "0" && scans > windowStride
    if overlapRequested {
      let absoluteLimit = maximumAdditionalBytes.map { $0 + allocatedBefore }
      let afterSecond = UInt64(device.currentAllocatedSize) + UInt64(windowBytes)
      if absoluteLimit.map({ afterSecond <= $0 }) ?? true {
        stagingB = try sharedBuffer(
          device: device, bytes: windowBytes, label: "native count staging b")
      }
    }

    if let stagingB {
      // One pread is in flight at a time; the semaphore also publishes a short
      // read to the loading thread before its buffer is encoded.
      let queued = DispatchQueue(label: "org.quantem.gpu.count-read", qos: .userInitiated)
      let status = UnsafeReadStatus()
      let incompleteWindow = incomplete
      func readAhead(_ first: Int, into buffer: MTLBuffer) {
        let count = min(windowStride, scans - first)
        let bytes = count * pixels * itemBytes
        let target = buffer.contents()
        let offset = camera.dataOffset + first * pixels * itemBytes
        // Cancellation is still sampled on the loading thread; a cancel is
        // honoured at the next window boundary exactly as it was before.
        let cancelled = shouldCancel()
        let done = DispatchSemaphore(value: 0)
        status.begin(done)
        queued.async {
          defer { done.signal() }
          let started = CFAbsoluteTimeGetCurrent()
          var read = 0
          var calls = 0
          if !cancelled {
            while read < bytes {
              let got = pread(fd, target.advanced(by: read), bytes - read, off_t(offset + read))
              guard got > 0 else { return status.fail() }
              read += got
              calls += 1
            }
          }
          status.finish(
            seconds: CFAbsoluteTimeGetCurrent() - started, bytes: UInt64(read), calls: calls)
        }
      }

      var stagingIndex = 0
      let staging = [stagingA, stagingB]
      var iterator = stride(from: 0, to: scans, by: windowStride).makeIterator()
      var current = iterator.next()
      if let first = current {
        readAhead(first, into: staging[0])
        status.wait()
        if status.failed { throw invalid(incompleteWindow) }
      }
      while let first = current {
        let next = iterator.next()
        if let next { readAhead(next, into: staging[1 - stagingIndex]) }
        guard !shouldCancel() else { throw invalid("Count loading cancelled.") }
        try encodeWindow(first, staging[stagingIndex])
        if next != nil {
          // The next window must be complete before its buffer is encoded, so
          // a short read still fails the load instead of publishing counts.
          status.wait()
          if status.failed { throw invalid(incompleteWindow) }
        }
        stagingIndex = 1 - stagingIndex
        current = next
      }
      profileReadSeconds += status.seconds
      profileReadBytes += status.bytes
      profileReadCalls += status.calls
    } else {
      let staging = stagingA
      for first in stride(from: 0, to: scans, by: windowStride) {
        guard !shouldCancel() else { throw invalid("Count loading cancelled.") }
        try readWindow(first, into: staging)
        try encodeWindow(first, staging)
      }
    }
    try camera.assertUnchanged()
    let built = try encoder.finish()
    if ProcessInfo.processInfo.environment["QGPU_RUNTIME_ANS_PROFILE"] == "1" {
      let record: [String: Any] = [
        "record": "runtime_ans_count_array_profile",
        "windows": profileWindows,
        "read_seconds": profileReadSeconds,
        "read_bytes": profileReadBytes,
        "read_calls": profileReadCalls,
        "append_seconds": profileEncodeSeconds,
        "spatial_index_seconds": profileIndexSeconds,
        "index_build_calls": index.buildCalls,
        "index_allocation_seconds": index.allocationSeconds,
        "index_reduction_wall_seconds": index.reductionWallSeconds,
        "index_reduction_gpu_seconds": index.reductionGPUSeconds,
        "index_cpu_prefix_seconds": index.prefixCPUSeconds,
        "index_pack_wall_seconds": index.packWallSeconds,
        "index_pack_gpu_seconds": index.packGPUSeconds,
        "index_payload_bytes": index.packPayloadBytes,
      ]
      if let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
        let line = String(data: data, encoding: .utf8)
      {
        FileHandle.standardError.write(Data(("QGPU_COUNT_ARRAY_PROFILE " + line + "\n").utf8))
      }
    }
    RuntimeANSEncoder.logProfile(
      built, logicalBytes: UInt64(scans) * UInt64(pixels) * UInt64(itemBytes))
    return try MetalRuntimeANSResidentSource(
      dataset: camera.dataset,
      identity: identity, built: built,
      totalSeconds: CFAbsoluteTimeGetCurrent() - started, device: device)
  }
}

/// Cross-thread read accounting and failure publication for the overlapped loader.
private final class UnsafeReadStatus: @unchecked Sendable {
  private let lock = NSLock()
  private var failedRead = false
  private var totalSeconds = 0.0
  private var totalBytes: UInt64 = 0
  private var totalCalls = 0
  private var completion = DispatchSemaphore(value: 0)

  func begin(_ semaphore: DispatchSemaphore) {
    lock.lock()
    completion = semaphore
    lock.unlock()
  }
  func fail() {
    lock.lock()
    failedRead = true
    lock.unlock()
  }
  func finish(seconds: Double, bytes: UInt64, calls: Int) {
    lock.lock()
    totalSeconds += seconds
    totalBytes += bytes
    totalCalls += calls
    lock.unlock()
  }
  func wait() {
    lock.lock()
    let semaphore = completion
    lock.unlock()
    semaphore.wait()
  }
  var failed: Bool { lock.lock(); defer { lock.unlock() }; return failedRead }
  var seconds: Double { lock.lock(); defer { lock.unlock() }; return totalSeconds }
  var bytes: UInt64 { lock.lock(); defer { lock.unlock() }; return totalBytes }
  var calls: Int { lock.lock(); defer { lock.unlock() }; return totalCalls }
}
