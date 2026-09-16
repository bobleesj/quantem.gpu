import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@available(macOS 15.0, *)
enum CameraBenchmark {
  static func run(_ url: URL) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal device required") }
    let start = CFAbsoluteTimeGetCurrent()
    let source: MetalRuntimeANSResidentSource
    if NativeANSSnapshot.matches(url) {
      source = try .load(snapshot: NativeANSSnapshot(url: url), device: device)
    } else {
      source = try .load(camera: NativeDM4Source(url: url), device: device)
    }
    let loadSeconds = CFAbsoluteTimeGetCurrent() - start
    defer { source.releaseResidentStorage() }
    let series = try MetalRuntimeANSSeries(sources: [source])
    defer { series.release() }
    let shape = source.shape, scans = shape[0] * shape[1], pixels = shape[2] * shape[3]
    let trials = min(1000, max(1, Int(ProcessInfo.processInfo.environment["K3_BENCH_TRIALS"] ?? "30") ?? 30))
    var records = [[String: Any]]()
    for scan in [0, scans / 2, scans - 1] {
      let start = CFAbsoluteTimeGetCurrent()
      let frame = try source.extractRawDiffraction(scanRow: scan / shape[1], scanColumn: scan % shape[1])
      records.append(["scan": scan, "seconds": CFAbsoluteTimeGetCurrent() - start, "sha256_u32": hash(frame)])
    }
    var diffractionMilliseconds = [Double]()
    for trial in 0..<trials {
      let scan = (trial * 7919) % scans
      let started = CFAbsoluteTimeGetCurrent()
      _ = try series.updatePriorityDiffractionBuffer(scanRow: scan / shape[1],
        scanColumn: scan % shape[1], priorityIndex: 0)
      diffractionMilliseconds.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
    }
    var audit: [String: Any] = [:]
    if let path = ProcessInfo.processInfo.environment["K3_AUDIT_DM4"] {
      let original = try NativeDM4Source(url: URL(fileURLWithPath: path))
      guard original.shape == shape else { throw NSError(domain: "CameraAudit", code: 1) }
      let handle = try FileHandle(forReadingFrom: original.url)
      defer { try? handle.close() }
      try handle.seek(toOffset: UInt64(original.dataOffset))
      let itemBytes = original.dataset.sourceDtype == "uint8" ? 1 : 2
      let started = CFAbsoluteTimeGetCurrent()
      var checked = 0
      for scan in 0..<scans {
        let raw = try handle.read(upToCount: pixels * itemBytes) ?? Data()
        guard raw.count == pixels * itemBytes else { throw NSError(domain: "CameraAudit", code: 2) }
        let buffer = try series.updatePriorityDiffractionBuffer(scanRow: scan / shape[1], scanColumn: scan % shape[1], priorityIndex: 0)
        let actual = buffer.contents().assumingMemoryBound(to: UInt32.self)
        let equal = raw.withUnsafeBytes { bytes in
          for pixel in 0..<pixels {
            let expected = itemBytes == 1 ? UInt32(bytes[pixel]) : UInt32(bytes.loadUnaligned(fromByteOffset: pixel * 2, as: UInt16.self).littleEndian)
            if actual[pixel] != expected { return false }
          }
          return true
        }
        guard equal else { throw NSError(domain: "CameraAudit", code: 3, userInfo: [NSLocalizedDescriptionKey: "Native count mismatch at scan \(scan)"]) }
        checked += pixels
        if scan % 512 == 0 { fputs("COUNT_AUDIT \(scan)/\(scans)\n", stderr) }
      }
      audit = ["exact": true, "counts_checked": checked, "seconds": CFAbsoluteTimeGetCurrent() - started, "original": path]
    }
    var queries = [[String: Any]]()
    for (name, inner, outer) in [("BF", 0.0, 126.0), ("ADF", 180.0, 360.0), ("DF", 126.0, 1222.0)] {
      var times = [Double](), gpu = [Double](), resultHash = ""
      var maskMilliseconds = [Double](), requestMilliseconds = [Double]()
      for trial in 0..<trials {
        let requestStarted = CFAbsoluteTimeGetCurrent()
        let row = Double(shape[2] - 1) / 2 + Double(trial) * 0.17
        let col = Double(shape[3] - 1) / 2 + Double(trial) * 0.31
        var mask = [UInt8](repeating: 0, count: pixels)
        for pixel in 0..<pixels {
          let y = Double(pixel / shape[3]) - row, x = Double(pixel % shape[3]) - col
          let squared = y * y + x * x
          mask[pixel] = squared >= inner * inner && squared <= outer * outer ? 1 : 0
        }
        maskMilliseconds.append((CFAbsoluteTimeGetCurrent() - requestStarted) * 1000)
        let result = try series.updatePriorityVirtualDetectorBuffer(mask: mask, priorityIndex: 0)
        requestMilliseconds.append((CFAbsoluteTimeGetCurrent() - requestStarted) * 1000)
        times.append(result.metrics.wallMilliseconds); gpu.append(result.metrics.gpuMilliseconds)
        if trial == trials - 1 {
          resultHash = hash(Array(UnsafeBufferPointer(start: result.buffer.contents().assumingMemoryBound(to: UInt32.self), count: scans)))
        }
      }
      queries.append(["product": name, "wall_ms": times, "gpu_ms": gpu,
        "mask_cpu_ms": maskMilliseconds, "request_ms": requestMilliseconds,
        "last_sha256_u32": resultHash])
    }
    let document: [String: Any] = ["path": url.path, "device": device.name, "shape": shape,
      "dtype": String(describing: source.logicalDtype), "load_seconds": loadSeconds,
      "backend_load_seconds": source.loadMetrics.totalSeconds, "encode_seconds": source.loadMetrics.fusedDecodeAndEncodeSeconds,
      "resident_bytes": source.residentBytes, "metal_allocated_bytes": device.currentAllocatedSize,
      "frames": records, "diffraction_buffer_ms": diffractionMilliseconds,
      "queries": queries, "trials": trials, "count_audit": audit,
      "timing_scope": "synchronized backend calls; no UI presentation; filesystem cache uncontrolled"]
    print(String(decoding: try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys]), as: UTF8.self))
  }
  static func hash(_ values: [UInt32]) -> String {
    values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
  }
}
