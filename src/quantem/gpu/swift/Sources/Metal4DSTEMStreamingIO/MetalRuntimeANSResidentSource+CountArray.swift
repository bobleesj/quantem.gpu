import Darwin
import Foundation
import Metal
import Native4DSTEMIO

extension MetalRuntimeANSResidentSource {
  /// Encode contiguous native counts with the same bounded Metal path as DM4.
  /// Example: `try load(array: NativeNPYSource(url: url), device: device)`.
  public static func load(array camera: any NativeCountArray, device: MTLDevice,
                          maximumAdditionalBytes: UInt64? = nil,
                          shouldCancel: () -> Bool = { false },
                          progress: (Int, Int) -> Void = { _, _ in }) throws -> MetalRuntimeANSResidentSource {
    let started = CFAbsoluteTimeGetCurrent(), shape = camera.shape
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 && $0 < 1 << 20 }),
      ["uint8", "uint16"].contains(camera.dataset.sourceDtype),
      shape == [camera.dataset.scanRows, camera.dataset.scanCols, camera.dataset.detectorRows, camera.dataset.detectorCols],
      camera.dataOffset >= 0 else {
      throw invalid("Expected a contiguous 4D uint8/uint16 array matching its metadata.")
    }
    let scans = shape[0] * shape[1], pixels = shape[2] * shape[3]
    let itemBytes = camera.dataset.sourceDtype == "uint8" ? 1 : 2
    guard scans < Int(UInt32.max), pixels < Int(UInt32.max),
      scans <= Int.max / pixels / itemBytes,
      UInt64(pixels) * (itemBytes == 1 ? 255 : 65_535) <= UInt64(UInt32.max),
      let identity = camera.dataset.sourceIdentitySHA256 else {
      throw invalid("This array exceeds native count/product storage or lacks a source identity. Use a supported source reader; large detectors require exact UInt64 detector sums.")
    }
    let encoder = try RuntimeANSEncoder(device: device, pixels: pixels, bytesPerValue: itemBytes,
      allocatedBefore: UInt64(device.currentAllocatedSize), maximumAdditionalBytes: maximumAdditionalBytes,
      singleFrameQueries: true)
    let index = try RuntimeSpatialIndex(device: device, shape: Array(shape[2...]),
                                       validity: [UInt8](repeating: 1, count: pixels))
    let raw = try sharedBuffer(device: device, bytes: min(512, scans) * pixels * itemBytes, label: "native count staging")
    let fd = open(camera.url.path, O_RDONLY)
    guard fd >= 0 else { throw invalid("Cannot open the detector payload: \(camera.url.lastPathComponent).") }
    defer { close(fd) }
    for first in stride(from: 0, to: scans, by: 512) {
      guard !shouldCancel() else { throw invalid("Count loading cancelled.") }
      let count = min(512, scans - first), bytes = count * pixels * itemBytes
      var read = 0
      while read < bytes {
        let got = pread(fd, raw.contents().advanced(by: read), bytes - read,
                        off_t(camera.dataOffset + first * pixels * itemBytes + read))
        guard got > 0 else { throw invalid("Incomplete detector payload; finish copying the file.") }
        read += got
      }
      try encoder.append(dense: raw, firstScan: first, scanCount: count)
      try encoder.addSpatialIndex(index.build(raw: raw, scans: count, itemBytes: itemBytes))
      progress(first + count, scans)
    }
    try camera.assertUnchanged()
    return try MetalRuntimeANSResidentSource(dataset: camera.dataset,
      identity: identity, built: encoder.finish(),
      totalSeconds: CFAbsoluteTimeGetCurrent() - started, device: device)
  }
}
