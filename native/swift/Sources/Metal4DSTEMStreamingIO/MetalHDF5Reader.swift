import Foundation
import Metal
import Native4DSTEMIO

/// Sequential bounded reads using the established native GPU HDF5 decoder.
public enum MetalHDF5Reader {
  /// Read each frame once. The buffer is borrowed until the closure returns;
  /// complete dependent GPU work or retain a GPU copy before returning.
  public static func read(
    source: Native4DSTEMIndexedSource, device: MTLDevice,
    maximumFrames: Int = 4096, shouldCancel: () -> Bool = { false },
    consume: (MTLBuffer, Range<Int>) throws -> Void
  ) throws {
    guard maximumFrames > 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("Use a positive maximum frame count.")
    }
    let started = Date.timeIntervalSinceReferenceDate
    let d = source.dataset
    let stamps = try OriginalHDF5Packing.inputStamps(source)
    let packing = try OriginalHDF5Packing(device: device, cachePlans: false)
    let windows = try source.windows(
      maximumDecodedBytes: UInt64(maximumFrames) * source.decodedBytesPerFrame,
      alignToScanRows: true)
    guard let largest = windows.map({ $0.globalFrameRange.count }).max() else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("No indexed frames are available.")
    }
    func buffer(_ bytes: Int) throws -> MTLBuffer {
      guard bytes > 0, bytes <= device.maxBufferLength,
        let value = device.makeBuffer(length: bytes, options: .storageModeShared)
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Cannot allocate an HDF5 region; reduce maximumFrames.")
      }
      return value
    }
    let bytes = largest * Int(source.decodedBytesPerFrame)
    let raw = try buffer(bytes)
    let scratch = try buffer(bytes)
    let mask = try buffer(d.detectorRows * d.detectorCols * 4)
    memset(mask.contents(), 0, mask.length)
    let audit = try buffer(largest * 8)
    let errors = try buffer(4)
    let moments = try buffer(largest * 32)
    var profile = OriginalHDF5Packing.Profile()
    // Reuse the native loader's bounded byte read-ahead; scientific decoding
    // and the consumer remain serialized on this thread.
    let slices = windows.flatMap(\.slices)
    let reader =
      ProcessInfo.processInfo.environment["QUANTEM_GPU_LOAD_REFERENCE"] == "1"
      ? nil : OriginalHDF5Packing.CompressedReadAhead(device: device, depth: 1)
    defer { reader?.cancelAndDrain() }
    var ordinal = 0
    if let reader, let first = slices.first {
      try reader.submit(packing.compressedReadPlan(first, source: source))
    }
    profile.readAheadEnabled = reader != nil
    profile.readAheadDepth = reader == nil ? 0 : 1
    let prepared = Date.timeIntervalSinceReferenceDate
    var consumeSeconds = 0.0
    for window in windows {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      try autoreleasepool {
        for slice in window.slices {
          var input: OriginalHDF5Packing.CompressedReadInput?
          if let reader {
            let beforeRead = Date.timeIntervalSinceReferenceDate
            input = try reader.take(
              expectedFrameStart: slice.globalFrameRange.lowerBound,
              shouldCancel: shouldCancel)
            profile.readWait += Date.timeIntervalSinceReferenceDate - beforeRead
            ordinal += 1
            if ordinal < slices.count {
              let plan = try packing.compressedReadPlan(slices[ordinal], source: source)
              profile.maximumConcurrentInputBytes = max(
                profile.maximumConcurrentInputBytes,
                (input?.reservedBytes ?? 0) + plan.reservedBytes)
              try reader.submit(plan)
            }
          }
          _ = try packing.decodeSlice(
            slice, source: source, firstFrame: window.globalFrameRange.lowerBound,
            dense: raw, mask: mask, audit: audit, scratch: scratch, errors: errors,
            partialDPC: nil, moments: moments, preparedInput: input,
            shouldCancel: shouldCancel, profile: &profile)
        }
        let beforeConsume = Date.timeIntervalSinceReferenceDate
        try consume(raw, window.globalFrameRange)
        consumeSeconds += Date.timeIntervalSinceReferenceDate - beforeConsume
      }
    }
    if ProcessInfo.processInfo.environment["QUANTEM_GPU_LOAD_PROFILE"] == "1" {
      var report = profile.json
      report["setup_seconds"] = prepared - started
      report["consumer_seconds"] = consumeSeconds
      report["total_seconds"] = Date.timeIntervalSinceReferenceDate - started
      print(
        "LOAD_PROFILE " + String(
          data: try JSONSerialization.data(withJSONObject: report, options: .sortedKeys),
          encoding: .utf8)!)
    }
    guard try OriginalHDF5Packing.inputStamps(source) == stamps else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The input changed while reading; reopen it and retry.")
    }
  }
}
