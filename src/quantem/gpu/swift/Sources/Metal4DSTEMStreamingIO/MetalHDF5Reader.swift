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
    for window in windows {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      try autoreleasepool {
        for slice in window.slices {
          _ = try packing.decodeSlice(
            slice, source: source, firstFrame: window.globalFrameRange.lowerBound,
            dense: raw, mask: mask, audit: audit, scratch: scratch, errors: errors,
            partialDPC: nil, moments: moments, shouldCancel: shouldCancel, profile: &profile)
        }
        try consume(raw, window.globalFrameRange)
      }
    }
    guard try OriginalHDF5Packing.inputStamps(source) == stamps else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The input changed while reading; reopen it and retry.")
    }
  }
}
