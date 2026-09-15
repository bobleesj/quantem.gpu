import Foundation
import Metal
import Metal4DSTEMStreamingIO

extension MetalImageOperations {
  /// Integrate restored scientific intensities without materializing a full 4D array.
  /// Weights use detector (row, column) order; all scan frames participate.
  public func virtualImage(
    source: MetalPackedSource, weights: [Float],
    shouldCancel: () -> Bool = { false }, progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> GPUImage {
    try virtualImages(
      source: source, weights: [weights], shouldCancel: shouldCancel,
      progress: progress)[0]
  }

  /// Integrate several detector masks in one bounded traversal of a calibrated resident.
  ///
  /// Each weight array is detector-row-major. Returned images preserve mask order.
  /// Decoded regions are shared across reductions, never retained as a full 4D array.
  public func virtualImages(
    source: MetalPackedSource, weights: [[Float]],
    shouldCancel: () -> Bool = { false }, progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> [GPUImage] {
    let shape = source.shape
    let pixels = shape[2] * shape[3]
    guard !weights.isEmpty,
      weights.allSatisfy({ $0.count == pixels && $0.allSatisfy(\.isFinite) }),
      !source.isReleased, source.readyFrames == shape[0] * shape[1]
    else {
      throw Self.invalid(
        "Use a complete calibrated resident and one finite weight per detector pixel.")
    }
    let results = try weights.map { _ in try image(rows: shape[0], columns: shape[1]) }
    let masks = try weights.map { try image(values: $0, rows: shape[2], columns: shape[3]) }
    for first in stride(from: 0, to: source.readyFrames, by: 512) {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      try autoreleasepool {
        let count = min(512, source.readyFrames - first)
        let values = try source.read(first..<(first + count))
        let command = try makeCommand()
        for (mask, result) in zip(masks, results) {
          try run(
            "calibrated_detector_sum", [values, mask.buffer, result.buffer],
            words: [UInt32(pixels), UInt32(first)], count: count * 128,
            groupSize: 128, command: command)
        }
        try complete(command)
      }
      progress(min(first + 512, source.readyFrames), source.readyFrames)
    }
    return results
  }

  /// Normalize a detector-weighted image by total intensity, mapping zero totals to zero.
  /// The input images remain unchanged, including their signed values.
  public func normalized(_ numerator: GPUImage, by denominator: GPUImage) throws -> GPUImage {
    guard !numerator.isComplex, !denominator.isComplex,
      numerator.rows == denominator.rows, numerator.columns == denominator.columns,
      numerator.buffer.device.registryID == device.registryID,
      denominator.buffer.device.registryID == device.registryID
    else { throw Self.invalid("Normalize matching real images on the operation's Metal device.") }
    let result = try image(rows: numerator.rows, columns: numerator.columns)
    let zero = try image(rows: 1, columns: 1)
    let command = try makeCommand()
    guard let copy = command.makeBlitCommandEncoder() else {
      throw Self.invalid("Cannot copy the weighted image for normalization.")
    }
    copy.copy(
      from: numerator.buffer, sourceOffset: 0, to: result.buffer,
      destinationOffset: 0, size: numerator.rows * numerator.columns * 4)
    copy.endEncoding()
    try run(
      "weighted_finish", [result.buffer, denominator.buffer, zero.buffer],
      words: [UInt32(numerator.rows * numerator.columns), 1],
      count: numerator.rows * numerator.columns, command: command)
    try complete(command)
    return result
  }
}
