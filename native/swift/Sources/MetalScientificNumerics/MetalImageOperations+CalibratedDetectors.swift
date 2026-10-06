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

/// Per-frame radial sums of a calibrated packed source around one detector
/// center. Built in one traversal; every annulus image afterwards is a range
/// sum over bins and takes milliseconds instead of a full traversal.
public final class RadialProfileBank {
  public let frames: Int
  public let binCount: Int
  public let binWidth: Float
  public let centerRow: Float
  public let centerCol: Float
  public let scanRows: Int
  public let scanColumns: Int
  let bins: MTLBuffer

  init(
    frames: Int, binCount: Int, binWidth: Float, centerRow: Float, centerCol: Float,
    scanRows: Int, scanColumns: Int, bins: MTLBuffer
  ) {
    self.frames = frames
    self.binCount = binCount
    self.binWidth = binWidth
    self.centerRow = centerRow
    self.centerCol = centerCol
    self.scanRows = scanRows
    self.scanColumns = scanColumns
    self.bins = bins
  }

  public func matches(centerRow: Float, centerCol: Float) -> Bool {
    abs(self.centerRow - centerRow) < 1e-4 && abs(self.centerCol - centerCol) < 1e-4
  }
}

extension MetalImageOperations {
  /// One traversal of the source producing per-frame ring sums around `center`.
  public func radialProfiles(
    source: MetalPackedSource, centerRow: Float, centerCol: Float, binWidth: Float = 0.5,
    shouldCancel: () -> Bool = { false }, progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> RadialProfileBank {
    let shape = source.shape
    let rows = shape[2]
    let columns = shape[3]
    let pixels = rows * columns
    guard binWidth > 0, centerRow.isFinite, centerCol.isFinite, !source.isReleased,
      source.readyFrames == shape[0] * shape[1]
    else {
      throw Self.invalid("Radial profiles need a complete calibrated resident and a finite center.")
    }
    var maxRadius: Float = 0
    for (r, c) in [(0, 0), (0, columns - 1), (rows - 1, 0), (rows - 1, columns - 1)] {
      let dr = Float(r) - centerRow
      let dc = Float(c) - centerCol
      maxRadius = max(maxRadius, (dr * dr + dc * dc).squareRoot())
    }
    let binCount = Int(maxRadius / binWidth) + 2
    // CSR layout: pixels grouped by ring so each threadgroup lane sums one ring.
    var binOfPixel = [Int](repeating: 0, count: pixels)
    var counts = [Int](repeating: 0, count: binCount)
    for pixel in 0..<pixels {
      let dr = Float(pixel / columns) - centerRow
      let dc = Float(pixel % columns) - centerCol
      let bin = min(binCount - 1, Int((dr * dr + dc * dc).squareRoot() / binWidth))
      binOfPixel[pixel] = bin
      counts[bin] += 1
    }
    var starts = [UInt32](repeating: 0, count: binCount + 1)
    for bin in 0..<binCount { starts[bin + 1] = starts[bin] + UInt32(counts[bin]) }
    var cursor = Array(starts.dropLast())
    var pixelIndex = [UInt32](repeating: 0, count: pixels)
    for pixel in 0..<pixels {
      let bin = binOfPixel[pixel]
      pixelIndex[Int(cursor[bin])] = UInt32(pixel)
      cursor[bin] += 1
    }
    let startBuffer = try buffer(starts.count * 4)
    starts.withUnsafeBytes {
      startBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
    }
    let indexBuffer = try buffer(pixelIndex.count * 4)
    pixelIndex.withUnsafeBytes {
      indexBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
    }
    let frames = source.readyFrames
    let bins = try buffer(frames * binCount * 4)
    for first in stride(from: 0, to: frames, by: 512) {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      try autoreleasepool {
        let count = min(512, frames - first)
        let values = try source.read(first..<(first + count))
        let command = try makeCommand()
        try run(
          "radial_bins", [values, startBuffer, indexBuffer, bins],
          words: [UInt32(pixels), UInt32(binCount), UInt32(first)], count: count * 128,
          groupSize: 128, command: command)
        try complete(command)
      }
      progress(min(first + 512, frames), frames)
    }
    return RadialProfileBank(
      frames: frames, binCount: binCount, binWidth: binWidth,
      centerRow: centerRow, centerCol: centerCol, scanRows: shape[0], scanColumns: shape[1],
      bins: bins)
  }

  /// Annulus image from a bank: rings whose radius range lies inside
  /// [innerRadius, outerRadius). Accurate to half a detector pixel at the two
  /// edges; callers refine with the exact path when interaction ends.
  public func annulusImage(bank: RadialProfileBank, innerRadius: Float, outerRadius: Float) throws
    -> GPUImage
  {
    guard innerRadius.isFinite, outerRadius.isFinite, outerRadius > innerRadius
    else { throw Self.invalid("Annulus needs finite radii with outer above inner.") }
    let result = try image(rows: bank.scanRows, columns: bank.scanColumns)
    let firstBin = min(bank.binCount, Int((max(0, innerRadius) / bank.binWidth).rounded(.up)))
    let endBin = min(
      bank.binCount, max(firstBin, Int((outerRadius / bank.binWidth).rounded(.down))))
    try run(
      "radial_range_sum", [bank.bins, result.buffer],
      words: [UInt32(bank.binCount), UInt32(firstBin), UInt32(endBin), UInt32(bank.frames)],
      count: bank.frames, groupSize: 256)
    return result
  }
}
