import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

extension MetalImageOperations {
  /// Load the original files through the existing native indexed HDF5 reader.
  public func loadEncoded(files: [URL], indexDirectory: URL) throws -> [MetalEncodedSource] {
    let catalog = Native4DSTEMCatalogBuilder(cacheDirectory: indexDirectory)
    return try files.map { file in
      let prepared = try catalog.prepare(input: file)
      guard prepared.datasets.count == 1 else {
        throw Self.invalid("Each file must identify one 4D acquisition.")
      }
      return try MetalEncodedSource.load(
        source: Native4DSTEMIndexedSource.open(dataset: prepared.datasets[0]), device: device)
    }
  }
  public func interiorWindow(rows: Int, columns: Int) throws -> GPUImage {
    let result = try allocate(rows, columns)
    try run(
      "interior_window", [result.buffer], words: [UInt32(rows), UInt32(columns)],
      count: rows * columns)
    return result
  }
  /// Translate a scan mask by the exact requested displacement, with zero exterior.
  public func shiftedScanMask(_ image: GPUImage, shifts: GPUImage, index: Int) throws -> GPUImage {
    let result = try allocate(image.rows, image.columns)
    try run(
      "shift_scan_mask", [image.buffer, shifts.buffer, result.buffer],
      words: [UInt32(image.rows), UInt32(image.columns), UInt32(index), 0],
      count: image.rows * image.columns)
    return result
  }
  public func uncoveredWeight(_ weights: [GPUImage]) throws -> GPUImage {
    let first = weights[0]
    let result = try image(rows: first.rows, columns: first.columns)
    for weight in weights {
      try run(
        "add_image", [weight.buffer, result.buffer], words: [UInt32(first.rows * first.columns)],
        count: first.rows * first.columns)
    }
    try run(
      "complement_clamp", [result.buffer], words: [UInt32(first.rows * first.columns)],
      count: first.rows * first.columns)
    return result
  }

  /// Accumulate one translated 4D contribution using caller-supplied weights.
  /// The caller defines the combination law and owns both accumulators.
  public func accumulateTranslated(
    source: MetalEncodedSource, outputRows: Range<Int>, scanShifts: GPUImage,
    detectorShifts: GPUImage, index: Int, scanWeight: GPUImage, detectorWeight: GPUImage,
    numerator: GPUImage, denominator: GPUImage
  ) throws {
    let shape = source.shape
    let columns = shape[1]
    let pixels = shape[2] * shape[3]
    // A two-scalar readback chooses the I/O window; interpolation is GPU-only.
    let shiftRow = scanShifts.buffer.contents().load(fromByteOffset: index * 8, as: Float.self)
    let delta = Int(floor(-shiftRow))
    let first = max(0, outputRows.lowerBound + delta)
    let stop = min(shape[0], outputRows.upperBound + delta + 1)
    if first >= stop { return }
    let raw = try source.read((first * columns)..<(stop * columns))
    let p = [
      shape[0], columns, shape[2], shape[3], outputRows.lowerBound, first, stop,
      source.itemBytes, outputRows.count, index,
    ].map { UInt32(bitPattern: Int32($0)) }
    try run(
      "sample_accumulate",
      [
        raw, scanWeight.buffer, detectorWeight.buffer,
        scanShifts.buffer, detectorShifts.buffer, numerator.buffer, denominator.buffer,
      ],
      words: p, count: outputRows.count * columns * pixels)
  }
  public func finishWeighted(_ numerator: GPUImage, denominator: GPUImage, uncovered: GPUImage)
    throws -> GPUImage
  {
    try run(
      "weighted_finish", [numerator.buffer, denominator.buffer, uncovered.buffer],
      words: [
        UInt32(numerator.rows * numerator.columns), UInt32(uncovered.rows * uncovered.columns),
      ], count: numerator.rows * numerator.columns)
    return numerator
  }
}
