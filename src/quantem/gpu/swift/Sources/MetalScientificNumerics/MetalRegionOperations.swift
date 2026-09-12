import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

struct TranslatedSamplingPlan {
  let signature: [UInt32]
  let scans: MTLBuffer
  let pixels: MTLBuffer
  let scanCoefficients: MTLBuffer
  let detectorCoefficients: MTLBuffer
}

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
    source: any MetalResidentCounts, outputRows: Range<Int>, scanShifts: GPUImage,
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
    if !referenceSampling {
      let scanPair = scanShifts.buffer.contents().assumingMemoryBound(to: Float.self)
      let detectorPair = detectorShifts.buffer.contents().assumingMemoryBound(to: Float.self)
      let signature =
        shape.map(UInt32.init) + [
          scanPair[index * 2].bitPattern, scanPair[index * 2 + 1].bitPattern,
          detectorPair[index * 2].bitPattern, detectorPair[index * 2 + 1].bitPattern,
        ]
      if translatedPlans[index]?.signature != signature {
        let plan = TranslatedSamplingPlan(
          signature: signature, scans: try buffer(shape[0] * columns * 16),
          pixels: try buffer(pixels * 16), scanCoefficients: try buffer(16),
          detectorCoefficients: try buffer(pixels * 16))
        try run(
          "translated_scan_plan", [scanShifts.buffer, plan.scans, plan.scanCoefficients],
          words: [UInt32(shape[0]), UInt32(columns), UInt32(index), 0], count: shape[0] * columns)
        try run(
          "translated_detector_plan",
          [detectorShifts.buffer, plan.pixels, plan.detectorCoefficients],
          words: [UInt32(shape[2]), UInt32(shape[3]), UInt32(index), 0], count: pixels)
        if translatedPlans.count >= 16, translatedPlans[index] == nil {
          translatedPlans.removeAll()
        }
        translatedPlans[index] = plan
      }
      let plan = translatedPlans[index]!
      let beforeRead = Date.timeIntervalSinceReferenceDate
      let bytes = (stop - first) * columns * pixels * source.itemBytes
      if translatedReadBuffer == nil || translatedReadBuffer!.length < bytes {
        translatedReadBuffer = try buffer(bytes)
      }
      let raw = translatedReadBuffer!
      // Operations are serialized; this internal scratch never escapes to callers.
      if profileSampling {
        let readCommand = try makeCommand()
        try source.encodeRead((first * columns)..<(stop * columns), into: raw, command: readCommand)
        try complete(readCommand)
        try source.checkErrors()
      }
      let readSeconds = Date.timeIntervalSinceReferenceDate - beforeRead
      let command = try makeCommand()
      if !profileSampling {
        try source.encodeRead((first * columns)..<(stop * columns), into: raw, command: command)
      }
      let enc = try encoder(
        command, "sample_prepared_u\(source.itemBytes * 8)",
        [
          raw, plan.scans, plan.pixels, plan.scanCoefficients, plan.detectorCoefficients,
          scanWeight.buffer, detectorWeight.buffer, numerator.buffer, denominator.buffer,
        ])
      var parameters = SIMD4<UInt32>(
        UInt32(pixels), UInt32(first * columns),
        UInt32(outputRows.lowerBound * columns), UInt32(outputRows.count * columns))
      enc.setBytes(&parameters, length: 16, index: 9)
      enc.dispatchThreads(
        MTLSize(width: pixels, height: outputRows.count * columns, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      enc.endEncoding()
      try complete(command)
      try source.checkErrors()
      if profileSampling {
        print(
          "SAMPLING_PROFILE decode=\(readSeconds) sampling_gpu=\(command.gpuEndTime - command.gpuStartTime)"
        )
      }
      return
    }
    let raw = try buffer((stop - first) * columns * pixels * source.itemBytes)
    let command = try makeCommand()
    try source.encodeRead((first * columns)..<(stop * columns), into: raw, command: command)
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
      words: p, count: outputRows.count * columns * pixels, command: command)
    try complete(command)
    try source.checkErrors()
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
