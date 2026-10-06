import Foundation
import Metal
import Metal4DSTEMStreamingIO

struct TranslatedSamplingPlan {
  let signature: [UInt32]
  let scans: MTLBuffer
  let pixels: MTLBuffer
  let scanCoefficients: MTLBuffer
  let detectorCoefficients: MTLBuffer
}

extension MetalImageOperations {
  /// Compatibility entry point for existing native workflow benchmarks.
  /// New callers should use `MetalEncodedSource.load(files:indexDirectory:device:)`.
  @available(
    *, deprecated,
    message: "Use MetalEncodedSource.load(files:indexDirectory:device:shouldCancel:progress:)."
  )
  public func loadEncoded(
    files: [URL], indexDirectory: URL,
    shouldCancel: () -> Bool = { false }, progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> [MetalEncodedSource] {
    try MetalEncodedSource.load(
      files: files, indexDirectory: indexDirectory, device: device,
      shouldCancel: shouldCancel, progress: progress)
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
      let plan = try samplingPlan(
        shape: shape, scanShifts: scanShifts, detectorShifts: detectorShifts, index: index)
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
  /// Sampling geometry for one source, prepared once per shape and displacement.
  func samplingPlan(shape: [Int], scanShifts: GPUImage, detectorShifts: GPUImage, index: Int)
    throws -> TranslatedSamplingPlan
  {
    let columns = shape[1]
    let pixels = shape[2] * shape[3]
    let scanPair = scanShifts.buffer.contents().assumingMemoryBound(to: Float.self)
    let detectorPair = detectorShifts.buffer.contents().assumingMemoryBound(to: Float.self)
    let signature =
      shape.map(UInt32.init) + [
        scanPair[index * 2].bitPattern, scanPair[index * 2 + 1].bitPattern,
        detectorPair[index * 2].bitPattern, detectorPair[index * 2 + 1].bitPattern,
      ]
    if let plan = translatedPlans[index], plan.signature == signature { return plan }
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
    return plan
  }

  /// Merge translated inputs straight into finished values: identical to filling the
  /// accumulators with zero, calling `accumulateTranslated` for every source in order
  /// and then `finishWeighted`, but the accumulators stay in registers. Each pass
  /// decodes a bounded window per source for `rowsPerPass` output rows. Returns false
  /// when the fused path does not apply (reference sampling, mixed or unsupported item
  /// sizes, or `QUANTEM_GPU_FUSED_MERGE=0`); the caller then uses the per-source path.
  public func mergeTranslated(
    sources: [any MetalResidentCounts], outputRows: Range<Int>, scanShifts: GPUImage,
    detectorShifts: GPUImage, scanWeights: [GPUImage], detectorWeights: [GPUImage],
    uncovered: GPUImage, output: GPUImage, rowsPerPass: Int = 4
  ) throws -> Bool {
    guard fusedMerge, !referenceSampling, !profileSampling, let first = sources.first,
      [1, 2].contains(first.itemBytes),
      sources.allSatisfy({ $0.itemBytes == first.itemBytes && $0.shape == first.shape }),
      scanWeights.count == sources.count, detectorWeights.count == sources.count,
      rowsPerPass > 0
    else { return false }
    let shape = first.shape
    let columns = shape[1]
    let pixels = shape[2] * shape[3]
    guard !outputRows.isEmpty, output.rows == outputRows.count * columns, output.columns == pixels
    else {
      throw Self.invalid("Merge output must hold the requested scan rows by detector pixels.")
    }
    // Windows span at most rowsPerPass + 2 scan rows; the kernel indexes them in 32 bits.
    guard (rowsPerPass + 2) * columns * pixels < Int(UInt32.max) else { return false }
    let kernel = "sample_fused_u\(first.itemBytes * 8)"
    for passStart in stride(from: outputRows.lowerBound, to: outputRows.upperBound, by: rowsPerPass)
    {
      let rows = passStart..<min(outputRows.upperBound, passStart + rowsPerPass)
      let command = try makeCommand()
      var table = [UInt64]()
      var used = [MTLBuffer]()
      var active = [any MetalResidentCounts]()
      for (index, source) in sources.enumerated() {
        let shiftRow = scanShifts.buffer.contents().load(fromByteOffset: index * 8, as: Float.self)
        let delta = Int(floor(-shiftRow))
        let windowFirst = max(0, rows.lowerBound + delta)
        let windowStop = min(shape[0], rows.upperBound + delta + 1)
        if windowFirst >= windowStop { continue }
        let plan = try samplingPlan(
          shape: shape, scanShifts: scanShifts, detectorShifts: detectorShifts, index: index)
        let bytes = (windowStop - windowFirst) * columns * pixels * source.itemBytes
        let slot = active.count
        if fusedReadBuffers.count <= slot { fusedReadBuffers.append(try buffer(bytes)) }
        if fusedReadBuffers[slot].length < bytes { fusedReadBuffers[slot] = try buffer(bytes) }
        let raw = fusedReadBuffers[slot]
        try source.encodeRead(
          (windowFirst * columns)..<(windowStop * columns), into: raw, command: command)
        let buffers = [
          raw, plan.scans, plan.pixels, plan.scanCoefficients, plan.detectorCoefficients,
          scanWeights[index].buffer, detectorWeights[index].buffer,
        ]
        table += buffers.map { $0.gpuAddress }
        table.append(UInt64(windowFirst * columns))  // windowStart, then padding
        used += buffers
        active.append(source)
      }
      let encoder = try self.encoder(command, kernel, [])
      if table.isEmpty { table = [UInt64](repeating: 0, count: 8) }
      table.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
      var count = UInt32(active.count)
      encoder.setBytes(&count, length: 4, index: 1)
      encoder.setBuffer(uncovered.buffer, offset: 0, index: 2)
      encoder.setBuffer(
        output.buffer, offset: (rows.lowerBound - outputRows.lowerBound) * columns * pixels * 4,
        index: 3)
      var parameters = SIMD4<UInt32>(
        UInt32(pixels), 0, UInt32(rows.lowerBound * columns), UInt32(rows.count * columns))
      encoder.setBytes(&parameters, length: 16, index: 4)
      encoder.useResources(used, usage: .read)
      encoder.dispatchThreads(
        MTLSize(width: pixels, height: rows.count * columns, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      encoder.endEncoding()
      try complete(command)
      for source in active { try source.checkErrors() }
      fusedMergePasses += 1
      fusedMergeGPUSeconds += max(0, command.gpuEndTime - command.gpuStartTime)
    }
    return true
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
