import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
import Native4DSTEMIO

/// Internal transcode into the existing blockwise bit-plane resident layout.
enum PairedRuntimePackedConversion {
  static func convert(
    dataset: Native4DSTEMDataset, moments: MetalCompactH5ExactDPCMoments,
    payload: MTLBuffer, offsets: MTLBuffer, modes: MTLBuffer, table: MTLBuffer,
    pixelOfRank: [UInt32]?, compactOffsets: Bool, library: MTLLibrary, queue: MTLCommandQueue,
    maximumAdditionalBytes: UInt64, staging: Bool, shouldCancel: () -> Bool
  ) throws -> MetalCompactH5ResidentSource {
    let started = ContinuousClock.now
    let device = queue.device
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    let scans = dataset.scanRows * dataset.scanCols
    let pixels = dataset.detectorRows * dataset.detectorCols
    let frames = 4096, packetsPerShard = frames / 512
    let tiles = frames / 32, checkpoints = (tiles + 31) / 32
    let stride = checkpoints + (tiles + 7) / 8
    guard scans > 0, scans % frames == 0, pixels > 0,
      dataset.sourceDtype == "uint16" || dataset.sourceDtype == "uint8",
      moments.total.count == scans, moments.detectorRowMoment.count == scans,
      moments.detectorColumnMoment.count == scans else {
      throw invalid("Resident conversion currently requires uint8/uint16 and a scan count divisible by 4096")
    }
    func checkpoint() throws {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      let held = UInt64(device.currentAllocatedSize)
      if held > allocatedBefore && held - allocatedBefore > maximumAdditionalBytes {
        throw invalid("Packed conversion exceeds available memory; release another acquisition")
      }
    }
    func buffer(_ bytes: Int, options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
      try checkpoint()
      let held = UInt64(device.currentAllocatedSize)
      let added = held > allocatedBefore ? held - allocatedBefore : 0
      guard bytes > 0, bytes <= device.maxBufferLength,
        added <= maximumAdditionalBytes, UInt64(bytes) <= maximumAdditionalBytes - added,
        let result = device.makeBuffer(length: bytes, options: options) else {
        throw invalid("Insufficient memory for bounded packed conversion; release another acquisition")
      }
      return result
    }
    try checkpoint()
    let constants = MTLFunctionConstantValues()
    var compact = compactOffsets
    constants.setConstantValue(&compact, type: .bool, index: 20)
    // Explicit bounded workspace for interactive conversion; the environment
    // override remains available to retained benchmark harnesses.
    var staged = staging || ProcessInfo.processInfo.environment["QGPU_PAIRED_CONVERSION_STAGING"] == "1"
    constants.setConstantValue(&staged, type: .bool, index: 80)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      try device.makeComputePipelineState(function: library.makeFunction(name: name, constantValues: constants))
    }
    let measure = try pipeline("paired_runtime_packed_measure")
    let headersPipeline = try pipeline("paired_runtime_packed_headers")
    let write = try pipeline("paired_runtime_packed_write")
    let staging = staged ? try buffer(frames * pixels * 2, options:.storageModePrivate) : nil
    let rankBuffer = try buffer(pixels * 4)
    let ranks = pixelOfRank ?? (0..<pixels).map(UInt32.init)
    ranks.withUnsafeBytes { rankBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    let failure = try buffer(4)
    let sums = try buffer(pixels * 4)
    let lengths = try buffer(pixels * 4)
    let widths = try buffer(pixels * 4)
    let scratchBytes = UInt64(rankBuffer.length + failure.length + sums.length + lengths.length + widths.length + (staging?.length ?? 0))
    var maximumWidths = [UInt8](repeating: 0, count: pixels)
    var detectorSum = [UInt64](repeating: 0, count: pixels)
    var shards: [(payload: MTLBuffer, headers: MTLBuffer)] = []
    var decodeSeconds = 0.0, writeSeconds = 0.0
    func complete(_ command: MTLCommandBuffer) throws -> Double {
      command.commit(); command.waitUntilCompleted()
      guard command.status == .completed, command.error == nil else {
        throw invalid("Metal packed conversion failed: \(String(describing: command.error))")
      }
      let status = failure.contents().load(as: UInt32.self)
      guard status == 0 else { throw invalid("ANS validation failed during conversion (code \(status))") }
      try checkpoint()
      return command.gpuEndTime - command.gpuStartTime
    }
    func encode(_ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
      _ buffers: [MTLBuffer], _ parameters: [UInt32], _ count: Int) throws {
      guard let encoder = command.makeComputeCommandEncoder() else { throw invalid("Cannot encode conversion") }
      encoder.setComputePipelineState(pipeline)
      for (index, value) in buffers.enumerated() { encoder.setBuffer(value, offset: 0, index: index) }
      var parameters = parameters
      encoder.setBytes(&parameters, length: parameters.count * 4, index: buffers.count)
      if buffers.count == 8, let staging { encoder.setBuffer(staging,offset:0,index:9) }
      encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: min(128, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
      encoder.endEncoding()
    }
    for firstScan in Swift.stride(from: 0, to: scans, by: frames) {
      try autoreleasepool {
        try checkpoint()
        let headers = try buffer(pixels * stride * 4)
        memset(failure.contents(), 0, 4); memset(sums.contents(), 0, sums.length)
        let parameters = [UInt32(pixels), UInt32(scans / 512), UInt32(payload.length),
          UInt32(packetsPerShard), UInt32(stride), UInt32(firstScan / 512)]
        guard let command = queue.makeCommandBuffer() else { throw invalid("Cannot measure packed sizes") }
        try encode(command, measure, [payload, offsets, modes, table, headers, sums, rankBuffer, failure],
          parameters, pixels * packetsPerShard)
        try encode(command, headersPipeline, [headers, lengths, widths], parameters, pixels)
        decodeSeconds += try complete(command)
        let sizes = lengths.contents().assumingMemoryBound(to: UInt32.self)
        let maxWidths = widths.contents().assumingMemoryBound(to: UInt32.self)
        let pixelSums = sums.contents().assumingMemoryBound(to: UInt32.self)
        let headerValues = headers.contents().assumingMemoryBound(to: UInt32.self)
        var words = UInt64(0)
        for pixel in 0..<pixels {
          guard words <= UInt64(UInt32.max >> 5) else { throw invalid("Packed shard offset overflow") }
          headerValues[pixel * stride] = UInt32(words)
          words += UInt64(sizes[pixel])
          maximumWidths[pixel] = max(maximumWidths[pixel], UInt8(maxWidths[pixel]))
          detectorSum[pixel] += UInt64(pixelSums[pixel])
        }
        guard words <= UInt64(UInt32.max >> 5) else { throw invalid("Packed shard exceeds addressable layout") }
        let output = try buffer(max(4, Int(words) * 4))
        guard let command = queue.makeCommandBuffer() else { throw invalid("Cannot write packed resident") }
        try encode(command, write, [payload, offsets, modes, table, headers, output, rankBuffer, failure],
          parameters, pixels * packetsPerShard)
        writeSeconds += try complete(command)
        shards.append((output, headers))
      }
    }
    // Existing DPC summaries are small scientific products, not decoded 4D data.
    var momentData = Data(count: scans * 32)
    momentData.withUnsafeMutableBytes { raw in
      let values = raw.bindMemory(to: UInt64.self)
      for scan in 0..<scans {
        values[scan * 4] = moments.total[scan]
        values[scan * 4 + 1] = moments.detectorRowMoment[scan]
        values[scan * 4 + 2] = moments.detectorColumnMoment[scan]
        values[scan * 4 + 3] = 0
      }
    }
    let maximumWidth = Int(maximumWidths.max() ?? 0)
    let maximum = UInt32((UInt64(1) << maximumWidth) - 1)
    let packed = OriginalPackedBuffers(origin: "residentANSConversion", countsRoundtripVerified: false,
      payloadLayout: 1, dataset: dataset, frames: frames, headerStride: stride, shards: shards,
      moments: momentData, detectorSum: detectorSum, maximum: maximum, maximumWidths: maximumWidths,
      calibration: OriginalHDF5Packing.measuredDetector(detectorSum, rows: dataset.detectorRows,
        columns: dataset.detectorCols, excludedFromEstimate: dataset.badPixelIndices),
      stagingBytes: scratchBytes, readSeconds: 0, decodeSeconds: decodeSeconds,
      decodeAndHeaderSeconds: 0, packingSeconds: writeSeconds, reusedDPC: true,
      combinedDecodePackingSeconds: nil)
    let result = try MetalCompactH5Loader.residentFromOriginal(packed, device: device, started: started,
      allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes, shouldCancel: shouldCancel)
    result.sourceHotPixelIndices = dataset.badPixelIndices
    return result
  }

  private static func invalid(_ message: String) -> NSError {
    NSError(domain: "paired-resident-conversion", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
  }
}
