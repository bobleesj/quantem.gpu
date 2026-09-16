import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
import Native4DSTEMIO

/// Stage timings for one exact original-HDF5 to paired-runtime build.
public struct MetalPairedRuntimeTANSBuildMetrics: Sendable {
  public let totalSeconds: Double
  public let fusedDecodeAndSizeSeconds: Double
  /// Provisional bounded prefix over stream-size metadata only, never counts.
  public let provisionalCPUPrefixSeconds: Double
  public let compactSeconds: Double
  public let consolidationSeconds: Double
  public let residentBytes: UInt64
}

/// Result of one bounded exact paired-runtime build.
struct MetalPairedRuntimeTANSBuildResult {
  let provider: PairedRuntimeTANSRecordProvider
  /// Detector pixel stored at each stream rank of a packet, or nil for pixel order.
  let streamPixels: [UInt32]?
  let dpcMoments: MetalCompactH5ExactDPCMoments
  let metrics: MetalPairedRuntimeTANSBuildMetrics
}

/// One-acquisition original-HDF5 paired-runtime builder.
///
/// The input remains full-resolution and native precision. Original counts are
/// decoded into one reusable private 16K window, immediately encoded, compacted,
/// and released before the next record. No dense full-volume or archive exists.
enum MetalPairedRuntimeTANSHDF5Builder {
  static func build(
    source: Native4DSTEMIndexedSource, device: MTLDevice,
    maximumAdditionalBytes: UInt64? = nil,
    configuration: PairedRuntimeConfiguration = .init(mode: nil),
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> MetalPairedRuntimeTANSBuildResult {
    let started = CFAbsoluteTimeGetCurrent()
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    try validate(source: source)
    guard let identity = source.dataset.sourceIdentitySHA256 else {
      throw invalid("Paired-runtime loading requires an exact source identity")
    }
    let logicalDtype: Metal4DSTEMIntegerDType =
      source.sourceBytesPerValue == 1 ? .uint8 : .uint16
    var validity = [UInt8](
      repeating: 1,
      count: source.dataset.detectorRows
        * source.dataset.detectorCols)
    for pixel in source.dataset.badPixelIndices where validity.indices.contains(pixel) {
      validity[pixel] = 0
    }
    let descriptor = try PairedRuntimeTANSSeriesDescriptor(
      sourceIdentitySHA256: [identity],
      shape: [
        1, source.dataset.scanRows, source.dataset.scanCols,
        source.dataset.detectorRows, source.dataset.detectorCols,
      ],
      logicalDtype: logicalDtype,
      detectorValidity: validity)
    let resources = try Resources(
      device: device, descriptor: descriptor,
      allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes,
      configuration: configuration)
    let packing = try OriginalHDF5Packing(device: device)

    var records: [PairedRuntimeTANSRecordBuffers] = []
    var extents: [PairedRuntimeTANSRecordExtent] = []
    var fusedSeconds = 0.0
    var prefixSeconds = 0.0
    var compactSeconds = 0.0
    var finalCommand: MTLCommandBuffer?
    var momentTotals: [UInt64] = []
    var momentRows: [UInt64] = []
    var momentColumns: [UInt64] = []
    momentTotals.reserveCapacity(source.logicalFrameCount)
    momentRows.reserveCapacity(source.logicalFrameCount)
    momentColumns.reserveCapacity(source.logicalFrameCount)
    try packing.forEachExactDecodedWindow(
      source: source,
      maximumFrames: PairedRuntimeTANSRecordABI.recordScans,
      includeDPCMoments: true,
      shouldCancel: shouldCancel,
      progress: progress
    ) { dense, moments, range, decodeCommand in
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      let recordIndex = records.count
      guard recordIndex < PairedRuntimeTANSRecordABI.recordsPerAcquisition,
        range.lowerBound == recordIndex * PairedRuntimeTANSRecordABI.recordScans,
        range.count == PairedRuntimeTANSRecordABI.recordScans
      else {
        throw invalid(
          "The first paired-runtime builder requires sixteen ordered 16K decode windows")
      }
      let output = try resources.encode(
        dense: dense, recordIndex: recordIndex, decodeCommand: decodeCommand,
        shouldCancel: shouldCancel)
      records.append(output.record)
      extents.append(output.extent)
      fusedSeconds += output.fusedSeconds
      prefixSeconds += output.prefixSeconds
      compactSeconds += output.compactSeconds
      finalCommand = output.completedCommand
      guard let moments else {
        throw invalid("Paired-runtime loading did not produce exact DPC moments")
      }
      let words = moments.contents().assumingMemoryBound(to: UInt64.self)
      for scan in 0..<range.count {
        momentTotals.append(words[scan * 4])
        momentRows.append(words[scan * 4 + 1])
        momentColumns.append(words[scan * 4 + 2])
      }
    }
    guard records.count == PairedRuntimeTANSRecordABI.recordsPerAcquisition,
      let finalCommand
    else {
      throw invalid("Paired-runtime loading did not produce all sixteen exact records")
    }
    let receipt = try PairedRuntimeTANSProducerReceipt(
      sourceIdentitySHA256: [identity], recordExtents: extents,
      completedCommand: finalCommand, failureFlag: resources.failure)
    if ProcessInfo.processInfo.environment["QGPU_RUNTIME_ANS_PROFILE"] == "1" {
      resources.logProfile(
        logicalBytes: UInt64(source.logicalFrameCount) * source.decodedBytesPerFrame)
    }
    let provider = try PairedRuntimeTANSRecordProvider(
      descriptor: descriptor, decodingTable: resources.decoding,
      records: records, receipt: receipt)
    let residentBytes =
      UInt64(resources.decoding.length)
      + records.reduce(UInt64(0)) {
        $0
          + UInt64(
            $1.payload.length + $1.offsets.length + $1.modes.length
              + ($1.workGroups?.length ?? 0))
      }
    return MetalPairedRuntimeTANSBuildResult(
      provider: provider,
      streamPixels: resources.streamPixels,
      dpcMoments: MetalCompactH5ExactDPCMoments(
        total: momentTotals,
        detectorRowMoment: momentRows,
        detectorColumnMoment: momentColumns,
        sourceIdentitySHA256: identity),
      metrics: MetalPairedRuntimeTANSBuildMetrics(
        totalSeconds: CFAbsoluteTimeGetCurrent() - started,
        fusedDecodeAndSizeSeconds: fusedSeconds,
        provisionalCPUPrefixSeconds: prefixSeconds,
        compactSeconds: compactSeconds, consolidationSeconds: 0,
        residentBytes: residentBytes))
  }

  private static func validate(source: Native4DSTEMIndexedSource) throws {
    guard source.dataset.scanRows == PairedRuntimeTANSRecordABI.scanRows,
      source.dataset.scanCols == PairedRuntimeTANSRecordABI.scanColumns,
      source.dataset.detectorRows == PairedRuntimeTANSRecordABI.detectorRows,
      source.dataset.detectorCols == PairedRuntimeTANSRecordABI.detectorColumns,
      source.logicalFrameCount
        == PairedRuntimeTANSRecordABI.recordScans
        * PairedRuntimeTANSRecordABI.recordsPerAcquisition,
      source.sourceBytesPerValue == 1 || source.sourceBytesPerValue == 2
    else {
      throw invalid(
        "Paired-runtime loading currently requires exact uint8 or uint16 "
          + "512x512x192x192 indexed data")
    }
  }

  final class Resources {
    struct EncodedRecord {
      let record: PairedRuntimeTANSRecordBuffers
      let extent: PairedRuntimeTANSRecordExtent
      let completedCommand: MTLCommandBuffer
      let fusedSeconds: Double
      let prefixSeconds: Double
      let compactSeconds: Double
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let encodePipeline: MTLComputePipelineState
    let compactPipeline: MTLComputePipelineState
    let frequencyStarts: MTLBuffer
    let encoding: MTLBuffer
    let decoding: MTLBuffer
    let failure: MTLBuffer
    let descriptor: PairedRuntimeTANSSeriesDescriptor
    let allocatedBefore: UInt64
    let maximumAdditionalBytes: UInt64?
    let scratchlessEncode: Bool
    let rankModesPipeline: MTLComputePipelineState
    let stagingPairs: MTLBuffer?
    let streamPixels: [UInt32]?
    let streamPixelsBuffer: MTLBuffer
    let scratch: MTLBuffer
    let sizes: MTLBuffer
    let stagingOffsets: MTLBuffer
    /// Command-buffer waits split into device execution and queue/stall time.
    private var encodeGPUSeconds = 0.0
    private var encodeStallSeconds = 0.0
    private var compactGPUSeconds = 0.0
    private var compactStallSeconds = 0.0
    private var encodeRecords = 0

    func logProfile(logicalBytes: UInt64) {
      let record: [String: Any] = [
        "record": "paired_runtime_encode_profile",
        "encode_records": encodeRecords,
        "encode_gpu_seconds": encodeGPUSeconds,
        "encode_stall_seconds": encodeStallSeconds,
        "compact_gpu_seconds": compactGPUSeconds,
        "compact_stall_seconds": compactStallSeconds,
        "logical_bytes": logicalBytes,
      ]
      guard
        let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
        let line = String(data: data, encoding: .utf8)
      else { return }
      FileHandle.standardError.write(Data(("QGPU_PAIRED_ENCODE_PROFILE " + line + "\n").utf8))
    }

    init(
      device: MTLDevice, descriptor: PairedRuntimeTANSSeriesDescriptor,
      allocatedBefore: UInt64, maximumAdditionalBytes: UInt64?,
      configuration: PairedRuntimeConfiguration
    ) throws {
      func pairedRuntimeEnvironment(_ name: String) -> String? { configuration.value(name) }
      self.device = device
      self.descriptor = descriptor
      self.allocatedBefore = allocatedBefore
      self.maximumAdditionalBytes = maximumAdditionalBytes
      guard let queue = device.makeCommandQueue() else {
        throw invalid("Metal could not create a paired-runtime HDF5 queue")
      }
      self.queue = queue
      let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
      let sparseSlackValue = configuration.value("QGPU_PAIRED_RUNTIME_SPARSE_SLACK") ?? "0"
      guard ["0", "4", "8", "12", "16", "24", "32", "48", "64"].contains(sparseSlackValue),
        let sparseSlack = UInt32(sparseSlackValue)
      else {
        throw invalid("QGPU_PAIRED_RUNTIME_SPARSE_SLACK must be 0, 4, 8, 12, 16, 24, 32, 48, or 64")
      }
      let scratchlessValue =
        pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_SCRATCHLESS_ENCODE") ?? "0"
      guard scratchlessValue == "0" || scratchlessValue == "1" else {
        throw invalid("QGPU_PAIRED_RUNTIME_SCRATCHLESS_ENCODE must be 0 or 1")
      }
      // A return from indexed residents can have enough room for the compact
      // replacement but not the buffered encoder's extra full-window scratch.
      // Use the same exact size-then-write encoder when the known next-window
      // working set cannot fit. This changes workspace, not stream selection.
      // Every actual allocation still goes through the hard admission guard.
      let windowSamples = PairedRuntimeTANSRecordABI.recordScans * descriptor.detectorPixels
      let bufferedWindowBytes = UInt64(
        windowSamples * MemoryLayout<UInt32>.stride
          + PairedRuntimeTANSRecordABI.streamScans * 2 * descriptor.streamsPerRecord
          + windowSamples * MemoryLayout<UInt16>.stride
          + descriptor.modesBytesPerRecord + descriptor.offsetsBytesPerRecord
          + descriptor.streamsPerRecord * MemoryLayout<UInt32>.stride)
      let boundedNormal =
        configuration.mode == .normal
        && maximumAdditionalBytes.map { $0 < bufferedWindowBytes } == true
      let useScratchlessEncode = scratchlessValue == "1" || boundedNormal
      scratchlessEncode = useScratchlessEncode
      let encodeConstants = MTLFunctionConstantValues()
      var specializedSparseSlack = sparseSlack
      encodeConstants.setConstantValue(&specializedSparseSlack, type: .uint, index: 4)
      var specializedScratchless = useScratchlessEncode
      encodeConstants.setConstantValue(&specializedScratchless, type: .bool, index: 7)
      let sparseMaxNonzeroValue =
        configuration.value("QGPU_PAIRED_RUNTIME_SPARSE_MAX_NONZERO") ?? "0"
      guard var sparseMaxNonzero = UInt32(sparseMaxNonzeroValue), sparseMaxNonzero <= 512
      else {
        throw invalid("QGPU_PAIRED_RUNTIME_SPARSE_MAX_NONZERO must be an integer from 0 to 512")
      }
      encodeConstants.setConstantValue(&sparseMaxNonzero, type: .uint, index: 27)
      let compactEventValue =
        pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_COMPACT_EVENT_MAX_NONZERO") ?? "0"
      guard var compactEventMaxNonzero = UInt32(compactEventValue), compactEventMaxNonzero <= 512
      else {
        throw invalid(
          "QGPU_PAIRED_RUNTIME_COMPACT_EVENT_MAX_NONZERO must be an integer from 0 to 512")
      }
      encodeConstants.setConstantValue(&compactEventMaxNonzero, type: .uint, index: 45)
      let streamOrder = pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_STREAM_ORDER") ?? "pixel"
      guard streamOrder == "pixel" || streamOrder == "radial1" else {
        throw invalid("QGPU_PAIRED_RUNTIME_STREAM_ORDER must be pixel or radial1")
      }
      if streamOrder == "radial1" {
        guard descriptor.detectorPixels == 192 * 192,
          let layout = PairedRuntimeTANSPolarPlan.indexLayout(
            leafPixels: 16, layoutKind: "radial1"),
          layout.permutation.count == descriptor.detectorPixels,
          Set(layout.permutation).count == descriptor.detectorPixels,
          layout.permutation.allSatisfy({ $0 >= 0 && Int($0) < descriptor.detectorPixels })
        else { throw invalid("Radial1 stream order requires a complete 192 x 192 permutation") }
        streamPixels = layout.permutation.map { UInt32($0) }
      } else {
        streamPixels = nil
      }
      var permutedDirectWrites = streamPixels != nil && useScratchlessEncode
      encodeConstants.setConstantValue(&permutedDirectWrites, type: .bool, index: 42)
      let encodeFunction = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSEncodeFunction,
        constantValues: encodeConstants)
      encodePipeline = try device.makeComputePipelineState(function: encodeFunction)
      let compactConstants = MTLFunctionConstantValues()
      var permutedStreams = streamPixels != nil
      compactConstants.setConstantValue(&permutedStreams, type: .bool, index: 42)
      compactPipeline = try device.makeComputePipelineState(
        function: try library.makeFunction(
          name: Metal4DSTEMKernels.pairedRuntimeTANSCompactFunction,
          constantValues: compactConstants))
      let tables = try PairedRuntimeTANSTables.build()
      var packedFrequencies = [UInt32](repeating: 0, count: tables.frequencies.count)
      for model in 0..<PairedRuntimeTANSTables.modelCount {
        var start = 0
        let base = model * PairedRuntimeTANSTables.symbolCount
        for symbol in 0..<PairedRuntimeTANSTables.symbolCount {
          let frequency = Int(tables.frequencies[base + symbol])
          packedFrequencies[base + symbol] = UInt32(start) | (UInt32(frequency) << 16)
          start += frequency
        }
      }
      frequencyStarts = try Self.upload(
        packedFrequencies, device: device, label: "paired-runtime frequency starts")
      encoding = try Self.upload(
        tables.encoding, device: device, label: "paired-runtime encoding")
      streamPixelsBuffer = try Self.upload(
        streamPixels ?? [0], device: device, label: "paired-runtime stream pixels")
      rankModesPipeline = try Self.pipeline(
        library: library, device: device, name: "paired_runtime_tans_rank_modes")
      decoding = try Self.privateUpload(
        tables.packedDecoding, device: device, queue: queue,
        label: "paired-runtime decoding")
      failure = try Self.makeBuffer(
        device: device, bytes: 4, options: .storageModeShared,
        label: "paired-runtime producer failure")
      let scratchBytes =
        useScratchlessEncode
        ? MemoryLayout<UInt32>.stride
        : PairedRuntimeTANSRecordABI.streamScans * 2 * descriptor.streamsPerRecord
      scratch = try Self.makeBoundedBuffer(
        device: device, bytes: scratchBytes, options: .storageModePrivate,
        label: "paired-runtime reusable scratch", allocatedBefore: allocatedBefore,
        maximumAdditionalBytes: maximumAdditionalBytes)
      sizes = try Self.makeBoundedBuffer(
        device: device, bytes: descriptor.streamsPerRecord * 4,
        options: .storageModeShared, label: "paired-runtime stream sizes",
        allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes)
      stagingOffsets = try Self.makeBoundedBuffer(
        device: device, bytes: descriptor.offsetsBytesPerRecord,
        options: .storageModeShared, label: "paired-runtime provisional CPU prefix",
        allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes)
      stagingPairs =
        streamPixels != nil && useScratchlessEncode
        ? try Self.makeBoundedBuffer(
          device: device, bytes: descriptor.streamsPerRecord * 8,
          options: .storageModeShared, label: "paired-runtime rank write ranges",
          allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes)
        : nil
    }

    func encode(
      dense: MTLBuffer, recordIndex: Int, decodeCommand: MTLCommandBuffer,
      shouldCancel: () -> Bool
    ) throws -> EncodedRecord {
      memset(failure.contents(), 0, 4)
      let modes = try Self.makeBoundedBuffer(
        device: device, bytes: descriptor.modesBytesPerRecord,
        options: .storageModePrivate, label: "paired-runtime modes \(recordIndex)",
        allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes)
      var parameters: [UInt32] = [
        UInt32(PairedRuntimeTANSRecordABI.recordScans),
        UInt32(descriptor.detectorPixels),
        UInt32(descriptor.streamsPerRecord),
        UInt32(descriptor.logicalDtype.bytesPerValue),
        UInt32(PairedRuntimeTANSRecordABI.streamScans * 2),
        0,
      ]
      guard let encoder = decodeCommand.makeComputeCommandEncoder() else {
        throw invalid("Metal could not encode paired-runtime stream sizes")
      }
      encoder.setComputePipelineState(encodePipeline)
      for (index, buffer) in [
        dense, frequencyStarts, encoding, scratch, sizes, modes, failure,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(&parameters, length: parameters.count * 4, index: 7)
      if scratchlessEncode { encoder.setBuffer(stagingOffsets, offset: 0, index: 8) }
      encoder.setBuffer(streamPixelsBuffer, offset: 0, index: 9)
      encoder.dispatchThreads(
        MTLSize(width: descriptor.streamsPerRecord, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
      let fusedStarted = CFAbsoluteTimeGetCurrent()
      try Self.finish(
        decodeCommand, failure: failure,
        message: "Paired-runtime decode and size encoding failed")
      let fusedSeconds = CFAbsoluteTimeGetCurrent() - fusedStarted
      encodeRecords += 1
      let fusedGPU = max(0, decodeCommand.gpuEndTime - decodeCommand.gpuStartTime)
      encodeGPUSeconds += fusedGPU
      encodeStallSeconds += max(0, fusedSeconds - fusedGPU)
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }

      let prefixStarted = CFAbsoluteTimeGetCurrent()
      let sizeWords = sizes.contents().bindMemory(
        to: UInt32.self, capacity: descriptor.streamsPerRecord)
      let offsetWords = stagingOffsets.contents().bindMemory(
        to: UInt32.self, capacity: descriptor.streamsPerRecord + 1)
      offsetWords[0] = 0
      var terminal = UInt64(0)
      let pixels = descriptor.detectorPixels
      if let streamPixels {
        // Rank-order prefix: output stream rank r copies encoded stream (packet, pixel perm[r]).
        try streamPixels.withUnsafeBufferPointer { permutation in
          var stream = 0
          for packetBase in stride(from: 0, to: descriptor.streamsPerRecord, by: pixels) {
            for rank in 0..<pixels {
              terminal += UInt64(sizeWords[packetBase + Int(permutation[rank])])
              guard terminal <= UInt64(UInt32.max) else {
                throw invalid("One paired-runtime record exceeds the UInt32 offset ABI")
              }
              offsetWords[stream + 1] = UInt32(terminal)
              stream += 1
            }
          }
        }
      } else {
        for stream in 0..<descriptor.streamsPerRecord {
          terminal += UInt64(sizeWords[stream])
          guard terminal <= UInt64(UInt32.max) else {
            throw invalid("One paired-runtime record exceeds the UInt32 offset ABI")
          }
          offsetWords[stream + 1] = UInt32(terminal)
        }
      }
      if let stagingPairs, let streamPixels {
        // Pixel-order stream s = packetBase + pixel writes to rank range [offsets[r], offsets[r + 1]).
        let pairs = stagingPairs.contents().bindMemory(
          to: UInt32.self, capacity: descriptor.streamsPerRecord * 2)
        for packetBase in stride(from: 0, to: descriptor.streamsPerRecord, by: pixels) {
          for rank in 0..<pixels {
            let source = packetBase + Int(streamPixels[rank])
            pairs[2 * source] = offsetWords[packetBase + rank]
            pairs[2 * source + 1] = offsetWords[packetBase + rank + 1]
          }
        }
      }
      let prefixSeconds = CFAbsoluteTimeGetCurrent() - prefixStarted
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }

      let payloadBytes = Int(terminal)
      let rankedModes =
        streamPixels == nil
        ? nil
        : try Self.makeBoundedBuffer(
          device: device, bytes: descriptor.modesBytesPerRecord,
          options: .storageModePrivate, label: "paired-runtime ranked modes \(recordIndex)",
          allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes)
      let residentPayload = try Self.makeBoundedBuffer(
        device: device, bytes: max(4, Self.alignedUInt32Bytes(payloadBytes)),
        options: .storageModePrivate, label: "paired-runtime payload \(recordIndex)",
        allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes)
      let residentOffsets = try Self.makeBoundedBuffer(
        device: device, bytes: descriptor.offsetsBytesPerRecord,
        options: .storageModePrivate, label: "paired-runtime offsets \(recordIndex)",
        allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes)
      guard let compactCommand = queue.makeCommandBuffer(),
        let compact = compactCommand.makeComputeCommandEncoder()
      else {
        throw invalid("Metal could not compact paired-runtime streams")
      }
      if scratchlessEncode {
        parameters[5] = 1
        compact.setComputePipelineState(encodePipeline)
        for (index, buffer) in [
          dense, frequencyStarts, encoding, residentPayload, sizes, modes, failure,
        ].enumerated() {
          compact.setBuffer(buffer, offset: 0, index: index)
        }
        compact.setBytes(&parameters, length: parameters.count * 4, index: 7)
        compact.setBuffer(stagingPairs ?? stagingOffsets, offset: 0, index: 8)
        compact.setBuffer(streamPixelsBuffer, offset: 0, index: 9)
      } else {
        var compactParameters: [UInt32] = [
          UInt32(descriptor.streamsPerRecord),
          UInt32(PairedRuntimeTANSRecordABI.streamScans * 2),
          UInt32(payloadBytes), UInt32(descriptor.detectorPixels),
        ]
        compact.setComputePipelineState(compactPipeline)
        for (index, buffer) in [scratch, sizes, stagingOffsets, residentPayload, failure]
          .enumerated()
        {
          compact.setBuffer(buffer, offset: 0, index: index)
        }
        compact.setBytes(&compactParameters, length: compactParameters.count * 4, index: 5)
        compact.setBuffer(streamPixelsBuffer, offset: 0, index: 6)
        compact.setBuffer(modes, offset: 0, index: 7)
        compact.setBuffer(rankedModes ?? modes, offset: 0, index: 8)
      }
      compact.dispatchThreads(
        MTLSize(width: descriptor.streamsPerRecord, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      if scratchlessEncode, let rankedModes {
        var rankParameters: [UInt32] = [
          UInt32(descriptor.streamsPerRecord), UInt32(descriptor.detectorPixels),
        ]
        compact.setComputePipelineState(rankModesPipeline)
        compact.setBuffer(modes, offset: 0, index: 0)
        compact.setBuffer(rankedModes, offset: 0, index: 1)
        compact.setBuffer(streamPixelsBuffer, offset: 0, index: 2)
        compact.setBuffer(failure, offset: 0, index: 3)
        compact.setBytes(&rankParameters, length: rankParameters.count * 4, index: 4)
        compact.dispatchThreads(
          MTLSize(width: descriptor.streamsPerRecord, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      }
      compact.endEncoding()
      guard let blit = compactCommand.makeBlitCommandEncoder() else {
        throw invalid("Metal could not retain paired-runtime offsets")
      }
      blit.copy(
        from: stagingOffsets, sourceOffset: 0,
        to: residentOffsets, destinationOffset: 0,
        size: descriptor.offsetsBytesPerRecord)
      blit.endEncoding()
      let compactStarted = CFAbsoluteTimeGetCurrent()
      try Self.finish(
        compactCommand, failure: failure,
        message: "Paired-runtime compaction failed")
      let compactSeconds = CFAbsoluteTimeGetCurrent() - compactStarted
      let compactGPU = max(0, compactCommand.gpuEndTime - compactCommand.gpuStartTime)
      compactGPUSeconds += compactGPU
      compactStallSeconds += max(0, compactSeconds - compactGPU)
      return EncodedRecord(
        record: PairedRuntimeTANSRecordBuffers(
          acquisitionIndex: 0, recordInAcquisition: recordIndex,
          payload: residentPayload, offsets: residentOffsets,
          modes: rankedModes ?? modes, workGroups: nil),
        extent: PairedRuntimeTANSRecordExtent(
          acquisitionIndex: 0, recordInAcquisition: recordIndex,
          terminalPayloadBytes: payloadBytes, workGroupCount: 0),
        completedCommand: compactCommand,
        fusedSeconds: fusedSeconds,
        prefixSeconds: prefixSeconds,
        compactSeconds: compactSeconds)
    }

    private static func pipeline(
      library: MTLLibrary, device: MTLDevice, name: String
    ) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw invalid("Paired-runtime Metal library is missing \(name)")
      }
      return try device.makeComputePipelineState(function: function)
    }

    private static func finish(
      _ command: MTLCommandBuffer, failure: MTLBuffer, message: String
    ) throws {
      command.commit()
      command.waitUntilCompleted()
      let code = failure.contents().load(as: UInt32.self)
      guard command.status == .completed, command.error == nil, code == 0 else {
        throw invalid(
          "\(message) with code \(code): "
            + (command.error?.localizedDescription ?? "invalid stream"))
      }
    }

    private static func upload<T>(
      _ values: [T], device: MTLDevice, label: String
    ) throws -> MTLBuffer {
      let result = values.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      }
      guard let result else { throw invalid("Metal could not allocate \(label)") }
      result.label = label
      return result
    }

    private static func privateUpload<T>(
      _ values: [T], device: MTLDevice, queue: MTLCommandQueue, label: String
    ) throws -> MTLBuffer {
      let staging = try upload(values, device: device, label: label + " staging")
      let result = try makeBuffer(
        device: device, bytes: staging.length, options: .storageModePrivate, label: label)
      guard let command = queue.makeCommandBuffer(),
        let blit = command.makeBlitCommandEncoder()
      else { throw invalid("Metal could not upload \(label)") }
      blit.copy(
        from: staging, sourceOffset: 0, to: result, destinationOffset: 0, size: staging.length)
      blit.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw invalid("Metal could not complete \(label) upload")
      }
      return result
    }

    private static func makeBoundedBuffer(
      device: MTLDevice, bytes: Int, options: MTLResourceOptions, label: String,
      allocatedBefore: UInt64, maximumAdditionalBytes: UInt64?
    ) throws -> MTLBuffer {
      guard bytes > 0, bytes <= device.maxBufferLength else {
        throw invalid("Paired-runtime \(label) exceeds Metal buffer limits")
      }
      if let maximumAdditionalBytes {
        let allocatedNow = UInt64(device.currentAllocatedSize)
        let active = allocatedNow > allocatedBefore ? allocatedNow - allocatedBefore : 0
        guard active <= maximumAdditionalBytes,
          UInt64(bytes) <= maximumAdditionalBytes - active
        else {
          throw invalid(
            "Paired-runtime \(label) needs \(active + UInt64(bytes)) bytes at this stage, "
              + "but the load budget is \(maximumAdditionalBytes) bytes")
        }
      }
      return try makeBuffer(device: device, bytes: bytes, options: options, label: label)
    }

    private static func makeBuffer(
      device: MTLDevice, bytes: Int, options: MTLResourceOptions, label: String
    ) throws -> MTLBuffer {
      guard bytes > 0, bytes <= device.maxBufferLength,
        let result = device.makeBuffer(length: bytes, options: options)
      else { throw invalid("Metal could not allocate \(bytes) bytes for \(label)") }
      result.label = label
      return result
    }

    private static func alignedUInt32Bytes(_ bytes: Int) -> Int {
      (bytes + 3) & ~3
    }
  }

  private static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }
}
