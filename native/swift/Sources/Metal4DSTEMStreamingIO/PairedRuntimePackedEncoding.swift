import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

extension MetalCompactH5ResidentSource {
  /// Re-encode retained exact counts into Normal ANS without opening source files.
  /// This synchronous prototype uses one reusable 16K-scan private window.
  /// Serialize with interactions/release. The original stays valid on failure.
  @_spi(PairedRuntimeTANSPrototype)
  public func makeANSResident(
    maximumAdditionalBytes: UInt64,
    shouldCancel: () -> Bool = { false }
  ) throws -> MetalPairedRuntimeTANSResidentSource {
    guard !isReleased, let dataset = originalDataset,
      dataset.scanRows == 512, dataset.scanCols == 512,
      dataset.detectorRows == 192, dataset.detectorCols == 192,
      dataset.sourceDtype == "uint16" || dataset.sourceDtype == "uint8",
      let identity = dataset.sourceIdentitySHA256,
      let moments = try preparedDPCMomentValues()
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "ANS conversion requires a live original-count 512x512x192x192 uint8/uint16 resident with exact DPC summaries"
      )
    }
    let started = CFAbsoluteTimeGetCurrent()
    let before = UInt64(device.currentAllocatedSize)
    let sum = before.addingReportingOverflow(maximumAdditionalBytes)
    let limit = min(
      device.recommendedMaxWorkingSetSize, sum.overflow ? UInt64.max : sum.partialValue)
    func check(_ bytes: UInt64 = 0) throws {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      let held = UInt64(device.currentAllocatedSize)
      guard held <= limit, bytes <= limit - held else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "ANS conversion exceeds available memory; release another acquisition before retrying")
      }
    }
    try check()
    let configuration = PairedRuntimeConfiguration(mode: .normal)
    let dtype: Metal4DSTEMIntegerDType = dataset.sourceDtype == "uint8" ? .uint8 : .uint16
    var valid = [UInt8](repeating: 1, count: metadata.detectorPixelCount)
    for pixel in sourceHotPixelIndices where valid.indices.contains(pixel) { valid[pixel] = 0 }
    let descriptor = try PairedRuntimeTANSSeriesDescriptor(
      sourceIdentitySHA256: [identity],
      shape: [1, 512, 512, 192, 192], logicalDtype: dtype, detectorValidity: valid)
    // Returning this small result lets the scratch window/encoder release before consolidation.
    let result: MetalPairedRuntimeTANSBuildResult = try autoreleasepool {
      let bytes =
        PairedRuntimeTANSRecordABI.recordScans * metadata.detectorPixelCount * dtype.bytesPerValue
      try check(UInt64(bytes))
      guard let window = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
        throw Metal4DSTEMStreamingIOError.allocationFailed(
          label: "ANS conversion window", bytes: UInt64(bytes))
      }
      let resources = try MetalPairedRuntimeTANSHDF5Builder.Resources(
        device: device,
        descriptor: descriptor, allocatedBefore: before,
        maximumAdditionalBytes: maximumAdditionalBytes,
        configuration: configuration)
      var records = [PairedRuntimeTANSRecordBuffers]()
      var extents = [PairedRuntimeTANSRecordExtent]()
      var final: MTLCommandBuffer?
      var encodeSeconds = 0.0
      var prefixSeconds = 0.0
      var compactSeconds = 0.0
      for record in 0..<PairedRuntimeTANSRecordABI.recordsPerAcquisition {
        try check()
        try autoreleasepool {
          guard let command = resources.queue.makeCommandBuffer() else {
            throw Metal4DSTEMStreamingIOError.metalUnavailable("Cannot encode ANS conversion")
          }
          try encodeConversionWindow(
            firstScan: record * PairedRuntimeTANSRecordABI.recordScans,
            count: PairedRuntimeTANSRecordABI.recordScans, into: window, commands: command,
            bytesPerValue: dtype.bytesPerValue)
          let encoded = try resources.encode(
            dense: window, recordIndex: record,
            decodeCommand: command, shouldCancel: shouldCancel)
          records.append(encoded.record)
          extents.append(encoded.extent)
          final = encoded.completedCommand
          encodeSeconds += encoded.fusedSeconds
          prefixSeconds += encoded.prefixSeconds
          compactSeconds += encoded.compactSeconds
          try check()
        }
      }
      let receipt = try PairedRuntimeTANSProducerReceipt(
        sourceIdentitySHA256: [identity],
        recordExtents: extents, completedCommand: final!, failureFlag: resources.failure)
      let provider = try PairedRuntimeTANSRecordProvider(
        descriptor: descriptor,
        decodingTable: resources.decoding, records: records, receipt: receipt)
      let residentBytes =
        UInt64(resources.decoding.length)
        + records.reduce(UInt64(0)) {
          $0
            + UInt64(
              $1.payload.length + $1.offsets.length + $1.modes.length + ($1.workGroups?.length ?? 0)
            )
        }
      return MetalPairedRuntimeTANSBuildResult(
        provider: provider, streamPixels: resources.streamPixels,
        dpcMoments: moments,
        metrics: MetalPairedRuntimeTANSBuildMetrics(
          totalSeconds: CFAbsoluteTimeGetCurrent() - started,
          fusedDecodeAndSizeSeconds: encodeSeconds, provisionalCPUPrefixSeconds: prefixSeconds,
          compactSeconds: compactSeconds, consolidationSeconds: 0, residentBytes: residentBytes))
    }
    try check(result.metrics.residentBytes + (UInt64(32) << 20))
    return try MetalPairedRuntimeTANSResidentSource(
      dataset: dataset, provider: result.provider,
      streamPixels: result.streamPixels, dpcMoments: result.dpcMoments, metrics: result.metrics,
      loadStarted: started, device: device, configuration: configuration,
      allocationLimit: limit, shouldCancel: shouldCancel)
  }
}
