import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

extension MetalRuntimeANSResidentSource {
  /// Restore a CUDA-written ANS snapshot directly on Metal, including exact indexes.
  public static func load(
    snapshot: NativeANSSnapshot, device: MTLDevice,
    maximumAdditionalBytes: UInt64? = nil
  ) throws -> MetalRuntimeANSResidentSource {
    let started = CFAbsoluteTimeGetCurrent()
    guard
      snapshot.dtype == "uint8"
        || UInt64(snapshot.shape[2] * snapshot.shape[3]) * 65_535 <= UInt64(UInt32.max)
    else {
      throw invalid(
        "This uint16 detector can exceed native UInt32 product storage. Use the Python MPS loader for exact UInt64 detector sums."
      )
    }
    if let budget = maximumAdditionalBytes, UInt64(snapshot.bodyBytes + (8 << 20)) > budget {
      throw invalid(
        "The encoded acquisition exceeds the Metal memory budget; release other datasets.")
    }
    let mapped = try snapshot.verifiedMapping()
    let chunks: [Chunk] = try mapped.withUnsafeBytes { bytes in
      try snapshot.chunks.map { chunk in
        let buffers = try chunk.arrays.enumerated().map { index, span -> MTLBuffer in
          // Spatial buffers use a four-byte minimum in the original encoder.
          // Restore that allocation without reading padding beyond the file span.
          let length = max(index >= 3 ? 4 : 1, span.bytes)
          guard
            let buffer = device.makeBuffer(
              length: length, options: .storageModeShared)
          else {
            throw invalid("Metal could not allocate the encoded snapshot.")
          }
          if span.bytes > 0 {
            memcpy(buffer.contents(),
              bytes.baseAddress!.advanced(by: snapshot.dataStart + span.offset), span.bytes)
          }
          if length > span.bytes {
            memset(buffer.contents().advanced(by: span.bytes), 0, length - span.bytes)
          }
          return buffer
        }
        return Chunk(
          firstScan: chunk.first, scanCount: chunk.scans,
          payload: buffers[0], offsets: buffers[1], models: buffers[2],
          spatial: Array(buffers[3...]))
      }
    }
    let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw invalid("Missing ANS Metal kernel \(name).")
      }
      return try device.makeComputePipelineState(function: function)
    }
    let tables = RuntimeANSEncoder.tables()
    let decoding = try tables.decoding.withUnsafeBytes { bytes -> MTLBuffer in
      guard
        let buffer = device.makeBuffer(
          bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      else {
        throw invalid("Metal could not allocate ANS decoding tables.")
      }
      return buffer
    }
    let built = try RuntimeANSEncoder.Output(
      chunks: chunks, decoding: decoding, fusedDecodeAndEncodeSeconds: 0,
      prefixSeconds: 0, compactSeconds: 0,
      // Snapshot restore reuses the saved payload; there is no encode pass.
      encodeGPUSeconds: 0, encodeStallSeconds: 0,
      compactGPUSeconds: 0, compactStallSeconds: 0, encodeWindows: 0,
      decodePipeline: pipeline(
        ProcessInfo.processInfo.environment["QGPU_K3_FRAME_PREFIX"] == "0"
          ? "streamed_counts_decode_range" : "camera_frame"),
      detectorDeltaPipeline: pipeline("streamed_counts_detector_delta"),
      detectorPacketPipeline: pipeline("streamed_counts_detector_packet4"), detectorPacketSIMDs: 4)
    return try MetalRuntimeANSResidentSource(
      dataset: snapshot.dataset, identity: snapshot.identity,
      built: built, totalSeconds: CFAbsoluteTimeGetCurrent() - started, device: device)
  }
}
