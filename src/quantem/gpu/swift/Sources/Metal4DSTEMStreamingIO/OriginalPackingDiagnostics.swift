import Foundation

/// Alternate execution is available only in explicitly instrumented test builds.
/// Ordinary consumers always use the validated defaults, independent of shell state.
enum OriginalPackingDiagnostics {
  static func enabled(_ name: String, byDefault defaultValue: Bool = false) -> Bool {
    #if QGPU_PACKING_DIAGNOSTICS
      switch ProcessInfo.processInfo.environment["QGPU_ORIGINAL_" + name] {
      case "1": return true
      case "0": return false
      default: return defaultValue
      }
    #else
      return defaultValue
    #endif
  }
}

extension OriginalHDF5Packing {
  struct Profile {
    var decodeWindowFrames = 0
    var readBytes: UInt64 = 0
    var read = 0.0, copy = 0.0, decodeGPU = 0.0, decodeWall = 0.0
    var decodeAndHeadersGPU = 0.0, decodeAndHeadersWall = 0.0
    var productsGPU = 0.0, productsWall = 0.0, packingGPU = 0.0, packingWall = 0.0
    var hashing = 0.0, writing = 0.0
    var headersGPU = 0.0, dpcGPU = 0.0, valuesGPU = 0.0, verifyGPU = 0.0
    var prefixWall = 0.0
    var scalarSlices = 0
    var scalarDecodeThreads = 128, bitshufflePackingThreads = 128
    var decodePipelineThreadLimit = 0
    var packingPipelineThreadLimit = 0
    var bitshufflePixelsPerThread = 1
    var alignedFillSlices = 0
    var zeroTailSlices = 0
    var alignedCopySlices = 0
    var transposeSlices = 0
    var directBitshuffleWindows = 0, directBitshuffleShortSlices = 0
    var directBitshuffleSIMDGatherWindows = 0
    var packedPayloadLayout: UInt32 = 0
    var maximumWidthHistogram: [Int] = []
    var directBitshuffleGPU = 0.0, directBitshuffleWall = 0.0
    var planRead = 0.0, planDecodeGPU = 0.0, planDecodeCPU = 0.0, planWrite = 0.0
    var planReadBytes: UInt64 = 0, planOutputBytes: UInt64 = 0
    var planWindows = 0, planFallbacks = 0
    var planOverlapWindows = 0
    var planOverlapHeaderBytes: UInt64 = 0
    var planStatus = "notRequested"
    var fusedWindows = 0
    var checkpointWindows = 0
    var fusedDPCWindows = 0
    var reusedDPC = false
    var privateDense = false
    var fusedDecodeHeaderWindows = 0
    var readAheadEnabled = false
    var readAheadDepth = 0
    var readWait = 0.0
    var maximumConcurrentInputBytes: UInt64 = 0
    var additionalReadReserveBytes: UInt64 = 0
    var json: [String: Any] {
      [
        "source_read_bytes": readBytes, "source_read_seconds": read,
        "decode_window_frames": decodeWindowFrames,
        "input_copy_seconds": copy, "decode_gpu_seconds": decodeGPU,
        "decode_and_headers_gpu_seconds": decodeAndHeadersGPU,
        "decode_and_headers_wall_seconds": decodeAndHeadersWall,
        "fused_decode_header_windows": fusedDecodeHeaderWindows,
        "decode_wall_seconds": decodeWall, "products_gpu_seconds": productsGPU,
        "products_wall_seconds": productsWall, "packing_gpu_seconds": packingGPU,
        "packing_wall_seconds": packingWall, "logical_hash_seconds": hashing,
        "output_write_and_hash_seconds": writing,
        "isolated_headers_gpu_seconds": headersGPU, "isolated_dpc_gpu_seconds": dpcGPU,
        "isolated_values_gpu_seconds": valuesGPU,
        "isolated_verify_and_retention_gpu_seconds": verifyGPU,
        "prefix_wall_seconds": prefixWall,
        "scalar_decode_slices": scalarSlices,
        "scalar_decode_threads": scalarDecodeThreads,
        "decode_pipeline_thread_limit": decodePipelineThreadLimit,
        "packing_pipeline_thread_limit": packingPipelineThreadLimit,
        "bitshuffle_packing_threads": bitshufflePackingThreads,
        "bitshuffle_pixels_per_thread": bitshufflePixelsPerThread,
        "aligned_repeat_fill_slices": alignedFillSlices,
        "zero_tail_slices": zeroTailSlices,
        "aligned_history_copy_slices": alignedCopySlices,
        "transpose_unshuffle_slices": transposeSlices,
        "direct_bitshuffle_windows": directBitshuffleWindows,
        "direct_bitshuffle_short_slices": directBitshuffleShortSlices,
        "direct_bitshuffle_simd_gather_windows": directBitshuffleSIMDGatherWindows,
        "packed_payload_layout": packedPayloadLayout,
        "maximum_width_histogram": maximumWidthHistogram,
        "direct_bitshuffle_gpu_seconds": directBitshuffleGPU,
        "direct_bitshuffle_wall_seconds": directBitshuffleWall,
        "direct_bitshuffle_dense_bytes": 0,
        "packing_plan_read_seconds": planRead,
        "packing_plan_decode_gpu_seconds": planDecodeGPU,
        "packing_plan_decode_cpu_seconds": planDecodeCPU,
        "packing_plan_overlap_windows": planOverlapWindows,
        "packing_plan_overlap_header_bytes": planOverlapHeaderBytes,
        "packing_plan_write_seconds": planWrite,
        "packing_plan_read_bytes": planReadBytes,
        "packing_plan_output_bytes": planOutputBytes,
        "packing_plan_reused_windows": planWindows,
        "packing_plan_fallbacks": planFallbacks,
        "packing_plan_status": planStatus,
        "fused_packing_windows": fusedWindows,
        "checkpoint_packing_windows": checkpointWindows,
        "fused_dpc_windows": fusedDPCWindows,
        "prepared_dpc_reused": reusedDPC,
        "private_dense_window": privateDense,
        "compressed_read_ahead": readAheadEnabled,
        "compressed_read_ahead_depth": readAheadDepth,
        "compressed_read_wait_seconds": readWait,
        "maximum_concurrent_compressed_input_bytes": maximumConcurrentInputBytes,
        "additional_compressed_read_reserve_bytes": additionalReadReserveBytes,
        "isolated_kernel_profile": OriginalPackingDiagnostics.enabled(
          "PROFILE_KERNELS", byDefault: false),
        "direct_compressed_read": true,
      ]
    }
  }
}
