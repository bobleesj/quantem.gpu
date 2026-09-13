import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

extension MetalCompactH5Loader {
  /// Prepare an original indexed Arina acquisition for exact compact residency.
  ///
  /// Uses bounded native-precision Metal decode windows, never a dense volume.
  /// Every count, including source-marked bad pixels, is retained. Existing
  /// destinations are immutable. The caller owns cache location and eviction.
  /// Cancellation removes only this invocation's incomplete temporary output.
  /// Example: `try MetalCompactH5Loader.prepare(source: indexed, destinationURL: cache, device: device)`.
  @discardableResult
  public static func prepare(
    source: Native4DSTEMIndexedSource,
    destinationURL: URL,
    device: MTLDevice,
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> URL {
    let packing = try OriginalHDF5Packing(device: device)
    _ = try packing.pack(
      source: source, destination: destinationURL,
      maximumAdditionalBytes: nil, shouldCancel: shouldCancel, progress: progress)
    return destinationURL
  }

  /// Load all original Arina counts directly into lossless packed Metal storage.
  ///
  /// No packed-count file or full dense volume is created. Every packed count is
  /// checked against its decoded input on Metal before the resident is returned.
  /// Logical SHA-256 fields remain nil: this path does not hash a dense tensor.
  /// When the supplied memory budget admits them, exact detector-region sums
  /// accelerate mask interiors. Boundaries and diffraction reads still use
  /// original counts. Their storage and preparation are included in metrics.
  /// Optional source-bound DPC sums avoid repeating that reduction. Missing or
  /// incompatible provenance/bounds discard the hint and recompute exact sums.
  /// `packingPlanURL` optionally saves bounded, disposable layout metadata,
  /// never count payloads. Every reopen still reads and decodes original counts.
  /// Invalid plans trigger one fresh-plan retry; the caller owns cache eviction.
  /// Example: `let resident = try MetalCompactH5Loader.load(source: indexed, device: device)`.
  public static func load(
    source: Native4DSTEMIndexedSource, device: MTLDevice,
    maximumAdditionalBytes: UInt64? = nil,
    preparedDPC: MetalCompactH5ExactDPCMoments? = nil,
    packingPlanURL: URL? = nil,
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> MetalCompactH5ResidentSource {
    let started = ContinuousClock.now
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    let initialInputs = try OriginalHDF5Packing.inputStamps(source)
    let planURL = OriginalHDF5Packing.safePlanURL(packingPlanURL, source: source)
    let packing = try OriginalHDF5Packing.forLoad(device: device, cachePlans: planURL != nil)
    let result: OriginalPackedBuffers?
    do {
      result = try packing.pack(
        source: source, destination: nil,
        maximumAdditionalBytes: maximumAdditionalBytes, preparedDPC: preparedDPC,
        packingPlanURL: planURL, shouldCancel: shouldCancel, progress: progress)
    } catch let mismatch as OriginalHDF5Packing.CacheMismatch {
      // A metadata hint is never count truth. Discard all partial residents and
      // rebuild once from original counts, including the failed attempt in metrics.
      guard try OriginalHDF5Packing.inputStamps(source) == initialInputs else {
        throw OriginalHDF5Packing.invalid(
          "Original data changed during loading; reopen its folder and retry")
      }
      result = try packing.pack(
        source: source, destination: nil,
        maximumAdditionalBytes: maximumAdditionalBytes, preparedDPC: preparedDPC,
        packingPlanURL: mismatch.retryWithoutPlan ? nil : planURL,
        ignoreCachedPlan: true, priorProfile: mismatch.profile,
        shouldCancel: shouldCancel, progress: progress)
    }
    guard let packed = result
    else { throw OriginalHDF5Packing.invalid("Original resident packing produced no buffers") }
    guard try OriginalHDF5Packing.inputStamps(source) == initialInputs else {
      throw OriginalHDF5Packing.invalid(
        "Original data changed during loading; reopen its folder and retry")
    }
    let resident = try residentFromOriginal(
      packed, device: device, started: started,
      allocatedBefore: allocatedBefore, maximumAdditionalBytes: maximumAdditionalBytes,
      shouldCancel: shouldCancel)
    resident.sourceHotPixelIndices = source.dataset.badPixelIndices
    do {
      // Region preparation extends readiness past the final original read.
      // Recheck the complete input binding before handing ownership over.
      guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }
      guard try OriginalHDF5Packing.inputStamps(source) == initialInputs else {
        throw OriginalHDF5Packing.invalid(
          "Original data changed during resident preparation; reopen its folder and retry")
      }
      return resident
    } catch {
      resident.releaseResidentStorage()
      throw error
    }
  }
}

/// Internal ownership transfer, never a persisted cache or public result type.
struct OriginalPackedBuffers {
  var payloadLayout: UInt32 = 0
  let dataset: Native4DSTEMDataset
  let frames: Int
  let headerStride: Int
  let shards: [(payload: MTLBuffer, headers: MTLBuffer)]
  let moments: Data
  let detectorSum: [UInt64]
  let maximum: UInt32
  let maximumWidths: [UInt8]
  let calibration: (Double, Double, Double)
  let stagingBytes: UInt64
  let readSeconds: Double
  let decodeSeconds: Double
  let decodeAndHeaderSeconds: Double
  let packingSeconds: Double
  let reusedDPC: Bool
  let combinedDecodePackingSeconds: Double?
}

final class OriginalHDF5Packing {
  #if QGPU_PACKING_DIAGNOSTICS
    private final class ReusablePackingBox: @unchecked Sendable {
      var value: OriginalHDF5Packing?
    }
    private static let reusablePackingLock = NSLock()
    private static let reusablePacking = ReusablePackingBox()
  #endif
  /// A cache hint must never become an output destination for source or index data.
  static func safePlanURL(_ candidate: URL?, source: Native4DSTEMIndexedSource) -> URL? {
    guard let candidate, candidate.isFileURL else { return nil }
    let inputs =
      (source.dataset.masterPath.map { [URL(fileURLWithPath: $0)] } ?? [])
      + source.shards.flatMap { [$0.sourceURL, $0.indexURL] }
    let resolved = candidate.standardizedFileURL.resolvingSymlinksInPath()
    guard !inputs.contains(where: { $0.standardizedFileURL.resolvingSymlinksInPath() == resolved })
    else { return nil }
    if let output = try? OriginalPackingLayoutCache.SourceStamp(url: candidate),
      inputs.contains(where: {
        guard let input = try? OriginalPackingLayoutCache.SourceStamp(url: $0) else { return false }
        return input.device == output.device && input.inode == output.inode
      })
    {
      return nil
    }
    return candidate
  }

  struct CacheMismatch: Error {
    let profile: Profile
    var retryWithoutPlan = false
  }

  /// Reuse immutable Metal queues and pipeline state only in an instrumented
  /// diagnostic process. Production/UI loads keep their existing ownership
  /// semantics; the opt-in probe isolates driver setup churn from resident
  /// allocation churn during repeated source switches.
  static func forLoad(device: MTLDevice, cachePlans: Bool) throws -> OriginalHDF5Packing {
    #if QGPU_PACKING_DIAGNOSTICS
      if !cachePlans && OriginalPackingDiagnostics.enabled("REUSE_PACKER", byDefault: false) {
        return try reusablePackingLock.withLock {
          if let value = reusablePacking.value, value.device.registryID == device.registryID {
            return value
          }
          let created = try OriginalHDF5Packing(device: device, cachePlans: cachePlans)
          reusablePacking.value = created
          return created
        }
      }
    #endif
    return try OriginalHDF5Packing(device: device, cachePlans: cachePlans)
  }

  struct Shape { var scans, pixels, columns, sourceBytes: UInt32 }
  struct InputStamp: Equatable {
    let device: dev_t
    let inode: ino_t
    let bytes: off_t
    let seconds: time_t
    let nanoseconds: Int
    let changeSeconds: time_t
    let changeNanoseconds: Int

    init(path: String) throws {
      let descriptor = path.withCString { Darwin.open($0, O_RDONLY | O_CLOEXEC) }
      guard descriptor >= 0 else {
        throw OriginalHDF5Packing.invalid(
          "Cannot open an acquisition input; reopen its folder and retry")
      }
      defer { Darwin.close(descriptor) }
      var value = stat()
      guard fstat(descriptor, &value) == 0, value.st_mode & S_IFMT == S_IFREG else {
        throw OriginalHDF5Packing.invalid(
          "Cannot inspect an acquisition input; reopen its folder and retry")
      }
      device = value.st_dev
      inode = value.st_ino
      bytes = value.st_size
      seconds = value.st_mtimespec.tv_sec
      nanoseconds = value.st_mtimespec.tv_nsec
      changeSeconds = value.st_ctimespec.tv_sec
      changeNanoseconds = value.st_ctimespec.tv_nsec
    }
  }
  static func inputStamps(_ source: Native4DSTEMIndexedSource) throws -> [InputStamp] {
    let paths =
      (source.dataset.masterPath.map { [$0] } ?? []) + source.shards.map { $0.sourceURL.path }
    return try paths.map { try InputStamp(path: $0) }
  }
  let device: MTLDevice
  let queue: MTLCommandQueue
  let decodeQueue: MTLCommandQueue
  let decode8, decode16, decode32, unshuffle32, headersPipeline,
    valuesPipeline: MTLComputePipelineState
  let fusedDecodeUnshuffle, fusedDecodeUnshuffleVector,
    fusedDecodeUnshuffleFrameCoop: MTLComputePipelineState?
  let verifyPipeline, momentsPipeline, narrowPipeline: MTLComputePipelineState
  let scalarDecode, scalarUnshuffle, orderedScalarDecode, distance3ScalarDecode,
    distance8ScalarDecode: MTLComputePipelineState?
  let standardPlaneValues, standardPlaneValuesU16, standardPlaneSummary: MTLComputePipelineState?
  let rangesPipeline, verifiedValuesPipeline: MTLComputePipelineState?
  let checkpointPacking: Bool
  let alignedRepeatFill: Bool
  let alignedHistoryCopy: Bool
  let transposeUnshuffle: Bool
  let dpcUnshuffle, dpcUnshuffle32, dpcReduce: MTLComputePipelineState?
  let planDecode, summaryValues, summaryReduce: MTLComputePipelineState?
  let tokenPlanBuild, tokenPlanExpand: MTLComputePipelineState?
  let cpuPlanDecode: Bool
  let bitshuffleHeaders, bitshuffleHeadersVector4: MTLComputePipelineState?
  let bitshuffleValues, bitshuffleValuesDPC, bitshuffleValuesWidthBounded,
    bitshuffleValuesWidthBoundedDPC, bitshuffleValuesZeroTailDPC,
    bitshuffleReduce: MTLComputePipelineState?
  let bitshuffleDPCFusedReduce: MTLComputePipelineState?
  let bitshuffleDirectCombined, compactFixedPlanes: MTLComputePipelineState?
  let bitshuffleDPC, bitshuffleDPCPruned, bitshuffleDPCWide: MTLComputePipelineState?
  let zeroTailDecode, zeroTailValues: MTLComputePipelineState?
  var bitshufflePayloadLayout: UInt32 = 0
  var bitshufflePixelsPerThread = 1
  let bitshuffleSIMDGather: Bool
  let scalarDecodeThreads: Int
  let bitshufflePackingThreads: Int
  let standardPackingThreads: Int

  init(device: MTLDevice, cachePlans: Bool = false) throws {
    self.device = device
    cpuPlanDecode = cachePlans && OriginalPackingDiagnostics.enabled("CPU_PLAN", byDefault: true)
    guard let queue = device.makeCommandQueue() else {
      throw Self.invalid("No Metal command queue")
    }
    guard let decodeQueue = device.makeCommandQueue() else {
      throw Self.invalid("No Metal decode command queue")
    }
    self.queue = queue
    self.decodeQueue = decodeQueue
    let decode = try Metal4DSTEMKernels.makeHDF5Library(device: device)
    let packing = try Metal4DSTEMKernels.makeOriginalPackingLibrary(device: device)
    func pipeline(
      _ library: MTLLibrary, _ name: String, boundedDecode: Bool = false,
      boundedPacking: Bool = false
    ) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw Self.invalid("Missing kernel \(name)")
      }
      if (boundedDecode
        && OriginalPackingDiagnostics.enabled("FIXED_DECODE_PIPELINE", byDefault: false))
        || (boundedPacking
          && OriginalPackingDiagnostics.enabled("FIXED_PACK_PIPELINE", byDefault: false))
      {
        let fixedThreads =
          Int(
            ProcessInfo.processInfo.environment[
              boundedDecode
                ? "QGPU_ORIGINAL_FIXED_DECODE_THREADS" : "QGPU_ORIGINAL_FIXED_PACK_THREADS"
            ] ?? "32") ?? 32
        guard [32, 64, 128, 256, 512, 1024].contains(fixedThreads) else {
          throw Self.invalid(
            "Fixed Metal pipeline threads must be one of 32, 64, 128, 256, 512, or 1024")
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.maxTotalThreadsPerThreadgroup = fixedThreads
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        return try device.makeComputePipelineState(
          descriptor: descriptor, options: [], reflection: nil)
      }
      return try device.makeComputePipelineState(function: function)
    }
    decode8 = try pipeline(decode, Metal4DSTEMKernels.decodeU8Function)
    decode16 = try pipeline(decode, Metal4DSTEMKernels.decodeU16Function)
    decode32 = try pipeline(decode, "h5lz4dc_full_u32_qh5idx")
    unshuffle32 = try pipeline(decode, "h5unshuffle_u32_qh5idx")
    let fusedDecode = OriginalPackingDiagnostics.enabled("FUSED_DECODE", byDefault: false)
    fusedDecodeUnshuffle =
      fusedDecode
      ? try pipeline(decode, "h5lz4dc_unshuffle_u16_single_block_qh5idx") : nil
    fusedDecodeUnshuffleVector =
      fusedDecode
        && OriginalPackingDiagnostics.enabled("FUSED_DECODE_VECTOR", byDefault: false)
      ? try pipeline(decode, "h5lz4dc_unshuffle_u16_single_block_vector_qh5idx") : nil
    fusedDecodeUnshuffleFrameCoop =
      fusedDecode
        && OriginalPackingDiagnostics.enabled("FUSED_DECODE_FRAME_COOP", byDefault: false)
      ? try pipeline(decode, Metal4DSTEMKernels.decodeU16FrameCooperativeFunction) : nil
    alignedRepeatFill = OriginalPackingDiagnostics.enabled("ALIGNED_FILL", byDefault: true)
    alignedHistoryCopy = OriginalPackingDiagnostics.enabled("ALIGNED_COPY", byDefault: false)
    let fastDecode =
      OriginalPackingDiagnostics.enabled("FAST_DECODE", byDefault: false)
    let fixedNineBlockDecode =
      OriginalPackingDiagnostics.enabled("DECODE_FIXED_BLOCKS9", byDefault: false)
    let shortTokenDecode =
      OriginalPackingDiagnostics.enabled("DECODE_SHORT_TOKENS", byDefault: false)
    let shortRepeatFill =
      OriginalPackingDiagnostics.enabled("DECODE_REPEAT32", byDefault: false)
    transposeUnshuffle = OriginalPackingDiagnostics.enabled("TRANSPOSE_UNSHUFFLE", byDefault: true)
    if OriginalPackingDiagnostics.enabled("SCALAR_DECODE", byDefault: true) {
      let function: String
      switch (alignedRepeatFill, alignedHistoryCopy) {
      case (true, true): function = "h5lz4dc_full_u16_aligned_fill_copy_qh5idx"
      case (true, false):
        if fastDecode {
          function = "h5lz4dc_full_u16_fast_qh5idx"
        } else if fixedNineBlockDecode {
          function = "h5lz4dc_full_u16_aligned_fill_fixed9_qh5idx"
        } else if shortTokenDecode {
          function = "h5lz4dc_full_u16_aligned_fill_short_tokens_qh5idx"
        } else {
          function =
            shortRepeatFill
            ? "h5lz4dc_full_u16_aligned_fill_repeat32_qh5idx"
            : "h5lz4dc_full_u16_aligned_fill_qh5idx"
        }
      case (false, true): function = "h5lz4dc_full_u16_aligned_copy_qh5idx"
      case (false, false): function = "h5lz4dc_full_u16_scalar_qh5idx"
      }
      scalarDecode = try pipeline(decode, function, boundedDecode: true)
      scalarUnshuffle = try pipeline(
        decode,
        transposeUnshuffle
          ? "h5unshuffle_u16_transpose_qh5idx" : "h5unshuffle_u16_scalar_qh5idx")
    } else {
      scalarDecode = nil
      scalarUnshuffle = nil
    }
    orderedScalarDecode =
      OriginalPackingDiagnostics.enabled("DECODE_ORDERED", byDefault: false)
      ? try pipeline(decode, "h5lz4dc_full_u16_aligned_fill_ordered_qh5idx", boundedDecode: true)
      : nil
    distance3ScalarDecode =
      alignedRepeatFill && !alignedHistoryCopy && !fastDecode
        && !shortTokenDecode
        && OriginalPackingDiagnostics.enabled("DECODE_DISTANCE3", byDefault: false)
      ? try pipeline(decode, "h5lz4dc_full_u16_aligned_fill_distance3_qh5idx", boundedDecode: true)
      : nil
    distance8ScalarDecode =
      alignedRepeatFill && !alignedHistoryCopy && !fastDecode
        && !shortTokenDecode
        && OriginalPackingDiagnostics.enabled("DECODE_DISTANCE8", byDefault: false)
      ? try pipeline(decode, "h5lz4dc_full_u16_aligned_fill_distance8_qh5idx", boundedDecode: true)
      : nil
    if OriginalPackingDiagnostics.enabled("FUSED_DPC", byDefault: true) {
      dpcUnshuffle = try pipeline(decode, "h5unshuffle_u16_dpc_qh5idx")
      dpcUnshuffle32 = try pipeline(decode, "h5unshuffle_u32_dpc_qh5idx")
      dpcReduce = try pipeline(decode, "h5reduce_u16_dpc_qh5idx")
    } else {
      dpcUnshuffle = nil
      dpcUnshuffle32 = nil
      dpcReduce = nil
    }
    checkpointPacking =
      cachePlans || OriginalPackingDiagnostics.enabled("CHECKPOINT_PACK", byDefault: true)
    if cachePlans || OriginalPackingDiagnostics.enabled("FUSED_PACK", byDefault: true) {
      rangesPipeline = try pipeline(packing, "original_packing_validate_ranges")
      verifiedValuesPipeline = try pipeline(
        packing,
        checkpointPacking
          ? "original_packing_values_verified_checkpoints" : "original_packing_values_verified")
    } else {
      rangesPipeline = nil
      verifiedValuesPipeline = nil
    }
    if cachePlans {
      planDecode =
        cpuPlanDecode ? nil : try pipeline(decode, "h5lz4dc_full_u16_aligned_fill_qh5idx")
      summaryValues = try pipeline(packing, "original_packing_values_verified_checkpoints_summary")
      summaryReduce = try pipeline(packing, "original_packing_reduce_verified_summary")
    } else {
      planDecode = nil
      summaryValues = nil
      summaryReduce = nil
    }
    let gpuTokenPlan = OriginalPackingDiagnostics.enabled("GPU_TOKEN_PLAN", byDefault: false)
    tokenPlanBuild =
      gpuTokenPlan
      ? try pipeline(decode, "h5lz4_build_token_plan_u16_qh5idx", boundedDecode: true)
      : nil
    tokenPlanExpand =
      gpuTokenPlan
      ? try pipeline(
        decode,
        "h5lz4_expand_token_plan_u16_qh5idx",
        boundedDecode: true)
      : nil
    if checkpointPacking, rangesPipeline != nil, verifiedValuesPipeline != nil,
      OriginalPackingDiagnostics.enabled("STANDARD_PLANES", byDefault: true)
    {
      let skipStoreVerification =
        OriginalPackingDiagnostics.enabled("SKIP_STANDARD_STORE_VERIFY", byDefault: false)
      standardPlaneValues = try pipeline(
        packing,
        skipStoreVerification
          ? "original_packing_values_planes_checkpoints_unchecked"
          : "original_packing_values_planes_checkpoints")
      standardPlaneValuesU16 =
        skipStoreVerification
        ? nil
        : try pipeline(
          packing,
          OriginalPackingDiagnostics.enabled("WIDTH56_LUT", byDefault: false)
            ? "original_packing_values_planes_checkpoints_u16_width56_diagnostic"
            : "original_packing_values_planes_checkpoints_u16")
      standardPlaneSummary = try pipeline(
        packing, "original_packing_values_planes_checkpoints_summary")
    } else {
      standardPlaneValues = nil
      standardPlaneValuesU16 = nil
      standardPlaneSummary = nil
    }
    let directScratchEnabled =
      OriginalPackingDiagnostics.enabled("DIRECT_SCRATCH", byDefault: false)
    if (cpuPlanDecode || directScratchEnabled)
      && OriginalPackingDiagnostics.enabled("DIRECT_BITSHUFFLE", byDefault: true)
    {
      bitshuffleHeaders = try pipeline(packing, "original_packing_bitshuffle_headers")
      let usePlanes =
        OriginalPackingDiagnostics.enabled("PLANES", byDefault: true)
        && OriginalPackingDiagnostics.enabled("SIMD_GATHER", byDefault: true)
      let vectorPlanes =
        usePlanes && OriginalPackingDiagnostics.enabled("PLANE_VECTOR4", byDefault: true)
      let vectorColumns = vectorPlanes ? 4 : 1
      bitshuffleHeadersVector4 =
        vectorPlanes
        ? try pipeline(packing, "original_packing_bitshuffle_headers_vector4", boundedPacking: true)
        : nil
      let skipDirectStoreVerify =
        OriginalPackingDiagnostics.enabled("SKIP_DIRECT_STORE_VERIFY", byDefault: false)
      let widthAwareHighPlanes =
        OriginalPackingDiagnostics.enabled("WIDTH_AWARE_HIGH_PLANES", byDefault: false)
      let low8Only = OriginalPackingDiagnostics.enabled("LOW8_ONLY", byDefault: false)
      let planeFunction =
        vectorPlanes
        ? (low8Only
          ? "original_packing_bitshuffle_planes_vector4_low8_summary"
          : (widthAwareHighPlanes
            ? "original_packing_bitshuffle_planes_vector4_widthaware_summary"
            : (skipDirectStoreVerify
              ? "original_packing_bitshuffle_planes_vector4_unchecked_summary"
              : "original_packing_bitshuffle_planes_vector4_summary")))
        : "original_packing_bitshuffle_planes_verified_summary"
      let cooperative =
        OriginalPackingDiagnostics.enabled("SIMD_GATHER", byDefault: true)
        ? try? pipeline(
          packing,
          usePlanes
            ? planeFunction
            : "original_packing_bitshuffle_transpose_verified_summary",
          boundedPacking: vectorPlanes) : nil
      if usePlanes && cooperative == nil {
        throw Self.invalid(
          "Exact bit-plane packing could not create its SIMD kernel; rebuild the Metal resources and retry"
        )
      }
      if let cooperative, cooperative.threadExecutionWidth == 32,
        cooperative.maxTotalThreadsPerThreadgroup
          >= (OriginalPackingDiagnostics.enabled("FIXED_PACK_PIPELINE", byDefault: false)
            ? 32 : 128)
      {
        bitshuffleValues = cooperative
        bitshuffleSIMDGather = true
        bitshufflePayloadLayout = usePlanes ? 1 : 0
        bitshufflePixelsPerThread = vectorColumns
      } else {
        bitshuffleValues = try pipeline(packing, "original_packing_bitshuffle_verified_summary")
        bitshuffleSIMDGather = false
      }
      bitshuffleReduce = try pipeline(packing, "original_packing_reduce_bitshuffle_summary")
      if vectorPlanes {
        bitshuffleValuesDPC = try pipeline(
          packing, "original_packing_bitshuffle_planes_vector4_dpc_summary", boundedPacking: true)
        bitshuffleValuesWidthBounded = try pipeline(
          packing, "original_packing_bitshuffle_planes_vector4_widthbounded_summary",
          boundedPacking: true)
        bitshuffleValuesWidthBoundedDPC = try pipeline(
          packing, "original_packing_bitshuffle_planes_vector4_widthbounded_dpc_summary",
          boundedPacking: true)
        bitshuffleValuesZeroTailDPC = try pipeline(
          packing, "original_packing_bitshuffle_planes_vector4_zero_tail_dpc_summary",
          boundedPacking: true)
        bitshuffleDPCFusedReduce = try pipeline(
          packing, "original_packing_reduce_bitshuffle_dpc_fused", boundedPacking: true)
        bitshuffleDirectCombined = try pipeline(
          packing, "original_packing_bitshuffle_direct_combined", boundedPacking: true)
        compactFixedPlanes = try pipeline(
          packing, "original_packing_compact_fixed_planes", boundedPacking: true)
      } else {
        bitshuffleValuesDPC = nil
        bitshuffleValuesWidthBounded = nil
        bitshuffleValuesWidthBoundedDPC = nil
        bitshuffleValuesZeroTailDPC = nil
        bitshuffleDPCFusedReduce = nil
        bitshuffleDirectCombined = nil
        compactFixedPlanes = nil
      }
    } else {
      bitshuffleHeaders = nil
      bitshuffleHeadersVector4 = nil
      bitshuffleValues = nil
      bitshuffleValuesDPC = nil
      bitshuffleValuesWidthBounded = nil
      bitshuffleValuesWidthBoundedDPC = nil
      bitshuffleValuesZeroTailDPC = nil
      bitshuffleReduce = nil
      bitshuffleDPCFusedReduce = nil
      bitshuffleDirectCombined = nil
      compactFixedPlanes = nil
      bitshuffleSIMDGather = false
    }
    headersPipeline = try pipeline(packing, "original_packing_headers")
    valuesPipeline = try pipeline(packing, "original_packing_values")
    verifyPipeline = try pipeline(packing, "original_packing_verify")
    momentsPipeline = try pipeline(packing, "original_packing_moments")
    bitshuffleDPC =
      OriginalPackingDiagnostics.enabled("DIRECT_DPC", byDefault: true)
      ? try pipeline(packing, "original_packing_bitshuffle_dpc") : nil
    bitshuffleDPCPruned =
      OriginalPackingDiagnostics.enabled("DPC_PRUNE_HIGH_PLANES", byDefault: false)
      ? try pipeline(packing, "original_packing_bitshuffle_dpc_pruned_high_planes") : nil
    bitshuffleDPCWide =
      OriginalPackingDiagnostics.enabled("DPC_WIDE_GROUPS", byDefault: false)
      ? try pipeline(packing, "original_packing_bitshuffle_dpc_wide_groups") : nil
    if OriginalPackingDiagnostics.enabled("ZERO_TAIL", byDefault: true) {
      zeroTailDecode = try pipeline(
        decode, "h5lz4dc_full_u16_zero_tail_qh5idx", boundedDecode: true)
      zeroTailValues = try pipeline(
        packing, "original_packing_bitshuffle_planes_zero_tail_summary", boundedPacking: true)
    } else {
      zeroTailDecode = nil
      zeroTailValues = nil
    }
    narrowPipeline = try pipeline(packing, "original_packing_u8")
    func diagnosticThreads(_ name: String, maximum: Int, defaultValue: Int = 32) throws -> Int {
      #if QGPU_PACKING_DIAGNOSTICS
        if let text = ProcessInfo.processInfo.environment["QGPU_ORIGINAL_" + name] {
          guard let value = Int(text), [32, 64, 96, 128, 256, 512].contains(value), value <= maximum
          else {
            throw Self.invalid(
              "Unsupported \(name); select a supported multiple of32 from32,64,96,128,256,512")
          }
          return value
        }
      #endif
      return defaultValue
    }
    scalarDecodeThreads = try diagnosticThreads(
      "DECODE_THREADS",
      maximum: scalarDecode?.maxTotalThreadsPerThreadgroup ?? 128,
      defaultValue: min(128, scalarDecode?.maxTotalThreadsPerThreadgroup ?? 128))
    bitshufflePackingThreads = try diagnosticThreads(
      "PACK_THREADS",
      maximum: bitshuffleValues?.maxTotalThreadsPerThreadgroup ?? 128,
      defaultValue: 32)
    standardPackingThreads = try diagnosticThreads(
      "STANDARD_PACK_THREADS",
      maximum: standardPlaneValues?.maxTotalThreadsPerThreadgroup ?? 128,
      defaultValue: 128)
  }

  func pack(
    source: Native4DSTEMIndexedSource, destination: URL?, maximumAdditionalBytes: UInt64?,
    preparedDPC: MetalCompactH5ExactDPCMoments? = nil,
    packingPlanURL: URL? = nil, ignoreCachedPlan: Bool = false, priorProfile: Profile? = nil,
    shouldCancel: () -> Bool, progress: (Int, Int) -> Void
  ) throws -> OriginalPackedBuffers? {
    let dataset = source.dataset
    let selectedStandardPlaneValues =
      source.sourceBytesPerValue == 2 ? standardPlaneValuesU16 : standardPlaneValues
    let standardPlanes = destination == nil && selectedStandardPlaneValues != nil
    guard let identity = dataset.sourceIdentitySHA256,
      source.logicalFrameCount.isMultiple(of: 32),
      ["uint8", "uint16", "uint32"].contains(dataset.sourceDtype)
    else {
      throw Self.invalid(
        "Original packed loading requires indexed uint8/uint16/uint32 counts and a scan count divisible by 32; no crop or bin was applied"
      )
    }
    guard destination == nil || source.sourceBytesPerValue != 4 else {
      throw Self.invalid(
        "uint32 source loading requires direct packed residency; on-disk packed export is not supported yet"
      )
    }
    // Reject master or data changes during loading, even when a writer restores
    // the modification timestamp. These cheap stamps are not content hashes.
    let masterStamp = try dataset.masterPath.map { try InputStamp(path: $0) }
    let dataStamps = try source.shards.map { try InputStamp(path: $0.sourceURL.path) }
    func validateInputs() throws {
      guard try dataset.masterPath.map({ try InputStamp(path: $0) }) == masterStamp else {
        throw Self.invalid(
          "The acquisition master changed while loading; reopen its folder and retry")
      }
      guard try source.shards.map({ try InputStamp(path: $0.sourceURL.path) }) == dataStamps else {
        throw Self.invalid("Original data changed while loading; reopen its folder and retry")
      }
    }
    if let destination, FileManager.default.fileExists(atPath: destination.path) {
      throw Self.invalid(
        "Prepared destination already exists; validate and reuse it or choose a new cache key")
    }
    let started = CFAbsoluteTimeGetCurrent()
    let pixels = dataset.detectorRows * dataset.detectorCols
    // Target 160 MiB of dense decode storage, with one 32-scan packing tile minimum.
    // Smaller detector geometries retain the existing 4096-frame window.
    // Only processing windows change, never the requested scan coverage.
    var frames = min(4096, source.logicalFrameCount)
    var denseWindowBudget = UInt64(160) << 20
    #if QGPU_PACKING_DIAGNOSTICS
      if let text = ProcessInfo.processInfo.environment["QGPU_ORIGINAL_WINDOW_MIB"],
        let value = UInt64(text), value > 0, value <= 1024
      {
        denseWindowBudget = value << 20
      }
    #endif
    while frames > 32 && UInt64(frames) * source.decodedBytesPerFrame > denseWindowBudget {
      frames = max(32, (frames / 2 / 32) * 32)
    }
    while source.logicalFrameCount % frames != 0 { frames -= 32 }
    let windows = try source.windows(
      maximumDecodedBytes: UInt64(frames) * source.decodedBytesPerFrame, alignToScanRows: false)
    guard frames * pixels * source.sourceBytesPerValue <= 512 << 20 else {
      throw Self.invalid(
        "One exact packing window exceeds 512 MiB; this detector geometry needs a smaller window")
    }
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    // Reserve bounded staging before any count storage is allocated. Grow the
    // resident only after checking each measured shard against the same budget.
    let forceNonScalarDecode =
      OriginalPackingDiagnostics.enabled("FORCE_NONSCALAR_DECODE", byDefault: false)
    let fusedDirectRequested =
      destination == nil
      && source.sourceBytesPerValue == 2
      && OriginalPackingDiagnostics.enabled("FUSED_DECODE", byDefault: false)
    let directScratchRequested =
      destination == nil
      && source.sourceBytesPerValue == 2
      && OriginalPackingDiagnostics.enabled("DIRECT_SCRATCH", byDefault: false)
      && !fusedDirectRequested
    let gpuTokenPlanRequested =
      directScratchRequested
      && tokenPlanBuild != nil && tokenPlanExpand != nil
    let useScalar =
      directScratchRequested
      ? scalarDecode != nil && pixels.isMultiple(of: 4096)
        && source.shards.allSatisfy { Int($0.index.metadata.nBlocksPerFrame) * 4096 == pixels }
      : (!fusedDirectRequested && !forceNonScalarDecode
        && scalarUnshuffle?.threadExecutionWidth == 32 && source.sourceBytesPerValue == 2
        && frames >= 2048 && pixels.isMultiple(of: 4096)
        && windows.contains { $0.slices.contains { $0.globalFrameRange.count >= 2048 } }
        && source.shards.allSatisfy { Int($0.index.metadata.nBlocksPerFrame) * 4096 == pixels })
    let scratchBytes =
      source.sourceBytesPerValue == 4 ? frames * pixels * 4 : (useScalar ? frames * pixels * 2 : 0)
    let cachedDPC = destination == nil ? validatedDPC(preparedDPC, source: source) : nil
    if destination == nil, !ignoreCachedPlan,
      cachedDPC != nil || bitshuffleDPC != nil, let packingPlanURL,
      let direct = try packBitshufflePlan(
        source: source, windows: windows, frames: frames,
        moments: cachedDPC, packingPlanURL: packingPlanURL,
        maximumAdditionalBytes: maximumAdditionalBytes, priorProfile: priorProfile,
        validateInputs: validateInputs, shouldCancel: shouldCancel, progress: progress)
    {
      return direct
    }
    let partialDPCBlocks = source.sourceBytesPerValue == 4 ? pixels / 2048 : pixels / 4096
    let partialBytes =
      cachedDPC == nil
        && ((source.sourceBytesPerValue == 4 && dpcUnshuffle32?.threadExecutionWidth == 32)
          || (useScalar && dpcUnshuffle?.threadExecutionWidth == 32))
      ? frames * partialDPCBlocks * 32 : 0
    // The existing plan summary uses uint32 partial sums. uint32 source data
    // builds fresh headers until that optional cache has a wide-sum schema.
    let cachePlans = destination == nil && packingPlanURL != nil && source.sourceBytesPerValue != 4
    // Covers one compressed record, upload, decoded header, codec/hash scratch.
    let planStaging: UInt64 = cachePlans ? 64 << 20 : 0
    // Keep the terms explicit so Swift 6.2 and Swift 6.3 can type-check the
    // expression consistently across the supported Apple hosts.
    let decodedStagingBytes =
      UInt64(frames) * UInt64(pixels) * UInt64(source.sourceBytesPerValue)
    let baseStagingBytes =
      decodedStagingBytes + UInt64(scratchBytes) + UInt64(partialBytes)
    let tokenPlanBlockBytes: UInt64 = 16 + 256 * 16
    let tokenPlanStaging =
      gpuTokenPlanRequested
      ? UInt64(frames)
        * UInt64(source.shards.map { Int($0.index.metadata.nBlocksPerFrame) }.max() ?? 0)
        * tokenPlanBlockBytes
      : 0
    let stagingReserve = baseStagingBytes + (768 << 20) + planStaging + tokenPlanStaging
    if let maximumAdditionalBytes, stagingReserve > maximumAdditionalBytes {
      if cachePlans {
        throw CacheMismatch(profile: priorProfile ?? Profile(), retryWithoutPlan: true)
      }
      throw Self.invalid(
        "Original decode staging exceeds the available memory budget; release another resident and retry"
      )
    }
    var temporary: URL?
    var output: FileHandle?
    if let destination {
      try FileManager.default.createDirectory(
        at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
      let path = destination.deletingLastPathComponent().appendingPathComponent(
        ".packing-\(UUID().uuidString).partial")
      guard FileManager.default.createFile(atPath: path.path, contents: nil) else {
        throw Self.invalid("Cannot create the local packed cache")
      }
      temporary = path
      output = try FileHandle(forWritingTo: path)
    }
    defer { if let temporary { try? FileManager.default.removeItem(at: temporary) } }
    defer { try? output?.close() }
    let reserved = 1 << 20
    try output?.write(contentsOf: Data(count: reserved))
    let writer = output.map { OutputWriter(output: $0) }
    defer { writer?.drain() }
    let slotCount = destination == nil ? 1 : 2
    // File preparation hashes dense bytes on the CPU; direct residency does
    // not read them there. Keep that distinction explicit for this experiment.
    let privateDense =
      destination == nil
      && OriginalPackingDiagnostics.enabled("PRIVATE_DENSE", byDefault: false)
    let denseSlots = try (0..<slotCount).map { _ in
      try buffer(frames * pixels * source.sourceBytesPerValue, privateStorage: privateDense)
    }
    let narrowSlots = try (0..<(destination == nil ? 0 : slotCount)).map { _ in
      try buffer(frames * pixels)
    }
    let availableSlots = (0..<slotCount).map { _ in DispatchSemaphore(value: 1) }
    let tiles = frames / 32
    let checkpoints = (tiles + 31) / 32
    let headerStride = checkpoints + (tiles + 7) / 8
    let headerBytes = pixels * headerStride * 4
    let headerCapacity = cachePlans ? ((headerBytes + 8191) / 8192) * 8192 : headerBytes
    // One spare byte lets the CPU metadata codec reject overlong decoded blocks.
    let headers = try buffer(headerCapacity + (cpuPlanDecode && cachePlans ? 1 : 0))
    let planSources =
      (dataset.masterPath.map { [URL(fileURLWithPath: $0)] } ?? [])
      + source.shards.map(\.sourceURL)
    let binding =
      cachePlans
      ? try? OriginalPackingLayoutCache.Binding.capture(
        sourceIdentity: identity, sourceFiles: planSources,
        scanRows: dataset.scanRows, scanColumns: dataset.scanCols,
        detectorRows: dataset.detectorRows, detectorColumns: dataset.detectorCols,
        sourceDtype: dataset.sourceDtype, framesPerWindow: frames,
        headerWordsPerPixel: headerStride) : nil
    let planReader =
      ignoreCachedPlan
      ? nil
      : binding.flatMap { binding in
        packingPlanURL.flatMap { OriginalPackingLayoutCache.Reader(url: $0, binding: binding) }
      }
    let planWriter =
      planReader == nil
      ? binding.flatMap { binding in
        packingPlanURL.flatMap { OriginalPackingLayoutCache.Writer(url: $0, binding: binding) }
      } : nil
    let planPartials = planReader != nil ? try buffer(pixels * checkpoints * 4) : nil
    let sizes = try buffer(pixels * 4)
    let sums = try buffer(pixels * 8)
    let moments = try buffer(frames * 32)
    let audit = try buffer(frames * 8)
    let mask = try buffer(pixels)
    let errors = try buffer(4)
    let payloadWords = try buffer(4)
    let scalarScratch =
      scratchBytes > 0
      ? try buffer(scratchBytes, privateStorage: true) : nil
    let tokenPlanBlocksPerFrame =
      source.shards.map {
        Int($0.index.metadata.nBlocksPerFrame)
      }.max() ?? 0
    let tokenPlanHeaders =
      gpuTokenPlanRequested && tokenPlanBlocksPerFrame > 0
      ? try buffer(frames * tokenPlanBlocksPerFrame * 16, privateStorage: true) : nil
    let tokenPlanOps =
      gpuTokenPlanRequested && tokenPlanBlocksPerFrame > 0
      ? try buffer(frames * tokenPlanBlocksPerFrame * 256 * 16, privateStorage: true) : nil
    let directScratch =
      destination == nil
      && source.sourceBytesPerValue == 2
      && scalarScratch != nil
      && bitshuffleHeaders != nil
      && bitshuffleValues != nil
      && bitshuffleDPC != nil
      && OriginalPackingDiagnostics.enabled("DIRECT_SCRATCH", byDefault: false)
    let fusedDirect =
      fusedDirectRequested
      && fusedDecodeUnshuffle != nil
      && source.sourceBytesPerValue == 2
      && pixels.isMultiple(of: 4096)
      && source.shards.allSatisfy {
        Int($0.index.metadata.nBlocksPerFrame) * 4096 == pixels
      }
    let directCombined =
      directScratch
      && OriginalPackingDiagnostics.enabled("COMBINED_DIRECT", byDefault: false)
      && bitshuffleDirectCombined != nil && bitshuffleDPCFusedReduce != nil
    // Diagnostic two-queue pipeline: while the pack queue materializes window
    // N, the decode queue prepares window N+1 in a second scratch slot. The
    // default remains serialized until this path passes parity and consistency
    // screens on the target machine.
    let pipelineWindows =
      directScratch && !directCombined && planReader == nil && planWriter == nil
      && OriginalPackingDiagnostics.enabled("WINDOW_PIPELINE", byDefault: false)
    let directWidthBounded =
      directScratch
      && OriginalPackingDiagnostics.enabled("WIDTH_BOUNDED", byDefault: false)
      && bitshuffleHeadersVector4 != nil
      && bitshuffleValuesWidthBoundedDPC != nil
      && bitshuffleDPCFusedReduce != nil
    let directWidthBoundedPayload =
      directScratch
      && OriginalPackingDiagnostics.enabled("WIDTH_BOUNDED_PAYLOAD", byDefault: false)
      && bitshuffleHeadersVector4 != nil
      && bitshuffleValuesWidthBounded != nil
    let directZeroTailEnabled =
      directScratch
      && !pipelineWindows
      && OriginalPackingDiagnostics.enabled("ZERO_TAIL_DIRECT", byDefault: false)
      && bitshuffleHeadersVector4 != nil
      && zeroTailDecode != nil
      && bitshuffleValuesZeroTailDPC != nil
      && bitshuffleDPCFusedReduce != nil
    guard !directScratch || standardPlanes else {
      throw Self.invalid(
        "Direct bitshuffle scratch loading requires the exact standard plane packer")
    }
    let directHighPlaneWords =
      directScratch && !directCombined
      ? try buffer((pixels + 31) / 32 * MemoryLayout<UInt32>.stride) : nil
    let pipelineScratch =
      pipelineWindows ? try buffer(scratchBytes, privateStorage: true) : nil
    let pipelineDecodeErrors = pipelineWindows ? try buffer(4) : nil
    let directCombinedFixedPayload =
      directCombined
      ? try buffer(pixels * (frames / 32) * 16 * MemoryLayout<UInt32>.stride, privateStorage: true)
      : nil
    let directCombinedDPCPartials =
      directCombined
      ? try buffer(frames * (pixels / 128) * MemoryLayout<UInt64>.stride * 4, privateStorage: true)
      : nil
    let directZeroTails =
      directZeroTailEnabled
      ? try buffer(frames * (pixels / 4096) * MemoryLayout<UInt32>.stride, privateStorage: true)
      : nil
    let directPackingDPCPartials =
      directScratch && !directCombined
        && OriginalPackingDiagnostics.enabled("FUSED_PACK_DPC", byDefault: true)
        && (directZeroTailEnabled ? bitshuffleValuesZeroTailDPC != nil : bitshuffleValuesDPC != nil)
        && bitshuffleDPCFusedReduce != nil
      ? try buffer(frames * (pixels / 128) * MemoryLayout<UInt64>.stride * 4, privateStorage: true)
      : nil
    let partialDPC = partialBytes > 0 ? try buffer(partialBytes, privateStorage: true) : nil
    let widths = try buffer(pixels * 4)
    let directPartialSums =
      directScratch ? try buffer(pixels * checkpoints * 4, privateStorage: true) : nil
    let directPartialMaximums =
      directScratch ? try buffer(pixels * checkpoints * 4, privateStorage: true) : nil
    memset(widths.contents(), 0, widths.length)
    memset(mask.contents(), 0, mask.length)  // Preserve even source-marked hot pixels.
    var maximum: UInt32 = 0
    var residentShards: [(payload: MTLBuffer, headers: MTLBuffer)] = []
    var residentMomentData = cachedDPC ?? Data()
    var residentBytes: UInt64 = 0
    var detectorSum = [UInt64](repeating: 0, count: pixels)
    var profile = priorProfile ?? Profile()
    profile.scalarDecodeThreads = scalarDecodeThreads
    profile.decodePipelineThreadLimit = scalarDecode?.maxTotalThreadsPerThreadgroup ?? 0
    profile.bitshufflePackingThreads = bitshufflePackingThreads
    profile.decodeWindowFrames = frames
    if ignoreCachedPlan { profile.planFallbacks += 1 }
    profile.planStatus = !cachePlans ? "notRequested" : (planReader != nil ? "hit" : "miss")
    profile.reusedDPC = cachedDPC != nil
    profile.privateDense = privateDense
    let countStaging =
      denseSlots.reduce(0) { $0 + $1.length }
      + narrowSlots.reduce(0) { $0 + $1.length }
    let metadataStaging = headers.length * 3 + sizes.length + sums.length
    let auxiliaryStaging =
      moments.length * 3 + audit.length + mask.length + errors.length + widths.length
      + payloadWords.length + (directPackingDPCPartials?.length ?? 0)
      + (directZeroTails?.length ?? 0)
      + (directCombinedFixedPayload?.length ?? 0) + (directCombinedDPCPartials?.length ?? 0)
      + (pipelineScratch?.length ?? 0) + (pipelineDecodeErrors?.length ?? 0)
      + (tokenPlanHeaders?.length ?? 0) + (tokenPlanOps?.length ?? 0)
    let fixedStaging =
      UInt64(countStaging + metadataStaging + auxiliaryStaging + scratchBytes + partialBytes)
      + UInt64(source.logicalFrameCount) * 32 + planStaging
    var largestInput: UInt64 = 0
    var largestPayload: UInt64 = 0
    var peakStaging = fixedStaging
    // Keep two compressed slices in flight for uint32 sources. Their decode and
    // packing work is GPU-bound, so a single serialized read leaves Metal idle
    // between slices while the SSD is still delivering the next input.
    let readAheadEnabled =
      destination == nil
      && OriginalPackingDiagnostics.enabled("READ_AHEAD", byDefault: false)
    var readAheadDepth = source.sourceBytesPerValue == 4 ? 2 : 1
    #if QGPU_PACKING_DIAGNOSTICS
      if let text = ProcessInfo.processInfo.environment["QGPU_ORIGINAL_READ_AHEAD_DEPTH"],
        let value = Int(text)
      {
        guard (1...8).contains(value) else {
          throw Self.invalid(
            "Unsupported READ_AHEAD_DEPTH; select an integer from 1 through 8")
        }
        readAheadDepth = value
      }
    #endif
    let reader =
      readAheadEnabled
      ? CompressedReadAhead(
        device: device, depth: readAheadDepth,
        coalesce: OriginalPackingDiagnostics.enabled("COALESCE_READS", byDefault: false),
        coalesceBatchSize: {
          #if QGPU_PACKING_DIAGNOSTICS
            if let text = ProcessInfo.processInfo.environment["QGPU_ORIGINAL_COALESCE_BATCH"],
              let value = Int(text), (1...4).contains(value)
            {
              return value
            }
          #endif
          return 2
        }()) : nil
    defer { reader?.cancelAndDrain() }
    let orderedSlices = readAheadEnabled ? windows.flatMap(\.slices) : []
    var sliceOrdinal = 0
    var pendingReadBytes: UInt64 = 0
    var additionalReadReserve: UInt64 = 0
    profile.readAheadEnabled = readAheadEnabled
    profile.readAheadDepth = readAheadEnabled ? readAheadDepth : 0
    func enqueueRead(_ slice: Native4DSTEMIndexedSlice, currentInputBytes: UInt64) throws {
      guard let reader else { return }
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      let plan = try compressedReadPlan(slice, source: source)
      additionalReadReserve = max(additionalReadReserve, plan.reservedBytes)
      // The allocated fixed buffers are known now. Admit both live inputs
      // and the existing conservative two-payload reserve before submission.
      let prospectiveInput = max(largestInput, currentInputBytes + plan.reservedBytes)
      let prospectiveStaging = fixedStaging + prospectiveInput + largestPayload * 2
      if let maximumAdditionalBytes,
        residentBytes + prospectiveStaging > maximumAdditionalBytes
      {
        if cachePlans { throw CacheMismatch(profile: profile, retryWithoutPlan: true) }
        throw Self.invalid(
          "Compressed read-ahead exceeds the available memory budget; release another resident and retry"
        )
      }
      largestInput = prospectiveInput
      peakStaging = fixedStaging + largestInput + largestPayload * 2
      profile.maximumConcurrentInputBytes = largestInput
      profile.additionalReadReserveBytes = additionalReadReserve
      try reader.submit(plan)
      pendingReadBytes += plan.reservedBytes
    }
    if readAheadEnabled {
      for slice in orderedSlices.prefix(readAheadDepth) {
        try enqueueRead(slice, currentInputBytes: pendingReadBytes)
      }
      sliceOrdinal = min(readAheadDepth, orderedSlices.count)
    }
    var shape = Shape(
      scans: UInt32(frames), pixels: UInt32(pixels), columns: UInt32(dataset.detectorCols),
      sourceBytes: UInt32(source.sourceBytesPerValue))
    let isolateKernels = OriginalPackingDiagnostics.enabled("PROFILE_KERNELS", byDefault: false)
    // Cached-DPC direct loading has no other product work to submit here.
    // Isolated profiling preserves separate decode/header command timings.
    let fuseDecodeHeaders =
      destination == nil && cachedDPC != nil && planReader == nil && !isolateKernels
      && OriginalPackingDiagnostics.enabled("FUSE_DECODE_HEADERS", byDefault: true)
    let batchDirectDecode =
      directScratch
      && (pipelineWindows
        || OriginalPackingDiagnostics.enabled("BATCH_DECODE", byDefault: false))
    var pendingPacking:
      (
        command: MTLCommandBuffer, started: CFAbsoluteTime, payload: MTLBuffer,
        headers: MTLBuffer, moments: MTLBuffer, errors: MTLBuffer, upperBound: Int
      )?
    func finalizePendingPacking() throws {
      guard let pending = pendingPacking else { return }
      pendingPacking = nil
      let elapsed = try wait(pending.command)
      profile.packingGPU += elapsed
      profile.packingWall += CFAbsoluteTimeGetCurrent() - pending.started
      guard pending.errors.contents().load(as: UInt32.self) == 0 else {
        throw Self.invalid("Packed counts differ from decoded source; no cache was published")
      }
      residentShards.append((pending.payload, pending.headers))
      residentBytes += UInt64(pending.payload.length + pending.headers.length)
      if cachedDPC == nil {
        residentMomentData.append(
          Data(bytes: pending.moments.contents(), count: pending.moments.length))
      }
      let allocatedNow = UInt64(device.currentAllocatedSize)
      if let maximumAdditionalBytes, allocatedNow > allocatedBefore,
        allocatedNow - allocatedBefore > maximumAdditionalBytes
      {
        throw Self.invalid(
          "Metal allocations exceeded the available load budget; release another resident")
      }
      progress(pending.upperBound, source.logicalFrameCount)
    }
    for (ordinal, window) in windows.enumerated() {
      let slot = ordinal % slotCount
      availableSlots[slot].wait()
      var submitted = false
      defer { if !submitted { availableSlots[slot].signal() } }
      let dense = denseSlots[slot]
      let narrow = narrowSlots.isEmpty ? nil : narrowSlots[slot]
      try autoreleasepool {
        let windowScratch: MTLBuffer? =
          pipelineWindows
          ? (ordinal.isMultiple(of: 2) ? scalarScratch : pipelineScratch)
          : scalarScratch
        let windowDecodeErrors: MTLBuffer =
          pipelineWindows ? pipelineDecodeErrors! : errors
        try writer?.check()
        if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
        let cachedPlan: OriginalPackingLayoutCache.Window?
        if let planReader {
          let readStarted = CFAbsoluteTimeGetCurrent()
          cachedPlan = planReader.read(window: ordinal)
          profile.planRead += CFAbsoluteTimeGetCurrent() - readStarted
          guard let cachedPlan, cachedPlan.headerBytes == headerBytes,
            cachedPlan.paddedHeaderBytes == headerCapacity
          else { throw CacheMismatch(profile: profile) }
          profile.planReadBytes += UInt64(
            cachedPlan.compressed.count + cachedPlan.metadata.count * 4)
          try decodePlan(cachedPlan, into: headers, errors: errors, profile: &profile)
          profile.planWindows += 1
        } else {
          cachedPlan = nil
        }
        memset(audit.contents(), 0, audit.length)
        let batchedDecodeCommand =
          batchDirectDecode
          ? try (pipelineWindows ? decodeCommandBuffer() : commandBuffer()) : nil
        let batchedDecodeStarted = batchedDecodeCommand.map { _ in CFAbsoluteTimeGetCurrent() }
        let fuseDPC =
          !fusedDirect && partialDPC != nil
          && (source.sourceBytesPerValue == 4
            || window.slices.allSatisfy { $0.globalFrameRange.count >= 2048 })
        for (sliceIndex, slice) in window.slices.enumerated() {
          if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
          let preparedInput: CompressedReadInput?
          if let reader {
            let waitStarted = CFAbsoluteTimeGetCurrent()
            preparedInput = try reader.take(
              expectedFrameStart: slice.globalFrameRange.lowerBound, shouldCancel: shouldCancel)
            profile.readWait += CFAbsoluteTimeGetCurrent() - waitStarted
            if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
            pendingReadBytes =
              pendingReadBytes >= preparedInput!.reservedBytes
              ? pendingReadBytes - preparedInput!.reservedBytes : 0
            if sliceOrdinal < orderedSlices.count {
              // The input just taken remains retained by this decode until the
              // slice command completes. Admission must count it together with
              // any pending read-ahead inputs; otherwise unified-memory
              // pressure is under-reported precisely during the overlap.
              try enqueueRead(
                orderedSlices[sliceOrdinal],
                currentInputBytes: pendingReadBytes + preparedInput!.reservedBytes)
              sliceOrdinal += 1
            }
          } else {
            preparedInput = nil
          }
          let staging = try decodeSlice(
            slice, source: source,
            firstFrame: window.globalFrameRange.lowerBound, dense: dense, mask: mask, audit: audit,
            scratch: windowScratch, errors: windowDecodeErrors,
            partialDPC: fuseDPC ? partialDPC : nil, moments: moments,
            preparedInput: preparedInput,
            zeroTails: directZeroTails,
            commandBufferOverride: batchedDecodeCommand,
            headersAfterDecode: fuseDecodeHeaders && sliceIndex == window.slices.count - 1
              ? (buffers: [dense, headers, sizes, sums, widths], shape: shape) : nil,
            forceScalar: directScratch, skipUnshuffle: directScratch,
            fusedDirect: fusedDirect,
            scratchOffset: directScratch
              ? (slice.globalFrameRange.lowerBound - window.globalFrameRange.lowerBound)
                * Int(source.decodedBytesPerFrame) : 0,
            tokenPlanBuild: gpuTokenPlanRequested ? tokenPlanBuild : nil,
            tokenPlanExpand: gpuTokenPlanRequested ? tokenPlanExpand : nil,
            tokenPlanHeaders: gpuTokenPlanRequested ? tokenPlanHeaders : nil,
            tokenPlanOps: gpuTokenPlanRequested ? tokenPlanOps : nil,
            tokenPlanBaseBlock: gpuTokenPlanRequested
              ? (slice.globalFrameRange.lowerBound - window.globalFrameRange.lowerBound)
                * Int(source.shards[slice.shardIndex].index.metadata.nBlocksPerFrame) : 0,
            shouldCancel: shouldCancel, profile: &profile)
          largestInput = max(largestInput, staging)
          peakStaging = fixedStaging + largestInput + largestPayload * 2
        }
        if let batchedDecodeCommand {
          let elapsed = try finish(batchedDecodeCommand)
          profile.decodeGPU += elapsed
          profile.decodeWall +=
            CFAbsoluteTimeGetCurrent() - (batchedDecodeStarted ?? CFAbsoluteTimeGetCurrent())
          if windowDecodeErrors.contents().load(as: UInt32.self) != 0 {
            throw Self.invalid(
              "Invalid compressed original counts; reopen an intact acquisition. No resident was published"
            )
          }
        }
        // In the two-queue diagnostic pipeline, decode for this window was
        // allowed to overlap packing of the previous one. Reclaim the shared
        // header/count buffers only after that previous pack has completed.
        if pipelineWindows { try finalizePendingPacking() }
        if directScratch {
          maximum = 65535
          guard let windowScratch else {
            throw Self.invalid("Missing direct bitshuffle scratch pipelines")
          }
          memset(errors.contents(), 0, 4)
          let directHeaderCommand = try commandBuffer()
          if directCombined {
            guard let bitshuffleDirectCombined, let directCombinedDPCPartials,
              let bitshuffleDPCFusedReduce,
              let combined = directHeaderCommand.makeComputeCommandEncoder()
            else { throw Self.invalid("Missing combined direct bitshuffle pipelines") }
            combined.setComputePipelineState(bitshuffleDirectCombined)
            combined.setBuffer(windowScratch, offset: 0, index: 0)
            combined.setBuffer(headers, offset: 0, index: 1)
            combined.setBuffer(sizes, offset: 0, index: 2)
            combined.setBuffer(sums, offset: 0, index: 3)
            combined.setBuffer(widths, offset: 0, index: 4)
            combined.setBuffer(errors, offset: 0, index: 5)
            combined.setBuffer(directCombinedFixedPayload!, offset: 0, index: 6)
            combined.setBuffer(directCombinedDPCPartials, offset: 0, index: 7)
            combined.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 8)
            combined.dispatchThreads(
              MTLSize(width: pixels / 4, height: 1, depth: 1),
              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            combined.endEncoding()
            guard let reduction = directHeaderCommand.makeComputeCommandEncoder() else {
              throw Self.invalid("Cannot encode combined direct DPC reduction")
            }
            reduction.setComputePipelineState(bitshuffleDPCFusedReduce)
            reduction.setBuffer(directCombinedDPCPartials, offset: 0, index: 0)
            reduction.setBuffer(moments, offset: 0, index: 1)
            reduction.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 2)
            reduction.setBuffer(errors, offset: 0, index: 3)
            reduction.dispatchThreads(
              MTLSize(width: frames, height: 1, depth: 1),
              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            reduction.endEncoding()
          } else {
            guard let bitshuffleHeaders, let directHighPlaneWords else {
              throw Self.invalid("Missing direct bitshuffle header pipelines")
            }
            memset(directHighPlaneWords.contents(), 0, directHighPlaneWords.length)
            if OriginalPackingDiagnostics.enabled("HEADER_VECTOR4", byDefault: false),
              let bitshuffleHeadersVector4,
              let headerVector = directHeaderCommand.makeComputeCommandEncoder()
            {
              headerVector.setComputePipelineState(bitshuffleHeadersVector4)
              headerVector.setBuffer(windowScratch, offset: 0, index: 0)
              headerVector.setBuffer(headers, offset: 0, index: 1)
              headerVector.setBuffer(sizes, offset: 0, index: 2)
              headerVector.setBuffer(sums, offset: 0, index: 3)
              headerVector.setBuffer(widths, offset: 0, index: 4)
              headerVector.setBuffer(errors, offset: 0, index: 5)
              headerVector.setBuffer(directHighPlaneWords, offset: 0, index: 6)
              headerVector.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 7)
              headerVector.setBuffer(directZeroTails, offset: 0, index: 8)
              var zeroTailEnabled = UInt32(directZeroTailEnabled ? 1 : 0)
              headerVector.setBytes(&zeroTailEnabled, length: 4, index: 9)
              headerVector.dispatchThreads(
                MTLSize(width: pixels / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
              headerVector.endEncoding()
            } else {
              try encode(
                directHeaderCommand, pipeline: bitshuffleHeaders,
                buffers: [
                  windowScratch, headers, sizes, sums, widths, errors, directHighPlaneWords,
                ],
                shape: &shape, count: pixels)
            }
            if cachedDPC == nil && directPackingDPCPartials == nil {
              guard let bitshuffleDPC,
                let dpc = directHeaderCommand.makeComputeCommandEncoder()
              else { throw Self.invalid("Cannot encode direct bitshuffle DPC") }
              let useWideDPC = bitshuffleDPCWide != nil
              if let bitshuffleDPCWide {
                dpc.setComputePipelineState(bitshuffleDPCWide)
              } else if let bitshuffleDPCPruned {
                dpc.setComputePipelineState(bitshuffleDPCPruned)
                dpc.setBuffer(directHighPlaneWords, offset: 0, index: 4)
              } else {
                dpc.setComputePipelineState(bitshuffleDPC)
              }
              dpc.setBuffer(windowScratch, offset: 0, index: 0)
              dpc.setBuffer(moments, offset: 0, index: 1)
              dpc.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 2)
              dpc.setBuffer(errors, offset: 0, index: 3)
              dpc.dispatchThreads(
                MTLSize(width: frames * (useWideDPC ? 128 : 32), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
              dpc.endEncoding()
            }
          }
          let directElapsed = try finish(directHeaderCommand)
          profile.headersGPU += directElapsed
          profile.productsGPU += directElapsed
          if cachedDPC == nil {
            profile.dpcGPU += directElapsed
          }
          if directPackingDPCPartials != nil || directCombined { profile.fusedDPCWindows += 1 }
        } else {
          let auditWords = audit.contents().assumingMemoryBound(to: UInt32.self)
          for index in stride(from: 0, to: frames * 2, by: 2) {
            maximum = max(maximum, auditWords[index])
          }
        }
        if !directScratch && !fuseDecodeHeaders && cachedPlan == nil {
          let productsStarted = CFAbsoluteTimeGetCurrent()
          var headerCommand = try commandBuffer()
          try encode(
            headerCommand, pipeline: headersPipeline,
            buffers: [dense, headers, sizes, sums, widths], shape: &shape, count: pixels)
          if isolateKernels {
            let elapsed = try finish(headerCommand)
            profile.headersGPU += elapsed
            profile.productsGPU += elapsed
            headerCommand = try commandBuffer()
          }
          if cachedDPC == nil && !fuseDPC {
            try encode(
              headerCommand, pipeline: momentsPipeline, buffers: [dense, moments], shape: &shape,
              count: frames * momentsPipeline.threadExecutionWidth)
          }
          if fuseDPC { profile.fusedDPCWindows += 1 }
          if isolateKernels {
            let elapsed = try finish(headerCommand)
            profile.dpcGPU += elapsed
            profile.productsGPU += elapsed
            headerCommand = try commandBuffer()
          }
          if maximum <= 255, let narrow {
            try encode(
              headerCommand, pipeline: narrowPipeline, buffers: [dense, narrow], shape: &shape,
              count: frames * pixels)
          }
          profile.productsGPU += try finish(headerCommand)
          profile.productsWall += CFAbsoluteTimeGetCurrent() - productsStarted
        }
        if cachedPlan != nil {
          if fuseDPC { profile.fusedDPCWindows += 1 }
          if cachedDPC == nil && !fuseDPC {
            let productsStarted = CFAbsoluteTimeGetCurrent()
            let dpcCommand = try commandBuffer()
            try encode(
              dpcCommand, pipeline: momentsPipeline, buffers: [dense, moments],
              shape: &shape, count: frames * momentsPipeline.threadExecutionWidth)
            profile.productsGPU += try finish(dpcCommand)
            profile.productsWall += CFAbsoluteTimeGetCurrent() - productsStarted
          }
        }
        let prefixStarted = CFAbsoluteTimeGetCurrent()
        var wordCount: UInt32 = 0
        let sizeWords = sizes.contents().assumingMemoryBound(to: UInt32.self)
        let headerWords = headers.contents().assumingMemoryBound(to: UInt32.self)
        let sumWords = sums.contents().assumingMemoryBound(to: UInt64.self)
        if let cachedPlan {
          wordCount = cachedPlan.payloadWordCount
        } else {
          for pixel in 0..<pixels {
            headerWords[pixel * headerStride] = wordCount
            let added = wordCount.addingReportingOverflow(sizeWords[pixel])
            guard !added.overflow else { throw Self.invalid("Packed shard offsets exceed uint32") }
            wordCount = added.partialValue
            detectorSum[pixel] += sumWords[pixel]
          }
          if wordCount == 0 {
            // Metal buffers cannot be empty. Give the final zero tile one
            // valid bit/count rather than retaining unreferenced padding.
            let lastWidth = (pixels - 1) * headerStride + checkpoints + (tiles - 1) / 8
            headerWords[lastWidth] = 1 << UInt32(((tiles - 1) % 8) * 4)
            let maximumWidths = widths.contents().assumingMemoryBound(to: UInt32.self)
            maximumWidths[pixels - 1] = max(1, maximumWidths[pixels - 1])
            wordCount = 1
          }
        }
        profile.prefixWall += CFAbsoluteTimeGetCurrent() - prefixStarted
        let payloadBytes = max(4, Int(wordCount) * 4)
        let prospectivePayload = max(largestPayload, UInt64(payloadBytes))
        let admissionStaging =
          readAheadEnabled
          ? fixedStaging + largestInput + prospectivePayload * 2
          : stagingReserve + additionalReadReserve
        #if QGPU_PACKING_DIAGNOSTICS
          if directScratch,
            ProcessInfo.processInfo.environment["QGPU_ORIGINAL_DEBUG_COMBINED"] == "1",
            ordinal == 0
          {
            let sizeWords = sizes.contents().assumingMemoryBound(to: UInt32.self)
            var maximumSize: UInt32 = 0
            for pixel in 0..<pixels { maximumSize = max(maximumSize, sizeWords[pixel]) }
            fputs(
              "COMBINED_DIRECT_DEBUG frames=\(frames) firstSize=\(sizeWords[0]) maxSize=\(maximumSize) wordCount=\(wordCount) payload=\(payloadBytes) fixed=\(directCombinedFixedPayload?.length ?? 0) staging=\(admissionStaging) resident=\(residentBytes) budget=\(maximumAdditionalBytes ?? 0)\n",
              stderr)
          }
        #endif
        if let maximumAdditionalBytes,
          residentBytes + admissionStaging + UInt64(payloadBytes + headerBytes)
            > maximumAdditionalBytes
        {
          if cachePlans { throw CacheMismatch(profile: profile, retryWithoutPlan: true) }
          throw Self.invalid(
            "Exact packed counts exceed the available memory budget; open fewer acquisitions. No binning was applied"
          )
        }
        let payload: MTLBuffer
        do {
          payload =
            destination == nil
            ? try buffer(payloadBytes, privateStorage: true)
            : try buffer(payloadBytes, privateStorage: false)
        } catch {
          if cachedPlan != nil { throw CacheMismatch(profile: profile, retryWithoutPlan: true) }
          throw error
        }
        largestPayload = max(largestPayload, UInt64(payload.length))
        peakStaging = fixedStaging + largestInput + largestPayload * 2
        if destination != nil { memset(payload.contents(), 0, payload.length) }
        memset(errors.contents(), 0, 4)
        let privateHeaders =
          destination == nil
          ? try buffer(headerBytes, privateStorage: true) : nil
        let packingStarted = CFAbsoluteTimeGetCurrent()
        var packingCommand = try commandBuffer()
        if let rangesPipeline, let verifiedValuesPipeline {
          payloadWords.contents().storeBytes(of: wordCount, as: UInt32.self)
          // Validate a disjoint, complete output partition before any writes.
          // The next encoder reads the tracked error buffer and rejects invalid
          // headers. Valid packing checks every actual volatile payload value
          // against the original count kept in registers, including zero tiles.
          try encode(
            packingCommand, pipeline: rangesPipeline, buffers: [headers, errors, payloadWords],
            shape: &shape, count: pixels)
          if cachedPlan != nil {
            guard let summaryValues, let summaryReduce, let planPartials else {
              throw CacheMismatch(profile: profile)
            }
            try encode(
              packingCommand, pipeline: standardPlanes ? standardPlaneSummary! : summaryValues,
              buffers: [dense, headers, payload, errors],
              shape: &shape, count: pixels * checkpoints, afterShape: planPartials)
            try encode(
              packingCommand, pipeline: summaryReduce,
              buffers: [planPartials, headers, sums, widths, errors], shape: &shape, count: pixels)
          } else if directScratch {
            if directCombined {
              guard let directCombinedFixedPayload, let compactFixedPlanes,
                let encoder = packingCommand.makeComputeCommandEncoder()
              else { throw Self.invalid("Missing combined direct compaction pipeline") }
              encoder.setComputePipelineState(compactFixedPlanes)
              encoder.setBuffer(directCombinedFixedPayload, offset: 0, index: 0)
              encoder.setBuffer(headers, offset: 0, index: 1)
              encoder.setBuffer(payload, offset: 0, index: 2)
              encoder.setBuffer(errors, offset: 0, index: 3)
              encoder.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 4)
              encoder.dispatchThreads(
                MTLSize(width: pixels, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
              encoder.endEncoding()
            } else {
              guard let bitshuffleValues,
                let directPartialSums, let directPartialMaximums
              else { throw Self.invalid("Missing direct bitshuffle packing buffers") }
              guard let encoder = packingCommand.makeComputeCommandEncoder() else {
                throw Self.invalid("Cannot encode direct bitshuffle packing")
              }
              let fusedDPC = directPackingDPCPartials != nil
              if fusedDPC {
                let dpcPipeline =
                  directZeroTailEnabled
                  ? bitshuffleValuesZeroTailDPC
                  : (directWidthBounded
                    ? bitshuffleValuesWidthBoundedDPC : bitshuffleValuesDPC)
                guard let dpcPipeline else {
                  throw Self.invalid("Missing fused direct bitshuffle DPC pipeline")
                }
                encoder.setComputePipelineState(dpcPipeline)
              } else if directWidthBoundedPayload {
                guard let bitshuffleValuesWidthBounded else {
                  throw Self.invalid("Missing width-bounded bitshuffle pipeline")
                }
                encoder.setComputePipelineState(bitshuffleValuesWidthBounded)
              } else {
                encoder.setComputePipelineState(bitshuffleValues)
              }
              encoder.setBuffer(windowScratch, offset: 0, index: 0)
              encoder.setBuffer(headers, offset: 0, index: 1)
              encoder.setBuffer(payload, offset: 0, index: 2)
              encoder.setBuffer(errors, offset: 0, index: 3)
              encoder.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 4)
              encoder.setBuffer(directPartialSums, offset: 0, index: 5)
              encoder.setBuffer(directPartialMaximums, offset: 0, index: 6)
              if let directPackingDPCPartials {
                if directZeroTailEnabled {
                  encoder.setBuffer(directZeroTails, offset: 0, index: 7)
                  encoder.setBuffer(directPackingDPCPartials, offset: 0, index: 8)
                } else {
                  encoder.setBuffer(directPackingDPCPartials, offset: 0, index: 7)
                }
              }
              encoder.dispatchThreads(
                MTLSize(
                  width: pixels / bitshufflePixelsPerThread * checkpoints,
                  height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                  width: min(
                    bitshufflePackingThreads, bitshuffleValues.maxTotalThreadsPerThreadgroup),
                  height: 1, depth: 1))
              encoder.endEncoding()
              if let directPackingDPCPartials, let bitshuffleDPCFusedReduce {
                guard let reduction = packingCommand.makeComputeCommandEncoder() else {
                  throw Self.invalid("Cannot encode fused direct DPC reduction")
                }
                reduction.setComputePipelineState(bitshuffleDPCFusedReduce)
                reduction.setBuffer(directPackingDPCPartials, offset: 0, index: 0)
                reduction.setBuffer(moments, offset: 0, index: 1)
                reduction.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: 2)
                reduction.setBuffer(errors, offset: 0, index: 3)
                reduction.dispatchThreads(
                  MTLSize(width: frames, height: 1, depth: 1),
                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                reduction.endEncoding()
              }
            }
          } else {
            try encode(
              packingCommand,
              pipeline: standardPlanes ? selectedStandardPlaneValues! : verifiedValuesPipeline,
              buffers: [dense, headers, payload, errors], shape: &shape,
              count: pixels * (checkpointPacking ? checkpoints : 1),
              threadsPerThreadgroup: standardPlanes ? standardPackingThreads : nil)
          }
          profile.fusedWindows += 1
          if checkpointPacking { profile.checkpointWindows += 1 }
        } else {
          try encode(
            packingCommand, pipeline: valuesPipeline, buffers: [dense, headers, payload],
            shape: &shape, count: pixels)
          if isolateKernels {
            let elapsed = try finish(packingCommand)
            profile.valuesGPU += elapsed
            profile.packingGPU += elapsed
            packingCommand = try commandBuffer()
          }
          try encode(
            packingCommand, pipeline: verifyPipeline, buffers: [dense, headers, payload, errors],
            shape: &shape, count: pixels)
        }
        if let privateHeaders {
          // Both compute encoders have finished before this blit. Retain the
          // same validated header bytes without another submission and wait.
          guard let blit = packingCommand.makeBlitCommandEncoder() else {
            throw Self.invalid("Cannot retain packed headers")
          }
          blit.copy(
            from: headers, sourceOffset: 0, to: privateHeaders, destinationOffset: 0,
            size: headerBytes)
          blit.endEncoding()
        }
        if pipelineWindows {
          guard let privateHeaders else {
            throw Self.invalid("Missing retained packed headers")
          }
          packingCommand.commit()
          pendingPacking = (
            packingCommand, packingStarted, payload, privateHeaders, moments, errors,
            window.globalFrameRange.upperBound
          )
        } else {
          let packingElapsed = try finish(packingCommand)
          profile.packingGPU += packingElapsed
          if isolateKernels && verifiedValuesPipeline == nil { profile.verifyGPU += packingElapsed }
          profile.packingWall += CFAbsoluteTimeGetCurrent() - packingStarted
        }
        #if QGPU_PACKING_DIAGNOSTICS
          if !pipelineWindows, directPackingDPCPartials != nil,
            ProcessInfo.processInfo.environment["QGPU_ORIGINAL_DEBUG_DPC"] == "1"
          {
            let dpc = moments.contents().assumingMemoryBound(to: UInt64.self)
            var maxTotal: UInt64 = 0
            var maxRow: UInt64 = 0
            var maxColumn: UInt64 = 0
            for scan in 0..<frames {
              maxTotal = max(maxTotal, dpc[scan * 4])
              maxRow = max(maxRow, dpc[scan * 4 + 1])
              maxColumn = max(maxColumn, dpc[scan * 4 + 2])
            }
            fputs(
              "FUSED_DPC_DEBUG first=\(dpc[0]),\(dpc[1]),\(dpc[2]) max=\(maxTotal),\(maxRow),\(maxColumn)\n",
              stderr)
          }
        #endif
        if !pipelineWindows {
          guard errors.contents().load(as: UInt32.self) == 0 else {
            if cachedPlan != nil { throw CacheMismatch(profile: profile) }
            throw Self.invalid("Packed counts differ from decoded source; no cache was published")
          }
          if cachedPlan != nil {
            // Fresh current-count sums, not cached images/calibration metadata.
            for pixel in 0..<pixels { detectorSum[pixel] += sumWords[pixel] }
          }
          if let planWriter {
            let writeStarted = CFAbsoluteTimeGetCurrent()
            if !planWriter.append(
              headerData: Data(bytes: headers.contents(), count: headerBytes),
              payloadWordCount: wordCount)
            {
              profile.planStatus = "notStored"
            }
            profile.planWrite += CFAbsoluteTimeGetCurrent() - writeStarted
          }
          // Small shared metadata is copied; the large count and payload buffers
          // stay owned until the writer signals this slot. No buffer can be reused
          // while its bytes are being authenticated or written.
          if let writer {
            writer.submit(
              rawData: data(dense), lowData: maximum <= 255 ? narrow.map(data) : nil,
              payload: data(payload),
              headers: Data(bytes: headers.contents(), count: headers.length),
              momentData: Data(bytes: moments.contents(), count: moments.length),
              release: availableSlots[slot])
            submitted = true
          } else {
            guard let privateHeaders else { throw Self.invalid("Missing retained packed headers") }
            residentShards.append((payload, privateHeaders))
            residentBytes += UInt64(payload.length + privateHeaders.length)
            if cachedDPC == nil {
              residentMomentData.append(Data(bytes: moments.contents(), count: moments.length))
            }
            let allocatedNow = UInt64(device.currentAllocatedSize)
            if let maximumAdditionalBytes, allocatedNow > allocatedBefore,
              allocatedNow - allocatedBefore > maximumAdditionalBytes
            {
              throw Self.invalid(
                "Metal allocations exceeded the available load budget; release another resident")
            }
          }
          progress(window.globalFrameRange.upperBound, source.logicalFrameCount)
        }
      }
    }
    if pipelineWindows { try finalizePendingPacking() }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    if destination == nil {
      try validateInputs()
      _ = try Native4DSTEMIndexedSource.open(dataset: dataset)
      if let binding, !binding.isCurrent(sourceFiles: planSources) {
        throw Self.invalid("Original source changed during loading; reopen its folder and retry")
      }
      let widthValues = widths.contents().assumingMemoryBound(to: UInt32.self)
      let maximumWidths = (0..<pixels).map { UInt8(widthValues[$0]) }
      if planReader != nil && maximum <= 255 && maximumWidths.contains(where: { $0 > 8 }) {
        // uint8 resident semantics cannot interpret a cached 16-bit escape as
        // 16. Never select physical encoding from a stale declared maximum.
        throw CacheMismatch(profile: profile)
      }
      if let planWriter {
        let writeStarted = CFAbsoluteTimeGetCurrent()
        profile.planStatus = planWriter.finish() ? "stored" : "notStored"
        profile.planWrite += CFAbsoluteTimeGetCurrent() - writeStarted
        if profile.planStatus == "stored", let packingPlanURL,
          let attributes = try? FileManager.default.attributesOfItem(atPath: packingPlanURL.path),
          let size = attributes[.size] as? NSNumber
        {
          profile.planOutputBytes = size.uint64Value
        }
      }
      profile.maximumWidthHistogram = maximumWidths.reduce(into: [Int](repeating: 0, count: 33)) {
        $0[Int($1)] += 1
      }
      profile.packedPayloadLayout = standardPlanes ? 1 : 0
      reportProfile(profile)
      var packed = OriginalPackedBuffers(
        payloadLayout: standardPlanes ? 1 : 0,
        dataset: dataset, frames: frames, headerStride: headerStride,
        shards: residentShards, moments: residentMomentData, detectorSum: detectorSum,
        maximum: maximum, maximumWidths: maximumWidths,
        calibration: measuredDetector(
          detectorSum, rows: dataset.detectorRows, columns: dataset.detectorCols,
          excludedFromEstimate: dataset.badPixelIndices), stagingBytes: peakStaging,
        readSeconds: profile.read, decodeSeconds: profile.decodeGPU,
        decodeAndHeaderSeconds: profile.decodeAndHeadersGPU,
        packingSeconds: profile.productsGPU + profile.packingGPU + profile.planDecodeGPU,
        reusedDPC: profile.reusedDPC,
        combinedDecodePackingSeconds: profile.directBitshuffleWindows > 0
          ? profile.directBitshuffleGPU : nil)
      return packed
    }
    guard let writer, let output, let temporary, let destination else {
      throw Self.invalid("Missing packed output destination")
    }
    let written = try writer.finish()
    try validateInputs()
    let records = written.records
    let momentData = written.moments
    profile.hashing = written.hashing
    profile.writing = written.writing
    // Revalidate source/index identity after the complete bounded read.
    _ = try Native4DSTEMIndexedSource.open(dataset: dataset)
    let is8 = maximum <= 255
    let working = is8 ? "uint8" : "uint16"
    let rawDigest = written.raw
    let workingDigest = is8 ? written.low : rawDigest
    let momentOffset = try output.offset()
    try output.write(contentsOf: momentData)
    let maximumValue: UInt64 = is8 ? 255 : 65535
    let totalBound = UInt64(pixels) * maximumValue
    let rowBound = UInt64(pixels * (dataset.detectorRows - 1) / 2) * maximumValue
    let columnBound = UInt64(pixels * (dataset.detectorCols - 1) / 2) * maximumValue
    let maskDigest = Self.digest(Data(repeating: 1, count: pixels))
    let workingField = is8 ? "working_uint8_sha256" : "working_logical_sha256"
    let prepared: [String: Any] = [
      "schema": "quantem.gpu.prepared-dpc-moments/v\(is8 ? 1 : 2)",
      "source_identity_sha256": identity,
      workingField: workingDigest, "working_dtype": working, "maximum_value": maximumValue,
      "detector_mask_sha256": maskDigest, "detector_selection": "all-nonexcluded-v1",
      "dtype": "little-endian-u32", "word_order": "little-endian-u32-pairs",
      "total_bound": String(totalBound), "row_moment_bound": String(rowBound),
      "column_moment_bound": String(columnBound),
      "scan_count": source.logicalFrameCount, "selected_detector_pixels": pixels,
      "detector_columns": dataset.detectorCols,
      "words_per_scan": 8, "narrow_integer": totalBound <= UInt32.max,
      "narrow_products": max(rowBound, columnBound) <= UInt32.max,
      "layout": [
        "total_lo", "total_hi", "row_lo", "row_hi", "column_lo", "column_hi", "padding_0",
        "padding_1",
      ],
      "file_offset": momentOffset, "file_bytes": momentData.count,
      "sha256": Self.digest(momentData),
    ]
    let calibration = measuredDetector(
      detectorSum, rows: dataset.detectorRows, columns: dataset.detectorCols,
      excludedFromEstimate: dataset.badPixelIndices)
    let manifest: [String: Any] = [
      "schema": "quantem.gpu.packed-detector-h5/v3", "payload_codec": "direct-bitpacked-u32",
      "status": "complete",
      "source_shape": [
        dataset.scanRows, dataset.scanCols, dataset.detectorRows, dataset.detectorCols,
      ],
      "source_dtype": dataset.sourceDtype, "working_dtype": working,
      "source_identity_sha256": identity,
      "source_raw_logical_sha256": rawDigest,
      is8 ? "prepared_uint8_sha256" : "working_logical_sha256": workingDigest,
      "scan_bin": 1, "detector_bin": 1, "crop": NSNull(), "scan_tile": 32,
      "scans_per_shard": frames, "shard_count": records.count,
      "working_value_definition":
        "all admitted source counts exactly; authenticated dead pixels set to zero",
      "masked_detector_pixels": [Int](), "detector_mask_sha256": maskDigest,
      "source_bad_pixel_indices": dataset.badPixelIndices,
      "bad_pixel_policy": "preserve_all_source_counts",
      "detector_calibration": [
        "schema": "quantem.gpu.detector-calibration/v1", "source_identity_sha256": identity,
        "detector_center_px": [calibration.0, calibration.1],
        "bright_field_radius_px": calibration.2,
        "method": "mean-DP half-p99 threshold; automatic initial detector, not angular calibration",
      ],
      "prepared_dpc_moments": prepared,
      "original_packing": [
        "version": 2, "every_count_roundtrip_verified": true, "maximum_count": maximum,
        "maximum_staging_bytes": peakStaging,
        "elapsed_seconds": CFAbsoluteTimeGetCurrent() - started,
        "phases": profile.json,
      ],
    ]
    let header = try JSONSerialization.data(
      withJSONObject: manifest, options: [.sortedKeys, .withoutEscapingSlashes])
    var binary = Data([81, 71, 73, 88, 0, 0, 0, 3])
    for value in [
      records.count, 0, dataset.scanRows, dataset.scanCols, dataset.detectorRows,
      dataset.detectorCols, frames, 32, is8 ? 1 : 2, 0,
    ] { binary.appendLE(UInt32(value)) }
    binary.append(Self.hexData(identity))
    for record in records {
      for value in [record.0, record.1, 0, 0, record.2, record.3, record.1] {
        binary.appendLE(value)
      }
      binary.appendLE(UInt32(headers.length / 4))
      binary.appendLE(UInt32(0))
      binary.append(record.4)
    }
    let binaryOffset = (24 + header.count + 7) & ~7
    guard binaryOffset + binary.count <= reserved else {
      throw Self.invalid("Packed manifest exceeds its reserved header")
    }
    var prelude = Data([81, 71, 80, 85, 72, 53, 0, 1])
    for value in [
      UInt32(header.count), Self.crc32(header), UInt32(binaryOffset), UInt32(binary.count),
    ] { prelude.appendLE(value) }
    try output.seek(toOffset: 0)
    try output.write(contentsOf: prelude)
    try output.write(contentsOf: header)
    try output.seek(toOffset: UInt64(binaryOffset))
    try output.write(contentsOf: binary)
    try output.synchronize()
    try output.close()
    _ = try MetalCompactH5Loader.inspect(sourceURL: temporary)
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    try FileManager.default.moveItem(at: temporary, to: destination)
    reportProfile(profile)
    return nil
  }

  static func makeDecodeOrder(
    metadata: MTLBuffer, frameCount: Int, blocks: Int, device: MTLDevice
  ) throws -> MTLBuffer {
    let count = frameCount * blocks
    guard count > 0, metadata.length >= count * MemoryLayout<UInt32>.stride * 2 else {
      throw invalid("Cannot construct the indexed decoder order")
    }
    let words = metadata.contents().assumingMemoryBound(to: UInt32.self)
    var order = (0..<count).map(UInt32.init)
    // The scalar decoder is launched in SIMD groups of 32, even when the
    // threadgroup contains 64/128/256 threads. Sort only within each SIMD
    // group so output ownership stays unchanged and no global permutation
    // buffer is needed.
    for start in stride(from: 0, to: count, by: 32) {
      let end = min(start + 32, count)
      order[start..<end].sort {
        let left = words[Int($0) * 2 + 1]
        let right = words[Int($1) * 2 + 1]
        return left == right ? $0 < $1 : left < right
      }
    }
    return try order.withUnsafeBytes { bytes in
      guard let base = bytes.baseAddress,
        let buffer = device.makeBuffer(
          bytes: base, length: bytes.count, options: .storageModeShared)
      else { throw invalid("Cannot allocate the indexed decoder order") }
      buffer.label = "QH5 decoder SIMD order"
      return buffer
    }
  }

  func decodeSlice(
    _ slice: Native4DSTEMIndexedSlice, source: Native4DSTEMIndexedSource,
    firstFrame: Int, dense: MTLBuffer, mask: MTLBuffer, audit: MTLBuffer,
    scratch: MTLBuffer?, errors: MTLBuffer,
    partialDPC: MTLBuffer?, moments: MTLBuffer,
    preparedInput: CompressedReadInput? = nil,
    zeroTails: MTLBuffer? = nil,
    commandBufferOverride: MTLCommandBuffer? = nil,
    headersAfterDecode: (buffers: [MTLBuffer], shape: Shape)? = nil,
    forceScalar: Bool = false, skipUnshuffle: Bool = false, fusedDirect: Bool = false,
    scratchOffset: Int = 0,
    tokenPlanBuild: MTLComputePipelineState? = nil,
    tokenPlanExpand: MTLComputePipelineState? = nil,
    tokenPlanHeaders: MTLBuffer? = nil,
    tokenPlanOps: MTLBuffer? = nil,
    tokenPlanBaseBlock: Int = 0,
    shouldCancel: () -> Bool, profile: inout Profile
  ) throws -> UInt64 {
    let pixels = source.dataset.detectorRows * source.dataset.detectorCols
    let shard = source.shards[slice.shardIndex]
    let input =
      try preparedInput
      ?? Self.readCompressed(
        compressedReadPlan(slice, source: source), device: device, isCancelled: shouldCancel)
    guard input.shardIndex == slice.shardIndex,
      input.frameRange == slice.globalFrameRange
    else {
      throw Self.invalid("Compressed input does not match the requested scan slice")
    }
    let compressed = input.compressed
    let metadata = input.metadata
    profile.read += input.readSeconds
    profile.copy += input.copySeconds
    profile.readBytes += input.readBytes
    if input.coalescedBatchSlices > 0 {
      profile.coalescedReadBatches += 1
      profile.coalescedReadSlices += input.coalescedBatchSlices
      profile.coalescedReadGapBytes += input.coalescedGapBytes
    }
    let decodeStarted = CFAbsoluteTimeGetCurrent()
    var zero: UInt32 = 0
    var blocks = UInt32(shard.index.metadata.nBlocksPerFrame)
    var pixelCount = UInt32(pixels)
    var frameCount = UInt32(slice.globalFrameRange.count)
    var auditOffset = UInt32(slice.globalFrameRange.lowerBound - firstFrame)
    let is32 = source.sourceBytesPerValue == 4
    let scalar =
      !fusedDirect
      && (forceScalar || is32 || (scratch != nil && scalarDecode != nil && frameCount >= 2048))
    let tokenPlanDecodeRequested =
      scalar && tokenPlanBuild != nil && tokenPlanExpand != nil
      && tokenPlanHeaders != nil && tokenPlanOps != nil
    let orderedDecodeRequested = scalar && orderedScalarDecode != nil
    let distance3DecodeRequested = scalar && !orderedDecodeRequested && distance3ScalarDecode != nil
    let distance8DecodeRequested =
      scalar && !orderedDecodeRequested
      && !distance3DecodeRequested && distance8ScalarDecode != nil
    let useZeroTailDecode =
      !orderedDecodeRequested && !distance3DecodeRequested && !distance8DecodeRequested
      && !is32 && scalar
      && zeroTails != nil && zeroTailDecode != nil
    let decodeOrder =
      orderedDecodeRequested
      ? try Self.makeDecodeOrder(
        metadata: metadata, frameCount: Int(frameCount), blocks: Int(blocks), device: device)
      : nil
    if scalar, alignedRepeatFill || alignedHistoryCopy, let scratch, scratch.gpuAddress % 16 != 0 {
      throw Self.invalid("Aligned decompression requires a 16-byte-aligned scratch buffer")
    }
    let denseOffset =
      (slice.globalFrameRange.lowerBound - firstFrame) * Int(source.decodedBytesPerFrame)
    let command: MTLCommandBuffer
    if let commandBufferOverride {
      command = commandBufferOverride
    } else {
      command = try commandBuffer()
    }
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Self.invalid("Cannot encode source decode")
    }
    var rangeStart = input.compressedRangeStart
    if tokenPlanDecodeRequested {
      guard let tokenPlanBuild, let tokenPlanExpand, let tokenPlanHeaders, let tokenPlanOps,
        tokenPlanBaseBlock >= 0,
        tokenPlanBaseBlock <= Int(UInt32.max)
      else { throw Self.invalid("GPU token-plan buffers are incomplete") }
      var planBlockOffset = UInt32(tokenPlanBaseBlock)
      encoder.setComputePipelineState(tokenPlanBuild)
      encoder.setBuffer(compressed, offset: 0, index: 0)
      encoder.setBuffer(metadata, offset: 0, index: 1)
      encoder.setBytes(&rangeStart, length: 8, index: 2)
      encoder.setBytes(&blocks, length: 4, index: 3)
      encoder.setBytes(&pixelCount, length: 4, index: 4)
      encoder.setBuffer(tokenPlanHeaders, offset: 0, index: 5)
      encoder.setBuffer(tokenPlanOps, offset: 0, index: 6)
      encoder.setBytes(&zero, length: 4, index: 7)
      memset(errors.contents(), 0, 4)
      encoder.setBuffer(errors, offset: 0, index: 10)
      encoder.setBytes(&frameCount, length: 4, index: 11)
      encoder.setBytes(&planBlockOffset, length: 4, index: 12)
      encoder.dispatchThreads(
        MTLSize(width: Int(frameCount) * Int(blocks), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: scalarDecodeThreads, height: 1, depth: 1))
      encoder.endEncoding()

      guard let expandEncoder = command.makeComputeCommandEncoder() else {
        throw Self.invalid("Cannot encode GPU token-plan expansion")
      }
      expandEncoder.setComputePipelineState(tokenPlanExpand)
      expandEncoder.setBuffer(compressed, offset: 0, index: 0)
      expandEncoder.setBytes(&blocks, length: 4, index: 1)
      expandEncoder.setBytes(&pixelCount, length: 4, index: 2)
      expandEncoder.setBuffer(scratch, offset: scratchOffset, index: 3)
      expandEncoder.setBuffer(tokenPlanHeaders, offset: 0, index: 4)
      expandEncoder.setBuffer(tokenPlanOps, offset: 0, index: 5)
      expandEncoder.setBytes(&planBlockOffset, length: 4, index: 6)
      expandEncoder.setBytes(&rangeStart, length: 8, index: 7)
      expandEncoder.setBuffer(errors, offset: 0, index: 10)
      expandEncoder.setBytes(&frameCount, length: 4, index: 11)
      let totalBlocks = Int(frameCount) * Int(blocks)
      expandEncoder.dispatchThreadgroups(
        MTLSize(width: (totalBlocks + 3) / 4, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      expandEncoder.endEncoding()
    } else {
      encoder.setComputePipelineState(
        fusedDirect
          ? (fusedDecodeUnshuffleFrameCoop
            ?? fusedDecodeUnshuffleVector
            ?? fusedDecodeUnshuffle)!
          : (is32
            ? decode32
            : (useZeroTailDecode
              ? zeroTailDecode!
              : (scalar
                ? (orderedDecodeRequested
                  ? orderedScalarDecode!
                  : (distance3DecodeRequested
                    ? distance3ScalarDecode!
                    : (distance8DecodeRequested ? distance8ScalarDecode! : scalarDecode!)))
                : (source.sourceBytesPerValue == 1 ? decode8 : decode16)))))
      encoder.setBuffer(compressed, offset: 0, index: 0)
      encoder.setBuffer(metadata, offset: 0, index: 1)
      encoder.setBytes(&rangeStart, length: 8, index: 2)
      encoder.setBytes(&blocks, length: 4, index: 3)
      encoder.setBytes(&pixelCount, length: 4, index: 4)
      encoder.setBuffer(
        scalar ? scratch : dense, offset: scalar ? scratchOffset : denseOffset, index: 5)
      encoder.setBytes(&zero, length: 4, index: 6)
      encoder.setBuffer(mask, offset: 0, index: 7)
      encoder.setBuffer(audit, offset: 0, index: 8)
      if source.sourceBytesPerValue > 1 { encoder.setBytes(&auditOffset, length: 4, index: 9) }
      if scalar {
        memset(errors.contents(), 0, 4)
        encoder.setBuffer(errors, offset: 0, index: 10)
        encoder.setBytes(&frameCount, length: 4, index: 11)
        if let decodeOrder { encoder.setBuffer(decodeOrder, offset: 0, index: 12) }
        if useZeroTailDecode {
          var tailOffset = UInt32(
            (slice.globalFrameRange.lowerBound - firstFrame) * Int(blocks))
          encoder.setBuffer(zeroTails, offset: 0, index: 12)
          encoder.setBytes(&tailOffset, length: 4, index: 13)
        }
        encoder.dispatchThreads(
          MTLSize(width: Int(frameCount) * Int(blocks), height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: scalarDecodeThreads, height: 1, depth: 1))
      } else if fusedDirect && fusedDecodeUnshuffleFrameCoop != nil {
        encoder.dispatchThreadgroups(
          MTLSize(width: Int(frameCount), height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      } else {
        encoder.dispatchThreadgroups(
          MTLSize(width: Int(frameCount), height: 1, depth: Int(blocks)),
          threadsPerThreadgroup: MTLSize(
            width: 32, height: source.sourceBytesPerValue == 1 ? 8 : 4, depth: 1))
      }
      encoder.endEncoding()
    }
    let selectedUnshuffle: MTLComputePipelineState?
    if is32 {
      selectedUnshuffle = partialDPC == nil ? unshuffle32 : dpcUnshuffle32
    } else {
      selectedUnshuffle = scalarUnshuffle
    }
    if !skipUnshuffle, scalar, let selectedUnshuffle {
      guard let unshuffle = command.makeComputeCommandEncoder() else {
        throw Self.invalid("Cannot encode exact unshuffle")
      }
      if partialDPC != nil {
        guard let dpcUnshuffle = is32 ? dpcUnshuffle32 : dpcUnshuffle else {
          throw Self.invalid("Missing exact uint32 DPC unshuffle kernel")
        }
        unshuffle.setComputePipelineState(dpcUnshuffle)
      } else {
        unshuffle.setComputePipelineState(selectedUnshuffle)
      }
      unshuffle.setBuffer(scratch, offset: 0, index: 0)
      unshuffle.setBytes(&blocks, length: 4, index: 3)
      unshuffle.setBytes(&pixelCount, length: 4, index: 4)
      unshuffle.setBuffer(dense, offset: denseOffset, index: 5)
      unshuffle.setBuffer(mask, offset: 0, index: 7)
      unshuffle.setBuffer(audit, offset: 0, index: 8)
      unshuffle.setBytes(&auditOffset, length: 4, index: 9)
      unshuffle.setBuffer(errors, offset: 0, index: 10)
      unshuffle.setBytes(&frameCount, length: 4, index: 11)
      if let partialDPC {
        var columns = UInt32(source.dataset.detectorCols)
        unshuffle.setBuffer(partialDPC, offset: 0, index: 12)
        unshuffle.setBytes(&columns, length: 4, index: 13)
      }
      unshuffle.dispatchThreadgroups(
        MTLSize(width: Int(frameCount), height: 1, depth: Int(blocks)),
        threadsPerThreadgroup: MTLSize(
          width: is32 || (transposeUnshuffle && partialDPC == nil) ? 128 : 64, height: 1, depth: 1))
      unshuffle.endEncoding()
      if let partialDPC, let dpcReduce {
        guard let reduction = command.makeComputeCommandEncoder() else {
          throw Self.invalid("Cannot encode exact DPC reduction")
        }
        reduction.setComputePipelineState(dpcReduce)
        reduction.setBuffer(partialDPC, offset: 0, index: 0)
        reduction.setBuffer(moments, offset: Int(auditOffset) * 32, index: 1)
        reduction.setBytes(&blocks, length: 4, index: 2)
        reduction.setBytes(&frameCount, length: 4, index: 3)
        reduction.setBuffer(errors, offset: 0, index: 4)
        reduction.dispatchThreads(
          MTLSize(width: Int(frameCount), height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        reduction.endEncoding()
      }
      profile.scalarSlices += 1
      if alignedRepeatFill { profile.alignedFillSlices += 1 }
      if alignedHistoryCopy { profile.alignedCopySlices += 1 }
      if transposeUnshuffle && partialDPC == nil { profile.transposeSlices += 1 }
      if useZeroTailDecode { profile.zeroTailSlices += 1 }
      if distance3DecodeRequested { profile.distance3DecodeSlices += 1 }
    } else if scalar {
      profile.scalarSlices += 1
      if alignedRepeatFill { profile.alignedFillSlices += 1 }
      if alignedHistoryCopy { profile.alignedCopySlices += 1 }
      if useZeroTailDecode { profile.zeroTailSlices += 1 }
      if distance3DecodeRequested { profile.distance3DecodeSlices += 1 }
    }
    if var headersAfterDecode {
      // The final slice and prior slices must complete before header reads.
      // Host error checking still precedes every prefix and payload allocation.
      let pixels = Int(headersAfterDecode.shape.pixels)
      try encode(
        command, pipeline: headersPipeline, buffers: headersAfterDecode.buffers,
        shape: &headersAfterDecode.shape, count: pixels)
    }
    if commandBufferOverride == nil {
      let elapsed = try finish(command)
      if headersAfterDecode != nil {
        profile.decodeAndHeadersGPU += elapsed
        profile.fusedDecodeHeaderWindows += 1
      } else {
        profile.decodeGPU += elapsed
      }
      if orderedDecodeRequested { profile.orderedDecodeSlices += 1 }
      if scalar, errors.contents().load(as: UInt32.self) != 0 {
        throw Self.invalid(
          "Invalid compressed original counts; reopen an intact acquisition. No resident was published"
        )
      }
      let wall = CFAbsoluteTimeGetCurrent() - decodeStarted
      if headersAfterDecode != nil {
        profile.decodeAndHeadersWall += wall
      } else {
        profile.decodeWall += wall
      }
    }
    return input.reservedBytes
  }

  func validatedDPC(
    _ value: MetalCompactH5ExactDPCMoments?, source: Native4DSTEMIndexedSource
  ) -> Data? {
    let dataset = source.dataset
    let scans = source.logicalFrameCount
    let pixels = dataset.detectorRows * dataset.detectorCols
    guard let value, let identity = value.sourceIdentitySHA256,
      identity == dataset.sourceIdentitySHA256,
      value.detectorMaskSHA256 == Self.digest(Data(repeating: 1, count: pixels)),
      value.total.count == scans, value.detectorRowMoment.count == scans,
      value.detectorColumnMoment.count == scans
    else { return nil }
    let maximum = (UInt64(1) << (source.sourceBytesPerValue * 8)) - 1
    let totalBound = UInt64(pixels) * maximum
    var words = [UInt64](repeating: 0, count: scans * 4)
    for scan in 0..<scans {
      let total = value.total[scan]
      let row = value.detectorRowMoment[scan]
      let column = value.detectorColumnMoment[scan]
      guard total <= totalBound,
        row <= total * UInt64(dataset.detectorRows - 1),
        column <= total * UInt64(dataset.detectorCols - 1)
      else { return nil }
      words[scan * 4] = total.littleEndian
      words[scan * 4 + 1] = row.littleEndian
      words[scan * 4 + 2] = column.littleEndian
    }
    return words.withUnsafeBytes { Data($0) }
  }

  func measuredDetector(_ sums: [UInt64], rows: Int, columns: Int, excludedFromEstimate: [Int]) -> (
    Double, Double, Double
  ) {
    let excluded = Set(excludedFromEstimate)
    let ordered = sums.enumerated().filter { !excluded.contains($0.offset) }.map(\.element).sorted()
    guard !ordered.isEmpty else { return (Double(rows - 1) / 2, Double(columns - 1) / 2, 1) }
    let threshold = Double(ordered[min(ordered.count - 1, ordered.count * 99 / 100)]) * 0.5
    let selected = sums.indices.filter { !excluded.contains($0) && Double(sums[$0]) > threshold }
    guard !selected.isEmpty else { return (Double(rows - 1) / 2, Double(columns - 1) / 2, 1) }
    let count = Double(selected.count)
    return (
      selected.reduce(0) { $0 + Double($1 / columns) } / count,
      selected.reduce(0) { $0 + Double($1 % columns) } / count, max(1, sqrt(count / .pi))
    )
  }
  func buffer(_ bytes: Int, privateStorage: Bool = false) throws -> MTLBuffer {
    let limit = device.maxBufferLength
    guard bytes > 0, bytes <= limit else { throw Self.invalid("Invalid packing buffer size") }
    let allocated = device.makeBuffer(
      length: bytes, options: privateStorage ? .storageModePrivate : .storageModeShared)
    guard let result = allocated else {
      throw Self.invalid("Cannot allocate bounded packing buffer of \(bytes) bytes")
    }
    return result
  }
  func commandBuffer() throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer() else {
      throw Self.invalid("Cannot create packing command")
    }
    return command
  }
  func decodeCommandBuffer() throws -> MTLCommandBuffer {
    guard let command = decodeQueue.makeCommandBuffer() else {
      throw Self.invalid("Cannot create decode command")
    }
    return command
  }
  func copiedBuffer(_ bytes: UnsafeRawBufferPointer) throws -> MTLBuffer {
    guard !bytes.isEmpty, bytes.count <= device.maxBufferLength,
      let result = device.makeBuffer(
        bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
    else { throw Self.invalid("Cannot copy bounded source input to Metal") }
    return result
  }
  @discardableResult func finish(_ command: MTLCommandBuffer) throws -> Double {
    command.commit()
    return try wait(command)
  }
  @discardableResult func wait(_ command: MTLCommandBuffer) throws -> Double {
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw Self.invalid(command.error?.localizedDescription ?? "Metal packing command failed")
    }
    return max(0, command.gpuEndTime - command.gpuStartTime)
  }
  func encode(
    _ command: MTLCommandBuffer, pipeline: MTLComputePipelineState, buffers: [MTLBuffer],
    shape: inout Shape, count: Int, threadsPerThreadgroup: Int? = nil,
    afterShape: MTLBuffer? = nil, sampledEncoder: MTLComputeCommandEncoder? = nil
  ) throws {
    guard let encoder = sampledEncoder ?? command.makeComputeCommandEncoder() else {
      throw Self.invalid("Cannot encode packing")
    }
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in buffers.enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&shape, length: MemoryLayout<Shape>.stride, index: buffers.count)
    if let afterShape { encoder.setBuffer(afterShape, offset: 0, index: buffers.count + 1) }
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(
          threadsPerThreadgroup ?? 128, pipeline.maxTotalThreadsPerThreadgroup),
        height: 1, depth: 1))
    encoder.endEncoding()
  }
  func data(_ buffer: MTLBuffer) -> Data {
    Data(
      bytesNoCopy: buffer.contents(), count: buffer.length,
      deallocator: .custom { [buffer] _, _ in withExtendedLifetime(buffer) {} })
  }
  static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError { .invalidRequest(message) }
  static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  static func hexData(_ text: String) -> Data {
    Data(
      stride(from: 0, to: text.count, by: 2).map { offset in
        let start = text.index(text.startIndex, offsetBy: offset)
        return UInt8(text[start..<text.index(start, offsetBy: 2)], radix: 16)!
      })
  }
  static func crc32(_ data: Data) -> UInt32 {
    var value = UInt32.max
    for byte in data {
      value ^= UInt32(byte)
      for _ in 0..<8 { value = (value >> 1) ^ ((value & 1) != 0 ? 0xedb8_8320 : 0) }
    }
    return ~value
  }
}

extension Data {
  mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
    var encoded = value.littleEndian
    Swift.withUnsafeBytes(of: &encoded) { append(contentsOf: $0) }
  }
}
