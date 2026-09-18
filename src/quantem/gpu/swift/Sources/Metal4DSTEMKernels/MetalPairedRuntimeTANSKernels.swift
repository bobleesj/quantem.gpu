import Foundation
import Metal

extension Metal4DSTEMKernels {
  public static let pairedRuntimeTANSEncodeFunction = "paired_runtime_tans_encode"
  public static let pairedRuntimeTANSCompactFunction = "paired_runtime_tans_compact"
  public static let pairedRuntimeTANSDecodeFunction = "paired_runtime_tans_decode"
  public static let pairedRuntimeTANSSelectedDPFunction = "paired_runtime_tans_selected_dp"
  public static let pairedRuntimeTANSDetectorPacketOwnerFunction =
    "paired_runtime_tans_detector_packet_owner"
  public static let pairedRuntimeTANSRebaseOffsetsFunction =
    "paired_runtime_tans_rebase_offsets"
  public static let pairedRuntimeTANSRebaseBlock32OffsetsFunction =
    "paired_runtime_tans_rebase_block32_offsets"
  public static let pairedRuntimeTANSDetectorPartialsFunction =
    "paired_runtime_tans_detector_partials"
  public static let pairedRuntimeTANSDetectorFinishFunction =
    "paired_runtime_tans_detector_finish"
  public static let pairedRuntimeTANSDetectorSparseScatterFunction =
    "paired_runtime_tans_detector_sparse_scatter"
  public static let pairedRuntimeTANSDetectorPacketOwner2Function =
    "paired_runtime_tans_detector_packet_owner2"
  public static let pairedRuntimeTANSEntropyChunkCensusFunction =
    "paired_runtime_tans_entropy_chunk_census"
  public static let pairedRuntimeTANSPolarRootsFunction = "paired_runtime_tans_polar_roots"
  public static let pairedRuntimeTANSPolarRootsInPlaceFunction =
    "paired_runtime_tans_polar_roots_in_place"
  public static let pairedRuntimeTANSPolarSizesFunction = "paired_runtime_tans_polar_sizes"
  public static let pairedRuntimeTANSPolarPackFunction = "paired_runtime_tans_polar_pack"
  public static let pairedRuntimeTANSPolarQueryFunction = "paired_runtime_tans_polar_query"
  public static let pairedRuntimeTANSPolarQueryScan512Function =
    "paired_runtime_tans_polar_query_scan512"
  public static let pairedRuntimeTANSPolarQueryField4Function =
    "paired_runtime_tans_polar_query_field4"
  public static let pairedRuntimeTANSPolarQueryPacketMajorFunction =
    "paired_runtime_tans_polar_query_packet_major"
  public static let pairedRuntimeTANSDetectorDenseCompactionFunction =
    "paired_runtime_tans_detector_dense_compaction"
  public static let pairedRuntimeTANSDetectorCooperativeFunction =
    "paired_runtime_tans_detector_cooperative"
  /// Function-constant index for the exact direct partial-store specialization.
  public static let pairedRuntimeTANSPartialStoresFunctionConstantIndex = 8
  /// Function-constant index controlling packet-owner split count.
  public static let pairedRuntimeTANSDetectorPacketSplitCountFunctionConstantIndex = 9
  /// Function-constant index for the overlapping-word refill specialization.
  public static let pairedRuntimeTANSSingleWordRefillFunctionConstantIndex = 10
  /// Function-constant index for lane-owned dense register reduction.
  public static let pairedRuntimeTANSRegisterReductionFunctionConstantIndex = 11
  /// Function-constant index for barrier-separated plain packet partials.
  public static let pairedRuntimeTANSPlainPacketOwner2FunctionConstantIndex = 12
  /// Function-constant index for the internally validated decode table.
  public static let pairedRuntimeTANSTrustedDecodeTableFunctionConstantIndex = 13
  /// Function-constant index for compact block-32 resident source offsets.
  public static let pairedRuntimeTANSCompactOffsetsFunctionConstantIndex = 20
  /// Experimental function-constant index for SIMD-uniform entropy groups.
  public static let pairedRuntimeTANSSIMDEntropyFastPathFunctionConstantIndex = 21
  /// Experimental function-constant index for one exact packed-pair SIMD reduction.
  public static let pairedRuntimeTANSVectorPairReductionFunctionConstantIndex = 25
  /// Experimental function-constant index for striped scan512 polar accumulation.
  public static let pairedRuntimeTANSPolarScan512StripesFunctionConstantIndex = 23
  /// Experimental function-constant index for contiguous scan512 polar quads.
  public static let pairedRuntimeTANSPolarScan512ContiguousQuadFunctionConstantIndex = 26
  /// Experimental function-constant index for the cadence window tANS reader.
  public static let pairedRuntimeTANSWindowReaderFunctionConstantIndex = 28
  /// Function-constant index for window-reader pairs per reload (1, 2, or 3).
  public static let pairedRuntimeTANSWindowReaderCadenceFunctionConstantIndex = 29

  /// Compile the exact paired runtime tANS prototype kernels.
  @_spi(PairedRuntimeTANSPrototype)
  public static func makePairedRuntimeTANSLibrary(device: MTLDevice) throws -> MTLLibrary {
    let resource = "paired_runtime_tans"
    let packagedURL = Bundle.main.resourceURL?
      .appendingPathComponent("MetalKernels_Metal4DSTEMKernels.bundle", isDirectory: true)
      .appendingPathComponent("Resources", isDirectory: true)
      .appendingPathComponent("\(resource).metal")
    let url =
      packagedURL.flatMap {
        FileManager.default.fileExists(atPath: $0.path) ? $0 : nil
      } ?? Bundle.module.url(
        forResource: resource, withExtension: "metal", subdirectory: "Resources")
      ?? Bundle.module.url(forResource: resource, withExtension: "metal")
    guard let url else {
      throw Metal4DSTEMKernelsError.missingResource("\(resource).metal")
    }
    do {
      let source = try String(contentsOf: url, encoding: .utf8)
      return try MetalLoadingLibraryCache.shared.library(device: device, source: source, strict: true)
    } catch {
      throw Metal4DSTEMKernelsError.libraryCompilation(
        resource: resource, message: error.localizedDescription)
    }
  }
}
