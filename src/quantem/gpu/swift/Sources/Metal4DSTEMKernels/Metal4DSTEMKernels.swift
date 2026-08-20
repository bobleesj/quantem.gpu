import Foundation
import Metal

/// Errors raised while loading shared 4D-STEM Metal resources.
public enum Metal4DSTEMKernelsError: LocalizedError {
  case missingResource(String)
  case libraryCompilation(resource: String, message: String)

  public var errorDescription: String? {
    switch self {
    case .missingResource(let name):
      "Metal4DSTEMKernels is missing \(name)."
    case .libraryCompilation(let resource, let message):
      "Metal 4D-STEM \(resource) compilation failed: \(message)"
    }
  }
}

/// Shared native 4D-STEM Metal libraries and their stable function names.
public enum Metal4DSTEMKernels {
  public static let decodeU8Function =
    "h5lz4dc_unshuffle_source_u8_qh5idx"
  public static let decodeU16Function =
    "h5lz4dc_unshuffle_u16_single_block_qh5idx"
  public static let decodeU16TwoBlockFunction =
    "h5lz4dc_unshuffle_u16_qh5idx"
  public static let decodeU16LosslessU8Function =
    "h5lz4dc_unshuffle_u16_lossless_u8_qh5idx"
  public static let decodeU16AuditedLow8Function =
    "h5lz4dc_unshuffle_u16_audited_low8_qh5idx"
  public static let decodeU16AuditedLow8Bin4U16WordMajorFunction =
    "h5lz4dc_unshuffle_u16_audited_low8_bin4_u16_word_major_qh5idx"
  public static let decodeU16AuditedLow8ScalarFunction =
    "h5lz4dc_u16_audited_low8_scalar_qh5idx"
  public static let binU16AuditedLow8ScalarU16WordMajorFunction =
    "h5lz4dc_bin_u16_audited_low8_scalar_u16_word_major_qh5idx"
  public static let binU16AuditedLow8ScalarU16WordMajorFrameOwnedFunction =
    "h5lz4dc_bin_u16_audited_low8_scalar_u16_word_major_frame_owned_qh5idx"
  public static let binU16AuditedLow8ScalarU16WordMajorFrameOwnedRow8Function =
    "h5lz4dc_bin_u16_audited_low8_scalar_u16_word_major_frame_owned_row8_qh5idx"
  public static let binU16AuditedLow8ScalarU16FrameMajorFunction =
    "h5lz4dc_bin_u16_audited_low8_scalar_u16_frame_major_qh5idx"
  public static let binU16AuditedLow8ScalarU16FrameMajorRow8Function =
    "h5lz4dc_bin_u16_audited_low8_scalar_u16_frame_major_row8_qh5idx"
  public static let clearU16WordMajorRangeFunction =
    "clear_u16_word_major_range_qh5idx"

  public static let detectorProductsU8Function = "detector_products_u8"
  public static let detectorProductsU8MomentsFunction =
    "detector_products_u8_with_u64_moments"
  public static let detectorProductsU16Function = "detector_products_u16"
  public static let detectorProductsU16MomentsFunction =
    "detector_products_u16_with_u64_moments"
  public static let detectorSumU8Function = "detector_sum_u8"
  public static let detectorSumU16Function = "detector_sum_u16"
  public static let transposeScanWordsFunction = "transpose_scan_words"
  public static let transposeScanWords32x8Function = "transpose_scan_words_32x8"
  public static let scanBinU8Function = "scan_bin_u8_to_u32_word_major"
  public static let scanBinU16Function = "scan_bin_u16_to_u32_word_major"
  public static let scanDetectorBinU8Function =
    "scan_detector_bin_u8_to_u32_word_major"
  public static let scanDetectorBinU8ToU16Function =
    "scan_detector_bin_u8_to_u16_word_major"
  public static let scanDetectorBinU16Function =
    "scan_detector_bin_u16_to_u32_word_major"
  public static let residentRebinU8Function =
    "resident_rebin_u8_word_major_to_u32_word_major"
  public static let residentRebinU16Function =
    "resident_rebin_u16_word_major_to_u32_word_major"
  public static let residentRebinU32Function =
    "resident_rebin_u32_word_major_to_u32_word_major"
  public static let detectorProductsU32Function = "detector_products_u32_word_major"
  public static let detectorProductsU16WordMajorFunction =
    "detector_products_u16_word_major"
  public static let detectorProductsU16WordMajorMomentsFunction =
    "detector_products_u16_word_major_with_u64_moments"
  public static let centerOfMassU8Function = "center_of_mass_u8_word_major"
  public static let centerOfMassU16Function = "center_of_mass_u16_word_major"
  public static let centerOfMassU32Function = "center_of_mass_u32_word_major"
  public static let centerOfMassU32MomentsFunction = "center_of_mass_u32_moments"
  public static let centerOfMassU64MomentsFunction = "center_of_mass_u64_moments"
  public static let fullSumU8Function = "full_sum_u8_word_major"
  public static let signedDeltaU8Function = "signed_delta_u8_word_major"
  public static let fullSumU16Function = "full_sum_u16_word_major"
  public static let signedDeltaU16Function = "signed_delta_u16_word_major"
  public static let fullSumU32Function = "full_sum_u32_word_major"
  public static let signedDeltaU32Function = "signed_delta_u32_word_major"
  public static let extractU8Function = "extract_u8_word_major_frame"
  public static let extractU16Function = "extract_u16_word_major_frame"
  public static let extractU32Function = "extract_u32_word_major_frame"
  public static let extractU8ToU32Function =
    "extract_u8_word_major_frame_to_u32"
  public static let extractU16ToU32Function =
    "extract_u16_word_major_frame_to_u32"
  public static let extractU32ToU32Function =
    "extract_u32_word_major_frame_to_u32"
  public static let scanRegionSumU8Function =
    "scan_region_sum_u8_word_major_to_u32"
  public static let scanRegionSumU16Function =
    "scan_region_sum_u16_word_major_to_u32"
  public static let scanRegionSumU32Function =
    "scan_region_sum_u32_word_major_to_u32"
  public static let dpcPackFunction = "dpc_pack_complex"
  public static let fftBitReverseRowsFunction = "fft_bit_reverse_rows"
  public static let fftBitReverseColumnsFunction = "fft_bit_reverse_columns"
  public static let fftButterflyRowsFunction = "fft_butterfly_rows"
  public static let fftButterflyColumnsFunction = "fft_butterfly_columns"
  public static let fftNormalizeFunction = "fft_normalize_2d"
  public static let bluesteinPrepareFunction = "bluestein_prepare_2d"
  public static let complexMultiplyFunction = "complex_multiply_in_place"
  public static let bluesteinExtractFunction = "bluestein_extract_2d"
  public static let dpcPoissonFunction = "dpc_poisson_frequency"
  public static let dpcExtractPhaseFunction = "dpc_extract_phase"

  /// Compile the fused QH5IDX bitshuffle/LZ4 decode library.
  public static func makeHDF5Library(device: MTLDevice) throws -> MTLLibrary {
    try makeLibrary(resource: "qh5idx", device: device)
  }

  /// Compile the detector reduction and interactive-drag library.
  public static func makeDetectorLibrary(device: MTLDevice) throws -> MTLLibrary {
    try makeLibrary(resource: "detector", device: device)
  }

  /// Compile the shared CoM/DPC/iDPC small-field library.
  public static func makeDPCLibrary(device: MTLDevice) throws -> MTLLibrary {
    try makeLibrary(resource: "dpc", device: device)
  }

  private static func makeLibrary(
    resource: String,
    device: MTLDevice
  ) throws -> MTLLibrary {
    let packagedURL = Bundle.main.resourceURL?
      .appendingPathComponent(
        "MetalKernels_Metal4DSTEMKernels.bundle",
        isDirectory: true
      )
      .appendingPathComponent("Resources", isDirectory: true)
      .appendingPathComponent("\(resource).metal")
    let url =
      packagedURL.flatMap {
        FileManager.default.fileExists(atPath: $0.path) ? $0 : nil
      } ?? Bundle.module.url(
        forResource: resource,
        withExtension: "metal",
        subdirectory: "Resources"
      ) ?? Bundle.module.url(forResource: resource, withExtension: "metal")
    guard let url else {
      throw Metal4DSTEMKernelsError.missingResource("\(resource).metal")
    }
    do {
      return try device.makeLibrary(
        source: String(contentsOf: url, encoding: .utf8),
        options: nil
      )
    } catch {
      throw Metal4DSTEMKernelsError.libraryCompilation(
        resource: resource,
        message: error.localizedDescription
      )
    }
  }
}
