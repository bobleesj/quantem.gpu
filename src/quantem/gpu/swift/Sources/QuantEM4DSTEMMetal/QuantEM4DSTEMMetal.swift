import Foundation
import Metal

/// Errors raised while loading shared 4D-STEM Metal resources.
public enum QuantEM4DSTEMMetalError: LocalizedError {
    case missingResource(String)
    case libraryCompilation(resource: String, message: String)

    public var errorDescription: String? {
        switch self {
        case .missingResource(let name):
            "QuantEM4DSTEMMetal is missing \(name)."
        case .libraryCompilation(let resource, let message):
            "QuantEM \(resource) Metal compilation failed: \(message)"
        }
    }
}

/// Shared native 4D-STEM Metal libraries and their stable function names.
public enum QuantEM4DSTEMMetal {
    public static let decodeU8Function =
        "h5lz4dc_unshuffle_source_u8_qh5idx"
    public static let decodeU16Function =
        "h5lz4dc_unshuffle_u16_qh5idx"

    public static let detectorProductsU8Function = "detector_products_u8"
    public static let detectorProductsU16Function = "detector_products_u16"
    public static let detectorSumU8Function = "detector_sum_u8"
    public static let detectorSumU16Function = "detector_sum_u16"
    public static let transposeScanWordsFunction = "transpose_scan_words"
    public static let fullSumU8Function = "full_sum_u8_word_major"
    public static let signedDeltaU8Function = "signed_delta_u8_word_major"
    public static let fullSumU16Function = "full_sum_u16_word_major"
    public static let signedDeltaU16Function = "signed_delta_u16_word_major"
    public static let extractU8Function = "extract_u8_word_major_frame"
    public static let extractU16Function = "extract_u16_word_major_frame"

    /// Compile the fused QH5IDX bitshuffle/LZ4 decode library.
    public static func makeHDF5Library(device: MTLDevice) throws -> MTLLibrary {
        try makeLibrary(resource: "qh5idx", device: device)
    }

    /// Compile the detector reduction and interactive-drag library.
    public static func makeDetectorLibrary(device: MTLDevice) throws -> MTLLibrary {
        try makeLibrary(resource: "detector", device: device)
    }

    private static func makeLibrary(
        resource: String,
        device: MTLDevice
    ) throws -> MTLLibrary {
        let url = Bundle.module.url(
            forResource: resource,
            withExtension: "metal",
            subdirectory: "Resources"
        ) ?? Bundle.module.url(forResource: resource, withExtension: "metal")
        guard let url else {
            throw QuantEM4DSTEMMetalError.missingResource("\(resource).metal")
        }
        do {
            return try device.makeLibrary(
                source: String(contentsOf: url, encoding: .utf8),
                options: nil
            )
        } catch {
            throw QuantEM4DSTEMMetalError.libraryCompilation(
                resource: resource,
                message: error.localizedDescription
            )
        }
    }
}
