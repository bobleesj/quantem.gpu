import CryptoKit
import Foundation

/// Portable numerical SSB phase and calibration, without Fourier evidence.
///
/// The record is `quantem.ssb-phase/2`, the JSON that sits beside its `.npy` phase and that
/// the Python reader `quantem.gpu.formats.ssb_phase.read_result` checks: snake_case keys,
/// aberrations as `C10_nm`, `C12_nm` and `phi12_deg`, and the physical
/// `scan_detector_rotation_deg` in [0, 360) with `scan_detector_rotation_flipped`. Every other schema,
/// including the version-1 `live.ssb` record, is refused, not converted.
/// Example: `let result = try SSBPhaseArtifact.load(from: downloadedURL)`.
/// Importing is not a new reconstruction or proof of cross-backend solver parity.
public struct SSBPhaseArtifact: Codable, Sendable {
  public static let recordSchema = "quantem.ssb-phase/2"

  /// Aberrations as the record states them, in nm and degrees (the engine works in Angstrom
  /// and radians).
  public struct Aberrations: Codable, Equatable, Sendable {
    public let c10Nanometers: Double
    public let c12Nanometers: Double
    public let phi12Degrees: Double

    private enum CodingKeys: String, CodingKey {
      case c10Nanometers = "C10_nm"
      case c12Nanometers = "C12_nm"
      case phi12Degrees = "phi12_deg"
    }
  }

  public let schema: String
  public let sourceIdentity: String
  /// `[rows, cols]` of the phase image.
  public let shape: [Int]
  public let phaseEncoding: String
  public let phaseUnit: String
  public let phase: Data
  public let phaseSHA256: String
  public let calibration: MetalSSBCalibration
  public let aberrations: Aberrations
  public let scanDetectorRotationDegrees: Double
  public let scanDetectorRotationFlipped: Bool
  public let provenance: [String: String]
  /// Original run metadata, retained rather than inferred from display settings.
  public let runMetadata: Data

  private enum CodingKeys: String, CodingKey {
    case schema
    case sourceIdentity = "source_identity"
    case shape
    case phaseEncoding = "phase_encoding"
    case phaseUnit = "phase_unit"
    case phase
    case phaseSHA256 = "phase_sha256"
    case calibration
    case aberrations
    case scanDetectorRotationDegrees = "scan_detector_rotation_deg"
    case scanDetectorRotationFlipped = "scan_detector_rotation_flipped"
    case provenance
    case runMetadata = "run_metadata"
  }

  /// Read a `.json` record and its `.npy` phase, then check them as the Python reader does.
  public static func load(from url: URL, matchingSourceIdentity: String? = nil) throws -> Self {
    let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard size > 0, size <= 32 << 20, url.pathExtension == "json" else {
      throw ArtifactError.invalid(
        "Choose the .json record of an SSB result, smaller than 32 MB, with its .npy phase beside it."
      )
    }
    let result = try loadPair(from: url)
    try result.validate(matchingSourceIdentity: matchingSourceIdentity)
    return result
  }

  public func validate(matchingSourceIdentity expected: String? = nil) throws {
    guard schema == Self.recordSchema, phaseEncoding == "float32-le-row-major",
      phaseUnit == "rad", shape.count == 2, shape.allSatisfy({ (1...4096).contains($0) }),
      phase.count == shape[0] * shape[1] * 4,
      sourceIdentity.count == 64,
      sourceIdentity.allSatisfy({ "0123456789abcdef".contains($0) }),
      expected == nil || expected == sourceIdentity,
      Self.digest(phase) == phaseSHA256
    else {
      throw ArtifactError.invalid(
        "SSB result is incomplete, unsupported, or belongs to different data. Export it again from the original run."
      )
    }
    let positive = [
      calibration.voltageKV, calibration.semiangleMrad,
      calibration.scanSamplingRowAngstroms, calibration.scanSamplingColumnAngstroms,
      calibration.detectorSamplingRowMrad, calibration.detectorSamplingColumnMrad,
    ]
    let finite = [
      calibration.brightfieldCenterRowPixels, calibration.brightfieldCenterColumnPixels,
      aberrations.c10Nanometers, aberrations.c12Nanometers, aberrations.phi12Degrees,
      scanDetectorRotationDegrees,
    ]
    guard positive.allSatisfy({ $0.isFinite && $0 > 0 }),
      finite.allSatisfy(\.isFinite), !provenance.isEmpty,
      calibration.brightfieldRadiusPixels.map({ $0.isFinite && $0 > 0 }) ?? true,
      (calibration.excludedDetectorPixels ?? []).allSatisfy({ $0 >= 0 }),
      (try? JSONSerialization.jsonObject(with: runMetadata)) != nil,
      phaseValues().allSatisfy(\.isFinite)
    else {
      throw ArtifactError.invalid(
        "SSB result has invalid calibration or phase values. Check the exported run; no guessed calibration is applied."
      )
    }
  }

  /// Decode only the small scientific image; never load raw acquisitions here.
  public func phaseValues() -> [Float] {
    phase.withUnsafeBytes { bytes in
      stride(from: 0, to: phase.count, by: 4).map {
        Float(
          bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self))
        )
      }
    }
  }

  public static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private enum ArtifactError: LocalizedError {
    case invalid(String)
    var errorDescription: String? {
      if case .invalid(let text) = self { return text }
      return nil
    }
  }
}
