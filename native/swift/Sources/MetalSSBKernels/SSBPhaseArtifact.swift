import CryptoKit
import Foundation

/// Portable numerical SSB phase and calibration, without Fourier evidence.
/// Example: `let result = try SSBPhaseArtifact.load(from: downloadedURL)`.
/// Importing is not a new reconstruction or proof of cross-backend solver parity.
public struct SSBPhaseArtifact: Codable, Sendable {
  public let schemaVersion: Int
  public let sourceIdentity: String
  public let rows: Int
  public let columns: Int
  public let phaseEncoding: String
  public let phaseUnits: String
  public let phase: Data
  public let phaseSHA256: String
  public let calibration: MetalSSBCalibration
  public let c10Nanometers: Double
  public let c12Nanometers: Double
  public let phi12Radians: Double
  public let rotationDegrees: Double
  public let provenance: [String: String]
  /// Original run metadata, retained rather than inferred from display settings.
  public let runMetadata: Data

  public static func load(from url: URL, matchingSourceIdentity: String? = nil) throws -> Self {
    let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard size > 0, size <= 32 << 20 else {
      throw ArtifactError.invalid("Choose a complete SSB result smaller than 32 MB.")
    }
    let result =
      try url.pathExtension == "json"
      ? loadPair(from: url)
      : JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    try result.validate(matchingSourceIdentity: matchingSourceIdentity)
    return result
  }

  public func validate(matchingSourceIdentity expected: String? = nil) throws {
    guard schemaVersion == 1, phaseEncoding == "float32-le-row-major",
      phaseUnits == "rad", rows > 0, columns > 0, rows <= 4096, columns <= 4096,
      phase.count == rows * columns * 4,
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
      calibration.beamEnergyKeV, calibration.semiangleMrad,
      calibration.scanStepRowAngstroms, calibration.scanStepColumnAngstroms,
      calibration.detectorStepRowMrad, calibration.detectorStepColumnMrad,
    ]
    let finite = [
      calibration.centerRow, calibration.centerColumn, c10Nanometers,
      c12Nanometers, phi12Radians, rotationDegrees,
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
