import CryptoKit
import Foundation
import Metal

/// Complete native direct ptychography result for local reopening without running the optimizer.
/// The app chooses storage location and retention. Original measurements are not modified.
/// Example: `try MetalDirectSavedRun(...).save(to: url)`; reopen with `load(from:matchingSourceIdentity:)`.
///
/// Aberrations are the engine's Angstrom and radians; the rotation is the scan-to-detector angle
/// the engine applied. Runs saved before `quantem.direct-saved-run/3` stored Angstrom under nm
/// names and are refused with that reason rather than read under another unit.
public struct MetalDirectSavedRun: Codable, Sendable {
  public static let recordSchema = "quantem.direct-saved-run/3"
  public let schema: String
  public let sourceIdentity: String
  public let backendRevision: String
  public let createdAt: Date
  public let geometry: MetalDirectPtychoGeometry
  public let calibration: MetalDirectPtychoCalibration?
  public let calibrationProvenance: [String: String]?
  public let aberrations: MetalDirectPtychoAberrations
  public let scanDetectorRotationDegrees: Float
  public let optimization: DirectOptimizationResult?
  public let optimizedScanDetectorRotationDegrees: Float?
  public let seed: UInt64
  public let provenance: MetalDirectPtychoProvenance
  public let reconstructionWallSeconds: Double
  public let reconstructionGPUSeconds: Double
  private let object: Data
  private let fourierSum: Data
  private let objectSHA256: String
  private let fourierSHA256: String

  public init(
    result: MetalDirectPtychoResult, sourceIdentity: String, backendRevision: String,
    geometry: MetalDirectPtychoGeometry, aberrations: MetalDirectPtychoAberrations,
    scanDetectorRotationDegrees: Float, optimization: DirectOptimizationResult? = nil,
    seed: UInt64 = 42,
    calibration: MetalDirectPtychoCalibration? = nil,
    calibrationProvenance: [String: String]? = nil,
    optimizedScanDetectorRotationDegrees: Float? = nil
  ) throws {
    guard result.object.storageMode == .shared, result.fourierSum.storageMode == .shared else {
      throw SavedRunError.invalid("Saving requires completed, CPU-readable reconstruction buffers.")
    }
    schema = Self.recordSchema
    self.sourceIdentity = sourceIdentity
    self.backendRevision = backendRevision
    createdAt = Date()
    self.geometry = geometry
    self.calibration = calibration
    self.calibrationProvenance = calibrationProvenance
    self.aberrations = aberrations
    self.scanDetectorRotationDegrees = scanDetectorRotationDegrees
    self.optimization = optimization
    self.optimizedScanDetectorRotationDegrees = optimizedScanDetectorRotationDegrees
    self.seed = seed
    provenance = result.provenance
    reconstructionWallSeconds = result.wallSeconds
    reconstructionGPUSeconds = result.gpuSeconds
    guard [128, 256, 512].contains(provenance.scanRows),
      provenance.scanColumns == provenance.scanRows
    else {
      throw SavedRunError.invalid("Unsupported native direct ptychography scan dimensions.")
    }
    let bytes = provenance.scanRows * provenance.scanColumns * MemoryLayout<SIMD2<Float>>.stride
    guard result.object.length >= bytes, result.fourierSum.length >= bytes else {
      throw SavedRunError.invalid(
        "The reconstruction buffers are incomplete. Run direct ptychography again before saving.")
    }
    object = Data(bytes: result.object.contents(), count: bytes)
    fourierSum = Data(bytes: result.fourierSum.contents(), count: bytes)
    objectSHA256 = Self.digest(object)
    fourierSHA256 = Self.digest(fourierSum)
    try validate(matchingSourceIdentity: sourceIdentity)
  }

  /// Atomically save numerical results and complete calibration/fit provenance.
  public func save(to url: URL) throws {
    try validate(matchingSourceIdentity: sourceIdentity)
    let encoder = PropertyListEncoder()
    encoder.outputFormat = .binary
    try encoder.encode(self).write(to: url, options: .atomic)
  }

  /// Reject another acquisition or an incomplete/unsupported artifact; never silently refit.
  public static func load(from url: URL, matchingSourceIdentity: String) throws -> Self {
    let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard size > 0, size <= 32 * 1024 * 1024 else {
      throw SavedRunError.invalid(
        "The saved direct ptychography file is empty or exceeds the supported 32 MB format.")
    }
    let data = try Data(contentsOf: url)
    // The schema is read first, so an earlier format is refused with its reason, not a key error.
    struct Header: Decodable { let schema: String }
    let schema = (try? PropertyListDecoder().decode(Header.self, from: data))?.schema
    guard schema == recordSchema else {
      throw SavedRunError.invalid(
        "This saved direct ptychography run is not \(recordSchema). Reconstruct it again from the original acquisition."
      )
    }
    let run = try PropertyListDecoder().decode(Self.self, from: data)
    try run.validate(matchingSourceIdentity: matchingSourceIdentity)
    return run
  }

  /// Restore only the two scientific image buffers. No source decode or optimizer runs.
  /// Result timings are historical reconstruction timings, not reopening measurements.
  public func reconstruction(device: MTLDevice) throws -> MetalDirectPtychoResult {
    try validate(matchingSourceIdentity: sourceIdentity)
    func buffer(_ data: Data) throws -> MTLBuffer {
      guard
        let result = data.withUnsafeBytes({ bytes in
          device.makeBuffer(
            bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        })
      else { throw MetalDirectPtychoError.allocation("saved direct ptychography image") }
      return result
    }
    return MetalDirectPtychoResult(
      object: try buffer(object), fourierSum: try buffer(fourierSum),
      wallSeconds: reconstructionWallSeconds, gpuSeconds: reconstructionGPUSeconds,
      provenance: provenance)
  }

  private func validate(matchingSourceIdentity expected: String) throws {
    guard schema == Self.recordSchema else {
      throw SavedRunError.invalid(
        "This saved direct ptycho format is unsupported. Open it with a compatible app version.")
    }
    guard !sourceIdentity.isEmpty, sourceIdentity == expected, !backendRevision.isEmpty else {
      throw SavedRunError.invalid(
        "This saved direct ptycho run is from another source, or lacks its compute revision.")
    }
    guard scanDetectorRotationDegrees.isFinite,
      optimizedScanDetectorRotationDegrees?.isFinite ?? true,
      aberrations.c10Angstroms.isFinite, aberrations.c12Angstroms.isFinite,
      aberrations.phi12Radians.isFinite
    else {
      throw SavedRunError.invalid(
        "Saved direct ptycho coefficients must be finite. Recompute or restore a valid result.")
    }
    guard [128, 256, 512].contains(provenance.scanRows),
      provenance.scanColumns == provenance.scanRows
    else {
      throw SavedRunError.invalid("Unsupported native direct ptychography scan dimensions.")
    }
    let bytes = provenance.scanRows * provenance.scanColumns * MemoryLayout<SIMD2<Float>>.stride
    guard geometry.qxByRow.count == provenance.scanRows,
      geometry.qyByColumn.count == provenance.scanColumns,
      object.count == bytes, fourierSum.count == bytes,
      Self.digest(object) == objectSHA256, Self.digest(fourierSum) == fourierSHA256
    else {
      throw SavedRunError.invalid(
        "Saved direct ptychography image data are incomplete or corrupted. Recompute from the original acquisition."
      )
    }
  }

  private static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private enum SavedRunError: LocalizedError {
    case invalid(String)
    var errorDescription: String? {
      if case .invalid(let message) = self { return message }
      return nil
    }
  }
}
