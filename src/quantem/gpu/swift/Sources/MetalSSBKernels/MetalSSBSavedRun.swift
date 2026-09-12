import CryptoKit
import Foundation
import Metal

/// Complete native SSB result for local reopening without running the optimizer.
/// The app chooses storage location and retention. Original measurements are not modified.
/// Example: `try MetalSSBSavedRun(...).save(to: url)`; reopen with `load(from:matchingSourceIdentity:)`.
public struct MetalSSBSavedRun: Codable, Sendable {
  public let schemaVersion: Int
  public let sourceIdentity: String
  public let backendRevision: String
  public let createdAt: Date
  public let geometry: MetalSSBGeometry
  public let calibration: MetalSSBCalibration?
  public let calibrationProvenance: [String: String]?
  public let aberrations: MetalSSBAberrations
  public let rotationDegrees: Float
  public let optimization: SSBOptimizationResult?
  public let optimizedRotationDegrees: Float?
  public let seed: UInt64
  public let provenance: MetalSSBProvenance
  public let reconstructionWallSeconds: Double
  public let reconstructionGPUSeconds: Double
  private let object: Data
  private let fourierSum: Data
  private let objectSHA256: String
  private let fourierSHA256: String

  public init(
    result: MetalSSBResult, sourceIdentity: String, backendRevision: String,
    geometry: MetalSSBGeometry, aberrations: MetalSSBAberrations,
    rotationDegrees: Float, optimization: SSBOptimizationResult? = nil, seed: UInt64 = 42,
    calibration: MetalSSBCalibration? = nil, calibrationProvenance: [String: String]? = nil,
    optimizedRotationDegrees: Float? = nil
  ) throws {
    guard result.object.storageMode == .shared, result.fourierSum.storageMode == .shared else {
      throw SavedRunError.invalid("Saving requires completed, CPU-readable reconstruction buffers.")
    }
    schemaVersion = 1
    self.sourceIdentity = sourceIdentity
    self.backendRevision = backendRevision
    createdAt = Date()
    self.geometry = geometry
    self.calibration = calibration
    self.calibrationProvenance = calibrationProvenance
    self.aberrations = aberrations
    self.rotationDegrees = rotationDegrees
    self.optimization = optimization
    self.optimizedRotationDegrees = optimizedRotationDegrees
    self.seed = seed
    provenance = result.provenance
    reconstructionWallSeconds = result.wallSeconds
    reconstructionGPUSeconds = result.gpuSeconds
    let bytes = 512 * 512 * MemoryLayout<SIMD2<Float>>.stride
    guard result.object.length >= bytes, result.fourierSum.length >= bytes else {
      throw SavedRunError.invalid(
        "The reconstruction buffers are incomplete. Run SSB again before saving.")
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
        "The saved SSB file is empty or exceeds the supported 32 MB format.")
    }
    let run = try PropertyListDecoder().decode(Self.self, from: Data(contentsOf: url))
    try run.validate(matchingSourceIdentity: matchingSourceIdentity)
    return run
  }

  /// Restore only the two scientific image buffers. No source decode or optimizer runs.
  /// Result timings are historical reconstruction timings, not reopening measurements.
  public func reconstruction(device: MTLDevice) throws -> MetalSSBResult {
    try validate(matchingSourceIdentity: sourceIdentity)
    func buffer(_ data: Data) throws -> MTLBuffer {
      guard
        let result = data.withUnsafeBytes({ bytes in
          device.makeBuffer(
            bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        })
      else { throw MetalSSBError.allocation("saved SSB image") }
      return result
    }
    return MetalSSBResult(
      object: try buffer(object), fourierSum: try buffer(fourierSum),
      wallSeconds: reconstructionWallSeconds, gpuSeconds: reconstructionGPUSeconds,
      provenance: provenance)
  }

  private func validate(matchingSourceIdentity expected: String) throws {
    guard schemaVersion == 1 else {
      throw SavedRunError.invalid(
        "This saved SSB format is unsupported. Open it with a compatible app version.")
    }
    guard !sourceIdentity.isEmpty, sourceIdentity == expected, !backendRevision.isEmpty else {
      throw SavedRunError.invalid(
        "This saved SSB run belongs to a different source, or lacks its compute revision.")
    }
    guard rotationDegrees.isFinite, optimizedRotationDegrees?.isFinite ?? true,
      aberrations.c10Nanometers.isFinite, aberrations.c12Nanometers.isFinite,
      aberrations.phi12Radians.isFinite else {
      throw SavedRunError.invalid("Saved SSB coefficients must be finite. Recompute or restore a valid result.")
    }
    let bytes = 512 * 512 * MemoryLayout<SIMD2<Float>>.stride
    guard provenance.scanRows == 512, provenance.scanColumns == 512,
      object.count == bytes, fourierSum.count == bytes,
      Self.digest(object) == objectSHA256, Self.digest(fourierSum) == fourierSHA256
    else {
      throw SavedRunError.invalid(
        "Saved SSB image data are incomplete or corrupted. Recompute from the original acquisition."
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
