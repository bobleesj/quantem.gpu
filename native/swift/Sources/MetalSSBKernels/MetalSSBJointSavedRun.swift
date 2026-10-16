import Foundation
import Metal

/// A phase-only native joint SSB result with complete calibration and source ROI.
/// Separate from the legacy complex-wave format: no object amplitude is inferred.
public struct MetalSSBJointSavedRun: Codable, Sendable {
  public let schema: String
  public let sourceIdentity: String
  public let backendRevision: String
  public let sourceRows: Int
  public let sourceColumns: Int
  public let region: SSBScanRegion?
  public let rows: Int
  public let columns: Int
  public let upsamplingFactor: Int
  public let calibration: MetalSSBCalibration
  public let parameters: SSBJointParameters
  public let rotationDegrees: Float
  public let optimization: SSBJointOptimizationResult?
  public let provenance: [String: String]
  public let phase: Data
  public let phaseSHA256: String

  public init(result: MetalSSBJointPhase, sourceIdentity: String, backendRevision: String,
              sourceRows: Int, sourceColumns: Int, region: SSBScanRegion?,
              calibration: MetalSSBCalibration, optimization: SSBJointOptimizationResult?,
              provenance: [String: String]) throws {
    guard result.phase.storageMode == .shared else { throw MetalSSBError.invalidGeometry("Complete the phase readback before saving.") }
    schema = "quantem.ssb-joint-run/1"
    self.sourceIdentity = sourceIdentity; self.backendRevision = backendRevision
    self.sourceRows = sourceRows; self.sourceColumns = sourceColumns; self.region = region
    rows = result.rows; columns = result.columns; upsamplingFactor = result.upsamplingFactor
    self.calibration = calibration; parameters = result.parameters; rotationDegrees = result.scanDetectorRotationDegrees
    self.optimization = optimization; self.provenance = provenance
    phase = Data(bytes: result.phase.contents(), count: rows * columns * 4)
    phaseSHA256 = SSBPhaseArtifact.digest(phase)
    try validate(sourceIdentity: sourceIdentity)
  }
  public func save(to url: URL) throws {
    try validate(sourceIdentity: sourceIdentity)
    let encoder = PropertyListEncoder(); encoder.outputFormat = .binary
    try encoder.encode(self).write(to: url, options: .atomic)
  }
  public static func load(from url: URL, sourceIdentity: String) throws -> Self {
    let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard size > 0, size <= 256 << 20 else { throw MetalSSBError.invalidGeometry("Choose a complete joint SSB result smaller than 256 MB.") }
    let result = try PropertyListDecoder().decode(Self.self, from: Data(contentsOf: url))
    try result.validate(sourceIdentity: sourceIdentity); return result
  }
  public func phaseBuffer(device: MTLDevice) throws -> MTLBuffer {
    guard let buffer = phase.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }) else {
      throw MetalSSBError.allocation("saved joint phase")
    }; return buffer
  }
  private func validate(sourceIdentity: String) throws {
    guard schema == "quantem.ssb-joint-run/1", self.sourceIdentity == sourceIdentity,
      !sourceIdentity.isEmpty, !backendRevision.isEmpty,
      [1, 2, 3, 4, 8].contains(upsamplingFactor), sourceRows > 0, sourceColumns > 0,
      rows > 0, columns > 0, rows <= 4096, columns <= 4096,
      rows % upsamplingFactor == 0, columns % upsamplingFactor == 0,
      rows / upsamplingFactor == (region?.size ?? sourceRows),
      columns / upsamplingFactor == (region?.size ?? sourceColumns),
      phase.count == rows * columns * 4, phaseSHA256 == SSBPhaseArtifact.digest(phase), rotationDegrees.isFinite else {
      throw MetalSSBError.invalidGeometry("This joint SSB result is incomplete, corrupted or belongs to another source.")
    }
    try region?.validate(sourceRows: sourceRows, sourceColumns: sourceColumns)
    try parameters.validate(); _ = try calibration.ssbUpsamplingFactor()
  }
}
