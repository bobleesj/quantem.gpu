import Foundation
import Metal

/// A phase-only native joint SSB result with complete calibration and source ROI.
/// Separate from the legacy complex-wave format: no object amplitude is inferred.
public struct MetalDirectPtychoJointSavedRun: Codable, Sendable {
  public let schema: String
  public let sourceIdentity: String
  public let backendRevision: String
  public let sourceRows: Int
  public let sourceColumns: Int
  public let region: DirectPtychoScanRegion?
  public let rows: Int
  public let columns: Int
  public let upsamplingFactor: Int
  /// Auto or an explicit override; absent in older files and when the factor was passed directly.
  public let outputSampling: DirectPtychoOutputSampling?
  /// The kernel that produced the phase; absent in older files.
  public let kernel: MetalKernelIdentity?
  public let calibration: MetalDirectPtychoCalibration
  public let parameters: DirectPtychoJointParameters
  /// Exact reconstruction point. Absent in older files that stored only the Float UI parameters.
  public let point: DirectPtychoModelPoint?
  public let rotationDegrees: Double
  public let optimization: DirectPtychoJointOptimizationResult?
  /// The fit that produced `parameters` (both rotation branches when they were compared).
  public let fit: DirectPtychoBranchedFit?
  public let provenance: [String: String]
  public let phase: Data
  public let phaseSHA256: String

  public init(
    result: MetalDirectPtychoJointPhase, sourceIdentity: String, backendRevision: String,
    sourceRows: Int, sourceColumns: Int, region: DirectPtychoScanRegion?,
    calibration: MetalDirectPtychoCalibration, optimization: DirectPtychoJointOptimizationResult?,
    fit: DirectPtychoBranchedFit? = nil, provenance: [String: String]
  ) throws {
    guard result.phase.storageMode == .shared else {
      throw MetalDirectPtychoError.invalidGeometry("Complete the phase readback before saving.")
    }
    schema = "quantem.ssb-joint-run/1"
    self.sourceIdentity = sourceIdentity
    self.backendRevision = backendRevision
    self.sourceRows = sourceRows
    self.sourceColumns = sourceColumns
    self.region = region
    rows = result.rows
    columns = result.columns
    upsamplingFactor = result.upsamplingFactor
    outputSampling = result.outputSampling
    kernel = result.kernel
    self.calibration = calibration
    parameters = result.parameters
    point = result.point
    rotationDegrees = result.scanDetectorRotationDegrees
    self.optimization = optimization
    self.fit = fit
    self.provenance = provenance
    phase = Data(bytes: result.phase.contents(), count: rows * columns * 4)
    phaseSHA256 = SSBPhaseArtifact.digest(phase)
    try validate(sourceIdentity: sourceIdentity)
  }
  public func save(to url: URL) throws {
    try validate(sourceIdentity: sourceIdentity)
    let encoder = PropertyListEncoder()
    encoder.outputFormat = .binary
    try encoder.encode(self).write(to: url, options: .atomic)
  }
  public static func load(from url: URL, sourceIdentity: String) throws -> Self {
    let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard size > 0, size <= 256 << 20 else {
      throw MetalDirectPtychoError.invalidGeometry(
        "Choose a complete joint SSB result smaller than 256 MB.")
    }
    let result = try PropertyListDecoder().decode(Self.self, from: Data(contentsOf: url))
    try result.validate(sourceIdentity: sourceIdentity)
    return result
  }
  public func phaseBuffer(device: MTLDevice) throws -> MTLBuffer {
    guard
      let buffer = phase.withUnsafeBytes({
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    else {
      throw MetalDirectPtychoError.allocation("saved joint phase")
    }
    return buffer
  }
  private func validate(sourceIdentity: String) throws {
    guard schema == "quantem.ssb-joint-run/1", self.sourceIdentity == sourceIdentity,
      !sourceIdentity.isEmpty, !backendRevision.isEmpty,
      [1, 2, 3, 4, 8].contains(upsamplingFactor), sourceRows > 0, sourceColumns > 0,
      rows > 0, columns > 0, rows <= 4096, columns <= 4096,
      rows % upsamplingFactor == 0, columns % upsamplingFactor == 0,
      rows / upsamplingFactor == (region?.size ?? sourceRows),
      columns / upsamplingFactor == (region?.size ?? sourceColumns),
      phase.count == rows * columns * 4, phaseSHA256 == SSBPhaseArtifact.digest(phase),
      rotationDegrees.isFinite,
      outputSampling.map({ $0 == .auto ? upsamplingFactor <= 4 : $0 == .factor(upsamplingFactor) })
        ?? true
    else {
      throw MetalDirectPtychoError.invalidGeometry(
        "This joint SSB result is incomplete, corrupted or belongs to another source.")
    }
    try region?.validate(sourceRows: sourceRows, sourceColumns: sourceColumns)
    try parameters.validate()
    if let point {
      try point.validate()
      guard point.jointParameters == parameters else {
        throw MetalDirectPtychoError.invalidGeometry(
          "The saved Double model point does not match its displayed SSB parameters.")
      }
    }
    _ = try calibration.ssbUpsamplingFactor()
  }
}
