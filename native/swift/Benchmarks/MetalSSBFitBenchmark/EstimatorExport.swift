import Foundation
import Metal
import MetalSSBKernels

/// For chosen limits of a `--limit-sweep` result: `phase_of_mean` of the full fitted point at the automatic factor
/// and at 1x, and `complex_wave` (phase and amplitude) at 1x of the same fit's C10/C12, the only terms that
/// estimator accepts (no tilt, depth or higher orders). Writes `<label>-limit<L>-<product>.f32` into `directory`.
func estimatorExport(
  sweep sweepURL: URL, limits: [Double], label: String, directory: URL, device: MTLDevice,
  calibration: MetalSSBCalibration, geometry: MetalSSBGeometry, counts: MTLBuffer,
  countType: MetalSSBCountType, rotationEstimate: Double
) throws {
  guard
    let sweep = try JSONSerialization.jsonObject(with: Data(contentsOf: sweepURL)) as? [String: Any],
    let rotation = sweep["rotation_deg"] as? Double,
    let records = sweep["limits"] as? [[String: Any]]
  else { throw failure("\(sweepURL.path) is not a limit sweep") }
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  let factor = try calibration.ssbUpsamplingFactor()
  let engine = try MetalSSBEngine(device: device, geometry: geometry)
  try engine.prepare(brightfield: counts, countType: countType)
  let model = try engine.jointModel(scanDetectorRotationDegrees: rotationEstimate)
  try model.retarget(scanDetectorRotationDegrees: rotation)
  func write(_ buffer: MTLBuffer, count: Int, _ name: String) throws {
    let values = Array(
      UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    try values.withUnsafeBytes { Data($0) }.write(to: directory.appendingPathComponent(name + ".f32"))
  }
  var products: [[String: Any]] = []
  for limit in limits {
    guard let record = records.first(where: { ($0["limit"] as? Double) == limit }),
      let c10 = record["C10_nm"] as? Double, let c12 = record["C12_nm"] as? Double,
      let phi12 = record["phi12_deg"] as? Double, let tilt = record["sample_tilt_mrad"] as? [Double],
      let depth = record["depth_spread_nm"] as? Double
    else { throw failure("No record for limit \(limit)") }
    let terms = (record["higher_order"] as? [[String: Any]] ?? []).compactMap { term -> SSBModelHigherOrder? in
      guard let order = term["order"] as? Int, let symmetry = term["symmetry"] as? Int,
        let magnitude = term["magnitude_A"] as? Double, let angle = term["angle_rad"] as? Double
      else { return nil }
      return SSBModelHigherOrder(
        order: order, symmetry: symmetry, magnitudeAngstroms: magnitude, angleRadians: angle)
    }
    let full = SSBModelPoint(
      c10Nanometers: c10, c12Nanometers: c12, phi12Degrees: phi12, sampleTiltMrad: tilt,
      depthSpreadNanometers: depth, higherOrder: terms)
    let lowOnly = SSBModelPoint(c10Nanometers: c10, c12Nanometers: c12, phi12Degrees: phi12)
    let tag = "\(label)-limit\(limit.formatted(.number.precision(.fractionLength(0...2))))"
    for (name, point, f, estimator) in [
      ("phase-auto", full, factor, SSBPhaseEstimator.phaseOfMean),
      ("phase-1x", full, 1, .phaseOfMean),
      ("complex-1x", lowOnly, 1, .complexWave),
    ] {
      let result = try model.reconstruct(point, upsamplingFactor: f, estimator: estimator)
      let count = result.rows * result.columns
      try write(result.phase, count: count, "\(tag)-\(name)-phase")
      if let amplitude = result.amplitude { try write(amplitude, count: count, "\(tag)-\(name)-amplitude") }
      products.append(["limit": limit, "product": name, "rows": result.rows, "factor": f])
    }
  }
  let summary: [String: Any] = [
    "sweep": sweepURL.path, "factor": factor,
    "pixel_A": calibration.scanSamplingColumnAngstroms, "products": products,
  ]
  try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]).write(
    to: directory.appendingPathComponent("\(label)-estimators.json"))
  log("wrote \(label) estimator products to \(directory.path)")
}
