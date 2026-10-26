import Foundation
import Metal
import MetalDirectPtychoKernels

/// The kept low-order fit and the fifth-order fit of a recorded run, reconstructed with `phase_of_mean` at the
/// calibration's automatic output sampling, so results of different searches are drawn by one code path.
/// Writes `<label>-<stage>.f32` (rows x columns, float32 radians) and `<label>.json` (shape, factor, pixel size,
/// fit parameters) into `directory`.
func exportPhases(
  run runURL: URL, label: String, directory: URL, device: MTLDevice,
  calibration: MetalDirectPtychoCalibration,
  geometry: MetalDirectPtychoGeometry, counts: MTLBuffer, countType: MetalDirectPtychoCountType,
  rotationEstimate: Double
) throws {
  guard
    let run = try JSONSerialization.jsonObject(with: Data(contentsOf: runURL)) as? [String: Any],
    let first = (run["results"] as? [[String: Any]])?.first,
    let fit = first["fit"] as? [String: Any], let rotation = fit["rotation_deg"] as? Double
  else { throw failure("\(runURL.path) has no recorded fit") }
  func point(_ record: [String: Any]) throws -> DirectPtychoModelPoint {
    guard let c10 = record["C10_nm"] as? Double, let c12 = record["C12_nm"] as? Double,
      let phi12 = record["phi12_deg"] as? Double,
      let tilt = record["sample_tilt_mrad"] as? [Double],
      let depth = record["depth_spread_nm"] as? Double
    else { throw failure("\(runURL.path) has an incomplete fit record") }
    let terms = (record["higher_order"] as? [[String: Any]] ?? []).compactMap {
      term -> DirectPtychoModelHigherOrder? in
      guard let order = term["order"] as? Int, let symmetry = term["symmetry"] as? Int,
        let magnitude = term["magnitude_A"] as? Double, let angle = term["angle_rad"] as? Double
      else { return nil }
      return DirectPtychoModelHigherOrder(
        order: order, symmetry: symmetry, magnitudeAngstroms: magnitude, angleRadians: angle)
    }
    return DirectPtychoModelPoint(
      c10Nanometers: c10, c12Nanometers: c12, phi12Degrees: phi12, sampleTiltMrad: tilt,
      depthSpreadNanometers: depth, higherOrder: terms)
  }
  let factor = try calibration.ssbUpsamplingFactor()
  let engine = try MetalDirectPtychoEngine(device: device, geometry: geometry)
  try engine.prepare(brightfield: counts, countType: countType)
  let model = try engine.jointModel(scanDetectorRotationDegrees: rotationEstimate)
  try model.retarget(scanDetectorRotationDegrees: rotation)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  var stages: [String: Any] = [:]
  for (stage, record) in [("low", first["initial"] as? [String: Any]), ("fifth", fit)] {
    guard let record else { continue }
    let phase = try model.reconstruct(try point(record), upsamplingFactor: factor)
    let values = Array(
      UnsafeBufferPointer(
        start: phase.phase.contents().assumingMemoryBound(to: Float.self),
        count: phase.rows * phase.columns))
    try values.withUnsafeBytes { Data($0) }.write(
      to: directory.appendingPathComponent("\(label)-\(stage).f32"))
    stages[stage] =
      [
        "rows": phase.rows, "columns": phase.columns, "C10_nm": record["C10_nm"] as Any,
        "C12_nm": record["C12_nm"] as Any, "phi12_deg": record["phi12_deg"] as Any,
        "sample_tilt_mrad": record["sample_tilt_mrad"] as Any,
        "depth_spread_nm": record["depth_spread_nm"] as Any,
        "sample_agreement": record["sample_agreement"] as Any,
      ] as [String: Any]
  }
  let summary: [String: Any] = [
    "run": runURL.path, "rotation_deg": rotation, "factor": factor,
    "pixel_A": calibration.scanSamplingColumnAngstroms / Double(factor),
    "fit_total_s": first["fit_total_s"] as Any, "stages": stages,
  ]
  try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]).write(
    to: directory.appendingPathComponent("\(label).json"))
  log("wrote \(label) phases at \(factor)x auto to \(directory.path)")
}
