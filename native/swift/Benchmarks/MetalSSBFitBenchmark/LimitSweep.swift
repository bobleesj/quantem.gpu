import Foundation
import Metal
import MetalSSBKernels

/// The fifth-order refinement of a recorded run's kept low-order fit, repeated with the 22 higher-order
/// coordinates bounded at +-limit (phase in radians at the aperture edge; the default fit uses 2). Each limit
/// starts from the same low-order point. Writes `<label>-limit<L>.f32` (phase_of_mean at the automatic output
/// factor) and `<label>-limits.json` (agreements, parameters, coordinates, timings) into `directory`.
func limitSweep(
  run runURL: URL, limits: [Double], label: String, directory: URL, device: MTLDevice,
  calibration: MetalSSBCalibration, geometry: MetalSSBGeometry, counts: MTLBuffer,
  countType: MetalSSBCountType, rotationEstimate: Double, search: SSBModelFit.HigherOrderSearch
) throws {
  guard
    let run = try JSONSerialization.jsonObject(with: Data(contentsOf: runURL)) as? [String: Any],
    let first = (run["results"] as? [[String: Any]])?.first,
    let initial = first["initial"] as? [String: Any],
    let rotation = (first["fit"] as? [String: Any])?["rotation_deg"] as? Double,
    let c10 = initial["C10_nm"] as? Double, let c12 = initial["C12_nm"] as? Double,
    let phi12 = initial["phi12_deg"] as? Double,
    let tilt = initial["sample_tilt_mrad"] as? [Double],
    let depth = initial["depth_spread_nm"] as? Double
  else { throw failure("\(runURL.path) has no kept low-order fit") }
  let start = SSBModelPoint(
    c10Nanometers: c10, c12Nanometers: c12, phi12Degrees: phi12, sampleTiltMrad: tilt,
    depthSpreadNanometers: depth)
  let factor = try calibration.ssbUpsamplingFactor()
  let engine = try MetalSSBEngine(device: device, geometry: geometry)
  try engine.prepare(brightfield: counts, countType: countType)
  let model = try engine.jointModel(scanDetectorRotationDegrees: rotationEstimate)
  try model.retarget(scanDetectorRotationDegrees: rotation)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  var records: [[String: Any]] = []
  for limit in limits {
    let started = Date()
    let refined = try SSBModelFit.refineFifthOrder(
      start, model: model, search: search,
      bounds: SSBModelFit.fifthOrderBounds(higherOrderLimit: limit))
    let seconds = Date().timeIntervalSince(started)
    let phase = try model.reconstruct(refined.point, upsamplingFactor: factor)
    let values = Array(
      UnsafeBufferPointer(
        start: phase.phase.contents().assumingMemoryBound(to: Float.self),
        count: phase.rows * phase.columns))
    let name = "\(label)-limit\(limit.formatted(.number.precision(.fractionLength(0...2))))"
    try values.withUnsafeBytes { Data($0) }.write(
      to: directory.appendingPathComponent(name + ".f32"))
    let z = refined.result.coordinates
    let p = refined.point
    records.append([
      "limit": limit, "file": name + ".f32", "rows": phase.rows, "columns": phase.columns,
      "seconds": seconds, "agreement": refined.agreement, "agreement64": refined.agreement64,
      "normalization": refined.normalization, "stop": refined.result.stop,
      "evaluations": refined.result.evaluations, "coordinates": z,
      "on_bound": z.dropFirst(6).filter { abs(abs($0) - limit) < 1e-9 }.count,
      "C10_nm": p.c10Nanometers, "C12_nm": p.c12Nanometers, "phi12_deg": p.phi12Degrees,
      "sample_tilt_mrad": p.sampleTiltMrad, "depth_spread_nm": p.depthSpreadNanometers,
      "higher_order": p.higherOrder.map {
        [
          "order": $0.order, "symmetry": $0.symmetry, "magnitude_A": $0.magnitudeAngstroms,
          "angle_rad": $0.angleRadians,
        ] as [String: Any]
      },
    ])
    log(
      String(
        format:
          "%@ limit %.1f: %.2f s, agreement64 %.9e, C10 %.4f nm, C12 %.4f nm, %d on bound, %@",
        label,
        limit, seconds, refined.agreement64, p.c10Nanometers, p.c12Nanometers,
        records.last!["on_bound"] as! Int, refined.result.stop))
  }
  let summary: [String: Any] = [
    "run": runURL.path, "rotation_deg": rotation, "factor": factor,
    "pixel_A": calibration.scanSamplingColumnAngstroms / Double(factor),
    "wavelength_A": model.wavelengthAngstroms, "semiangle_rad": model.semiangleRadians,
    "search": search.rawValue, "limits": records,
  ]
  try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]).write(
    to: directory.appendingPathComponent("\(label)-limits.json"))
}
