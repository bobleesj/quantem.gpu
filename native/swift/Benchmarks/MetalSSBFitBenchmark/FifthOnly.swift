import Foundation
import Metal
import MetalSSBKernels

/// The fifth-order stage alone, from the kept low-order fit and rotation of an earlier benchmark result.
func fifthOnly(
  baselineURL: URL, options: Options, device: MTLDevice,
  setup: (geometry: MetalSSBGeometry, pixels: [Int]), counts: MTLBuffer,
  countType: MetalSSBCountType, rotationEstimate: Double
) throws -> [[String: Any]] {
  guard
    let baseline = try JSONSerialization.jsonObject(with: Data(contentsOf: baselineURL))
      as? [String: Any],
    let first = (baseline["results"] as? [[String: Any]])?.first,
    let initial = first["initial"] as? [String: Any],
    let fit = first["fit"] as? [String: Any],
    let rotation = fit["rotation_deg"] as? Double,
    let c10 = initial["C10_nm"] as? Double, let c12 = initial["C12_nm"] as? Double,
    let phi12 = initial["phi12_deg"] as? Double, let tilt = initial["sample_tilt_mrad"] as? [Double],
    let depth = initial["depth_spread_nm"] as? Double
  else { throw failure("\(baselineURL.path) has no kept initial fit") }
  let point = SSBModelPoint(
    c10Nanometers: c10, c12Nanometers: c12, phi12Degrees: phi12, sampleTiltMrad: tilt,
    depthSpreadNanometers: depth)
  var results: [[String: Any]] = []
  for repeatIndex in 0..<options.repeats {
    try autoreleasepool {
      let engine = try MetalSSBEngine(device: device, geometry: setup.geometry)
      try engine.prepare(brightfield: counts, countType: countType)
      let model = try engine.jointModel(scanDetectorRotationDegrees: rotationEstimate)
      try model.retarget(scanDetectorRotationDegrees: rotation)
      var result: [String: Any] = [
        "repeat": repeatIndex, "device": device.name, "rotation_deg": rotation,
        "bf_pixels": setup.pixels.count, "band_count": model.bandCount,
        "search": options.search.rawValue,
      ]
      if options.gradientCheck {
        let start =
          [
            point.c10Angstroms / 100, point.c12Angstroms / 100 * cos(2 * point.phi12Radians),
            point.c12Angstroms / 100 * sin(2 * point.phi12Radians), point.sampleTiltMrad[0] / 5,
            point.sampleTiltMrad[1] / 5, point.depthSpreadAngstroms / 100,
          ] + Array(repeating: 0.0, count: 22)
        var checks = [
          try SSBModelFit.fifthOrderGradientCheck(
            model: model, z: start, steps: [1e-2, 3e-3, 1e-3, 3e-4])
        ]
        if let refinement = fit["refinement"] as? [String: Any],
          let z = refinement["coordinates"] as? [Double]
        {
          checks.append(
            try SSBModelFit.fifthOrderGradientCheck(
              model: model, z: z, steps: [1e-2, 3e-3, 1e-3, 3e-4]))
        }
        result["gradient_checks"] = checks
        for check in checks {
          let differences = check["central_differences"] as! [[String: Any]]
          log(
            String(
              format: "agreement %.9g: objective %.2f ms, value+gradient %.2f ms (x28: %.1f / %.1f ms)",
              check["agreement"] as! Double, 1e3 * (check["objective_s"] as! Double),
              1e3 * (check["value_and_gradient_s"] as! Double),
              1e3 * (check["objective_28_s"] as! Double),
              1e3 * (check["value_and_gradient_28_s"] as! Double)))
          for d in differences {
            log(
              String(
                format: "  h %.0e: max |central - exact| / max |central| = %.3e",
                d["step"] as! Double, d["max_relative_error"] as! Double))
          }
        }
      } else {
        let started = Date()
        let refined = try SSBModelFit.refineFifthOrder(
          point, model: model, search: options.search,
          maxEvaluations: SSBFitOptions().maxHigherOrderEvaluations,
          gradientSettings: options.gradientSettings, log: options.verbose ? { log($0) } : nil)
        let seconds = Date().timeIntervalSince(started)
        let r = refined.result
        result["fifth_order_s"] = seconds
        result["agreement"] = refined.agreement
        result["normalization"] = refined.normalization
        result["refinement"] = [
          "evaluations": r.evaluations, "executed_objectives": r.executedObjectives,
          "iterations": r.iterations, "stop": r.stop, "normalized_agreement": r.agreement,
          "coordinates": r.coordinates, "subsampled_gradients": r.subsampledGradients ?? 0,
          "gradient_evaluations": r.gradientEvaluations ?? 0,
          "hessian_stride": r.hessianStride ?? 0,
          "trace": r.trace.map { [$0.evaluation, $0.agreement, $0.elapsedSeconds] },
        ] as [String: Any]
        if let base = fit["refinement"] as? [String: Any],
          let z = base["coordinates"] as? [Double],
          let agreement = fit["sample_agreement"] as? Double
        {
          let dz = zip(r.coordinates, z).map { abs($0 - $1) }
          result["baseline_agreement"] = agreement
          result["agreement_relative_to_baseline"] = refined.agreement / agreement - 1
          result["max_abs_dz"] = dz.max() ?? 0
          result["abs_dz"] = dz
          log(
            String(
              format:
                "repeat %d: fifth-order %.2f s, %d evaluations, %d iterations (%@), agreement %.9g (baseline %.9g, rel %+.3e), max |dz| %.3e",
              repeatIndex, seconds, r.evaluations, r.iterations, r.stop, refined.agreement,
              agreement, refined.agreement / agreement - 1, dz.max() ?? 0))
        }
      }
      results.append(result)
    }
  }
  return results
}
