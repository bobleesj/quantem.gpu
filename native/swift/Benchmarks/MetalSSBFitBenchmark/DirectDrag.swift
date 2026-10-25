import Foundation
import Metal
@_spi(SSBBenchmark) import MetalSSBKernels

/// `--direct-drag CASE_DIR [--optics NAME] [--factors 3] [--steps 30]`: the optics-change cost of the regularized
/// direct methods at a fixture's optics, as a C10 slider drag sees it. Each step moves C10 by 0.05 nm, so every
/// step prepares the normal equations again; the stage times of every step are reported.
func directDrag(
  directory: URL, output: URL?, device: MTLDevice, calibration: MetalSSBCalibration,
  geometry: MetalSSBGeometry,
  counts: MTLBuffer, countType: MetalSSBCountType, rotationEstimate: Double, only: String?,
  factors: [Int]?,
  steps: Int, windowControls: [SSBDragControl]
) throws {
  guard
    let optics = try JSONSerialization.jsonObject(
      with: Data(contentsOf: directory.appendingPathComponent("optics.json")))
      as? [String: Any],
    let sets = optics["optics_sets"] as? [[String: Any]]
  else { throw failure("\(directory.path) is not a fixture (optics.json)") }
  guard let set = sets.first(where: { only == nil || ($0["name"] as? String) == only }),
    let name = set["name"] as? String, var record = set["aberrations"] as? [String: Any],
    let rotation = set["rotation_deg"] as? Double
  else { throw failure("No optics set \(only ?? "")") }
  record["higher_order"] = set["higher_order"]
  record["sample_tilt_mrad"] = set["sample_tilt_mrad"]
  record["depth_spread_nm"] = set["depth_spread_nm"]
  let point = try modelPoint(record)
  let auto = try calibration.ssbUpsamplingFactor()
  let engine = try MetalSSBEngine(device: device, geometry: geometry)
  try engine.prepare(brightfield: counts, countType: countType)
  let model = try engine.jointModel(scanDetectorRotationDegrees: rotationEstimate)
  try model.retarget(scanDetectorRotationDegrees: rotation)
  func stages(_ t: SSBDirectTiming?) -> [String: Double] {
    guard let t else { return [:] }
    return [
      "layout_ms": t.layoutSeconds * 1e3, "prepare_ms": t.prepareSeconds * 1e3,
      "prepare_gpu_ms": t.prepareGPUSeconds * 1e3, "solve_ms": t.solveSeconds * 1e3,
      "wave_ms": t.waveSeconds * 1e3,
    ]
  }
  func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted.isEmpty ? .nan : sorted[sorted.count / 2]
  }
  var rows: [[String: Any]] = []
  for factor in factors ?? [auto] {
    for method in [SSBReconstructionMethod.phase, .complex] {
      var started = Date()
      do {
        _ = try model.reconstruct(point, upsamplingFactor: factor, method: method)
      } catch SSBDirectError.diverged {}
      let first = Date().timeIntervalSince(started) * 1e3
      var walls: [Double] = []
      var prepares: [Double] = []
      var gpus: [Double] = []
      var solves: [Double] = []
      var waves: [Double] = []
      for step in 1...steps {
        var moved = point
        moved.c10Angstroms += 0.5 * Double(step)
        started = Date()
        do {
          _ = try model.reconstruct(moved, upsamplingFactor: factor, method: method)
        } catch SSBDirectError.diverged {}
        walls.append(Date().timeIntervalSince(started) * 1e3)
        let t = stages(model.lastDirectTiming)
        prepares.append(t["prepare_ms"] ?? .nan)
        gpus.append(t["prepare_gpu_ms"] ?? .nan)
        solves.append(t["solve_ms"] ?? .nan)
        waves.append(t["wave_ms"] ?? .nan)
      }
      let row: [String: Any] = [
        "optics": name, "method": method.rawValue, "factor": factor, "first_ms": first,
        "steps": steps,
        "median_wall_ms": median(walls), "median_prepare_ms": median(prepares),
        "median_prepare_gpu_ms": median(gpus), "median_solve_ms": median(solves),
        "median_wave_ms": median(waves),
        "max_wall_ms": walls.max() ?? .nan,
      ]
      log(
        String(
          format:
            "%@ %@ %dx: first %.1f ms; per C10 step wall %.1f (prepare %.1f, GPU %.1f, solve %.1f, wave %.1f) ms",
          name, method.rawValue, factor, first, median(walls), median(prepares), median(gpus),
          median(solves),
          median(waves)))
      rows.append(row)
    }
  }
  // Drag windows: build along each control, then preview frames across the window against the exact solve.
  for factor in factors ?? [auto] {
    for method in [SSBReconstructionMethod.complex, .phase] {
      for control in windowControls {
        model.releaseDragWindows()
        var started = Date()
        var state = try model.buildDragWindow(
          along: control, at: point, upsamplingFactor: factor, method: method, nodes: 1000)
        let build = Date().timeIntervalSince(started) * 1e3
        let center = control.value(of: point)
        let span = state.range.upperBound.isFinite ? state.range.upperBound - center : 1
        var frames: [Double] = []
        var stages: [Double] = []
        var rms: [Double] = []
        var maxima: [Double] = []
        var amplitudeRMS: [Double] = []
        for step in 0..<steps {
          // values across 90 % of the window, not on the nodes
          let fraction = -0.9 + 1.8 * (Double(step) + 0.37) / Double(steps)
          let moved = control.setting(center + fraction * span, in: point)
          started = Date()
          guard
            let preview = try model.dragPreview(
              moved, along: control, upsamplingFactor: factor, method: method)
          else { throw failure("no preview at \(fraction)") }
          frames.append(Date().timeIntervalSince(started) * 1e3)
          stages.append((model.lastDirectTiming?.prepareSeconds ?? .nan) * 1e3)
          if step % max(1, steps / 4) == 0 {
            let exact = try model.reconstruct(moved, upsamplingFactor: factor, method: method)
            let count = exact.rows * exact.columns
            let a = Array(
              UnsafeBufferPointer(
                start: preview.phase.contents().assumingMemoryBound(to: Float.self), count: count))
            let b = Array(
              UnsafeBufferPointer(
                start: exact.phase.contents().assumingMemoryBound(to: Float.self), count: count))
            let agreement = parity(a, b.map(Double.init), wrap: true)
            rms.append(agreement["rms"]! / agreement["reference_std"]!)
            maxima.append(agreement["max_abs"]!)
            if let pa = preview.amplitude, let ea = exact.amplitude {
              let x = Array(
                UnsafeBufferPointer(
                  start: pa.contents().assumingMemoryBound(to: Float.self), count: count))
              let y = Array(
                UnsafeBufferPointer(
                  start: ea.contents().assumingMemoryBound(to: Float.self), count: count))
              let p = parity(x, y.map(Double.init), wrap: false)
              amplitudeRMS.append(p["rms"]! / p["reference_mean"]!)
            }
          }
        }
        state =
          model.dragWindowState(along: control, at: point, upsamplingFactor: factor, method: method)
          ?? state
        let row: [String: Any] = [
          "method": method.rawValue, "factor": factor, "control": "\(control)",
          "nodes": state.total,
          "build_ms": build, "range": [state.range.lowerBound, state.range.upperBound],
          "median_frame_ms": median(frames), "max_frame_ms": frames.max() ?? .nan,
          "median_evaluate_ms": median(stages), "max_rms_over_std": rms.max() ?? .nan,
          "max_abs_rad": maxima.max() ?? .nan, "max_amplitude_rel_rms": amplitudeRMS.max() ?? .nan,
        ]
        log(
          String(
            format:
              "window %@ %@ %dx: %d nodes built in %.0f ms (range %.3f..%.3f); preview median %.1f ms (evaluate %.1f), max %.1f ms; vs exact rms/std %.2e, max %.2e rad, amp %.2e",
            method.rawValue, "\(control)", factor, state.total, build, state.range.lowerBound,
            state.range.upperBound,
            median(frames), median(stages), frames.max() ?? .nan, rms.max() ?? .nan,
            maxima.max() ?? .nan,
            amplitudeRMS.max() ?? .nan))
        rows.append(row)
      }
    }
  }
  // JSON has no NaN: missing numbers are null
  func clean(_ value: Any) -> Any {
    if let number = value as? Double { return number.isFinite ? number : NSNull() }
    if let row = value as? [String: Any] { return row.mapValues(clean) }
    if let list = value as? [Any] { return list.map(clean) }
    return value
  }
  let summary: [String: Any] = [
    "device": device.name, "fixture": directory.path, "rows": rows.map(clean),
  ]
  let data = try JSONSerialization.data(
    withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
  if let output { try data.write(to: output) } else { print(String(decoding: data, as: UTF8.self)) }
}
