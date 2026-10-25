import Foundation
import Metal
import MetalSSBKernels

// Agreement-objective micro-benchmark (`--objective-bench RUN.json`): the joint model at a fit's kept rotation, then
// fixed, deterministic candidate batches of the fifth-order objective (`fifthOrderAgreements`) and the low-order
// objective (`agreements`). Prints and records every agreement value with median GPU time (command-buffer GPU
// start to end) and wall time per candidate, so kernels are compared at identical parameters.
//
// Candidates come from RUN.json (a `metal-ssb-fit-benchmark --output` result): the kept branch's low-order fit
// (`initial`), the fifth-order start built from it, the final fifth-order coordinates, a 57-point axis stencil
// around the final point (the Newton diagonal design) and 64 seeded random points within +-0.1 of it. Then the
// exact-gradient search's value and gradient calls at the start and final points (`search_kernels`).

/// SSBFifthOrderCoordinates.pack: dimensionless coordinates to the 28 Cartesian float32 coefficients.
private func pack(_ z: [Double], wavelength: Double, semiangle: Double) -> [Float] {
  let orders: [(Int, Int)] = [
    (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4),
    (4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4), (5, 6),
  ]
  var packed = [Float](repeating: 0, count: 28)
  packed[0] = Float(z[0] * 50)
  packed[2] = Float(z[1] * 50)
  packed[3] = Float(z[2] * 50)
  var j = 6
  for (i, (n, m)) in orders.enumerated() where n >= 2 {
    let scale = wavelength / (2 * Double.pi * pow(semiangle, Double(n + 1)))
    packed[2 * i] = Float(z[j] * scale)
    j += 1
    if m != 0 {
      packed[2 * i + 1] = Float(z[j] * scale)
      j += 1
    }
  }
  return packed
}

/// The low-order part of SSBFifthOrderCoordinates.point (the higher orders travel in the packed coefficients).
private func lowOrder(_ z: [Double]) -> SSBModelPoint {
  SSBModelPoint(
    c10Angstroms: z[0] * 100, c12Angstroms: hypot(z[1], z[2]) * 100,
    phi12Radians: atan2(z[2], z[1]) / 2,
    sampleTiltMrad: [z[3] * 5, z[4] * 5], depthSpreadAngstroms: z[5] * 100)
}

func objectiveBenchmark(
  run: URL, device: MTLDevice, geometry: MetalSSBGeometry, counts: MTLBuffer,
  countType: MetalSSBCountType, repeats: Int, searchKernelsOnly: Bool = false, output: URL?
) throws {
  guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: run)) as? [String: Any],
    let result = (json["results"] as? [[String: Any]])?.first,
    let fit = result["fit"] as? [String: Any], let initial = result["initial"] as? [String: Any],
    let rotation = fit["rotation_deg"] as? Double,
    let zFinal = (fit["refinement"] as? [String: Any])?["coordinates"] as? [Double],
    zFinal.count == 28,
    let c10 = initial["C10_nm"] as? Double, let c12 = initial["C12_nm"] as? Double,
    let phi12 = initial["phi12_deg"] as? Double,
    let tilt = initial["sample_tilt_mrad"] as? [Double],
    let depth = initial["depth_spread_nm"] as? Double
  else { throw failure("\(run.path) lacks a kept fifth-order fit with its low-order start") }
  let prepared = Date()
  let engine = try MetalSSBEngine(device: device, geometry: geometry)
  try engine.prepare(brightfield: counts, countType: countType)
  let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)
  log(
    String(
      format: "objective bench: %@, rotation %.6f deg, %d band frequencies, prepared in %.2f s",
      device.name, rotation, model.bandCount, Date().timeIntervalSince(prepared)))

  let start = SSBModelPoint(
    c10Angstroms: c10 * 10, c12Angstroms: c12 * 10, phi12Radians: phi12 * Double.pi / 180,
    sampleTiltMrad: tilt, depthSpreadAngstroms: depth * 10)
  let zStart =
    [
      start.c10Angstroms / 100, start.c12Angstroms / 100 * cos(2 * start.phi12Radians),
      start.c12Angstroms / 100 * sin(2 * start.phi12Radians), tilt[0] / 5, tilt[1] / 5,
      start.depthSpreadAngstroms / 100,
    ] + Array(repeating: 0.0, count: 22)
  let bounds = SSBFifthOrderSearch.bounds
  func clip(_ z: [Double]) -> [Double] {
    zip(z, bounds).map { min($1.upperBound, max($1.lowerBound, $0)) }
  }
  var state: UInt64 = 0x9E37_79B9_7F4A_7C15
  func uniform() -> Double {
    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return Double(state >> 11) / Double(1 << 53)
  }
  let random = (0..<64).map { _ in clip(zFinal.map { $0 + (uniform() - 0.5) * 0.2 }) }
  let steps = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05] + Array(repeating: 0.02, count: 22)
  var stencil = [zFinal]
  for i in 0..<28 {
    for sign in [1.0, -1.0] {
      var z = zFinal
      z[i] += sign * steps[i]
      stencil.append(clip(z))
    }
  }
  let lowRandom = (0..<64).map { _ -> SSBModelPoint in
    SSBModelPoint(
      c10Angstroms: start.c10Angstroms + (uniform() - 0.5) * 10,
      c12Angstroms: abs(start.c12Angstroms + (uniform() - 0.5) * 4),
      phi12Radians: start.phi12Radians + (uniform() - 0.5) * 0.4,
      sampleTiltMrad: tilt.map { $0 + (uniform() - 0.5) * 1.0 },
      depthSpreadAngstroms: max(0, start.depthSpreadAngstroms + (uniform() - 0.5) * 40))
  }
  let wl = model.wavelengthAngstroms
  let semi = model.semiangleRadians
  let fifthSets: [(String, [[Double]])] = [
    ("fifth final", [zFinal]), ("fifth start+final", [zStart, zFinal]),
    ("fifth random4", Array(random.prefix(4))), ("fifth stencil57", stencil),
    ("fifth random64", random),
  ]
  let lowSets: [(String, [SSBModelPoint])] = [
    ("low initial", [start]), ("low random8", Array(lowRandom.prefix(8))),
    ("low random64", lowRandom),
  ]
  var sets: [[String: Any]] = []
  func measure(_ name: String, count: Int, _ evaluate: () throws -> [Double]) throws {
    _ = try evaluate()  // pipelines, workspaces
    var gpu: [Double] = []
    var wall: [Double] = []
    var values: [Double] = []
    for _ in 0..<max(1, repeats) {
      let started = Date()
      values = try evaluate()
      wall.append(Date().timeIntervalSince(started))
      gpu.append(model.lastAgreementGPUSeconds)
    }
    let gpuMedian = gpu.sorted()[gpu.count / 2] / Double(count) * 1e3
    let wallMedian = wall.sorted()[wall.count / 2] / Double(count) * 1e3
    log(
      String(
        format:
          "  %-18@ %3d candidates  GPU %8.3f ms/candidate  wall %8.3f ms/candidate  first %.9e",
        name as NSString, count, gpuMedian, wallMedian, values[0]))
    sets.append([
      "name": name, "candidates": count, "gpu_ms_per_candidate": gpuMedian,
      "wall_ms_per_candidate": wallMedian, "gpu_s": gpu, "wall_s": wall, "agreements": values,
    ])
  }
  for (name, zs) in searchKernelsOnly ? [] : fifthSets {
    let points = zs.map(lowOrder)
    let packed = zs.map { pack($0, wavelength: wl, semiangle: semi) }
    try measure(name, count: zs.count) {
      try model.fifthOrderAgreements(points, packedCoefficients: packed)
    }
  }
  for (name, points) in searchKernelsOnly ? [] : lowSets {
    try measure(name, count: points.count) { try model.agreements(points) }
  }
  // The exact low-order value, gradient and Hessian (the primary Newton's centers) in Newton z.
  func newtonZ(_ p: SSBModelPoint) -> [Double] {
    [
      p.c10Angstroms, p.c12Angstroms * cos(2 * p.phi12Radians),
      p.c12Angstroms * sin(2 * p.phi12Radians),
      p.sampleTiltMrad[0], p.sampleTiltMrad[1], p.depthSpreadAngstroms,
    ]
  }
  // The fused pass (Newton centers whose derivatives it uses) and the value pass followed by the derivative pass.
  let derivativeSets: [(String, [[Double]], Bool)] = [
    ("derivatives initial", [newtonZ(start)], false),
    ("derivatives random3", lowRandom.prefix(3).map(newtonZ), false),
    ("two-stage initial", [newtonZ(start)], true),
    ("two-stage random3", lowRandom.prefix(3).map(newtonZ), true),
  ]
  for (name, rows, twoStage) in searchKernelsOnly ? [] : derivativeSets {
    _ = try model.lowOrderAgreementDerivatives(rows, twoStage: twoStage)
    var gpu: [Double] = []
    var valueGPU: [Double] = []
    var wall: [Double] = []
    var results: [SSBAgreementSearch.Derivatives] = []
    for _ in 0..<max(1, repeats) {
      let started = Date()
      results = try model.lowOrderAgreementDerivatives(rows, twoStage: twoStage)
      wall.append(Date().timeIntervalSince(started))
      gpu.append(model.lastLowOrderGPUSeconds)
      valueGPU.append(model.lastLowOrderValueGPUSeconds)
    }
    let gpuMedian = gpu.sorted()[gpu.count / 2] / Double(rows.count) * 1e3
    let valueMedian = valueGPU.sorted()[valueGPU.count / 2] / Double(rows.count) * 1e3
    let wallMedian = wall.sorted()[wall.count / 2] / Double(rows.count) * 1e3
    log(
      String(
        format:
          "  %-20@ %3d centers  GPU %8.3f ms/center (value stage %.3f)  wall %8.3f ms/center  first %.9e",
        name as NSString, rows.count, gpuMedian, valueMedian, wallMedian, results[0].value))
    sets.append([
      "name": name, "candidates": rows.count, "gpu_ms_per_candidate": gpuMedian,
      "value_stage_gpu_ms_per_candidate": valueMedian,
      "wall_ms_per_candidate": wallMedian, "gpu_s": gpu, "wall_s": wall, "rows": rows,
      "agreements": results.map(\.value), "gradients": results.map(\.gradient),
      "hessians": results.map(\.hessian),
    ])
  }
  // The exact-gradient search's calls (value, gradient after it, both; all and every 8th BF pixel; the
  // Hessian batch on every 64th) at its start and at the final point.
  var searchKernels: [[String: Any]] = []
  for (label, z) in [("start", zStart), ("final", zFinal)] {
    for entry in try SSBModelFit.fifthOrderKernelTimings(
      model: model, z: z, repeats: max(3, repeats))
    {
      var entry = entry
      entry["point"] = label
      log(
        String(
          format: "  %-6@ %-28@ %3d candidates  GPU %8.3f ms  wall %8.3f ms  first %.9e",
          label as NSString, entry["name"] as! NSString, entry["candidates"] as! Int,
          entry["gpu_ms"] as! Double, entry["wall_ms"] as! Double,
          (entry["agreements64"] as! [Double])[0]))
      searchKernels.append(entry)
    }
  }
  let summary: [String: Any] = [
    "schema": "quantem-gpu-ssb-objective-benchmark/v1", "device": device.name, "run": run.path,
    "rotation_deg": rotation, "band_count": model.bandCount, "repeats": repeats,
    "timing_boundary":
      "gpu: command-buffer gpuStartTime..gpuEndTime of one agreements call; wall: the call",
    "sets": sets, "search_kernels": searchKernels,
  ]
  let data = try JSONSerialization.data(
    withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
  if let output { try data.write(to: output) } else { FileHandle.standardOutput.write(data) }
}
