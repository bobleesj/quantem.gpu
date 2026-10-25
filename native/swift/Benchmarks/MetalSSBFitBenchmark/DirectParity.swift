import Foundation
import Metal
@_spi(SSBBenchmark) import MetalSSBKernels

/// A float32 or float64 `.npy` array (C order) as Double values and its shape.
func readNPY(_ url: URL) throws -> (values: [Double], shape: [Int]) {
  let data = try Data(contentsOf: url)
  guard data.count > 10, data.prefix(6) == Data([0x93] + Array("NUMPY".utf8)) else {
    throw failure("\(url.path) is not an .npy file")
  }
  let major = data[6]
  let headerLength: Int
  let start: Int
  if major == 1 {
    headerLength = Int(data[8]) | Int(data[9]) << 8
    start = 10
  } else {
    headerLength = Int(data[8]) | Int(data[9]) << 8 | Int(data[10]) << 16 | Int(data[11]) << 24
    start = 12
  }
  let header = String(decoding: data[start..<(start + headerLength)], as: UTF8.self)
  guard !header.contains("'fortran_order': True") else { throw failure("\(url.path) is Fortran ordered") }
  guard let open = header.range(of: "'shape': ("), let close = header[open.upperBound...].firstIndex(of: ")")
  else { throw failure("\(url.path) has no shape") }
  let shape = header[open.upperBound..<close].split(separator: ",").compactMap {
    Int($0.trimmingCharacters(in: .whitespaces))
  }
  let body = data[(start + headerLength)...]
  let count = shape.reduce(1, *)
  let values: [Double]
  if header.contains("'<f4'") {
    values = body.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)).map(Double.init) }
  } else if header.contains("'<f8'") {
    values = body.withUnsafeBytes { Array($0.bindMemory(to: Double.self).prefix(count)) }
  } else {
    throw failure("\(url.path): only little-endian float32 or float64 arrays are read")
  }
  guard values.count == count else { throw failure("\(url.path) is truncated") }
  return (values, shape)
}

/// Native-vs-reference agreement of one image: Pearson correlation, rms difference (wrapped for phases), the
/// reference's std and mean.
func parity(_ native: [Float], _ reference: [Double], wrap: Bool) -> [String: Double] {
  precondition(native.count == reference.count)
  let count = Double(native.count)
  var sumA = 0.0, sumB = 0.0
  for i in native.indices {
    sumA += Double(native[i])
    sumB += reference[i]
  }
  let meanA = sumA / count, meanB = sumB / count
  var ab = 0.0, aa = 0.0, bb = 0.0, dd = 0.0, maximum = 0.0
  for i in native.indices {
    let a = Double(native[i]) - meanA, b = reference[i] - meanB
    ab += a * b
    aa += a * a
    bb += b * b
    var d = Double(native[i]) - reference[i]
    if wrap { d -= 2 * .pi * (d / (2 * .pi)).rounded() }
    dd += d * d
    maximum = max(maximum, abs(d))
  }
  return [
    "correlation": ab / (aa * bb).squareRoot(), "rms": (dd / count).squareRoot(), "max_abs": maximum,
    "reference_std": (bb / count).squareRoot(), "reference_mean": meanB,
  ]
}

func modelPoint(_ record: [String: Any]) throws -> SSBModelPoint {
  guard let c10 = record["C10_nm"] as? Double, let c12 = record["C12_nm"] as? Double,
    let phi12 = record["phi12_deg"] as? Double, let tilt = record["sample_tilt_mrad"] as? [Double],
    let depth = record["depth_spread_nm"] as? Double
  else { throw failure("Incomplete optics record") }
  let terms = (record["higher_order"] as? [[String: Any]] ?? []).compactMap { term -> SSBModelHigherOrder? in
    guard let order = term["order"] as? Int, let symmetry = term["symmetry"] as? Int,
      let magnitude = term["magnitude_A"] as? Double, let angle = term["angle_rad"] as? Double
    else { return nil }
    return SSBModelHigherOrder(order: order, symmetry: symmetry, magnitudeAngstroms: magnitude, angleRadians: angle)
  }
  return SSBModelPoint(
    c10Nanometers: c10, c12Nanometers: c12, phi12Degrees: phi12, sampleTiltMrad: tilt,
    depthSpreadNanometers: depth, higherOrder: terms)
}

/// `--direct-parity LIMITS.json --cuda-dir DIR`: each method (ssb, phase, complex) of the chosen limits' optics at
/// the automatic factor, against CUDA's `limit<L>-<method>-{phase,amplitude}.npy` from the same optics, with stage
/// timings: the first call (layout and kernel specialization), a repeat (prepared system kept) and a damping change.
func directParity(
  sweep sweepURL: URL, limits: [Double], cudaDirectory: URL, output: URL?, phaseOut: URL?, device: MTLDevice,
  calibration: MetalSSBCalibration, geometry: MetalSSBGeometry, counts: MTLBuffer, countType: MetalSSBCountType,
  rotationEstimate: Double, factors: [Int]?
) throws {
  guard
    let sweep = try JSONSerialization.jsonObject(with: Data(contentsOf: sweepURL)) as? [String: Any],
    let rotation = sweep["rotation_deg"] as? Double,
    let records = sweep["limits"] as? [[String: Any]]
  else { throw failure("\(sweepURL.path) is not a limit sweep") }
  let auto = try calibration.ssbUpsamplingFactor()
  let engine = try MetalSSBEngine(device: device, geometry: geometry)
  try engine.prepare(brightfield: counts, countType: countType)
  let model = try engine.jointModel(scanDetectorRotationDegrees: rotationEstimate)
  try model.retarget(scanDetectorRotationDegrees: rotation)
  if let phaseOut { try FileManager.default.createDirectory(at: phaseOut, withIntermediateDirectories: true) }
  var rows: [[String: Any]] = []
  func values(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
    Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
  }
  func timing(_ t: SSBDirectTiming?) -> [String: Any] {
    guard let t else { return [:] }
    return [
      "layout_ms": t.layoutSeconds * 1e3, "prepare_ms": t.prepareSeconds * 1e3,
      "prepare_gpu_ms": t.prepareGPUSeconds * 1e3, "solve_ms": t.solveSeconds * 1e3,
      "wave_ms": t.waveSeconds * 1e3, "prepared_from_cache": t.preparedFromCache,
    ]
  }
  for factor in factors ?? [auto] {
    for limit in limits {
      guard let record = records.first(where: { ($0["limit"] as? Double) == limit }) else {
        throw failure("No record for limit \(limit)")
      }
      let point = try modelPoint(record)
      let tag = "limit\(Int(limit))"
      for method in SSBReconstructionMethod.allCases {
        var row: [String: Any] = ["limit": limit, "method": method.rawValue, "factor": factor]
        let regularization = SSBDirectRegularization()
        var started = Date()
        let result: MetalSSBJointPhase
        do {
          result = try model.reconstruct(point, upsamplingFactor: factor, method: method, regularization: regularization)
        } catch {
          log("\(method.rawValue) failed: \(error); stages \(timing(model.lastDirectTiming))")
          continue
        }
        row["first_ms"] = Date().timeIntervalSince(started) * 1e3
        if method != .ssb { row["first_stages"] = timing(model.lastDirectTiming) }
        started = Date()
        _ = try model.reconstruct(point, upsamplingFactor: factor, method: method, regularization: regularization)
        row["repeat_ms"] = Date().timeIntervalSince(started) * 1e3
        if method != .ssb {
          row["repeat_stages"] = timing(model.lastDirectTiming)
          started = Date()
          _ = try model.reconstruct(
            point, upsamplingFactor: factor, method: method,
            regularization: SSBDirectRegularization(phaseDamping: 0.01, amplitudeDamping: 0.01))
          row["damping_change_ms"] = Date().timeIntervalSince(started) * 1e3
        }
        row["rows"] = result.rows
        let count = result.rows * result.columns
        let phase = values(result.phase, count)
        if let phaseOut {
          try phase.withUnsafeBytes { Data($0) }.write(
            to: phaseOut.appendingPathComponent("\(tag)-\(method.rawValue)-\(factor)x-phase.f32"))
        }
        let phaseURL = cudaDirectory.appendingPathComponent("\(tag)-\(method.rawValue)-phase.npy")
        if FileManager.default.fileExists(atPath: phaseURL.path) {
          let reference = try readNPY(phaseURL)
          if reference.shape == [result.rows, result.columns] {
            row["phase"] = parity(phase, reference.values, wrap: true)
          } else {
            row["phase_reference_shape"] = reference.shape
          }
        }
        if let amplitude = result.amplitude {
          let native = values(amplitude, count)
          if let phaseOut {
            try native.withUnsafeBytes { Data($0) }.write(
              to: phaseOut.appendingPathComponent("\(tag)-\(method.rawValue)-\(factor)x-amplitude.f32"))
          }
          let url = cudaDirectory.appendingPathComponent("\(tag)-\(method.rawValue)-amplitude.npy")
          if FileManager.default.fileExists(atPath: url.path) {
            let reference = try readNPY(url)
            if reference.shape == [result.rows, result.columns] {
              var p = parity(native, reference.values, wrap: false)
              p["relative_rms"] = p["rms"]! / p["reference_mean"]!
              row["amplitude"] = p
            }
          }
        }
        log(String(describing: row))
        rows.append(row)
      }
    }
  }
  let summary: [String: Any] = [
    "device": device.name, "sweep": sweepURL.path, "cuda": cudaDirectory.path, "rotation_deg": rotation,
    "auto_factor": auto, "rows": rows,
  ]
  let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
  if let output { try data.write(to: output) } else { print(String(decoding: data, as: UTF8.self)) }
}
