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

/// The parity gates of the native port: phase correlation >= 0.99999 and rms <= 1e-4 rad and <= 1 % of the reference
/// std; amplitude rms <= 1e-4 of its mean.
func gates(phase: [String: Double]?, amplitude: [String: Double]?) -> [String: Bool] {
  var result: [String: Bool] = [:]
  if let phase {
    result["phase_correlation"] = phase["correlation"]! >= 0.99999
    result["phase_rms"] = phase["rms"]! <= 1e-4 && phase["rms"]! <= 0.01 * phase["reference_std"]!
  }
  if let amplitude { result["amplitude_relative_rms"] = amplitude["rms"]! <= 1e-4 * amplitude["reference_mean"]! }
  return result
}

/// `--fixture-parity CASE_DIR [--oracle-dir DIR]`: every reconstruction listed in the fixture's `references.json`
/// (optics sets of `optics.json`, methods, 1x and Auto, regularization settings) natively, against the CUDA arrays and,
/// where present, the float64 oracle's arrays of the same names, with the first-call and repeat wall times.
func fixtureParity(
  directory: URL, oracleDirectory: URL?, output: URL?, phaseOut: URL?, device: MTLDevice,
  calibration: MetalSSBCalibration, geometry: MetalSSBGeometry, counts: MTLBuffer, countType: MetalSSBCountType,
  rotationEstimate: Double, only: String?
) throws {
  guard
    let optics = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("optics.json")))
      as? [String: Any],
    let sets = optics["optics_sets"] as? [[String: Any]],
    let manifest = try JSONSerialization.jsonObject(
      with: Data(contentsOf: directory.appendingPathComponent("references.json"))) as? [String: Any],
    let references = manifest["references"] as? [[String: Any]]
  else { throw failure("\(directory.path) is not a fixture (optics.json, references.json)") }
  let auto = try calibration.ssbUpsamplingFactor()
  let engine = try MetalSSBEngine(device: device, geometry: geometry)
  try engine.prepare(brightfield: counts, countType: countType)
  let model = try engine.jointModel(scanDetectorRotationDegrees: rotationEstimate)
  func values(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
    Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
  }
  var rows: [[String: Any]] = []
  var failures = 0
  for set in sets {
    guard let name = set["name"] as? String, var record = set["aberrations"] as? [String: Any],
      let rotation = set["rotation_deg"] as? Double
    else { throw failure("Incomplete optics set") }
    if let only, only != name { continue }
    record["higher_order"] = set["higher_order"]
    record["sample_tilt_mrad"] = set["sample_tilt_mrad"]
    record["depth_spread_nm"] = set["depth_spread_nm"]
    let point = try modelPoint(record)
    try model.retarget(scanDetectorRotationDegrees: rotation)
    for reference in references where (reference["optics"] as? String) == name {
      guard let file = reference["file"] as? String, let methodName = reference["method"] as? String,
        let method = SSBReconstructionMethod(rawValue: methodName), let factor = reference["upsampling"] as? Int
      else { throw failure("Incomplete reference entry") }
      let requested = reference["requested"] as? [String: Any] ?? [:]
      let regularization = SSBDirectRegularization(
        phaseDamping: requested["phase_damping"] as? Double ?? 0.03,
        amplitudeDamping: requested["amplitude_damping"] as? Double ?? 0.03,
        smoothnessInverseAngstroms: requested["smoothness_inv_A"] as? Double ?? 1,
        detectorWeighting: (requested["detector_weighting"] as? String).flatMap(SSBDetectorWeighting.init) ?? .uniform)
      var row: [String: Any] = [
        "optics": name, "file": file, "method": methodName, "factor": factor,
        "setting": reference["setting"] as? String ?? "", "auto": (reference["factor"] as? String) == "auto",
      ]
      if (reference["factor"] as? String) == "auto" && factor != auto {
        row["error"] = "CUDA auto factor \(factor) differs from the native \(auto)"
      }
      var started = Date()
      let result: MetalSSBJointPhase
      do {
        result = try model.reconstruct(
          point, upsamplingFactor: factor, method: method, regularization: regularization)
      } catch {
        row["error"] = "\(error)"
        failures += 1
        rows.append(row)
        log(String(describing: row))
        continue
      }
      row["first_ms"] = Date().timeIntervalSince(started) * 1e3
      if method != .ssb, let t = model.lastDirectTiming {
        row["first_stages"] = [
          "layout_ms": t.layoutSeconds * 1e3, "prepare_ms": t.prepareSeconds * 1e3,
          "prepare_gpu_ms": t.prepareGPUSeconds * 1e3, "solve_ms": t.solveSeconds * 1e3,
          "wave_ms": t.waveSeconds * 1e3, "prepared_from_cache": t.preparedFromCache,
        ] as [String: Any]
      }
      started = Date()
      _ = try model.reconstruct(point, upsamplingFactor: factor, method: method, regularization: regularization)
      row["repeat_ms"] = Date().timeIntervalSince(started) * 1e3
      let count = result.rows * result.columns
      let phase = values(result.phase, count)
      let amplitude = result.amplitude.map { values($0, count) }
      let base = (file as NSString).lastPathComponent.replacingOccurrences(of: "-phase.npy", with: "")
      if let phaseOut {
        let folder = phaseOut.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try phase.withUnsafeBytes { Data($0) }.write(to: folder.appendingPathComponent(base + "-phase.f32"))
        if let amplitude {
          try amplitude.withUnsafeBytes { Data($0) }.write(to: folder.appendingPathComponent(base + "-amplitude.f32"))
        }
      }
      let cuda = try readNPY(directory.appendingPathComponent(file))
      guard cuda.shape == [result.rows, result.columns] else {
        row["error"] = "shape \(cuda.shape) vs native \([result.rows, result.columns])"
        failures += 1
        rows.append(row)
        continue
      }
      let phaseParity = parity(phase, cuda.values, wrap: true)
      row["phase"] = phaseParity
      var amplitudeParity: [String: Double]?
      if let amplitude, let amplitudeFile = reference["amplitude_file"] as? String {
        var p = parity(amplitude, try readNPY(directory.appendingPathComponent(amplitudeFile)).values, wrap: false)
        p["relative_rms"] = p["rms"]! / p["reference_mean"]!
        amplitudeParity = p
        row["amplitude"] = p
      } else if method == .complex {
        row["error"] = "complex result without amplitude"
      }
      let gate = gates(phase: phaseParity, amplitude: amplitudeParity)
      row["gates"] = gate
      if gate.values.contains(false) && method != .ssb { failures += 1 }
      if let oracleDirectory {
        let oracle = oracleDirectory.appendingPathComponent(name).appendingPathComponent(
          (file as NSString).lastPathComponent)
        if FileManager.default.fileExists(atPath: oracle.path) {
          let reference = try readNPY(oracle).values
          var nativeOracle = parity(phase, reference, wrap: true)
          let cudaOracle = parity(cuda.values.map(Float.init), reference, wrap: true)
          nativeOracle["cuda_rms"] = cudaOracle["rms"]
          nativeOracle["ratio_to_cuda"] = nativeOracle["rms"]! / max(cudaOracle["rms"]!, 1e-30)
          row["phase_vs_oracle"] = nativeOracle
          if amplitude != nil, let amplitudeFile = reference_amplitude(oracle) {
            let ref = try readNPY(amplitudeFile).values
            var p = parity(amplitude!, ref, wrap: false)
            let c = parity(
              try readNPY(directory.appendingPathComponent(row["file"] as! String)
                .deletingLastPathComponent().appendingPathComponent(amplitudeFile.lastPathComponent)).values
                .map(Float.init), ref, wrap: false)
            p["cuda_rms"] = c["rms"]
            p["ratio_to_cuda"] = p["rms"]! / max(c["rms"]!, 1e-30)
            row["amplitude_vs_oracle"] = p
          }
        }
      }
      log(
        "\(name) \(methodName) \(factor)x \(row["setting"]!): corr \(phaseParity["correlation"]!) rms \(phaseParity["rms"]!)"
          + (amplitudeParity.map { " amp \($0["relative_rms"]!)" } ?? "")
          + " first \(String(format: "%.1f", row["first_ms"] as! Double)) ms")
      rows.append(row)
    }
  }
  let summary: [String: Any] = [
    "device": device.name, "fixture": directory.path, "auto_factor": auto, "rows": rows, "failures": failures,
  ]
  let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
  if let output { try data.write(to: output) } else { print(String(decoding: data, as: UTF8.self)) }
  log("\(rows.count) reconstructions, \(failures) failing a gate")
}

/// The oracle's amplitude array beside its phase array, if present.
func reference_amplitude(_ phase: URL) -> URL? {
  let url = phase.deletingLastPathComponent().appendingPathComponent(
    phase.lastPathComponent.replacingOccurrences(of: "-phase.npy", with: "-amplitude.npy"))
  return FileManager.default.fileExists(atPath: url.path) ? url : nil
}
