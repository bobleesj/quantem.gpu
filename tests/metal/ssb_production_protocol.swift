// Production-config SSB fit measurement harness (evidence only, no gate).
//
// Two questions are answered here with measurement instead of argument.
//
// 1. `ab`: why two harnesses that both report "8937 BF" differ by ~3.4x in
//    wall time for the same 200-trial + Nelder-Mead protocol. Both are built
//    here in one process, from one exported scan, on one GPU lock:
//
//      parity : case.json `metal_geometry` (documented detector sampling,
//               antialiased aperture, 2464 nonzero terms after exact
//               zero-aperture pruning) + dense uint16 source + the historical
//               `SSBOptimizer.run(start:evaluate:)` call shape.
//      app    : `MetalSSBCalibration` + `matchApertureToBrightfieldDisk()`
//               (hard-edge disk, all 8937 terms active) + the same dense
//               uint16 source + `engine.optimize(start:rotationDegrees:)`,
//               which is what the app calls.
//
//    The exported columns are the same 8937 detector pixels in the same order
//    for both, so the executed-term count is the only per-eval work that
//    changes; everything else (scan content, pixel order, cache layout,
//    source encoding, timer) is held fixed.
//
// 2. `protocol`: on the app configuration, what each block of TPE trials buys
//    ({25, 50, 100, 200} trials + Nelder-Mead, plus an NM-only arm warm
//    started from the 200-trial optimum), forward and reverse.
//
// The harness never changes the objective, the evaluator, the summation order
// or any search arithmetic; it only selects the geometry/source/call shape
// that is being measured.
//
// Usage: ssb-production-protocol <case-dir> <out-dir> <ab|protocol>

import CryptoKit
import Foundation
import Metal
import MetalSSBKernels

private func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data("ssb-production-protocol: \(message)\n".utf8))
  exit(1)
}

private func loadAverage() -> Double {
  var loads = [Double](repeating: 0, count: 3)
  getloadavg(&loads, 3)
  return loads[0]
}

private func sha256Hex(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private struct CaseFile: Decodable {
  let scan_shape: [Int]
  let detector_shape: [Int]
  let bf_count: Int
  let active_bf_count: Int
  let dc_value_exact: Double
  let counts_sum: Int64
  let semiangle_mrad: Double
  let voltage_kV: Double
  let scan_sampling_A: [Double]
  let rotation_angle_deg: Double
  let det_sampling_mrad: Double
  let bf_center: [Double]
  let bf_radius_px: Double
  let detector_dead_pixels: [[Int]]
  let brightfield_kx_unrotated: [Float]
  let brightfield_ky_unrotated: [Float]
  let metal_geometry: MetalSSBGeometry
}

private struct MicroReport: Encodable {
  let configuration: String
  let executedBrightfieldCount: Int
  let evaluations: Int
  let medianSeconds: Double
  let minSeconds: Double
  let maxSeconds: Double
  let loadAverageBefore: Double
  let loadAverageAfter: Double
}

private struct FitReport: Encodable {
  let configuration: String
  let pass: String
  let globalTrials: Int
  let executedBrightfieldCount: Int
  let evaluations: Int
  let refinementEvaluations: Int
  let lossSeconds: Double
  let wallSeconds: Double
  let bestC10Nanometers: Double
  let bestC12Nanometers: Double
  let bestPhi12Radians: Double
  let bestLoss: Double
  let loadAverageBefore: Double
  let loadAverageAfter: Double
  let trajectorySha256: String
}

private struct AppGeometryReport: Encodable {
  let detectorStepRowMrad: Double
  let semiangleMrad: Double
  let selectedPixels: Int
  let minimumApertureWeight: Double
  let dcValueReal: Double
  let executedBrightfieldCount: Int
  let unrotatedKxMaxRelativeDeviation: Double
  let unrotatedScaleFactor: Double
}

private struct ABReport: Encodable {
  let device: String
  let recommendedMaxWorkingSetBytes: UInt64
  let currentAllocatedBytesBefore: Int
  let caseJsonSha256: String
  let bfCount: Int
  let activeBfCount: Int
  let micro: [MicroReport]
  let fits: [FitReport]
  let appGeometry: AppGeometryReport
}

private struct TrialRecord: Encodable {
  let index: Int
  let stage: String
  let c10Nanometers: Double
  let c12Nanometers: Double
  let phi12Radians: Double
  let loss: Double
}

private struct ArmReport: Encodable {
  let arm: String
  let pass: String
  let globalTrials: Int
  let evaluations: Int
  let refinementEvaluations: Int
  let wallSeconds: Double
  let bestC10Nanometers: Double
  let bestC12Nanometers: Double
  let bestPhi12Radians: Double
  let bestLoss: Double
  let loadAverageBefore: Double
  let loadAverageAfter: Double
  let trajectorySha256: String
}

private struct ProtocolReport: Encodable {
  let device: String
  let caseJsonSha256: String
  let configuration: String
  let startPoint: [Double]
  let warmStartPoint: [Double]
  let referenceArm: String
  let loadAverageAtStart: Double
  let loadAverageAtEnd: Double
  let arms: [ArmReport]
  let trajectories: [String: [TrialRecord]]
}

private func trajectoryDigest(_ trials: [SSBOptimizationTrial]) -> String {
  var text = ""
  for trial in trials {
    text += String(
      format: "%@|%.17g|%.17g|%.17g|%.17g\n", trial.stage, trial.point.c10Nanometers,
      trial.point.c12Nanometers, trial.point.phi12Radians, trial.loss)
  }
  return sha256Hex(Data(text.utf8))
}

private func record(_ trials: [SSBOptimizationTrial]) -> [TrialRecord] {
  trials.enumerated().map { index, trial in
    TrialRecord(
      index: index, stage: trial.stage,
      c10Nanometers: trial.point.c10Nanometers,
      c12Nanometers: trial.point.c12Nanometers,
      phi12Radians: trial.point.phi12Radians,
      loss: trial.loss)
  }
}

private func microBenchmark(
  configuration: String, engine: MetalSSBEngine, aberrations: MetalSSBAberrations,
  rotationDegrees: Float?, evaluations: Int
) throws -> MicroReport {
  var samples: [Double] = []
  samples.reserveCapacity(evaluations)
  let loadBefore = loadAverage()
  for _ in 0..<evaluations {
    let start = Date()
    _ = try engine.phaseVariance(aberrations: aberrations, rotationDegrees: rotationDegrees)
    samples.append(Date().timeIntervalSince(start))
  }
  let loadAfter = loadAverage()
  let sorted = samples.sorted()
  let report = MicroReport(
    configuration: configuration,
    executedBrightfieldCount: engine.executedBrightfieldCount,
    evaluations: evaluations,
    medianSeconds: sorted[sorted.count / 2],
    minSeconds: sorted.first ?? 0,
    maxSeconds: sorted.last ?? 0,
    loadAverageBefore: loadBefore,
    loadAverageAfter: loadAfter)
  print(
    String(
      format: "micro %@ bf %d evals %d median %.6f s min %.6f max %.6f load %.2f->%.2f",
      configuration, report.executedBrightfieldCount, evaluations, report.medianSeconds,
      report.minSeconds, report.maxSeconds, loadBefore, loadAfter))
  fflush(stdout)
  return report
}

private func loadDenseSource(
  device: MTLDevice, caseDirectory: URL, payload: CaseFile
) throws -> MTLBuffer {
  let side = payload.scan_shape[0]
  let counts = try Data(
    contentsOf: caseDirectory.appendingPathComponent("source/bf_columns.u16"),
    options: .mappedIfSafe)
  guard counts.count == payload.bf_count * side * side * MemoryLayout<UInt16>.size else {
    fail("bf_columns.u16 has an unexpected size")
  }
  return counts.withUnsafeBytes {
    device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
  }
}

@main enum SSBProductionProtocol {
  static func main() throws {
    let arguments = CommandLine.arguments
    guard arguments.count >= 4 else {
      fail("usage: ssb-production-protocol <case-dir> <out-dir> <ab|protocol>")
    }
    let caseDirectory = URL(fileURLWithPath: arguments[1])
    let outDirectory = URL(fileURLWithPath: arguments[2])
    let mode = arguments[3]
    try? FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)

    let caseData = try Data(contentsOf: caseDirectory.appendingPathComponent("case.json"))
    let payload = try JSONDecoder().decode(CaseFile.self, from: caseData)
    let side = payload.scan_shape[0]
    guard payload.scan_shape[0] == payload.scan_shape[1] else { fail("scan must be square") }
    guard let device = MTLCreateSystemDefaultDevice() else { fail("an Apple GPU is required") }

    let start = SSBOptimizationPoint(c10Nanometers: 0, c12Nanometers: 50, phi12Radians: 0)
    let seed: UInt64 = 42
    let rotation = Float(payload.rotation_angle_deg)

    switch mode {
    case "ab":
      let report = try runAB(
        device: device, caseDirectory: caseDirectory, payload: payload, side: side,
        caseData: caseData, start: start, seed: seed, rotation: rotation)
      try write(report, to: outDirectory.appendingPathComponent("ab-report.json"))
    case "identity":
      let report = try runIdentity(
        device: device, caseDirectory: caseDirectory, payload: payload, side: side,
        caseData: caseData, start: start, seed: seed, rotation: rotation)
      try write(report, to: outDirectory.appendingPathComponent("identity-report.json"))
    case "protocol":
      let report = try runProtocol(
        device: device, caseDirectory: caseDirectory, payload: payload, side: side,
        caseData: caseData, start: start, seed: seed, rotation: rotation)
      try write(report, to: outDirectory.appendingPathComponent("protocol-report.json"))
    default:
      fail("unknown mode \(mode)")
    }
  }

  private static func write<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(value).write(to: url)
    print("wrote \(url.path)")
  }

  // MARK: - app configuration

  /// Build the engine the way `SSBExplorer` builds it: a calibration whose
  /// aperture is matched to the bright-field disk, then `geometry(...)` with
  /// the acquisition's dead pixels excluded.
  private static func makeAppEngine(
    device: MTLDevice, payload: CaseFile, source: MTLBuffer
  ) throws -> (MetalSSBEngine, AppGeometryReport) {
    let detectorSide = payload.detector_shape[0]
    let excluded = Set(payload.detector_dead_pixels.map { $0[0] * detectorSide + $0[1] })
    var calibration = MetalSSBCalibration(
      beamEnergyKeV: payload.voltage_kV,
      semiangleMrad: payload.semiangle_mrad,
      scanStepRowAngstroms: payload.scan_sampling_A[0],
      scanStepColumnAngstroms: payload.scan_sampling_A[1],
      detectorStepRowMrad: 1, detectorStepColumnMrad: 1,
      centerRow: payload.bf_center[0], centerColumn: payload.bf_center[1],
      brightfieldRadiusPixels: payload.bf_radius_px)
    try calibration.matchApertureToBrightfieldDisk()
    // The detector sum only feeds the disk selection and the DC term; the
    // acquisition's exact total over the selected disk reproduces both.
    var sums = [UInt64](repeating: 0, count: detectorSide * detectorSide)
    let beamCenterIndex =
      Int(payload.bf_center[0].rounded()) * detectorSide + Int(payload.bf_center[1].rounded())
    sums[beamCenterIndex] = UInt64(payload.counts_sum)
    let setup = try calibration.geometry(
      detectorRows: detectorSide, detectorColumns: detectorSide, detectorSum: sums,
      excludedPixels: excluded, scanRows: payload.scan_shape[0],
      scanColumns: payload.scan_shape[1])
    guard setup.pixels.count == payload.bf_count else {
      fail("app geometry selected \(setup.pixels.count) pixels, expected \(payload.bf_count)")
    }
    let aperture = setup.geometry.brightfieldAperture
    guard aperture.allSatisfy({ $0 > 0 }) else { fail("app geometry has a zero aperture term") }
    // Structural check: the unrotated radial scale must differ from the
    // documented detector sampling only by the disk-matched scale factor.
    let scale = (payload.semiangle_mrad / payload.bf_radius_px) * 1e-3
    let documented = payload.det_sampling_mrad * 1e-3
    var maxDeviation = 0.0
    for index in 0..<min(setup.geometry.brightfieldKX.count, payload.brightfield_kx_unrotated.count) {
      let expected = payload.brightfield_kx_unrotated[index] * Float(scale / documented)
      let actual = setup.geometry.brightfieldKX[index]
      let denominator = max(abs(Double(expected)), 1e-12)
      maxDeviation = max(maxDeviation, abs(Double(actual - expected)) / denominator)
    }
    let engine = try MetalSSBEngine(
      device: device, geometry: setup.geometry, cacheBudgetBytes: nil)
    try engine.prepare(brightfield: source, countType: .uint16)
    guard engine.executedBrightfieldCount == payload.bf_count else {
      fail("app engine executed \(engine.executedBrightfieldCount) terms, expected \(payload.bf_count)")
    }
    let report = AppGeometryReport(
      detectorStepRowMrad: calibration.detectorStepRowMrad,
      semiangleMrad: calibration.semiangleMrad,
      selectedPixels: setup.pixels.count,
      minimumApertureWeight: Double(aperture.min() ?? 0),
      dcValueReal: Double(setup.geometry.dcValue.x),
      executedBrightfieldCount: engine.executedBrightfieldCount,
      unrotatedKxMaxRelativeDeviation: maxDeviation,
      unrotatedScaleFactor: scale / documented)
    print(
      String(
        format:
          "app geometry: detectorStep %.9f mrad, pixels %d, min aperture %.6f, dc %.4f, executed %d, kx deviation %.3g",
        report.detectorStepRowMrad, report.selectedPixels, report.minimumApertureWeight,
        report.dcValueReal, report.executedBrightfieldCount,
        report.unrotatedKxMaxRelativeDeviation))
    fflush(stdout)
    return (engine, report)
  }

  // MARK: - mode ab

  private static func runAB(
    device: MTLDevice, caseDirectory: URL, payload: CaseFile, side: Int, caseData: Data,
    start: SSBOptimizationPoint, seed: UInt64, rotation: Float
  ) throws -> ABReport {
    let probe = MetalSSBAberrations(
      c10Nanometers: 73.18188621458395, c12Nanometers: 14.020962948808993,
      phi12Radians: 0.4700365259977606)
    var micro: [MicroReport] = []
    var fits: [FitReport] = []

    // 1. Parity-case configuration: documented sampling, pruned aperture.
    do {
      let source = try loadDenseSource(device: device, caseDirectory: caseDirectory, payload: payload)
      let engine = try MetalSSBEngine(
        device: device, geometry: payload.metal_geometry, cacheBudgetBytes: nil)
      try engine.prepare(brightfield: source, countType: .uint16)
      guard engine.executedBrightfieldCount == payload.active_bf_count else {
        fail(
          "parity engine executed \(engine.executedBrightfieldCount) terms, expected \(payload.active_bf_count)")
      }
      micro.append(
        try microBenchmark(
          configuration: "parity-2464-executed", engine: engine, aberrations: probe,
          rotationDegrees: nil, evaluations: 16))

      let loadBefore = loadAverage()
      let optimizer = SSBOptimizer(globalTrials: 200, seed: seed)
      let result = try optimizer.run(
        start: start,
        evaluate: { point in
          Double(
            try engine.phaseVariance(
              aberrations: MetalSSBAberrations(
                c10Nanometers: Float(point.c10Nanometers),
                c12Nanometers: Float(point.c12Nanometers),
                phi12Radians: Float(point.phi12Radians))).loss)
        })
      let loadAfter = loadAverage()
      let parityLegacy = FitReport(
        configuration: "parity-2464-executed", pass: "SSBOptimizer.run",
        globalTrials: result.globalTrials,
        executedBrightfieldCount: engine.executedBrightfieldCount,
        evaluations: result.trials.count, refinementEvaluations: result.refinementEvaluations,
        lossSeconds: result.elapsedSeconds, wallSeconds: result.elapsedSeconds,
        bestC10Nanometers: result.best.c10Nanometers,
        bestC12Nanometers: result.best.c12Nanometers,
        bestPhi12Radians: result.best.phi12Radians,
        bestLoss: result.loss,
        loadAverageBefore: loadBefore, loadAverageAfter: loadAfter,
        trajectorySha256: trajectoryDigest(result.trials))
      print(
        String(
          format: "fit parity/legacy bf %d evals %d wall %.3f s loss %.17g load %.2f->%.2f",
          engine.executedBrightfieldCount, result.trials.count, result.elapsedSeconds,
          result.loss, loadBefore, loadAfter))
      fflush(stdout)

      let engineLoadBefore = loadAverage()
      let outerStart = Date()
      let fit = try engine.optimize(
        start: MetalSSBAberrations(
          c10Nanometers: Float(start.c10Nanometers), c12Nanometers: Float(start.c12Nanometers),
          phi12Radians: Float(start.phi12Radians)),
        rotationDegrees: nil, globalTrials: 200, seed: seed)
      let outerSeconds = Date().timeIntervalSince(outerStart)
      let engineLoadAfter = loadAverage()
      let parityEngineCall = FitReport(
        configuration: "parity-2464-executed", pass: "engine.optimize",
        globalTrials: fit.globalTrials,
        executedBrightfieldCount: engine.executedBrightfieldCount,
        evaluations: fit.trials.count, refinementEvaluations: fit.refinementEvaluations,
        lossSeconds: fit.elapsedSeconds, wallSeconds: outerSeconds,
        bestC10Nanometers: fit.best.c10Nanometers,
        bestC12Nanometers: fit.best.c12Nanometers,
        bestPhi12Radians: fit.best.phi12Radians,
        bestLoss: fit.loss,
        loadAverageBefore: engineLoadBefore, loadAverageAfter: engineLoadAfter,
        trajectorySha256: trajectoryDigest(fit.trials))
      print(
        String(
          format:
            "fit parity/engine.optimize bf %d evals %d optimizer %.3f s outer %.3f s loss %.17g load %.2f->%.2f",
          engine.executedBrightfieldCount, fit.trials.count, fit.elapsedSeconds, outerSeconds,
          fit.loss, engineLoadBefore, engineLoadAfter))
      fflush(stdout)
      fits.append(parityLegacy)
      fits.append(parityEngineCall)
    }

    // 2. Production (app) configuration: disk-matched sampling, full disk.
    do {
      let source = try loadDenseSource(device: device, caseDirectory: caseDirectory, payload: payload)
      let (engine, appGeometry) = try makeAppEngine(
        device: device, payload: payload, source: source)
      micro.append(
        try microBenchmark(
          configuration: "app-8937-executed", engine: engine, aberrations: probe,
          rotationDegrees: rotation, evaluations: 16))

      let loadBefore = loadAverage()
      let outerStart = Date()
      let fit = try engine.optimize(
        start: MetalSSBAberrations(
          c10Nanometers: Float(start.c10Nanometers), c12Nanometers: Float(start.c12Nanometers),
          phi12Radians: Float(start.phi12Radians)),
        rotationDegrees: rotation, globalTrials: 200, seed: seed)
      let outerSeconds = Date().timeIntervalSince(outerStart)
      let loadAfter = loadAverage()
      let appFit = FitReport(
        configuration: "app-8937-executed", pass: "engine.optimize",
        globalTrials: fit.globalTrials,
        executedBrightfieldCount: engine.executedBrightfieldCount,
        evaluations: fit.trials.count, refinementEvaluations: fit.refinementEvaluations,
        lossSeconds: fit.elapsedSeconds, wallSeconds: outerSeconds,
        bestC10Nanometers: fit.best.c10Nanometers,
        bestC12Nanometers: fit.best.c12Nanometers,
        bestPhi12Radians: fit.best.phi12Radians,
        bestLoss: fit.loss,
        loadAverageBefore: loadBefore, loadAverageAfter: loadAfter,
        trajectorySha256: trajectoryDigest(fit.trials))
      print(
        String(
          format:
            "fit app/engine.optimize bf %d evals %d optimizer %.3f s outer %.3f s loss %.17g load %.2f->%.2f",
          engine.executedBrightfieldCount, fit.trials.count, fit.elapsedSeconds, outerSeconds,
          fit.loss, loadBefore, loadAfter))
      fflush(stdout)
      fits.append(appFit)

      let report = ABReport(
        device: device.name,
        recommendedMaxWorkingSetBytes: device.recommendedMaxWorkingSetSize,
        currentAllocatedBytesBefore: device.currentAllocatedSize,
        caseJsonSha256: sha256Hex(caseData),
        bfCount: payload.bf_count, activeBfCount: payload.active_bf_count,
        micro: micro, fits: fits, appGeometry: appGeometry)
      return report
    }
  }


  // MARK: - mode identity

  /// Single-run optimum bit-identity on the production configuration:
  /// 25 TPE trials + NM against 200 TPE trials + NM, plus NM alone warm
  /// started from the 200-trial optimum. Values only; no timing protocol.
  private static func runIdentity(
    device: MTLDevice, caseDirectory: URL, payload: CaseFile, side: Int, caseData: Data,
    start: SSBOptimizationPoint, seed: UInt64, rotation: Float
  ) throws -> ProtocolReport {
    let source = try loadDenseSource(device: device, caseDirectory: caseDirectory, payload: payload)
    let (engine, _) = try makeAppEngine(device: device, payload: payload, source: source)

    func runArm(name: String, trials: Int, startPoint: SSBOptimizationPoint)
      throws -> (ArmReport, [TrialRecord], SSBOptimizationResult)
    {
      let loadBefore = loadAverage()
      let outerStart = Date()
      let fit = try engine.optimize(
        start: MetalSSBAberrations(
          c10Nanometers: Float(startPoint.c10Nanometers),
          c12Nanometers: Float(startPoint.c12Nanometers),
          phi12Radians: Float(startPoint.phi12Radians)),
        rotationDegrees: rotation, globalTrials: trials, seed: seed)
      let outerSeconds = Date().timeIntervalSince(outerStart)
      let loadAfter = loadAverage()
      let report = ArmReport(
        arm: name, pass: "single",
        globalTrials: fit.globalTrials,
        evaluations: fit.trials.count, refinementEvaluations: fit.refinementEvaluations,
        wallSeconds: outerSeconds,
        bestC10Nanometers: fit.best.c10Nanometers,
        bestC12Nanometers: fit.best.c12Nanometers,
        bestPhi12Radians: fit.best.phi12Radians,
        bestLoss: fit.loss,
        loadAverageBefore: loadBefore, loadAverageAfter: loadAfter,
        trajectorySha256: trajectoryDigest(fit.trials))
      print(
        String(
          format:
            "arm %@ trials %d evals %d refine %d wall %.3f s loss %.17g best %.17g %.17g %.17g load %.2f->%.2f",
          name, trials, fit.trials.count, fit.refinementEvaluations, outerSeconds, fit.loss,
          fit.best.c10Nanometers, fit.best.c12Nanometers, fit.best.phi12Radians, loadBefore,
          loadAfter))
      fflush(stdout)
      return (report, record(fit.trials), fit)
    }

    let loadAtStart = loadAverage()
    var arms: [ArmReport] = []
    var trajectories: [String: [TrialRecord]] = [:]

    let (referenceReport, referenceTrials, reference) = try runArm(
      name: "tpe200", trials: 200, startPoint: start)
    arms.append(referenceReport)
    trajectories["tpe200"] = referenceTrials

    let (shortReport, shortTrials, shortFit) = try runArm(
      name: "tpe25", trials: 25, startPoint: start)
    arms.append(shortReport)
    trajectories["tpe25"] = shortTrials

    let warmStart = SSBOptimizationPoint(
      c10Nanometers: reference.best.c10Nanometers,
      c12Nanometers: reference.best.c12Nanometers,
      phi12Radians: reference.best.phi12Radians)
    let (warmReport, warmTrials, _) = try runArm(
      name: "nmWarm", trials: 0, startPoint: warmStart)
    arms.append(warmReport)
    trajectories["nmWarm"] = warmTrials
    let loadAtEnd = loadAverage()

    // Value comparisons: float32 bit patterns of the optimum triple, the
    // loss, and the evaluation count.
    func bits(_ c10: Double, _ c12: Double, _ phi: Double) -> String {
      String(
        format: "%08x %08x %08x", Float(c10).bitPattern, Float(c12).bitPattern,
        Float(phi).bitPattern)
    }
    func bits(_ point: SSBOptimizationPoint) -> String {
      bits(point.c10Nanometers, point.c12Nanometers, point.phi12Radians)
    }
    func trail(_ trials: [SSBOptimizationTrial]) -> String {
      var text = ""
      for trial in trials where trial.stage == "tpe" {
        text += String(
          format: "%08x %08x %08x %08x\n", Float(trial.point.c10Nanometers).bitPattern,
          Float(trial.point.c12Nanometers).bitPattern, Float(trial.point.phi12Radians).bitPattern,
          Float(trial.loss).bitPattern)
      }
      return sha256Hex(Data(text.utf8))
    }
    let tpeTrials200 = reference.trials.filter { $0.stage == "tpe" }
    let tpeTrials25 = shortFit.trials.filter { $0.stage == "tpe" }
    var differingTrials = 0
    for index in 0..<min(tpeTrials25.count, 25) {
      let a = tpeTrials25[index]
      let b = tpeTrials200[index]
      if Float(a.point.c10Nanometers).bitPattern != Float(b.point.c10Nanometers).bitPattern
        || Float(a.point.c12Nanometers).bitPattern != Float(b.point.c12Nanometers).bitPattern
        || Float(a.point.phi12Radians).bitPattern != Float(b.point.phi12Radians).bitPattern
        || Float(a.loss).bitPattern != Float(b.loss).bitPattern
      {
        differingTrials += 1
      }
    }
    let optimumIdentical =
      bits(shortFit.best) == bits(reference.best) && shortFit.loss.bitPattern == reference.loss.bitPattern
    let warmIdentical =
      bits(
        warmReport.bestC10Nanometers, warmReport.bestC12Nanometers, warmReport.bestPhi12Radians)
      == bits(reference.best) && warmReport.bestLoss.bitPattern == reference.loss.bitPattern
    let comparisons: [String: String] = [
      "tpe25_optimum_bits": bits(shortFit.best),
      "tpe200_optimum_bits": bits(reference.best),
      "tpe25_loss_bits": String(format: "%016llx", shortFit.loss.bitPattern),
      "tpe200_loss_bits": String(format: "%016llx", reference.loss.bitPattern),
      "tpe25_evaluations": String(shortFit.trials.count),
      "tpe200_evaluations": String(reference.trials.count),
      "tpe25_refinement_evaluations": String(shortFit.refinementEvaluations),
      "tpe200_refinement_evaluations": String(reference.refinementEvaluations),
      "tpe_prefix_bitwise_identical": differingTrials == 0 ? "yes" : "no",
      "tpe_prefix_differing_trials": String(differingTrials),
      "tpe25_prefix_trajectory_sha256": trail(shortFit.trials),
      "tpe200_prefix_trajectory_sha256": trail(reference.trials),
      "nmWarm_optimum_bits": bits(
        warmReport.bestC10Nanometers, warmReport.bestC12Nanometers, warmReport.bestPhi12Radians),
      "nmWarm_loss_bits": String(format: "%016llx", warmReport.bestLoss.bitPattern),
      "nmWarm_evaluations": String(warmReport.evaluations),
      "optimum_bitwise_identical": optimumIdentical ? "yes" : "no",
      "warm_optimum_bitwise_identical": warmIdentical ? "yes" : "no",
    ]
    let verdict =
      "25 trials + NM vs 200 trials + NM on the production config: "
      + (optimumIdentical ? "BIT-IDENTICAL optimum" : "NOT identical")
    print(verdict)
    for key in comparisons.keys.sorted() {
      print("  \(key) = \(comparisons[key]!)")
    }
    fflush(stdout)

    return ProtocolReport(
      device: device.name,
      caseJsonSha256: sha256Hex(caseData),
      configuration:
        "app disk-matched aperture, \(engine.executedBrightfieldCount) executed BF terms, rotation \(rotation) deg",
      startPoint: [start.c10Nanometers, start.c12Nanometers, start.phi12Radians],
      warmStartPoint: [
        warmStart.c10Nanometers, warmStart.c12Nanometers, warmStart.phi12Radians,
      ],
      referenceArm: "tpe200",
      loadAverageAtStart: loadAtStart, loadAverageAtEnd: loadAtEnd,
      arms: arms, trajectories: trajectories)
  }

  // MARK: - mode protocol

  private static func runProtocol(
    device: MTLDevice, caseDirectory: URL, payload: CaseFile, side: Int, caseData: Data,
    start: SSBOptimizationPoint, seed: UInt64, rotation: Float
  ) throws -> ProtocolReport {
    let source = try loadDenseSource(device: device, caseDirectory: caseDirectory, payload: payload)
    let (engine, appGeometry) = try makeAppEngine(
      device: device, payload: payload, source: source)

    func runArm(name: String, trials: Int, startPoint: SSBOptimizationPoint, pass: String)
      throws -> (ArmReport, [TrialRecord])
    {
      let loadBefore = loadAverage()
      let outerStart = Date()
      let fit = try engine.optimize(
        start: MetalSSBAberrations(
          c10Nanometers: Float(startPoint.c10Nanometers),
          c12Nanometers: Float(startPoint.c12Nanometers),
          phi12Radians: Float(startPoint.phi12Radians)),
        rotationDegrees: rotation, globalTrials: trials, seed: seed)
      let outerSeconds = Date().timeIntervalSince(outerStart)
      let loadAfter = loadAverage()
      let report = ArmReport(
        arm: name, pass: pass, globalTrials: fit.globalTrials,
        evaluations: fit.trials.count, refinementEvaluations: fit.refinementEvaluations,
        wallSeconds: outerSeconds,
        bestC10Nanometers: fit.best.c10Nanometers,
        bestC12Nanometers: fit.best.c12Nanometers,
        bestPhi12Radians: fit.best.phi12Radians,
        bestLoss: fit.loss,
        loadAverageBefore: loadBefore, loadAverageAfter: loadAfter,
        trajectorySha256: trajectoryDigest(fit.trials))
      print(
        String(
          format:
            "arm %@ pass %@ trials %d evals %d refine %d wall %.3f s loss %.17g best %.6f %.6f %.9f load %.2f->%.2f",
          name, pass, trials, fit.trials.count, fit.refinementEvaluations, outerSeconds,
          fit.loss, fit.best.c10Nanometers, fit.best.c12Nanometers, fit.best.phi12Radians,
          loadBefore, loadAfter))
      fflush(stdout)
      return (report, record(fit.trials))
    }

    let loadAtStart = loadAverage()
    var arms: [ArmReport] = []
    var trajectories: [String: [TrialRecord]] = [:]
    let order = [("tpe25", 25), ("tpe50", 50), ("tpe100", 100), ("tpe200", 200)]
    for (name, trials) in order {
      let (report, trialsRecorded) = try runArm(
        name: name, trials: trials, startPoint: start, pass: "forward")
      arms.append(report)
      trajectories["\(name).forward"] = trialsRecorded
    }
    // The warm start is this configuration's own 200-trial optimum, so the
    // NM-only arm measures refinement alone on the production objective.
    let reference = arms.first { $0.arm == "tpe200" }!
    let warmStart = SSBOptimizationPoint(
      c10Nanometers: reference.bestC10Nanometers,
      c12Nanometers: reference.bestC12Nanometers,
      phi12Radians: reference.bestPhi12Radians)
    let (warmReport, warmTrials) = try runArm(
      name: "nmWarm", trials: 0, startPoint: warmStart, pass: "forward")
    arms.append(warmReport)
    trajectories["nmWarm.forward"] = warmTrials

    let reverseOrder = [("nmWarm", 0), ("tpe200", 200), ("tpe100", 100), ("tpe50", 50), ("tpe25", 25)]
    for (name, trials) in reverseOrder {
      let startPoint = name == "nmWarm" ? warmStart : start
      let (report, trialsRecorded) = try runArm(
        name: name, trials: trials, startPoint: startPoint, pass: "reverse")
      arms.append(report)
      trajectories["\(name).reverse"] = trialsRecorded
    }
    let loadAtEnd = loadAverage()

    let report = ProtocolReport(
      device: device.name,
      caseJsonSha256: sha256Hex(caseData),
      configuration:
        "app disk-matched aperture, \(engine.executedBrightfieldCount) executed BF terms, rotation \(rotation) deg",
      startPoint: [start.c10Nanometers, start.c12Nanometers, start.phi12Radians],
      warmStartPoint: [
        warmStart.c10Nanometers, warmStart.c12Nanometers, warmStart.phi12Radians,
      ],
      referenceArm: "tpe200",
      loadAverageAtStart: loadAtStart, loadAverageAtEnd: loadAtEnd,
      arms: arms, trajectories: trajectories)
    print(
      String(
        format: "protocol complete: %d evals, executed BF %d, load %.2f->%.2f",
        arms.reduce(0) { $0 + $1.evaluations } + arms.reduce(0) { $0 + $1.refinementEvaluations },
        engine.executedBrightfieldCount, loadAtStart, loadAtEnd))
    _ = (side, appGeometry)
    return report
  }
}
