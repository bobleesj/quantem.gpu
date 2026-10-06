// SSB fit-trajectory parity harness.
//
// The strict objective gate (`ssb_parity_check.swift`) proves that one
// objective evaluation agrees with an independent double oracle. That is not
// enough to protect a fit: the search draws candidate t+1 from a history that
// may or may not already contain candidate t. `SSBOptimizer.run` draws
// `min(2, ...)` candidates from the *same* history whenever an `evaluateBatch`
// closure is supplied, so enabling batching changes the trajectory even when
// every individual loss is unchanged.
//
// This harness records, on one exported exact BF-column artifact and one seed:
//
//   * the full 200-trial trajectory plus Nelder-Mead refinement of the
//     sequential path (`evaluateBatch == nil`, what `MetalSSBEngine.optimize`
//     uses today),
//   * the same run with the batched pair draw (`evaluateBatch != nil`),
//   * whether the production `optimize` entry point reproduces the sequential
//     trajectory exactly,
//   * whether the objective is a pure function of the float32 aberration
//     triple (same point, same engine; same point, fresh engine; the same
//     float32 triple reached from a different double),
//   * how far the two trajectories and their final optima diverge.
//
// The batched draw is measured twice when the tree exposes a native batch
// objective (`MetalSSBEngine.phaseVarianceBatch`, compiled in only when the
// symbol exists, see `check_ssb_fit_trajectory.sh`):
//
//   * `closure` batches the *same* single-candidate evaluation through
//     `evaluateBatch`, which isolates the sampling effect from every numeric
//     effect, and
//   * `engine` evaluates the pair with the native batch objective, which also
//     tests its documented per-candidate bit-identity claim and whether it
//     leaves any state that perturbs later single evaluations.
//
// Nothing is fitted, seeded, binned or cropped differently between the runs,
// and no frozen number is written here.
//
// Usage: ssb-fit-trajectory <case-dir> <out-dir> [closure|engine|both]

import CryptoKit
import Foundation
import Metal
import MetalSSBKernels

private func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data("ssb-fit-trajectory: \(message)\n".utf8))
  exit(1)
}

private struct CaseFile: Decodable {
  let scan_shape: [Int]
  let bf_count: Int
  let active_bf_count: Int
  let metal_geometry: MetalSSBGeometry
}

private struct Trial: Encodable {
  let index: Int
  let stage: String
  let c10Nanometers: Double
  let c12Nanometers: Double
  let phi12Radians: Double
  let loss: Double
}

private struct RunReport: Encodable {
  let bestC10Nanometers: Double
  let bestC12Nanometers: Double
  let bestPhi12Radians: Double
  let bestLoss: Double
  let globalTrials: Int
  let refinementEvaluations: Int
  let totalEvaluations: Int
  let elapsedSeconds: Double
  let trials: [Trial]
}

private struct Comparison: Encodable {
  let trialsCompared: Int
  let trialsWithDifferentFloat32Point: Int
  let trialsWithDifferentLossAtSameIndex: Int
  let maxLossDifferenceAtSameIndex: Double
  let bestSoFarDifferentCount: Int
  let finalBestPointDeltaC10: Double
  let finalBestPointDeltaC12: Double
  let finalBestPointDeltaPhi12: Double
  let finalBestLossDelta: Double
  let sharedCandidateLossMismatches: Int
}

private struct Purity: Encodable {
  let probes: Int
  let repeatSameEngineBitwiseMismatches: Int
  let freshEngineBitwiseMismatches: Int
  let float32AliasBitwiseMismatches: Int
  let lossMaxUlpDifference: UInt32
  let firstProbeLoss: Double
}

private struct BatchObjective: Encodable {
  let requested: String
  let compiledIn: Bool
  let probes: Int
  let bitwiseIdentityMismatches: Int
  let maxLossBitDifference: UInt32
  let postBatchPurityMismatches: Int
}

private struct FitReport: Encodable {
  let caseName: String
  let scanSide: Int
  let logicalBrightfieldCount: Int
  let activeBrightfieldCount: Int
  let seed: UInt64
  let globalTrials: Int
  let start: [String: Double]
  let caseJsonSha256: String
  let sequential: RunReport
  let batched: RunReport?
  let batchedEngine: RunReport?
  let batchObjective: BatchObjective?
  let productionOptimizeAlwaysMatchesSequential: Bool
  let objectivePurity: Purity
  let comparison: Comparison?
  let comparisonEngine: Comparison?
}

@main enum SSBFitTrajectory {
  static func main() throws {
    let arguments = CommandLine.arguments
    guard arguments.count >= 3 else {
      fail("usage: ssb-fit-trajectory <case-dir> <out-dir> [closure|engine|both]")
    }
    let caseDirectory = URL(fileURLWithPath: arguments[1])
    let outDirectory = URL(fileURLWithPath: arguments[2])
    let requestedBatch = arguments.count >= 4 ? arguments[3] : "closure"
    guard ["closure", "engine", "both"].contains(requestedBatch) else {
      fail("batch objective must be closure, engine or both")
    }
    try? FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)

    let caseURL = caseDirectory.appendingPathComponent("case.json")
    let caseData = try Data(contentsOf: caseURL)
    let caseDigest = SHA256.hash(data: caseData).map { String(format: "%02x", $0) }.joined()
    let payload = try JSONDecoder().decode(CaseFile.self, from: caseData)
    let side = payload.scan_shape[0]
    guard payload.scan_shape[0] == payload.scan_shape[1] else { fail("scan must be square") }
    let counts = try Data(
      contentsOf: caseDirectory.appendingPathComponent("source/bf_columns.u16"),
      options: .mappedIfSafe)
    let expectedBytes =
      payload.bf_count * side * side * MemoryLayout<UInt16>.size
    guard counts.count == expectedBytes else {
      fail("bf_columns.u16 is \(counts.count) bytes, expected \(expectedBytes)")
    }
    guard let device = MTLCreateSystemDefaultDevice() else { fail("an Apple GPU is required") }

    let source = counts.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    let engine = try MetalSSBEngine(
      device: device, geometry: payload.metal_geometry, cacheBudgetBytes: nil)
    try engine.prepare(brightfield: source, countType: .uint16)

    let start = SSBOptimizationPoint(
      c10Nanometers: 0, c12Nanometers: 50, phi12Radians: 0)
    let seed: UInt64 = 42
    let globalTrials = 200

    func loss(_ point: SSBOptimizationPoint) throws -> Double {
      Double(
        try engine.phaseVariance(
          aberrations: MetalSSBAberrations(
            c10Nanometers: Float(point.c10Nanometers),
            c12Nanometers: Float(point.c12Nanometers),
            phi12Radians: Float(point.phi12Radians))).loss)
    }

    #if SSB_HAS_BATCH_OBJECTIVE
      let batchObjectiveCompiledIn = true
      let nativeBatch: (([SSBOptimizationPoint]) throws -> [Double])? = { points in
        try engine.phaseVarianceBatch(
          aberrations: points.map {
            MetalSSBAberrations(
              c10Nanometers: Float($0.c10Nanometers),
              c12Nanometers: Float($0.c12Nanometers),
              phi12Radians: Float($0.phi12Radians))
          }
        ).map { Double($0.loss) }
      }
    #else
      let batchObjectiveCompiledIn = false
      let nativeBatch: (([SSBOptimizationPoint]) throws -> [Double])? = nil
    #endif

    func trials(_ result: SSBOptimizationResult) -> [Trial] {
      result.trials.enumerated().map { index, trial in
        Trial(
          index: index, stage: trial.stage,
          c10Nanometers: trial.point.c10Nanometers,
          c12Nanometers: trial.point.c12Nanometers,
          phi12Radians: trial.point.phi12Radians,
          loss: trial.loss)
      }
    }

    func runReport(_ result: SSBOptimizationResult) -> RunReport {
      RunReport(
        bestC10Nanometers: result.best.c10Nanometers,
        bestC12Nanometers: result.best.c12Nanometers,
        bestPhi12Radians: result.best.phi12Radians,
        bestLoss: result.loss,
        globalTrials: result.globalTrials,
        refinementEvaluations: result.refinementEvaluations,
        totalEvaluations: result.trials.count,
        elapsedSeconds: result.elapsedSeconds,
        trials: trials(result))
    }

    /// Index-by-index comparison of two runs on one artifact and one seed.
    func compare(
      _ left: SSBOptimizationResult, _ right: SSBOptimizationResult
    ) -> Comparison {
      let leftTrials = trials(left)
      let rightTrials = trials(right)
      let sharedCount = min(leftTrials.count, rightTrials.count)
      var differentPoints = 0
      var differentLosses = 0
      var maxLossDelta = 0.0
      var bestSoFarDifferent = 0
      var leftBest = leftTrials[0].loss
      var rightBest = rightTrials[0].loss
      var leftLossByPoint: [String: Double] = [:]
      var rightLossByPoint: [String: Double] = [:]
      for index in 0..<sharedCount {
        let one = leftTrials[index]
        let two = rightTrials[index]
        if trialKey(one) != trialKey(two) { differentPoints += 1 }
        if one.loss != two.loss {
          differentLosses += 1
          maxLossDelta = max(maxLossDelta, abs(one.loss - two.loss))
        }
        leftBest = min(leftBest, one.loss)
        rightBest = min(rightBest, two.loss)
        if leftBest != rightBest { bestSoFarDifferent += 1 }
        leftLossByPoint[trialKey(one)] = one.loss
        rightLossByPoint[trialKey(two)] = two.loss
      }
      var sharedMismatches = 0
      for (key, value) in leftLossByPoint {
        if let other = rightLossByPoint[key], other != value { sharedMismatches += 1 }
      }
      return Comparison(
        trialsCompared: sharedCount,
        trialsWithDifferentFloat32Point: differentPoints,
        trialsWithDifferentLossAtSameIndex: differentLosses,
        maxLossDifferenceAtSameIndex: maxLossDelta,
        bestSoFarDifferentCount: bestSoFarDifferent,
        finalBestPointDeltaC10: right.best.c10Nanometers - left.best.c10Nanometers,
        finalBestPointDeltaC12: right.best.c12Nanometers - left.best.c12Nanometers,
        finalBestPointDeltaPhi12: right.best.phi12Radians - left.best.phi12Radians,
        finalBestLossDelta: right.loss - left.loss,
        sharedCandidateLossMismatches: sharedMismatches)
    }

    // 1. Sequential draw: exactly what `MetalSSBEngine.optimize` runs today.
    let sequential = try SSBOptimizer(globalTrials: globalTrials, seed: seed).run(
      start: start, evaluate: loss)

    // 2. Objective purity, measured before any batch run touches the engine.
    let probePoints = [
      start,
      sequential.trials[1].point,
      sequential.trials[50].point,
      sequential.best,
      SSBOptimizationPoint(
        c10Nanometers: 73.18188621458395, c12Nanometers: 14.020962948808993,
        phi12Radians: 0.4700365259977606),
    ]
    var firstPass: [Double] = []
    for point in probePoints { firstPass.append(try loss(point)) }
    var repeatMismatches = 0
    for (index, point) in probePoints.enumerated() {
      if try loss(point) != firstPass[index] { repeatMismatches += 1 }
    }
    let freshEngine = try MetalSSBEngine(
      device: device, geometry: payload.metal_geometry, cacheBudgetBytes: nil)
    try freshEngine.prepare(brightfield: source, countType: .uint16)
    var freshMismatches = 0
    for (index, point) in probePoints.enumerated() {
      let fresh = Double(
        try freshEngine.phaseVariance(
          aberrations: MetalSSBAberrations(
            c10Nanometers: Float(point.c10Nanometers),
            c12Nanometers: Float(point.c12Nanometers),
            phi12Radians: Float(point.phi12Radians))).loss)
      if fresh != firstPass[index] { freshMismatches += 1 }
    }
    // The same float32 triple reached from a neighbouring double must give the
    // same loss: the objective is a function of the quantised point only.
    var aliasMismatches = 0
    var maxUlp: UInt32 = 0
    for (index, point) in probePoints.enumerated() {
      let nudgedDouble = point.c10Nanometers.nextUp
      let nudged = SSBOptimizationPoint(
        c10Nanometers: nudgedDouble,
        c12Nanometers: point.c12Nanometers,
        phi12Radians: point.phi12Radians)
      let nudgeLoss = try loss(nudged)
      let before = Float(point.c10Nanometers).bitPattern
      let after = Float(nudgedDouble).bitPattern
      if before == after {
        if nudgeLoss != firstPass[index] { aliasMismatches += 1 }
      }
      let ulp = before > after ? before - after : after - before
      maxUlp = max(maxUlp, ulp)
    }

    // 3. Batched pair draw through the single-candidate closure.
    var batched: SSBOptimizationResult? = nil
    if requestedBatch == "closure" || requestedBatch == "both" {
      batched = try SSBOptimizer(globalTrials: globalTrials, seed: seed).run(
        start: start,
        evaluate: loss,
        evaluateBatch: { points in try points.map(loss) })
    }

    // 4. Batched pair draw through the native batch objective, if the tree has one.
    var batchedEngine: SSBOptimizationResult? = nil
    var batchObjective: BatchObjective? = nil
    if requestedBatch == "engine" || requestedBatch == "both" {
      guard let nativeBatch else {
        fail(
          "this tree exposes no MetalSSBEngine.phaseVarianceBatch; rebuild with "
            + "-D SSB_HAS_BATCH_OBJECTIVE to measure the native batch objective")
      }
      batchedEngine = try SSBOptimizer(globalTrials: globalTrials, seed: seed).run(
        start: start, evaluate: loss, evaluateBatch: nativeBatch)

      // The batch objective is documented as bit-identical per candidate. Test
      // it on the probe points and on the first pairs the batched search drew.
      var identityMismatches = 0
      var identityProbes = 0
      var maxLossBits: UInt32 = 0
      var pairs: [[SSBOptimizationPoint]] = [Array(probePoints.prefix(4))]
      var trajectoryPairs: [[SSBOptimizationPoint]] = []
      let engineTrials = trials(batchedEngine!)
      var index = 1
      while index + 1 < engineTrials.count && trajectoryPairs.count < 32 {
        if engineTrials[index].stage == "tpe" && engineTrials[index + 1].stage == "tpe" {
          trajectoryPairs.append([
            point(of: engineTrials[index]), point(of: engineTrials[index + 1]),
          ])
          index += 2
        } else {
          index += 1
        }
      }
      pairs.append(contentsOf: trajectoryPairs)
      for pair in pairs {
        let batchedLosses = try nativeBatch(pair)
        for (offset, point) in pair.enumerated() {
          let single = try loss(point)
          identityProbes += 1
          if single != batchedLosses[offset] {
            identityMismatches += 1
            let left = Float(single).bitPattern
            let right = Float(batchedLosses[offset]).bitPattern
            maxLossBits = max(maxLossBits, left > right ? left - right : right - left)
          }
        }
      }
      // ... and that running the batch path left no state behind.
      var postBatchMismatches = 0
      for (probeIndex, point) in probePoints.enumerated() {
        if try loss(point) != firstPass[probeIndex] { postBatchMismatches += 1 }
      }
      batchObjective = BatchObjective(
        requested: requestedBatch,
        compiledIn: batchObjectiveCompiledIn,
        probes: identityProbes,
        bitwiseIdentityMismatches: identityMismatches,
        maxLossBitDifference: maxLossBits,
        postBatchPurityMismatches: postBatchMismatches)
    }

    // 5. The production entry point must reproduce the sequential trajectory.
    let production = try engine.optimize(
      start: MetalSSBAberrations(c10Nanometers: 0, c12Nanometers: 50, phi12Radians: 0),
      globalTrials: globalTrials, seed: seed)
    let productionMatches =
      production.trials.count == sequential.trials.count
      && zip(production.trials, sequential.trials).allSatisfy {
        $0.point == $1.point && $0.loss == $1.loss && $0.stage == $1.stage
      }
      && production.best == sequential.best && production.loss == sequential.loss

    let report = FitReport(
      caseName: caseDirectory.lastPathComponent,
      scanSide: side,
      logicalBrightfieldCount: payload.metal_geometry.logicalBrightfieldCount,
      activeBrightfieldCount: payload.active_bf_count,
      seed: seed,
      globalTrials: globalTrials,
      start: [
        "C10": start.c10Nanometers, "C12": start.c12Nanometers,
        "phi12": start.phi12Radians,
      ],
      caseJsonSha256: caseDigest,
      sequential: runReport(sequential),
      batched: batched.map(runReport),
      batchedEngine: batchedEngine.map(runReport),
      batchObjective: batchObjective,
      productionOptimizeAlwaysMatchesSequential: productionMatches,
      objectivePurity: Purity(
        probes: probePoints.count,
        repeatSameEngineBitwiseMismatches: repeatMismatches,
        freshEngineBitwiseMismatches: freshMismatches,
        float32AliasBitwiseMismatches: aliasMismatches,
        lossMaxUlpDifference: maxUlp,
        firstProbeLoss: firstPass[0]),
      comparison: batched.map { compare(sequential, $0) },
      comparisonEngine: batchedEngine.map { compare(sequential, $0) })

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(to: outDirectory.appendingPathComponent("fit-trajectory.json"))

    print("case=\(report.caseName) side=\(side) logical=\(report.logicalBrightfieldCount) "
      + "active=\(report.activeBrightfieldCount) trials=\(globalTrials) seed=\(seed)")
    print("case.json sha256=\(caseDigest)")
    print("sequential best=\(sequential.best.c10Nanometers) \(sequential.best.c12Nanometers) "
      + "\(sequential.best.phi12Radians) loss=\(sequential.loss) evals=\(sequential.trials.count)")
    for (name, result) in [("batched-closure", batched), ("batched-engine", batchedEngine)] {
      guard let result else { continue }
      print("\(name) best=\(result.best.c10Nanometers) \(result.best.c12Nanometers) "
        + "\(result.best.phi12Radians) loss=\(result.loss) evals=\(result.trials.count)")
    }
    print("production optimize matches sequential=\(productionMatches)")
    for (name, entry) in [("closure", report.comparison), ("engine", report.comparisonEngine)] {
      guard let entry else { continue }
      print("comparison[\(name)]: differentPoints=\(entry.trialsWithDifferentFloat32Point)/"
        + "\(entry.trialsCompared) differentLoss=\(entry.trialsWithDifferentLossAtSameIndex) "
        + "maxLossDelta=\(entry.maxLossDifferenceAtSameIndex) "
        + "bestSoFarDifferent=\(entry.bestSoFarDifferentCount) "
        + "sharedMismatches=\(entry.sharedCandidateLossMismatches) "
        + "optimumDelta=\(entry.finalBestPointDeltaC10),\(entry.finalBestPointDeltaC12),"
        + "\(entry.finalBestPointDeltaPhi12) lossDelta=\(entry.finalBestLossDelta)")
    }
    if let batchObjective {
      print("batch objective: compiledIn=\(batchObjective.compiledIn) "
        + "probes=\(batchObjective.probes) identityMismatches="
        + "\(batchObjective.bitwiseIdentityMismatches) "
        + "maxLossBits=\(batchObjective.maxLossBitDifference) "
        + "postBatchPurityMismatches=\(batchObjective.postBatchPurityMismatches)")
    }
    print("purity: repeat=\(repeatMismatches) fresh=\(freshMismatches) alias=\(aliasMismatches) "
      + "maxUlp=\(maxUlp)")

    var summary: [String: Any] = [
      "case": report.caseName,
      "caseJsonSha256": caseDigest,
      "sequentialTrajectory": trajectoryDigest(report.sequential),
      "sequentialBest": [
        sequential.best.c10Nanometers, sequential.best.c12Nanometers,
        sequential.best.phi12Radians,
      ],
      "sequentialLoss": report.sequential.bestLoss,
    ]
    if let engine = report.batchedEngine {
      summary["batchedEngineTrajectory"] = trajectoryDigest(engine)
    }
    if let entry = report.comparisonEngine ?? report.comparison {
      summary["comparison"] = [
        "trialsWithDifferentFloat32Point": entry.trialsWithDifferentFloat32Point,
        "maxLossDifferenceAtSameIndex": entry.maxLossDifferenceAtSameIndex,
        "bestSoFarDifferentCount": entry.bestSoFarDifferentCount,
        "finalBestLossDelta": entry.finalBestLossDelta,
      ]
    }
    let summaryData = try JSONSerialization.data(
      withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
    try summaryData.write(to: outDirectory.appendingPathComponent("fit-trajectory-summary.json"))
  }

  /// SHA-256 over the recorded float32 points, losses and stages of one run.
  fileprivate static func trajectoryDigest(_ run: RunReport) -> String {
    var hasher = SHA256()
    for trial in run.trials {
      hasher.update(data: Data(trialKey(trial).utf8))
      var loss = trial.loss.bitPattern.littleEndian
      withUnsafeBytes(of: &loss) { hasher.update(data: Data($0)) }
      hasher.update(data: Data(trial.stage.utf8))
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  fileprivate static func point(of trial: Trial) -> SSBOptimizationPoint {
    SSBOptimizationPoint(
      c10Nanometers: trial.c10Nanometers,
      c12Nanometers: trial.c12Nanometers,
      phi12Radians: trial.phi12Radians)
  }

  fileprivate static func trialKey(_ trial: Trial) -> String {
    let c10 = Float(trial.c10Nanometers).bitPattern
    let c12 = Float(trial.c12Nanometers).bitPattern
    let phi = Float(trial.phi12Radians).bitPattern
    return "\(c10)-\(c12)-\(phi)"
  }

  fileprivate static func floatKey(_ point: SSBOptimizationPoint) -> String {
    let c10 = Float(point.c10Nanometers).bitPattern
    let c12 = Float(point.c12Nanometers).bitPattern
    let phi = Float(point.phi12Radians).bitPattern
    return "\(c10)-\(c12)-\(phi)"
  }
}
