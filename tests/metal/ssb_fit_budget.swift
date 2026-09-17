// SSB trial-budget sensitivity harness (evidence only, no gate).
//
// The production fit is 200 TPE trials followed by Nelder-Mead refinement.
// This harness measures what each block of trials actually buys on the real
// 8937-BF ARINA artifact, without changing the objective, the evaluator, the
// summation order or any search arithmetic: every arm runs the same
// `MetalSSBEngine.phaseVariance` closure and the same `SSBOptimizer` code path
// the production entry point uses. Only `globalTrials` changes, plus one
// refinement-only arm that starts at the recorded 200-trial optimum.
//
// Arms:
//   tpe25/tpe50/tpe100/tpe200 : globalTrials = 25/50/100/200, then NM
//   nmWarm                    : globalTrials = 0 (no TPE), NM from the
//                               recorded 8937-BF 200-trial optimum
//
// Because the seed and the evaluator are fixed, the TPE prefixes nest: the
// first 25 trials of tpe200 are exactly the tpe25 trials. The harness runs the
// whole sequence forward and then in reverse (ABBA pairing for wall time) and
// records both, so the losses double as an order-independence control.
//
// Usage: ssb-fit-budget <case-dir> <out-dir> <warm-start.json>

import CryptoKit
import Foundation
import Metal
import MetalSSBKernels

private func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data("ssb-fit-budget: \(message)\n".utf8))
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

private struct ArmReport: Encodable {
  let arm: String
  let pass: String
  let globalTrials: Int
  let startC10Nanometers: Double
  let startC12Nanometers: Double
  let startPhi12Radians: Double
  let bestC10Nanometers: Double
  let bestC12Nanometers: Double
  let bestPhi12Radians: Double
  let bestLoss: Double
  let totalEvaluations: Int
  let refinementEvaluations: Int
  let elapsedSeconds: Double
  let loadAverageBefore: Double
  let loadAverageAfter: Double
  let trials: [Trial]
}

private struct BudgetReport: Encodable {
  let caseName: String
  let scanSide: Int
  let logicalBrightfieldCount: Int
  let activeBrightfieldCount: Int
  let seed: UInt64
  let start: [String: Double]
  let warmStart: [String: Double]
  let caseJsonSha256: String
  let loadAverageAtStart: Double
  let loadAverageAtEnd: Double
  let arms: [ArmReport]
}

private func loadAverage() -> Double {
  var loads = [Double](repeating: 0, count: 3)
  getloadavg(&loads, 3)
  return loads[0]
}

private func sha256Hex(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

@main enum SSBFitBudget {
  static func main() throws {
    let arguments = CommandLine.arguments
    guard arguments.count >= 4 else {
      fail("usage: ssb-fit-budget <case-dir> <out-dir> <warm-start.json>")
    }
    let caseDirectory = URL(fileURLWithPath: arguments[1])
    let outDirectory = URL(fileURLWithPath: arguments[2])
    let warmStartPath = URL(fileURLWithPath: arguments[3])
    try? FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)

    let caseData = try Data(contentsOf: caseDirectory.appendingPathComponent("case.json"))
    let payload = try JSONDecoder().decode(CaseFile.self, from: caseData)
    let side = payload.scan_shape[0]
    guard payload.scan_shape[0] == payload.scan_shape[1] else { fail("scan must be square") }
    let counts = try Data(
      contentsOf: caseDirectory.appendingPathComponent("source/bf_columns.u16"),
      options: .mappedIfSafe)
    guard counts.count == payload.bf_count * side * side * MemoryLayout<UInt16>.size else {
      fail("bf_columns.u16 has an unexpected size")
    }
    guard let device = MTLCreateSystemDefaultDevice() else { fail("an Apple GPU is required") }
    let source = counts.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    let engine = try MetalSSBEngine(
      device: device, geometry: payload.metal_geometry, cacheBudgetBytes: nil)
    try engine.prepare(brightfield: source, countType: .uint16)

    // Warm start: the recorded 8937-BF 200-trial optimum of the production fit.
    let warmData = try Data(contentsOf: warmStartPath)
    guard
      let warmRoot = try JSONSerialization.jsonObject(with: warmData) as? [String: Any],
      let warmSequential = warmRoot["sequential"] as? [String: Any],
      let warmC10 = warmSequential["bestC10Nanometers"] as? Double,
      let warmC12 = warmSequential["bestC12Nanometers"] as? Double,
      let warmPhi = warmSequential["bestPhi12Radians"] as? Double
    else {
      fail("warm-start file has no sequential.best* point")
    }

    let start = SSBOptimizationPoint(c10Nanometers: 0, c12Nanometers: 50, phi12Radians: 0)
    let warmStart = SSBOptimizationPoint(
      c10Nanometers: warmC10, c12Nanometers: warmC12, phi12Radians: warmPhi)
    let seed: UInt64 = 42

    func loss(_ point: SSBOptimizationPoint) throws -> Double {
      Double(
        try engine.phaseVariance(
          aberrations: MetalSSBAberrations(
            c10Nanometers: Float(point.c10Nanometers),
            c12Nanometers: Float(point.c12Nanometers),
            phi12Radians: Float(point.phi12Radians))).loss)
    }

    func runArm(
      name: String, trials: Int, startPoint: SSBOptimizationPoint, pass: String
    ) throws -> ArmReport {
      let loadBefore = loadAverage()
      let optimizer = SSBOptimizer(globalTrials: trials, seed: seed)
      let result = try optimizer.run(start: startPoint, evaluate: loss)
      let loadAfter = loadAverage()
      let recorded = result.trials.enumerated().map { index, trial in
        Trial(
          index: index, stage: trial.stage,
          c10Nanometers: trial.point.c10Nanometers,
          c12Nanometers: trial.point.c12Nanometers,
          phi12Radians: trial.point.phi12Radians,
          loss: trial.loss)
      }
      print(
        String(
          format: "arm %@ pass %@ trials %d evals %d loss %.12g elapsed %.3fs load %.2f->%.2f",
          name, pass, trials, result.trials.count, result.loss, result.elapsedSeconds,
          loadBefore, loadAfter))
      fflush(stdout)
      return ArmReport(
        arm: name, pass: pass, globalTrials: result.globalTrials,
        startC10Nanometers: startPoint.c10Nanometers,
        startC12Nanometers: startPoint.c12Nanometers,
        startPhi12Radians: startPoint.phi12Radians,
        bestC10Nanometers: result.best.c10Nanometers,
        bestC12Nanometers: result.best.c12Nanometers,
        bestPhi12Radians: result.best.phi12Radians,
        bestLoss: result.loss,
        totalEvaluations: result.trials.count,
        refinementEvaluations: result.refinementEvaluations,
        elapsedSeconds: result.elapsedSeconds,
        loadAverageBefore: loadBefore,
        loadAverageAfter: loadAfter,
        trials: recorded)
    }

    let loadAtStart = loadAverage()
    var arms: [ArmReport] = []
    let forward: [(String, Int, SSBOptimizationPoint)] = [
      ("tpe25", 25, start),
      ("tpe50", 50, start),
      ("tpe100", 100, start),
      ("tpe200", 200, start),
      ("nmWarm", 0, warmStart),
    ]
    for (name, trials, point) in forward {
      arms.append(try runArm(name: name, trials: trials, startPoint: point, pass: "forward"))
    }
    for (name, trials, point) in forward.reversed() {
      arms.append(try runArm(name: name, trials: trials, startPoint: point, pass: "reverse"))
    }
    let loadAtEnd = loadAverage()

    let report = BudgetReport(
      caseName: caseDirectory.lastPathComponent,
      scanSide: side,
      logicalBrightfieldCount: payload.bf_count,
      activeBrightfieldCount: payload.active_bf_count,
      seed: seed,
      start: [
        "C10": start.c10Nanometers, "C12": start.c12Nanometers,
        "phi12": start.phi12Radians,
      ],
      warmStart: [
        "C10": warmStart.c10Nanometers, "C12": warmStart.c12Nanometers,
        "phi12": warmStart.phi12Radians,
      ],
      caseJsonSha256: sha256Hex(caseData),
      loadAverageAtStart: loadAtStart,
      loadAverageAtEnd: loadAtEnd,
      arms: arms)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(
      to: outDirectory.appendingPathComponent("fit-budget.json"))
    print("wrote \(outDirectory.appendingPathComponent("fit-budget.json").path)")
  }
}
