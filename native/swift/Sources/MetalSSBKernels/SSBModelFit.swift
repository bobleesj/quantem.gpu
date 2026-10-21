import Foundation
import Metal

/// What a fit maximizes or minimizes.
public enum SSBObjective: String, Codable, Sendable {
  /// The least-squares agreement of the transfer model with the measured double-overlap data (the default).
  case agreement
  /// The older SSB objective: the variance of the per-pixel phases, minimized.
  case phaseVariance = "phase_variance"
}

/// The settings of `SSBModelFit.findAberrations`, Python `SSB.find_aberrations` in one vocabulary:
///
/// | Options | Method |
/// |---|---|
/// | `SSBFitOptions()` | model agreement, tilt-corrected (sample tilt and depth spread) |
/// | `SSBFitOptions(tilt: false)` | model agreement, sample tilt and depth spread held at 0 |
/// | `SSBFitOptions(tilt: false, objective: .phaseVariance)` | the older phase-variance SSB |
public struct SSBFitOptions: Codable, Equatable, Sendable {
  public var tilt: Bool
  public var objective: SSBObjective
  public var trials: Int
  public var seed: Int
  /// Also run the 200-trial search at depth spread 0 that only measures `tiltFitGain`.
  public var compareStandard: Bool
  public var tiltLimitMrad: Double

  public init(
    tilt: Bool = true, objective: SSBObjective = .agreement, trials: Int = 200, seed: Int = 42,
    compareStandard: Bool = true, tiltLimitMrad: Double = 25
  ) {
    self.tilt = tilt
    self.objective = objective
    self.trials = trials
    self.seed = seed
    self.compareStandard = compareStandard
    self.tiltLimitMrad = tiltLimitMrad
  }

  /// A short name for the method, as the results and the app state it.
  public var method: String {
    switch (tilt, objective) {
    case (true, _): return "model, tilt-corrected"
    case (false, .agreement): return "model, tilt 0"
    case (false, .phaseVariance): return "phase variance"
    }
  }

  func validate() throws {
    guard trials > 0, tiltLimitMrad.isFinite, tiltLimitMrad > 0 else {
      throw MetalSSBError.invalidGeometry("Use at least one trial and a positive tilt limit.")
    }
    if tilt && objective == .phaseVariance {
      throw MetalSSBError.invalidGeometry(
        "The phase-variance objective does not see the sample tilt. Use tilt: false with it, or the agreement objective.")
    }
  }
}

/// One fit at one scan-detector rotation.
public struct SSBFit: Codable, Sendable {
  public let scanDetectorRotationDegrees: Double
  public let point: SSBModelPoint
  public let options: SSBFitOptions
  /// The model agreement at the fit (agreement objective), larger is better.
  public let sampleAgreement: Double?
  /// The best agreement of the standard comparison and the ratio of the two (tilt fits only).
  public let standardAgreement: Double?
  public let tiltFitGain: Double?
  /// The phase variance at the fit (phase-variance objective).
  public let phaseVarianceLoss: Double?
  public let trialCount: Int
  public let timings: [String: Double]
}

/// Both 180-degree branches fitted independently and the branch kept (Python `RotationBranches`).
public struct SSBBranchedFit: Codable, Sendable {
  public let fits: [SSBFit]
  public let branches: [SSBRotationBranches.Branch]
  public let chosen: Int
  public let kept: Int
  public let decidedBy: SSBRotationBranches.Decision
  public let columnCount: Int
  public var fit: SSBFit { fits[kept] }
}

/// Python `SSB.find_aberrations` on a prepared native session: the default fit, optionally on both
/// 180-degree branches of an estimated rotation with the branch test.
public enum SSBModelFit {
  public typealias Progress = (_ stage: String, _ completed: Int, _ total: Int) -> Void

  /// Fit at the model's current rotation. `engine` serves the phase-variance objective.
  public static func fit(
    model: MetalSSBJointModel, engine: MetalSSBEngine, options: SSBFitOptions,
    progress: Progress = { _, _, _ in }, isCancelled: () -> Bool = { false }
  ) throws -> SSBFit {
    try options.validate()
    var count = 0
    let total = options.trials
    func evaluate(_ points: [[Double]]) throws -> [Double] {
      if isCancelled() { throw CancellationError() }
      let values = try model.agreements(
        points.map {
          SSBModelPoint(
            c10Angstroms: $0[0], c12Angstroms: $0[1], phi12Radians: $0[2],
            sampleTiltMrad: [$0[3], $0[4]], depthSpreadAngstroms: $0[5])
        })
      count += points.count
      progress(count <= total ? "Search trials" : "Newton refinement", min(count, total), total)
      return values
    }
    var timings: [String: Double] = [:]
    let rotation = model.scanDetectorRotationDegrees
    switch (options.tilt, options.objective) {
    case (true, _):
      var standard: SSBAgreementSearch.Result?
      if options.compareStandard {
        let started = Date()
        standard = try SSBAgreementSearch.standardComparison(
          evaluate, seed: options.seed, isCancelled: isCancelled)
        timings["standard_comparison_s"] = Date().timeIntervalSince(started)
        count = 0
      }
      let result = try SSBAgreementSearch.fitTilt(
        evaluate, trials: options.trials, seed: options.seed, tiltLimitMrad: options.tiltLimitMrad,
        isCancelled: isCancelled)
      timings["coarse_search_s"] = result.searchSeconds
      timings["refinement_s"] = result.refinementSeconds
      let p = result.point
      return SSBFit(
        scanDetectorRotationDegrees: rotation,
        point: SSBModelPoint(
          c10Angstroms: p[0], c12Angstroms: p[1], phi12Radians: p[2], sampleTiltMrad: [p[3], p[4]],
          depthSpreadAngstroms: p[5]),
        options: options, sampleAgreement: result.agreement, standardAgreement: standard?.agreement,
        tiltFitGain: standard.map { result.agreement / $0.agreement }, phaseVarianceLoss: nil,
        trialCount: options.trials, timings: timings)
    case (false, .agreement):
      let result = try SSBAgreementSearch.fitStandard(
        evaluate, trials: options.trials, seed: options.seed, isCancelled: isCancelled)
      timings["coarse_search_s"] = result.searchSeconds
      timings["refinement_s"] = result.refinementSeconds
      let p = result.point
      return SSBFit(
        scanDetectorRotationDegrees: rotation,
        point: SSBModelPoint(c10Angstroms: p[0], c12Angstroms: p[1], phi12Radians: p[2]),
        options: options, sampleAgreement: result.agreement, standardAgreement: nil, tiltFitGain: nil,
        phaseVarianceLoss: nil, trialCount: options.trials, timings: timings)
    case (false, .phaseVariance):
      // The native phase-variance search (seeded TPE and Nelder-Mead); its proposals are its own.
      let started = Date()
      let result = try engine.optimize(
        start: MetalSSBAberrations(c10Angstroms: 0, c12Angstroms: 50, phi12Radians: 0),
        scanDetectorRotationDegrees: Float(rotation), globalTrials: options.trials,
        seed: UInt64(options.seed),
        progress: { progress($0.stage == "tpe" ? "Search trials" : "Refinement", $0.completed, $0.total) },
        isCancelled: isCancelled)
      timings["fit_s"] = Date().timeIntervalSince(started)
      return SSBFit(
        scanDetectorRotationDegrees: rotation,
        point: SSBModelPoint(
          c10Angstroms: result.best.c10Angstroms, c12Angstroms: result.best.c12Angstroms,
          phi12Radians: SSBAgreementSearch.wrapPhi(result.best.phi12Radians)),
        options: options, sampleAgreement: nil, standardAgreement: nil, tiltFitGain: nil,
        phaseVarianceLoss: result.loss, trialCount: options.trials, timings: timings)
    }
  }

  /// Fit `rotation` and the opposite branch independently with the same options, reconstruct each fit's
  /// phase at the native sampling and keep the branch the dark-field column test (else the phase tail)
  /// selects. The model is left at the kept rotation. The standard comparison, which only measures
  /// `tiltFitGain` and depends on the rotation and seed alone, runs once, for the kept branch, after the
  /// decision (Python `_measure_tilt_fit_gain`).
  public static func findAberrations(
    model: MetalSSBJointModel, engine: MetalSSBEngine, rotationDegrees: Double,
    darkField: [Float]?, scanSampling: (Double, Double), options: SSBFitOptions = SSBFitOptions(),
    keepFirst: Bool = false, progress: Progress = { _, _, _ in }, isCancelled: () -> Bool = { false }
  ) throws -> SSBBranchedFit {
    let rotations = [rotationDegrees, SSBRotationBranches.opposite(rotationDegrees)]
    var fits: [SSBFit] = []
    var phases: [[Float]] = []
    var size = 0
    var branchOptions = options
    branchOptions.compareStandard = false
    for (index, rotation) in rotations.enumerated() {
      try model.retarget(scanDetectorRotationDegrees: rotation)
      let fitted = try fit(
        model: model, engine: engine, options: branchOptions,
        progress: { stage, completed, total in
          progress("Branch \(index + 1)/2 · \(stage)", completed, total)
        }, isCancelled: isCancelled)
      let started = Date()
      let phase = try model.reconstruct(fitted.point, upsamplingFactor: 1, isCancelled: isCancelled)
      var timings = fitted.timings
      timings["native_phase_s"] = Date().timeIntervalSince(started)
      size = phase.rows
      phases.append(Array(UnsafeBufferPointer(
        start: phase.phase.contents().assumingMemoryBound(to: Float.self), count: phase.rows * phase.columns)))
      fits.append(
        SSBFit(
          scanDetectorRotationDegrees: fitted.scanDetectorRotationDegrees, point: fitted.point,
          options: fitted.options, sampleAgreement: fitted.sampleAgreement,
          standardAgreement: fitted.standardAgreement, tiltFitGain: fitted.tiltFitGain,
          phaseVarianceLoss: fitted.phaseVarianceLoss, trialCount: fitted.trialCount, timings: timings))
    }
    let decision = SSBRotationBranches.compare(
      phases: phases, rotations: rotations, rows: size, columns: size, darkField: darkField,
      scanSampling: scanSampling, keepFirst: keepFirst)
    try model.retarget(scanDetectorRotationDegrees: rotations[decision.kept])
    if options.tilt && options.compareStandard, let agreement = fits[decision.kept].sampleAgreement {
      let started = Date()
      let standard = try SSBAgreementSearch.standardComparison(
        { points in
          try model.agreements(
            points.map { SSBModelPoint(c10Angstroms: $0[0], c12Angstroms: $0[1], phi12Radians: $0[2]) })
        }, seed: options.seed, isCancelled: isCancelled)
      let kept = fits[decision.kept]
      var timings = kept.timings
      timings["standard_comparison_s"] = Date().timeIntervalSince(started)
      fits[decision.kept] = SSBFit(
        scanDetectorRotationDegrees: kept.scanDetectorRotationDegrees, point: kept.point, options: options,
        sampleAgreement: agreement, standardAgreement: standard.agreement,
        tiltFitGain: standard.agreement > 0 ? agreement / standard.agreement : .nan,
        phaseVarianceLoss: nil, trialCount: kept.trialCount, timings: timings)
    }
    return SSBBranchedFit(
      fits: fits, branches: decision.branches, chosen: decision.chosen, kept: decision.kept,
      decidedBy: decision.decidedBy, columnCount: decision.columns)
  }
}
