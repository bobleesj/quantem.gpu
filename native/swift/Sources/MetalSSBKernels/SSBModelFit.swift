import Foundation
import Metal

/// What a fit maximizes or minimizes.
public enum SSBObjective: String, Codable, Sendable {
  /// The least-squares agreement of the transfer model with the measured double-overlap data, the only
  /// objective a fit runs.
  case agreement
  /// Retired: the older native SSB objective, the variance of the per-pixel phases. Saved fits that used it still
  /// decode; `SSBModelFit` refuses it.
  case phaseVariance = "phase_variance"
}

/// The settings of `SSBModelFit.findAberrations`, Python `SSB.find_aberrations` in one vocabulary:
///
/// | Options | Method |
/// |---|---|
/// | `SSBFitOptions()` | 200 primary TPE trials, then joint fifth-order Newton with tilt/depth |
/// | `SSBFitOptions(maximumOrder: 1)` | original low-order model agreement with tilt/depth |
/// | `SSBFitOptions(tilt: false)` | model agreement, sample tilt and depth spread held at 0 |
///
/// The older phase-variance search (`objective: .phaseVariance`) is retired; saved options naming it decode, and
/// `validate()` refuses them.
public struct SSBFitOptions: Codable, Equatable, Sendable {
  public var tilt: Bool
  public var objective: SSBObjective
  public var trials: Int
  public var seed: Int
  /// Also run the 200-trial search at depth spread 0 that only measures `tiltFitGain`.
  public var compareStandard: Bool
  public var tiltLimitMrad: Double
  /// Standard fits initialize the low-order model, then jointly refine through fifth order.
  /// Set 1 to retain the original low-order fit, including its reference trajectory.
  public var maximumOrder: Int
  public var maxHigherOrderEvaluations: Int
  /// The fifth-order search: exact gradients, converged on every BF pixel after a start on every 8th one.
  /// Saved runs without this key used the finite-difference Newton search, which stays available as the
  /// Python-trajectory reference.
  public var higherOrderSearch: SSBModelFit.HigherOrderSearch = .exactGradientCoarseToFine

  public init(
    tilt: Bool = true, objective: SSBObjective = .agreement, trials: Int = 200, seed: Int = 42,
    compareStandard: Bool? = nil, tiltLimitMrad: Double = 25,
    maximumOrder: Int? = nil, maxHigherOrderEvaluations: Int = 5000
  ) {
    self.tilt = tilt
    self.objective = objective
    self.trials = trials
    self.seed = seed
    self.maximumOrder = maximumOrder ?? (tilt && objective == .agreement ? 5 : 1)
    self.maxHigherOrderEvaluations = maxHigherOrderEvaluations
    self.compareStandard = compareStandard ?? (self.maximumOrder == 1)
    self.tiltLimitMrad = tiltLimitMrad
  }

  private enum CodingKeys: String, CodingKey {
    case tilt, objective, trials, seed, compareStandard, tiltLimitMrad, maximumOrder,
      maxHigherOrderEvaluations, higherOrderSearch
  }

  /// Old saved runs describe the original primary fit, never an implicit fifth-order fit.
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    tilt = try c.decode(Bool.self, forKey: .tilt)
    objective = try c.decode(SSBObjective.self, forKey: .objective)
    trials = try c.decode(Int.self, forKey: .trials)
    seed = try c.decode(Int.self, forKey: .seed)
    compareStandard = try c.decode(Bool.self, forKey: .compareStandard)
    tiltLimitMrad = try c.decode(Double.self, forKey: .tiltLimitMrad)
    maximumOrder = try c.decodeIfPresent(Int.self, forKey: .maximumOrder) ?? 1
    maxHigherOrderEvaluations =
      try c.decodeIfPresent(Int.self, forKey: .maxHigherOrderEvaluations) ?? 5000
    higherOrderSearch =
      try c.decodeIfPresent(SSBModelFit.HigherOrderSearch.self, forKey: .higherOrderSearch)
      ?? .finiteDifferenceNewton
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
    guard objective == .agreement else {
      throw MetalSSBError.invalidGeometry(
        "The phase-variance SSB fit is retired. Fit with the model agreement objective (the default)."
      )
    }
    guard [1, 5].contains(maximumOrder), maxHigherOrderEvaluations > 0 else {
      throw MetalSSBError.invalidGeometry(
        "Choose maximumOrder 1 or 5 and a positive higher-order evaluation cap.")
    }
    if maximumOrder == 5 && !tilt {
      throw MetalSSBError.invalidGeometry(
        "The standard fifth-order fit jointly refines tilt and depth. Use maximumOrder: 1 for the primary-only alternatives."
      )
    }
    if maximumOrder == 5 && tiltLimitMrad != 25 {
      throw MetalSSBError.invalidGeometry(
        "The standard fifth-order fit uses its shared ±25 mrad tilt bounds. Use maximumOrder: 1 for a custom primary tilt range."
      )
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
  public let trialCount: Int
  public var timings: [String: Double]
  /// The trust-region Newton starts of the primary fit (iterations, objective evaluations, convergence).
  public var newtonStarts: [SSBAgreementSearch.NewtonStart]? = nil
  /// The branch whose fit this one is the exact 180-degree image of (`SSBRotationBranches.oppositeBranch`);
  /// nil for a fit searched at its own rotation.
  public var mappedFromBranch: Int? = nil
  /// The bounded fifth-order Newton trajectory; nil for a primary-only fit.
  /// Agreements in this trace are divided by `higherOrderNormalization`.
  public var higherOrderRefinement: SSBFifthOrderSearch.Result? = nil
  /// Initial agreement for the shared fifth-order model, using the strict 0.2...0.9 inverse-Angstrom band.
  public var higherOrderNormalization: Double? = nil
}

/// Both 180-degree branches fitted independently and the branch kept (Python `RotationBranches`).
public struct SSBBranchedFit: Codable, Sendable {
  /// One fit (the rotation as given), or the given rotation then the opposite branch.
  public let fits: [SSBFit]
  public let branches: [SSBRotationBranches.Branch]
  public let chosen: Int
  public let kept: Int
  public let decidedBy: SSBRotationBranches.Decision
  public let columnCount: Int
  /// Branch diagnostics retain the initial native-grid fits; refinement holds the selected rotation.
  public var refinedFit: SSBFit? = nil
  public var fit: SSBFit { refinedFit ?? fits[kept] }
  /// Whether the 180-degree branches were compared.
  public var comparedBranches: Bool { fits.count == 2 }
}

/// Python `SSB.find_aberrations` on a prepared native session: the default fit, optionally on both
/// 180-degree branches of an estimated rotation with the branch test.
public enum SSBModelFit {
  public typealias Progress = (_ stage: String, _ completed: Int, _ total: Int) -> Void

  /// Fit at the model's current rotation.
  public static func fit(
    model: MetalSSBJointModel, options: SSBFitOptions,
    progress: Progress = { _, _, _ in }, onInitialFit: (SSBFit) throws -> Void = { _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> SSBFit {
    try options.validate()
    let primary = try fitPrimary(
      model: model, options: options,
      progress: progress, isCancelled: isCancelled)
    try onInitialFit(primary)
    return try refineHigherOrder(
      primary, model: model, options: options,
      progress: progress, isCancelled: isCancelled)
  }

  private static func fitPrimary(
    model: MetalSSBJointModel, options: SSBFitOptions,
    progress: Progress, isCancelled: () -> Bool
  ) throws -> SSBFit {
    try options.validate()
    var count = 0
    let total = options.trials
    // Wall time of the search's objective evaluations; the rest of the search is the host's proposals.
    var searchObjectiveSeconds = 0.0
    func evaluate(_ points: [[Double]]) throws -> [Double] {
      if isCancelled() { throw CancellationError() }
      let started = Date()
      defer { if count <= total { searchObjectiveSeconds += Date().timeIntervalSince(started) } }
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
    // The Newton climbs on the exact gradient and Hessian of the same objective, asking for the derivatives
    // only at the centers whose step it uses.
    // Newton centers by pass: fused value and derivatives, value only, and derivatives after a value-only pass.
    var fusedCenters = 0
    var valueCenters = 0
    var derivativeCenters = 0
    var latestFused = false
    func values(_ rows: [[Double]], withDerivatives: Bool) throws -> [Double] {
      if isCancelled() { throw CancellationError() }
      if withDerivatives { fusedCenters += rows.count } else { valueCenters += rows.count }
      latestFused = withDerivatives
      let values = try model.lowOrderValues(
        rows, withDerivatives: withDerivatives, isCancelled: isCancelled)
      count += rows.count
      progress("Newton refinement", total, total)
      return values
    }
    func derivatives(_ indices: [Int]) throws -> [SSBAgreementSearch.Derivatives] {
      if isCancelled() { throw CancellationError() }
      if !latestFused { derivativeCenters += indices.count }
      return try model.lowOrderDerivatives(at: indices, isCancelled: isCancelled).map {
        SSBAgreementSearch.Derivatives(value: $0.value, gradient: $0.gradient, hessian: $0.hessian)
      }
    }
    var timings: [String: Double] = [:]
    let rotation = model.scanDetectorRotationDegrees
    if options.tilt {
      var standard: SSBAgreementSearch.Result?
      if options.compareStandard {
        let started = Date()
        standard = try SSBAgreementSearch.standardComparison(
          evaluate, seed: options.seed, isCancelled: isCancelled)
        timings["standard_comparison_s"] = Date().timeIntervalSince(started)
        count = 0
        searchObjectiveSeconds = 0
      }
      let result = try withoutActuallyEscaping(values) { values in
        try withoutActuallyEscaping(derivatives) { derivatives in
          try SSBAgreementSearch.fitTilt(
            evaluate, trials: options.trials, seed: options.seed,
            tiltLimitMrad: options.tiltLimitMrad,
            staged: SSBAgreementSearch.StagedDerivatives(values: values, derivatives: derivatives),
            isCancelled: isCancelled)
        }
      }
      timings["coarse_search_s"] = result.searchSeconds
      timings["coarse_search_objective_s"] = searchObjectiveSeconds
      timings["refinement_s"] = result.refinementSeconds
      timings["newton_fused_centers"] = Double(fusedCenters)
      timings["newton_value_only_centers"] = Double(valueCenters)
      timings["newton_derivative_only_centers"] = Double(derivativeCenters)
      let p = result.point
      var fit = SSBFit(
        scanDetectorRotationDegrees: rotation,
        point: SSBModelPoint(
          c10Angstroms: p[0], c12Angstroms: p[1], phi12Radians: p[2], sampleTiltMrad: [p[3], p[4]],
          depthSpreadAngstroms: p[5]),
        options: options, sampleAgreement: result.agreement, standardAgreement: standard?.agreement,
        tiltFitGain: standard.map { result.agreement / $0.agreement },
        trialCount: options.trials, timings: timings)
      fit.newtonStarts = result.newtonStarts
      return fit
    } else {
      let result = try withoutActuallyEscaping(values) { values in
        try withoutActuallyEscaping(derivatives) { derivatives in
          try SSBAgreementSearch.fitStandard(
            evaluate, trials: options.trials, seed: options.seed,
            staged: SSBAgreementSearch.StagedDerivatives(values: values, derivatives: derivatives),
            isCancelled: isCancelled)
        }
      }
      timings["coarse_search_s"] = result.searchSeconds
      timings["coarse_search_objective_s"] = searchObjectiveSeconds
      timings["refinement_s"] = result.refinementSeconds
      timings["newton_fused_centers"] = Double(fusedCenters)
      timings["newton_value_only_centers"] = Double(valueCenters)
      timings["newton_derivative_only_centers"] = Double(derivativeCenters)
      let p = result.point
      var fit = SSBFit(
        scanDetectorRotationDegrees: rotation,
        point: SSBModelPoint(c10Angstroms: p[0], c12Angstroms: p[1], phi12Radians: p[2]),
        options: options, sampleAgreement: result.agreement, standardAgreement: nil,
        tiltFitGain: nil,
        trialCount: options.trials, timings: timings)
      fit.newtonStarts = result.newtonStarts
      return fit
    }
  }

  /// Fit `rotation`, give the opposite branch its fit, reconstruct each fit's phase at the native sampling
  /// and keep the branch the dark-field column test (else the phase tail) selects. The model is left at the
  /// kept rotation. The standard comparison, which only measures `tiltFitGain` and depends on the rotation
  /// and seed alone, runs once, for the kept branch, after the decision (Python `_measure_tilt_fit_gain`).
  ///
  /// The agreement objective is exactly symmetric between the branches
  /// (`SSBRotationBranches.oppositeBranch`): every point of one branch has an equal-agreement point on the
  /// other, so the opposite branch's fit is the mapped fit of `rotation` (`mappedFromBranch` 0), its
  /// agreement evaluated at its own rotation, not a second search.
  ///
  /// `checkRotation: false` fits `rotation` alone (Python `check_rotation=False`), the standard comparison
  /// included.
  public static func findAberrations(
    model: MetalSSBJointModel, rotationDegrees: Double,
    darkField: [Float]?, scanSampling: (Double, Double), options: SSBFitOptions = SSBFitOptions(),
    checkRotation: Bool = true, keepFirst: Bool = false, progress: Progress = { _, _, _ in },
    onInitialFit: (SSBFit) throws -> Void = { _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> SSBBranchedFit {
    if !checkRotation {
      try model.retarget(scanDetectorRotationDegrees: rotationDegrees)
      let fitted = try fit(
        model: model, options: options, progress: progress,
        onInitialFit: onInitialFit, isCancelled: isCancelled
      )
      return SSBBranchedFit(
        fits: [fitted], branches: [], chosen: 0, kept: 0, decidedBy: .phaseTail, columnCount: 0)
    }
    let rotations = [rotationDegrees, SSBRotationBranches.opposite(rotationDegrees)]
    var fits: [SSBFit] = []
    var phases: [[Float]] = []
    var size = 0
    var branchOptions = options
    branchOptions.compareStandard = false
    for (index, rotation) in rotations.enumerated() {
      try model.retarget(scanDetectorRotationDegrees: rotation)
      let fitted: SSBFit
      if index == 1 {
        let started = Date()
        let point = SSBRotationBranches.oppositeBranch(fits[0].point)
        let agreement = try model.agreements([point], isCancelled: isCancelled)[0]
        var opposite = SSBFit(
          scanDetectorRotationDegrees: rotation, point: point, options: branchOptions,
          sampleAgreement: agreement, standardAgreement: nil, tiltFitGain: nil,
          trialCount: fits[0].trialCount,
          timings: ["mapped_agreement_s": Date().timeIntervalSince(started)])
        opposite.mappedFromBranch = 0
        fitted = opposite
      } else {
        fitted = try fitPrimary(
          model: model, options: branchOptions, progress: progress, isCancelled: isCancelled)
      }
      let started = Date()
      let phase = try model.reconstruct(fitted.point, upsamplingFactor: 1, isCancelled: isCancelled)
      var branchFit = fitted
      branchFit.timings["native_phase_s"] = Date().timeIntervalSince(started)
      size = phase.rows
      phases.append(
        Array(
          UnsafeBufferPointer(
            start: phase.phase.contents().assumingMemoryBound(to: Float.self),
            count: phase.rows * phase.columns)))
      fits.append(branchFit)
    }
    let decision = SSBRotationBranches.compare(
      phases: phases, rotations: rotations, rows: size, columns: size, darkField: darkField,
      scanSampling: scanSampling, keepFirst: keepFirst)
    try model.retarget(scanDetectorRotationDegrees: rotations[decision.kept])
    if options.tilt && options.compareStandard, let agreement = fits[decision.kept].sampleAgreement
    {
      let started = Date()
      let standard = try SSBAgreementSearch.standardComparison(
        { points in
          try model.agreements(
            points.map {
              SSBModelPoint(c10Angstroms: $0[0], c12Angstroms: $0[1], phi12Radians: $0[2])
            })
        }, seed: options.seed, isCancelled: isCancelled)
      let kept = fits[decision.kept]
      var timings = kept.timings
      timings["standard_comparison_s"] = Date().timeIntervalSince(started)
      fits[decision.kept] = SSBFit(
        scanDetectorRotationDegrees: kept.scanDetectorRotationDegrees, point: kept.point,
        options: options,
        sampleAgreement: agreement, standardAgreement: standard.agreement,
        tiltFitGain: standard.agreement > 0 ? agreement / standard.agreement : .nan,
        trialCount: kept.trialCount, timings: timings)
      fits[decision.kept].newtonStarts = kept.newtonStarts
      fits[decision.kept].mappedFromBranch = kept.mappedFromBranch
    }
    var result = SSBBranchedFit(
      fits: fits, branches: decision.branches, chosen: decision.chosen, kept: decision.kept,
      decidedBy: decision.decidedBy, columnCount: decision.columns)
    try onInitialFit(result.fit)
    if options.maximumOrder > 1 {
      result.refinedFit = try refineHigherOrder(
        result.fit, model: model, options: options,
        progress: progress, isCancelled: isCancelled)
    }
    return result
  }

  /// Refine a low-order fit through fifth order without repeating its seeded search: the same result as
  /// `findAberrations` with `maximumOrder: 5`, which reruns the low-order search to the same point first.
  /// `model` must be at the kept branch's rotation, as `findAberrations` leaves it.
  public static func refineToFifthOrder(
    _ fitted: SSBBranchedFit, model: MetalSSBJointModel, options: SSBFitOptions = SSBFitOptions(),
    progress: Progress = { _, _, _ in }, isCancelled: () -> Bool = { false }
  ) throws -> SSBBranchedFit {
    try options.validate()
    let kept = fitted.fits[fitted.kept]
    guard options.maximumOrder == 5, kept.options.tilt, kept.options.objective == .agreement,
      kept.higherOrderRefinement == nil
    else {
      throw MetalSSBError.invalidGeometry(
        "Refine a low-order tilt-corrected model fit with maximumOrder: 5.")
    }
    guard model.scanDetectorRotationDegrees == kept.scanDetectorRotationDegrees else {
      throw MetalSSBError.invalidGeometry(
        "The joint model is at another rotation than the kept branch. Fit again before refining.")
    }
    var result = fitted
    result.refinedFit = try refineHigherOrder(
      kept, model: model, options: options, progress: progress, isCancelled: isCancelled)
    return result
  }

  private static func refineHigherOrder(
    _ primary: SSBFit, model: MetalSSBJointModel, options: SSBFitOptions,
    progress: Progress, isCancelled: () -> Bool
  ) throws -> SSBFit {
    guard options.maximumOrder == 5 else { return primary }
    guard model.bandInverseAngstroms == Float(0.2)...Float(0.9) else {
      throw MetalSSBError.invalidGeometry(
        "The fifth-order fit requires the 0.2...0.9 inverse-Angstrom fitting band. Prepare the joint model with its default band."
      )
    }
    let started = Date()
    let refined = try refineFifthOrder(
      primary.point, model: model, search: options.higherOrderSearch,
      maxEvaluations: options.maxHigherOrderEvaluations,
      progress: { progress("Fifth-order Newton", $0, $1) }, isCancelled: isCancelled)
    var timings = primary.timings
    timings["higher_order_refinement_s"] = Date().timeIntervalSince(started)
    var fit = SSBFit(
      scanDetectorRotationDegrees: primary.scanDetectorRotationDegrees,
      point: refined.point, options: options,
      sampleAgreement: refined.agreement, standardAgreement: nil, tiltFitGain: nil,
      trialCount: primary.trialCount, timings: timings)
    fit.higherOrderRefinement = refined.result
    fit.higherOrderNormalization = refined.normalization
    return fit
  }
}
