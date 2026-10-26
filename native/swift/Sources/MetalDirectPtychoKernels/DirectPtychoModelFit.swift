import Foundation
import Metal

/// What a fit maximizes or minimizes.
public enum DirectPtychoObjective: String, Codable, Sendable {
  /// The least-squares agreement of the transfer model with the measured double-overlap data, the only
  /// objective a fit runs.
  case agreement
  /// Retired: the older native SSB objective, the variance of the per-pixel phases. Saved fits that used it still
  /// decode; `DirectPtychoModelFit` refuses it.
  case phaseVariance = "phase_variance"
}

/// The settings of `DirectPtychoModelFit.findAberrations`, Python `SSB.find_aberrations` in one vocabulary:
///
/// | Options | Method |
/// |---|---|
/// | `DirectPtychoFitOptions()` | 200 primary TPE trials and exact low-order Newton, then the exact-gradient fifth-order search with tilt/depth |
/// | `DirectPtychoFitOptions(maximumOrder: 1)` | original low-order model agreement with tilt/depth |
/// | `DirectPtychoFitOptions(tilt: false)` | model agreement, sample tilt and depth spread held at 0 |
///
/// The older phase-variance search (`objective: .phaseVariance`) is retired; saved options naming it decode, and
/// `validate()` refuses them.
///
/// `higherOrderLimit` bounds each of the 22 higher-order coordinates of the fifth-order fit (C21 through C56) to
/// ±L rad of probe phase at the aperture edge: `wideHigherOrderLimit` (±20, the default; past about 14 the box no
/// longer binds) or `standardHigherOrderLimit` (±2). A wide fit is checked by default
/// (`checkHigherOrderConsistency`, `DirectPtychoHigherOrderConsistency`): it is refitted on two halves of the band
/// frequencies, and when the halves disagree the fit falls back to ±2 with its higher orders marked not trusted.
public struct DirectPtychoFitOptions: Codable, Equatable, Sendable {
  /// ±2 rad of phase at the aperture edge per higher-order coordinate: the original box.
  public static let standardHigherOrderLimit = 2.0
  /// ±20 rad: wide enough that the box no longer binds (it stops binding near 14).
  public static let wideHigherOrderLimit = 20.0
  /// The higher-order limits a fit accepts.
  public static let higherOrderLimits = [standardHigherOrderLimit, wideHigherOrderLimit]
  /// The limit of `DirectPtychoFitOptions()`.
  public static let defaultHigherOrderLimit = wideHigherOrderLimit

  public var tilt: Bool
  public var objective: DirectPtychoObjective
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
  /// Saved runs without this key used the retired finite-difference Newton (`.finiteDifferenceNewton`); they decode,
  /// and `validate()` refuses a fifth-order fit with it.
  public var higherOrderSearch: DirectPtychoModelFit.HigherOrderSearch = .exactGradientCoarseToFine
  /// The bound of each higher-order coordinate, in rad of probe phase at the aperture edge: one of
  /// `higherOrderLimits`. Saved runs without this key used the standard ±2.
  public var higherOrderLimit = DirectPtychoFitOptions.defaultHigherOrderLimit
  /// Check a wide (±20) fifth-order fit against its two half-band refits and fall back to ±2 when they disagree
  /// (`DirectPtychoHigherOrderConsistency`; two further fifth-order fits, and a third on fallback). Saved options without this
  /// key decode as false: those fits were not checked.
  public var checkHigherOrderConsistency = true

  public init(
    tilt: Bool = true, objective: DirectPtychoObjective = .agreement, trials: Int = 200,
    seed: Int = 42,
    compareStandard: Bool? = nil, tiltLimitMrad: Double = 25,
    maximumOrder: Int? = nil, maxHigherOrderEvaluations: Int = 5000,
    higherOrderLimit: Double = DirectPtychoFitOptions.defaultHigherOrderLimit,
    checkHigherOrderConsistency: Bool = true
  ) {
    self.higherOrderLimit = higherOrderLimit
    self.checkHigherOrderConsistency = checkHigherOrderConsistency
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
      maxHigherOrderEvaluations, higherOrderSearch, higherOrderLimit, checkHigherOrderConsistency
  }

  /// Old saved runs describe the original primary fit, never an implicit fifth-order fit. A retired
  /// `computeHigherOrderGuard` key is ignored.
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    tilt = try c.decode(Bool.self, forKey: .tilt)
    objective = try c.decode(DirectPtychoObjective.self, forKey: .objective)
    trials = try c.decode(Int.self, forKey: .trials)
    seed = try c.decode(Int.self, forKey: .seed)
    compareStandard = try c.decode(Bool.self, forKey: .compareStandard)
    tiltLimitMrad = try c.decode(Double.self, forKey: .tiltLimitMrad)
    maximumOrder = try c.decodeIfPresent(Int.self, forKey: .maximumOrder) ?? 1
    maxHigherOrderEvaluations =
      try c.decodeIfPresent(Int.self, forKey: .maxHigherOrderEvaluations) ?? 5000
    higherOrderSearch =
      try c.decodeIfPresent(DirectPtychoModelFit.HigherOrderSearch.self, forKey: .higherOrderSearch)
      ?? .finiteDifferenceNewton
    higherOrderLimit =
      try c.decodeIfPresent(Double.self, forKey: .higherOrderLimit) ?? Self.standardHigherOrderLimit
    checkHigherOrderConsistency =
      try c.decodeIfPresent(Bool.self, forKey: .checkHigherOrderConsistency) ?? false
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
      throw MetalDirectPtychoError.invalidGeometry(
        "Use at least one trial and a positive tilt limit.")
    }
    guard objective == .agreement else {
      throw MetalDirectPtychoError.invalidGeometry(
        "The phase-variance SSB fit is retired. Fit with the model agreement objective (the default)."
      )
    }
    guard Self.higherOrderLimits.contains(higherOrderLimit) else {
      throw MetalDirectPtychoError.invalidGeometry(
        "Choose a higher-order limit of 2 (standard) or 20 (wide) rad of phase at the aperture edge, not \(higherOrderLimit)."
      )
    }
    guard [1, 5].contains(maximumOrder), maxHigherOrderEvaluations > 0 else {
      throw MetalDirectPtychoError.invalidGeometry(
        "Choose maximumOrder 1 or 5 and a positive higher-order evaluation cap.")
    }
    guard maximumOrder == 1 || higherOrderSearch != .finiteDifferenceNewton else {
      throw MetalDirectPtychoError.invalidGeometry(
        DirectPtychoModelFit.HigherOrderSearch.retiredMessage)
    }
    if maximumOrder == 5 && !tilt {
      throw MetalDirectPtychoError.invalidGeometry(
        "The standard fifth-order fit jointly refines tilt and depth. Use maximumOrder: 1 for the primary-only alternatives."
      )
    }
    if maximumOrder == 5 && tiltLimitMrad != 25 {
      throw MetalDirectPtychoError.invalidGeometry(
        "The standard fifth-order fit uses its shared ±25 mrad tilt bounds. Use maximumOrder: 1 for a custom primary tilt range."
      )
    }
  }
}

/// One fit at one scan-detector rotation.
///
/// `sampleAgreement` is the fraction of G's power in the fit band that the fitted model explains, from 0 to 1 (Python
/// `sample_agreement`): the raw agreement S over the band power P (`MetalDirectPtychoJointModel.bandPower`), the same quantity
/// for the low-order and fifth-order fits. Saved fits store it under `sampleAgreementFraction` and keep the raw S
/// under `sampleAgreement`, the key that held it before. A fit saved before the band power was measured (no
/// `bandPower` key) decodes with `sampleAgreement` and `standardAgreement` nil and its raw values in
/// `rawSampleAgreement` and `rawStandardAgreement`: a raw agreement is never read as a fraction.
public struct DirectPtychoFit: Codable, Sendable {
  public let scanDetectorRotationDegrees: Double
  public let point: DirectPtychoModelPoint
  public let options: DirectPtychoFitOptions
  /// The fraction of the band's power the model explains at the fit, in [0, 1] (S / P); nil for a fit saved before
  /// the band power was measured.
  public let sampleAgreement: Double?
  /// The best agreement of the standard comparison (tilt and depth spread 0), as the same fraction; nil without a
  /// comparison or for a fit saved before the band power was measured.
  public let standardAgreement: Double?
  /// The fit's agreement over the best standard-SSB agreement on the same data (> 1: the tilt explains more). For a
  /// fifth-order fit, the low-order fit's gain times the fifth-order agreement over the low-order fit's (Python
  /// `tilt_fit_gain`); nil without a standard comparison (`DirectPtychoFitOptions.compareStandard`).
  public let tiltFitGain: Double?
  /// G's power over the 0.2...0.9 1/A fit band, sum_q w_q sum_k |G(q, k)|^2 (Python `band_power`); nil for a fit
  /// saved before it was measured.
  public let bandPower: Double?
  /// The raw agreement S at the fit (larger is better; S = `sampleAgreement` x `bandPower`).
  public let rawSampleAgreement: Double?
  /// The raw best agreement of the standard comparison.
  public let rawStandardAgreement: Double?
  public let trialCount: Int
  public var timings: [String: Double]
  /// The trust-region Newton starts of the primary fit (iterations, objective evaluations, convergence).
  public var newtonStarts: [DirectPtychoAgreementSearch.NewtonStart]? = nil
  /// The branch whose fit this one is the exact 180-degree image of (`DirectPtychoRotationBranches.oppositeBranch`);
  /// nil for a fit searched at its own rotation.
  public var mappedFromBranch: Int? = nil
  /// The fifth-order search's trajectory (`higherOrderSearch`); nil for a primary-only fit.
  /// Agreements in this trace are divided by `higherOrderNormalization`.
  public var higherOrderRefinement: DirectPtychoFifthOrderGradientSearch.Result? = nil
  /// Initial raw agreement for the shared fifth-order model, using the strict 0.2...0.9 inverse-Angstrom band.
  public var higherOrderNormalization: Double? = nil
  /// The half-band check of a wide fit (`DirectPtychoFitOptions.checkHigherOrderConsistency`): whether its higher orders are
  /// trusted, and the limit of this fit (±2 after a fallback, when `options.higherOrderLimit` is 2). nil otherwise.
  public var higherOrderConsistency: DirectPtychoHigherOrderConsistency? = nil

  /// A fit from its raw agreements and the band power P that turns them into fractions.
  public init(
    scanDetectorRotationDegrees: Double, point: DirectPtychoModelPoint,
    options: DirectPtychoFitOptions,
    agreement: Double, standardAgreement: Double?, bandPower: Double, tiltFitGain: Double?,
    trialCount: Int, timings: [String: Double]
  ) {
    self.scanDetectorRotationDegrees = scanDetectorRotationDegrees
    self.point = point
    self.options = options
    rawSampleAgreement = agreement
    rawStandardAgreement = standardAgreement
    self.bandPower = bandPower
    sampleAgreement = agreement / bandPower
    self.standardAgreement = standardAgreement.map { $0 / bandPower }
    self.tiltFitGain = tiltFitGain
    self.trialCount = trialCount
    self.timings = timings
  }

  private enum CodingKeys: String, CodingKey {
    case scanDetectorRotationDegrees, point, options, tiltFitGain, bandPower, trialCount, timings,
      newtonStarts, mappedFromBranch, higherOrderRefinement, higherOrderNormalization,
      higherOrderConsistency
    case sampleAgreement = "sampleAgreementFraction"
    case standardAgreement = "standardAgreementFraction"
    // The keys that held the raw agreements before the band power was measured; they still do.
    case rawSampleAgreement = "sampleAgreement"
    case rawStandardAgreement = "standardAgreement"
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    scanDetectorRotationDegrees = try c.decode(Double.self, forKey: .scanDetectorRotationDegrees)
    point = try c.decode(DirectPtychoModelPoint.self, forKey: .point)
    options = try c.decode(DirectPtychoFitOptions.self, forKey: .options)
    tiltFitGain = try c.decodeIfPresent(Double.self, forKey: .tiltFitGain)
    bandPower = try c.decodeIfPresent(Double.self, forKey: .bandPower)
    rawSampleAgreement = try c.decodeIfPresent(Double.self, forKey: .rawSampleAgreement)
    rawStandardAgreement = try c.decodeIfPresent(Double.self, forKey: .rawStandardAgreement)
    // Fractions only with the band power that made them; an older fit's raw values stay raw.
    let measured = bandPower != nil
    sampleAgreement =
      measured ? try c.decodeIfPresent(Double.self, forKey: .sampleAgreement) : nil
    standardAgreement =
      measured ? try c.decodeIfPresent(Double.self, forKey: .standardAgreement) : nil
    trialCount = try c.decode(Int.self, forKey: .trialCount)
    timings = try c.decode([String: Double].self, forKey: .timings)
    newtonStarts = try c.decodeIfPresent(
      [DirectPtychoAgreementSearch.NewtonStart].self, forKey: .newtonStarts)
    mappedFromBranch = try c.decodeIfPresent(Int.self, forKey: .mappedFromBranch)
    higherOrderRefinement = try c.decodeIfPresent(
      DirectPtychoFifthOrderGradientSearch.Result.self, forKey: .higherOrderRefinement)
    higherOrderNormalization = try c.decodeIfPresent(Double.self, forKey: .higherOrderNormalization)
    higherOrderConsistency = try c.decodeIfPresent(
      DirectPtychoHigherOrderConsistency.self, forKey: .higherOrderConsistency)
  }
}

/// Both 180-degree branches fitted independently and the branch kept (Python `RotationBranches`).
public struct DirectPtychoBranchedFit: Codable, Sendable {
  /// One fit (the rotation as given), or the given rotation then the opposite branch.
  public let fits: [DirectPtychoFit]
  public let branches: [DirectPtychoRotationBranches.Branch]
  public let chosen: Int
  public let kept: Int
  public let decidedBy: DirectPtychoRotationBranches.Decision
  public let columnCount: Int
  /// Branch diagnostics retain the initial native-grid fits; refinement holds the selected rotation.
  public var refinedFit: DirectPtychoFit? = nil
  public var fit: DirectPtychoFit { refinedFit ?? fits[kept] }
  /// Whether the 180-degree branches were compared.
  public var comparedBranches: Bool { fits.count == 2 }
}

/// Python `SSB.find_aberrations` on a prepared native session: the default fit, optionally on both
/// 180-degree branches of an estimated rotation with the branch test.
public enum DirectPtychoModelFit {
  public typealias Progress = (_ stage: String, _ completed: Int, _ total: Int) -> Void

  /// The box of the fifth-order search in its dimensionless coordinates (`DirectPtychoFifthOrderCoordinates`): C10, Cartesian
  /// C12, tilt, depth spread, then the 22 higher-order Cartesian coordinates, each bounded to ±`higherOrderLimit` rad
  /// of probe phase at the aperture edge. A limit of 2 is the original box; 20 no longer binds.
  public static func fifthOrderBounds(higherOrderLimit: Double) -> [ClosedRange<Double>] {
    [-3...3, -2...2, -2...2, -5...5, -5...5, 0.2...6]
      + Array(repeating: -higherOrderLimit...higherOrderLimit, count: 22)
  }

  /// Fit at the model's current rotation. `onRefinedFit` receives a fifth-order fit before its guard is computed.
  public static func fit(
    model: MetalDirectPtychoJointModel, options: DirectPtychoFitOptions,
    progress: Progress = { _, _, _ in }, onInitialFit: (DirectPtychoFit) throws -> Void = { _ in },
    onRefinedFit: (DirectPtychoFit) throws -> Void = { _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> DirectPtychoFit {
    try options.validate()
    let primary = try fitPrimary(
      model: model, options: options,
      progress: progress, isCancelled: isCancelled)
    try onInitialFit(primary)
    return try refineHigherOrder(
      primary, model: model, options: options,
      progress: progress, onRefinedFit: onRefinedFit, isCancelled: isCancelled)
  }

  private static func fitPrimary(
    model: MetalDirectPtychoJointModel, options: DirectPtychoFitOptions,
    progress: Progress, isCancelled: () -> Bool
  ) throws -> DirectPtychoFit {
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
          DirectPtychoModelPoint(
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
    func derivatives(_ indices: [Int]) throws -> [DirectPtychoAgreementSearch.Derivatives] {
      if isCancelled() { throw CancellationError() }
      if !latestFused { derivativeCenters += indices.count }
      return try model.lowOrderDerivatives(at: indices, isCancelled: isCancelled).map {
        DirectPtychoAgreementSearch.Derivatives(
          value: $0.value, gradient: $0.gradient, hessian: $0.hessian)
      }
    }
    var timings: [String: Double] = [:]
    let rotation = model.scanDetectorRotationDegrees
    if options.tilt {
      var standard: DirectPtychoAgreementSearch.Result?
      if options.compareStandard {
        let started = Date()
        standard = try DirectPtychoAgreementSearch.standardComparison(
          evaluate, seed: options.seed, isCancelled: isCancelled)
        timings["standard_comparison_s"] = Date().timeIntervalSince(started)
        count = 0
        searchObjectiveSeconds = 0
      }
      let result = try withoutActuallyEscaping(values) { values in
        try withoutActuallyEscaping(derivatives) { derivatives in
          try DirectPtychoAgreementSearch.fitTilt(
            evaluate, trials: options.trials, seed: options.seed,
            tiltLimitMrad: options.tiltLimitMrad,
            staged: DirectPtychoAgreementSearch.StagedDerivatives(
              values: values, derivatives: derivatives),
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
      var fit = DirectPtychoFit(
        scanDetectorRotationDegrees: rotation,
        point: DirectPtychoModelPoint(
          c10Angstroms: p[0], c12Angstroms: p[1], phi12Radians: p[2], sampleTiltMrad: [p[3], p[4]],
          depthSpreadAngstroms: p[5]),
        options: options, agreement: result.agreement, standardAgreement: standard?.agreement,
        bandPower: model.bandPower, tiltFitGain: standard.map { result.agreement / $0.agreement },
        trialCount: options.trials, timings: timings)
      fit.newtonStarts = result.newtonStarts
      return fit
    } else {
      let result = try withoutActuallyEscaping(values) { values in
        try withoutActuallyEscaping(derivatives) { derivatives in
          try DirectPtychoAgreementSearch.fitStandard(
            evaluate, trials: options.trials, seed: options.seed,
            staged: DirectPtychoAgreementSearch.StagedDerivatives(
              values: values, derivatives: derivatives),
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
      var fit = DirectPtychoFit(
        scanDetectorRotationDegrees: rotation,
        point: DirectPtychoModelPoint(c10Angstroms: p[0], c12Angstroms: p[1], phi12Radians: p[2]),
        options: options, agreement: result.agreement, standardAgreement: nil,
        bandPower: model.bandPower, tiltFitGain: nil,
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
  /// (`DirectPtychoRotationBranches.oppositeBranch`): every point of one branch has an equal-agreement point on the
  /// other, so the opposite branch's fit is the mapped fit of `rotation` (`mappedFromBranch` 0), its
  /// agreement evaluated at its own rotation, not a second search.
  ///
  /// `checkRotation: false` fits `rotation` alone (Python `check_rotation=False`), the standard comparison
  /// included.
  ///
  /// `onInitialFit` receives the kept low-order fit, `onRefinedFit` the fifth-order fit before its half-band check
  /// runs, so a viewer can show each while the rest runs.
  public static func findAberrations(
    model: MetalDirectPtychoJointModel, rotationDegrees: Double,
    darkField: [Float]?, scanSampling: (Double, Double),
    options: DirectPtychoFitOptions = DirectPtychoFitOptions(),
    checkRotation: Bool = true, keepFirst: Bool = false, progress: Progress = { _, _, _ in },
    onInitialFit: (DirectPtychoFit) throws -> Void = { _ in },
    onRefinedFit: (DirectPtychoFit) throws -> Void = { _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> DirectPtychoBranchedFit {
    if !checkRotation {
      try model.retarget(scanDetectorRotationDegrees: rotationDegrees)
      let fitted = try fit(
        model: model, options: options, progress: progress,
        onInitialFit: onInitialFit, onRefinedFit: onRefinedFit, isCancelled: isCancelled
      )
      return DirectPtychoBranchedFit(
        fits: [fitted], branches: [], chosen: 0, kept: 0, decidedBy: .phaseTail, columnCount: 0)
    }
    let rotations = [rotationDegrees, DirectPtychoRotationBranches.opposite(rotationDegrees)]
    var fits: [DirectPtychoFit] = []
    var phases: [[Float]] = []
    var size = 0
    var branchOptions = options
    branchOptions.compareStandard = false
    for (index, rotation) in rotations.enumerated() {
      try model.retarget(scanDetectorRotationDegrees: rotation)
      let fitted: DirectPtychoFit
      if index == 1 {
        let started = Date()
        let point = DirectPtychoRotationBranches.oppositeBranch(fits[0].point)
        let agreement = try model.agreements([point], isCancelled: isCancelled)[0]
        var opposite = DirectPtychoFit(
          scanDetectorRotationDegrees: rotation, point: point, options: branchOptions,
          agreement: agreement, standardAgreement: nil, bandPower: model.bandPower,
          tiltFitGain: nil, trialCount: fits[0].trialCount,
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
    let decision = DirectPtychoRotationBranches.compare(
      phases: phases, rotations: rotations, rows: size, columns: size, darkField: darkField,
      scanSampling: scanSampling, keepFirst: keepFirst)
    try model.retarget(scanDetectorRotationDegrees: rotations[decision.kept])
    if options.tilt && options.compareStandard,
      let agreement = fits[decision.kept].rawSampleAgreement
    {
      let started = Date()
      let standard = try DirectPtychoAgreementSearch.standardComparison(
        { points in
          try model.agreements(
            points.map {
              DirectPtychoModelPoint(c10Angstroms: $0[0], c12Angstroms: $0[1], phi12Radians: $0[2])
            })
        }, seed: options.seed, isCancelled: isCancelled)
      let kept = fits[decision.kept]
      var timings = kept.timings
      timings["standard_comparison_s"] = Date().timeIntervalSince(started)
      fits[decision.kept] = DirectPtychoFit(
        scanDetectorRotationDegrees: kept.scanDetectorRotationDegrees, point: kept.point,
        options: options,
        agreement: agreement, standardAgreement: standard.agreement, bandPower: model.bandPower,
        tiltFitGain: standard.agreement > 0 ? agreement / standard.agreement : .nan,
        trialCount: kept.trialCount, timings: timings)
      fits[decision.kept].newtonStarts = kept.newtonStarts
      fits[decision.kept].mappedFromBranch = kept.mappedFromBranch
    }
    var result = DirectPtychoBranchedFit(
      fits: fits, branches: decision.branches, chosen: decision.chosen, kept: decision.kept,
      decidedBy: decision.decidedBy, columnCount: decision.columns)
    try onInitialFit(result.fit)
    if options.maximumOrder > 1 {
      result.refinedFit = try refineHigherOrder(
        result.fit, model: model, options: options,
        progress: progress, onRefinedFit: onRefinedFit, isCancelled: isCancelled)
    }
    return result
  }

  /// Refine a low-order fit through fifth order without repeating its seeded search: the same result as
  /// `findAberrations` with `maximumOrder: 5`, which reruns the low-order search to the same point first.
  /// `model` must be at the kept branch's rotation, as `findAberrations` leaves it. A fit already refined is refined
  /// again from its kept low-order fit (e.g. at another `higherOrderLimit`). `onRefinedFit` receives the fifth-order
  /// fit before its guard is computed.
  public static func refineToFifthOrder(
    _ fitted: DirectPtychoBranchedFit, model: MetalDirectPtychoJointModel,
    options: DirectPtychoFitOptions = DirectPtychoFitOptions(),
    progress: Progress = { _, _, _ in }, onRefinedFit: (DirectPtychoFit) throws -> Void = { _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> DirectPtychoBranchedFit {
    try options.validate()
    let kept = fitted.fits[fitted.kept]
    guard options.maximumOrder == 5, kept.options.tilt, kept.options.objective == .agreement,
      kept.higherOrderRefinement == nil
    else {
      throw MetalDirectPtychoError.invalidGeometry(
        "Refine a low-order tilt-corrected model fit with maximumOrder: 5.")
    }
    guard model.scanDetectorRotationDegrees == kept.scanDetectorRotationDegrees else {
      throw MetalDirectPtychoError.invalidGeometry(
        "The joint model is at another rotation than the kept branch. Fit again before refining.")
    }
    var result = fitted
    result.refinedFit = try refineHigherOrder(
      kept, model: model, options: options, progress: progress, onRefinedFit: onRefinedFit,
      isCancelled: isCancelled)
    return result
  }

  private static func refineHigherOrder(
    _ primary: DirectPtychoFit, model: MetalDirectPtychoJointModel, options: DirectPtychoFitOptions,
    progress: Progress, onRefinedFit: (DirectPtychoFit) throws -> Void, isCancelled: () -> Bool
  ) throws -> DirectPtychoFit {
    guard options.maximumOrder == 5 else { return primary }
    guard model.bandInverseAngstroms == Float(0.2)...Float(0.9) else {
      throw MetalDirectPtychoError.invalidGeometry(
        "The fifth-order fit requires the 0.2...0.9 inverse-Angstrom fitting band. Prepare the joint model with its default band."
      )
    }
    func refine(_ limit: Double) throws -> (
      fit: DirectPtychoFit, coordinates: [Double], seconds: Double
    ) {
      let started = Date()
      let refined = try refineFifthOrder(
        primary.point, model: model, search: options.higherOrderSearch,
        maxEvaluations: options.maxHigherOrderEvaluations,
        bounds: fifthOrderBounds(higherOrderLimit: limit),
        progress: { progress("Fifth-order fit", $0, $1) }, isCancelled: isCancelled)
      var fitOptions = options
      fitOptions.higherOrderLimit = limit
      var fit = DirectPtychoFit(
        scanDetectorRotationDegrees: primary.scanDetectorRotationDegrees,
        point: refined.point, options: fitOptions,
        agreement: refined.agreement, standardAgreement: nil, bandPower: model.bandPower,
        // The low-order gain times the fifth-order agreement over the low-order point's (Python
        // `fit_fifth_order`): `normalization` is the fifth-order objective at the low-order fit.
        tiltFitGain: primary.tiltFitGain.map { $0 * refined.agreement / refined.normalization },
        trialCount: primary.trialCount, timings: primary.timings)
      fit.higherOrderRefinement = refined.result
      fit.higherOrderNormalization = refined.normalization
      return (fit, refined.result.coordinates, Date().timeIntervalSince(started))
    }
    let requested = try refine(options.higherOrderLimit)
    var fit = requested.fit
    fit.timings["higher_order_refinement_s"] = requested.seconds
    try onRefinedFit(fit)
    if options.checkHigherOrderConsistency
      && options.higherOrderLimit == DirectPtychoFitOptions.wideHigherOrderLimit
    {
      let started = Date()
      let check = try higherOrderConsistency(
        start: primary.point, checked: requested.coordinates, model: model,
        search: options.higherOrderSearch,
        maxEvaluations: options.maxHigherOrderEvaluations, limit: options.higherOrderLimit,
        progress: { progress("Half-band check", $0, $1) }, isCancelled: isCancelled)
      let checkSeconds = Date().timeIntervalSince(started)
      if !check.trusted {
        let shown = try refine(check.shownLimit)
        fit = shown.fit
        fit.timings["higher_order_refinement_s"] = requested.seconds
        fit.timings["higher_order_fallback_s"] = shown.seconds
      }
      fit.timings["higher_order_consistency_s"] = checkSeconds
      fit.higherOrderConsistency = check
    }
    return fit
  }
}
