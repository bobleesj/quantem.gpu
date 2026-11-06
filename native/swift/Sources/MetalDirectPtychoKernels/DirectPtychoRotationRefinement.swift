import Foundation

/// Evidence for an explicit scan-detector rotation prior. Angles stay unwrapped around the supplied center.
public struct DirectPtychoRotationRefinement: Codable, Sendable {
  public let centerDegrees: Double
  public let halfWidthDegrees: Double
  public let fittedDegrees: Double
  public let evaluatedDegrees: [Double]
  public let initialAgreement: Double
  public let finalAgreement: Double
  public let seconds: Double
  /// The complete initial search, separate from the warm-started angular refinements.
  public var initialTrialCount: Int? = nil
  public var initialTimings: [String: Double]? = nil
  /// Within one angular search tolerance of a bound; inspect the prior before interpreting this fit.
  public var hitBoundary: Bool {
    abs(fittedDegrees - centerDegrees) >= halfWidthDegrees - 0.05
  }
}

/// One-degree bracketing followed by bounded Brent refinement, matching the Python rotation workflow.
/// The objective is maximized; the initial value is retained even if no candidate improves it.
enum DirectPtychoBoundedRotationSearch {
  static func search(
    halfWidth: Double, tolerance: Double = 0.05,
    evaluate: (Double) throws -> Double, isCancelled: () -> Bool = { false }
  ) throws -> Double {
    guard halfWidth.isFinite, halfWidth > 0, halfWidth < 90,
      tolerance.isFinite, tolerance > 0
    else {
      throw MetalDirectPtychoError.invalidGeometry(
        "Use a finite rotation half-width greater than 0 and less than 90 degrees.")
    }
    var scores: [Double: Double] = [:]
    var best = 0.0
    func value(_ offset: Double) throws -> Double {
      if isCancelled() { throw CancellationError() }
      if let score = scores[offset] { return -score }
      let score = try evaluate(offset)
      guard score.isFinite else {
        throw MetalDirectPtychoError.invalidGeometry(
          "Rotation fitting returned a nonfinite agreement; check the calibration and source data.")
      }
      if scores.isEmpty || score > scores[best]! { best = offset }
      scores[offset] = score
      return -score
    }
    let step = min(1.0, halfWidth)
    let center = try value(0)
    let up = try value(step)
    let down = try value(-step)
    var low = -step
    var high = step
    if center > min(up, down) {
      let direction = up < down ? step : -step
      var at = direction
      var score = min(up, down)
      while abs(at) < halfWidth {
        let next = max(-halfWidth, min(halfWidth, at + direction))
        let candidate = try value(next)
        if candidate >= score { break }
        at = next
        score = candidate
      }
      low = max(-halfWidth, at - step)
      high = min(halfWidth, at + step)
    }
    // Safeguarded parabolic interpolation; golden steps guarantee progress when interpolation is unsuitable.
    let golden = (3 - sqrt(5.0)) / 2
    var x = low + golden * (high - low)
    var w = x
    var v = x
    var fx = try value(x)
    var fw = fx
    var fv = fx
    var d = 0.0
    var e = 0.0
    for _ in 0..<128 {
      let midpoint = (low + high) / 2
      let tol = sqrt(Double.ulpOfOne) * abs(x) + tolerance / 3
      let tol2 = 2 * tol
      if abs(x - midpoint) <= tol2 - (high - low) / 2 { break }
      var parabolic = false
      if abs(e) > tol {
        let r = (x - w) * (fx - fv)
        var q = (x - v) * (fx - fw)
        var p = (x - v) * q - (x - w) * r
        q = 2 * (q - r)
        if q > 0 { p = -p }
        q = abs(q)
        let previous = e
        e = d
        if abs(p) < abs(0.5 * q * previous), p > q * (low - x), p < q * (high - x) {
          d = p / q
          parabolic = true
          let trial = x + d
          if trial - low < tol2 || high - trial < tol2 { d = midpoint >= x ? tol : -tol }
        }
      }
      if !parabolic {
        e = x >= midpoint ? low - x : high - x
        d = golden * e
      }
      let u = x + (abs(d) >= tol ? d : (d >= 0 ? tol : -tol))
      let fu = try value(u)
      if fu <= fx {
        if u >= x { low = x } else { high = x }
        v = w
        fv = fw
        w = x
        fw = fx
        x = u
        fx = fu
      } else {
        if u < x { low = u } else { high = u }
        if fu <= fw || w == x {
          v = w
          fv = fw
          w = u
          fw = fu
        } else if fu <= fv || v == x || v == w {
          v = u
          fv = fu
        }
      }
    }
    return best
  }
}

extension DirectPtychoModelFit {
  static func findWithinRotationPrior(
    model: MetalDirectPtychoJointModel, center: Double,
    halfWidth: Double, options: DirectPtychoFitOptions, progress: Progress,
    onInitialFit: (DirectPtychoFit) throws -> Void, onRefinedFit: (DirectPtychoFit) throws -> Void,
    isCancelled: () -> Bool
  ) throws -> DirectPtychoBranchedFit {
    guard center.isFinite, halfWidth.isFinite, halfWidth > 0, halfWidth < 90 else {
      throw MetalDirectPtychoError.invalidGeometry(
        "Enter a finite rotation center and a half-width between 0 and 90 degrees.")
    }
    try options.validate()
    let started = Date()
    try model.retarget(scanDetectorRotationDegrees: center)
    var primaryOptions = options
    primaryOptions.maximumOrder = 1
    primaryOptions.compareStandard = false
    let initial = try fitPrimary(
      model: model, options: primaryOptions, progress: progress, isCancelled: isCancelled)
    var fits = [0.0: initial]
    var evaluated = [center]
    let offset = try DirectPtychoBoundedRotationSearch.search(
      halfWidth: halfWidth,
      evaluate: { offset in
        if let fit = fits[offset] { return fit.rawSampleAgreement! }
        let nearest = fits.keys.min { a, b in
          abs(a - offset) == abs(b - offset) ? a < b : abs(a - offset) < abs(b - offset)
        }!
        try model.retarget(scanDetectorRotationDegrees: center + offset)
        progress("Refining rotation", evaluated.count, 0)
        let fit = try fitPrimary(
          model: model, options: primaryOptions, progress: { _, _, _ in },
          isCancelled: isCancelled, start: fits[nearest]!.point)
        fits[offset] = fit
        evaluated.append(center + offset)
        return fit.rawSampleAgreement!
      }, isCancelled: isCancelled)
    let selected = fits[offset]!
    try model.retarget(scanDetectorRotationDegrees: selected.scanDetectorRotationDegrees)
    var timings: [String: Double] = [:]
    for offset in fits.keys.sorted() {
      for (key, value) in fits[offset]!.timings {
        timings[key, default: 0] += value
      }
    }
    timings["rotation_refinement_s"] = Date().timeIntervalSince(started)
    var standardAgreement: Double? = nil
    if options.tilt && options.compareStandard {
      let comparisonStarted = Date()
      let standard = try DirectPtychoAgreementSearch.standardComparison(
        { points in
          try model.agreements(
            points.map {
              DirectPtychoModelPoint(c10Angstroms: $0[0], c12Angstroms: $0[1], phi12Radians: $0[2])
            })
        }, seed: options.seed, isCancelled: isCancelled)
      standardAgreement = standard.agreement
      timings["standard_comparison_s"] = Date().timeIntervalSince(comparisonStarted)
    }
    primaryOptions.compareStandard = options.compareStandard
    var kept = DirectPtychoFit(
      scanDetectorRotationDegrees: selected.scanDetectorRotationDegrees, point: selected.point,
      options: primaryOptions, agreement: selected.rawSampleAgreement!,
      standardAgreement: standardAgreement, bandPower: model.bandPower,
      tiltFitGain: standardAgreement.map { $0 > 0 ? selected.rawSampleAgreement! / $0 : .nan },
      trialCount: initial.trialCount, timings: timings)
    kept.newtonStarts = fits.keys.sorted().flatMap { fits[$0]!.newtonStarts ?? [] }
    var result = DirectPtychoBranchedFit(
      fits: [kept], branches: [], chosen: 0, kept: 0,
      decidedBy: .phaseTail, columnCount: 0)
    result.rotationRefinement = DirectPtychoRotationRefinement(
      centerDegrees: center,
      halfWidthDegrees: halfWidth, fittedDegrees: kept.scanDetectorRotationDegrees,
      evaluatedDegrees: evaluated, initialAgreement: initial.sampleAgreement!,
      finalAgreement: kept.sampleAgreement!, seconds: Date().timeIntervalSince(started),
      initialTrialCount: initial.trialCount, initialTimings: initial.timings)
    try onInitialFit(kept)
    if options.maximumOrder > 1 {
      result.refinedFit = try refineHigherOrder(
        kept, model: model, options: options, progress: progress,
        onRefinedFit: onRefinedFit, isCancelled: isCancelled)
    }
    return result
  }
}
