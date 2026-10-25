import Foundation

/// Bounded trust-region quasi-Newton refinement of the fifth-order agreement with exact gradients.
/// The exact full-data gradient defines the optimum; the Hessian only steers the steps. It starts from forward
/// differences of exact gradients on a band subsample at the initial point and is kept current with symmetric
/// rank-one updates from the full-band gradients of accepted steps. Steps respect the box of
/// `SSBFifthOrderSearch.bounds` through the same projected trust-region solve. A step is accepted only when the
/// full objective increases, and the search stops when the bounded model step predicts a gain below
/// `Settings.gainTolerance`, at a stationary point of the full objective within its bounds.
public enum SSBFifthOrderGradientSearch {
  public enum Request: Sendable, Equatable {
    /// The full-band agreement and its gradient.
    case valueAndGradient
    /// The full-band agreement only (empty gradient); its projections are kept for `gradientOfLastValue`.
    case value
    /// The gradients of the points of the last `value` request, adding only the gradient pass.
    case gradientOfLastValue
    /// Only the gradient of the agreement over every n-th band frequency, scaled to the full band weight
    /// (value NaN): a steering curvature, never the objective.
    case subsampledGradient(stride: Int)
  }

  /// Values and gradients (d value / d z) of candidate coordinates.
  public typealias Evaluate = (_ points: [[Double]], _ request: Request) throws -> [(
    value: Double, gradient: [Double]
  )]

  public struct Settings: Sendable {
    /// Band subsample of the initial difference Hessian (1: the full band).
    public var hessianStride = 8
    /// Trust radii in units of the coordinate scales of `SSBFifthOrderSearch` (0.02, tilt and depth 0.05).
    public var initialRadius = 8.0
    public var maxRadius = 64.0
    /// The smallest normalized gain a step must predict (the agreement is divided by its initial value; one
    /// float32 agreement resolves about 1e-7 of it).
    public var gainTolerance = 1e-9
    /// Evaluate a proposal's value first and its gradient only when it is accepted.
    public var valueFirst = true
    public var maxObjectives = 400
    public var maxIterations = 200
    public init() {}
  }

  /// - Returns: the best evaluated point; `evaluations` and `executedObjectives` count full-band objective
  ///   passes, `gradientEvaluations` full-band gradient passes, `subsampledGradients` the Hessian's band-subsample
  ///   gradient evaluations.
  public static func refine(
    _ evaluate: Evaluate, start: [Double], bounds: [ClosedRange<Double>] = SSBFifthOrderSearch.bounds,
    settings: Settings = .init(),
    progress: (_ completed: Int, _ total: Int) -> Void = { _, _ in },
    log: ((String) -> Void)? = nil,
    isCancelled: () -> Bool = { false }
  ) throws -> SSBFifthOrderSearch.Result {
    let n = start.count
    guard n >= 6, bounds.count == n, settings.maxObjectives > 1, settings.hessianStride >= 1,
      zip(start, bounds).allSatisfy({ $0.isFinite && $1.contains($0) }),
      bounds.allSatisfy({
        $0.lowerBound.isFinite && $0.upperBound.isFinite && $0.upperBound > $0.lowerBound
      })
    else {
      throw MetalSSBError.invalidGeometry(
        "The initial fit must lie inside the higher-order bounds. Check calibration and the initial fit."
      )
    }
    // The trust-region scaling of `SSBFifthOrderSearch`; half of it steps the difference Hessian.
    let h = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05] + Array(repeating: 0.02, count: n - 6)
    var objectives = 0
    var gradients = 0
    var subsampled = 0
    var best = -Double.infinity
    var bestPoint = start
    var trace: [SSBFifthOrderSearch.Sample] = []
    let started = Date()
    func track(_ points: [[Double]], _ request: Request) throws -> [(
      value: Double, gradient: [Double]
    )] {
      if isCancelled() { throw CancellationError() }
      guard
        points.allSatisfy({ row in
          row.count == n
            && zip(row, bounds).allSatisfy {
              $0.isFinite && $0 >= $1.lowerBound - 1e-9 && $0 <= $1.upperBound + 1e-9
            }
        })
      else { throw MetalSSBError.invalidGeometry("A Newton candidate left its calibrated bounds.") }
      let values = try evaluate(points, request)
      let withGradient = request != .value
      guard values.count == points.count,
        values.allSatisfy({
          (!withGradient || ($0.gradient.count == n && $0.gradient.allSatisfy(\.isFinite)))
        })
      else {
        throw MetalSSBError.invalidGeometry(
          "The higher-order gradient is not finite. Check the calibration and fitting band.")
      }
      switch request {
      case .subsampledGradient:
        subsampled += points.count
        return values
      case .gradientOfLastValue:
        gradients += points.count
        return values
      case .valueAndGradient, .value:
        if request == .valueAndGradient { gradients += points.count }
      }
      guard values.allSatisfy(\.value.isFinite) else {
        throw MetalSSBError.invalidGeometry(
          "The higher-order objective returned nonfinite scores. Check the calibration and fitting band."
        )
      }
      for (point, value) in zip(points, values) {
        objectives += 1
        if value.value > best {
          best = value.value
          bestPoint = point
        }
        trace.append(
          .init(
            evaluation: objectives, agreement: best,
            elapsedSeconds: Date().timeIntervalSince(started)))
      }
      progress(objectives, settings.maxObjectives)
      return values
    }
    func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
    func times(_ m: [[Double]], _ v: [Double]) -> [Double] { m.map { dot($0, v) } }

    var center = start
    var (value, gradient) = try track([start], .valueAndGradient)[0]
    // Forward differences of exact gradients, stepping inward at an upper bound.
    let offsets = (0..<n).map { i in
      center[i] + 0.5 * h[i] <= bounds[i].upperBound ? 0.5 * h[i] : -0.5 * h[i]
    }
    let design = (0..<n).map { i -> [Double] in
      var p = center
      p[i] += offsets[i]
      return p
    }
    let stride = settings.hessianStride
    // On a band subsample the differences use the subsample gradient at the center too.
    let columns =
      stride > 1
      ? try track([center] + design, .subsampledGradient(stride: stride))
      : try track(design, .valueAndGradient)
    let reference = stride > 1 ? columns[0].gradient : gradient
    let differenced = stride > 1 ? Array(columns.dropFirst()) : columns
    var hessian = Array(repeating: Array(repeating: 0.0, count: n), count: n)
    for i in 0..<n {
      for j in 0..<n {
        hessian[i][j] =
          0.5
          * ((differenced[j].gradient[i] - reference[i]) / offsets[j]
            + (differenced[i].gradient[j] - reference[j]) / offsets[i])
      }
    }
    log?(
      String(
        format: "start %.12f, Hessian from %d gradients at q stride %d, %.3f s", value,
        differenced.count, stride, Date().timeIntervalSince(started)))
    var radius = settings.initialRadius
    var iterations = 0
    var stop = "iteration limit"
    while iterations < settings.maxIterations {
      if objectives >= settings.maxObjectives {
        stop = "evaluation cap"
        break
      }
      iterations += 1
      let delta = SSBFifthOrderSearch.boundedStep(
        center: center, gradient: gradient, hessian: hessian, steps: h, radius: radius,
        bounds: bounds)
      let proposal = (0..<n).map {
        min(bounds[$0].upperBound, max(bounds[$0].lowerBound, center[$0] + delta[$0]))
      }
      let step = (0..<n).map { proposal[$0] - center[$0] }
      let scaledNorm = (0..<n).map { step[$0] * step[$0] / (h[$0] * h[$0]) }.reduce(0, +)
        .squareRoot()
      let predicted = dot(gradient, step) + 0.5 * dot(step, times(hessian, step))
      if predicted < settings.gainTolerance || scaledNorm < 1e-6 {
        stop = "converged"
        log?(String(format: "%3d predicted %+.3e: converged", iterations, predicted))
        break
      }
      var next = try track([proposal], settings.valueFirst ? .value : .valueAndGradient)[0]
      let actual = next.value - value
      let ratio = actual / predicted
      log?(
        String(
          format: "%3d radius %7.3f |step| %7.3f predicted %+.3e actual %+.3e ratio %+.3f %@ %.3f s",
          iterations, radius, scaledNorm, predicted, actual, ratio, actual > 0 ? "accept" : "reject",
          Date().timeIntervalSince(started)))
      if actual > 0 && settings.valueFirst {
        next = try track([proposal], .gradientOfLastValue)[0]
      }
      if !next.gradient.isEmpty {
        // Symmetric rank-one update from every step with a full-band gradient.
        let y = (0..<n).map { next.gradient[$0] - gradient[$0] }
        let hs = times(hessian, step)
        let r = (0..<n).map { y[$0] - hs[$0] }
        let rs = dot(r, step)
        if abs(rs) > 1e-8 * dot(r, r).squareRoot() * dot(step, step).squareRoot() {
          for i in 0..<n {
            for j in 0..<n { hessian[i][j] += r[i] * r[j] / rs }
          }
        }
      }
      if actual > 0 {
        center = proposal
        value = next.value
        gradient = next.gradient
        if ratio > 0.75 && scaledNorm > 0.8 * radius {
          radius = min(radius * 2, settings.maxRadius)
        } else if ratio < 0.25 {
          radius *= 0.5
        }
      } else {
        radius = 0.25 * min(radius, scaledNorm)
      }
      if radius < 1e-4 {
        stop = "trust radius exhausted"
        break
      }
    }
    var result = SSBFifthOrderSearch.Result(
      coordinates: bestPoint, agreement: best, evaluations: objectives,
      executedObjectives: objectives, iterations: iterations, stop: stop, trace: trace)
    result.gradientEvaluations = gradients
    result.subsampledGradients = subsampled
    result.hessianStride = stride
    let projected = projectedGradient(gradient, at: center, bounds: bounds)
    log?(
      String(
        format:
          "%@ after %d iterations: %d objectives, %d gradients, %d subsampled gradients, %.3f s; max |projected gradient * h| %.3e",
        stop, iterations, objectives, gradients, subsampled, Date().timeIntervalSince(started),
        (0..<n).map { abs(projected[$0] * h[$0]) }.max() ?? 0))
    return result
  }

  /// The projected gradient: zero where a bound holds the coordinate and the ascent direction leaves the box.
  public static func projectedGradient(
    _ gradient: [Double], at z: [Double], bounds: [ClosedRange<Double>] = SSBFifthOrderSearch.bounds
  ) -> [Double] {
    (0..<gradient.count).map { i in
      if z[i] <= bounds[i].lowerBound && gradient[i] < 0 { return 0 }
      if z[i] >= bounds[i].upperBound && gradient[i] > 0 { return 0 }
      return gradient[i]
    }
  }
}

extension SSBModelFit {
  /// Which search refines the fifth-order model.
  public enum HigherOrderSearch: String, Codable, Sendable {
    /// `SSBFifthOrderSearch`: bounded finite-difference Newton (the reference).
    case finiteDifferenceNewton
    /// `SSBFifthOrderGradientSearch`: exact gradients, trust-region quasi-Newton.
    case exactGradient
  }

  /// Refine a low-order fit through fifth order on `model` at its current rotation.
  /// - Returns: the refined point, its agreement, the initial agreement that normalizes the trace, and the trace.
  public static func refineFifthOrder(
    _ primary: SSBModelPoint, model: MetalSSBJointModel, search: HigherOrderSearch,
    maxEvaluations: Int = 5000,
    gradientSettings: SSBFifthOrderGradientSearch.Settings = .init(),
    log: ((String) -> Void)? = nil,
    progress: (_ completed: Int, _ total: Int) -> Void = { _, _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> (
    point: SSBModelPoint, agreement: Double, normalization: Double,
    result: SSBFifthOrderSearch.Result
  ) {
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms,
      semiangleRadians: model.semiangleRadians)
    let start = coordinates.initial(primary)
    let normalization = try model.fifthOrderAgreements(
      [primary], packedCoefficients: [coordinates.pack(start)], isCancelled: isCancelled)[0]
    guard normalization.isFinite, normalization > 0 else {
      throw MetalSSBError.invalidGeometry(
        "The higher-order fit needs a positive finite initial agreement. Check calibration and signal in the fitting band."
      )
    }
    let result: SSBFifthOrderSearch.Result
    switch search {
    case .finiteDifferenceNewton:
      result = try SSBFifthOrderSearch.refine(
        { rows in
          try model.fifthOrderAgreements(
            rows.map(coordinates.point),
            packedCoefficients: rows.map(coordinates.pack), isCancelled: isCancelled
          ).map { $0 / normalization }
        }, start: start, maxEvaluations: maxEvaluations, progress: progress,
        isCancelled: isCancelled)
    case .exactGradient:
      var settings = gradientSettings
      settings.maxObjectives = min(settings.maxObjectives, maxEvaluations)
      result = try SSBFifthOrderGradientSearch.refine(
        { rows, request in
          let points = rows.map(coordinates.point)
          let packed = rows.map(coordinates.pack)
          let values: [(agreement: Double, gradient: [Double])]
          switch request {
          case .valueAndGradient, .value:
            values = try model.fifthOrderAgreementsAndGradients(
              points, packedCoefficients: packed, gradients: request == .valueAndGradient,
              isCancelled: isCancelled)
          case .gradientOfLastValue:
            values = try model.fifthOrderRetainedGradients(
              points, packedCoefficients: packed, isCancelled: isCancelled)
          case .subsampledGradient(let stride):
            values = try model.fifthOrderAgreementsAndGradients(
              points, packedCoefficients: packed, qStride: stride, isCancelled: isCancelled)
          }
          return values.map {
            (
              value: $0.agreement / normalization,
              gradient: $0.gradient.isEmpty
                ? [] : coordinates.gradient(row: $0.gradient).map { $0 / normalization }
            )
          }
        }, start: start, settings: settings, progress: progress, log: log,
        isCancelled: isCancelled)
    }
    return (
      coordinates.point(result.coordinates), result.agreement * normalization, normalization, result
    )
  }

  /// Benchmark diagnostics: the exact gradient (d agreement / d z) and central differences of the float32
  /// objective at `z` for each step in `steps`, and the wall time of one objective and one value-and-gradient
  /// evaluation (median of `repeats`).
  public static func fifthOrderGradientCheck(
    model: MetalSSBJointModel, z: [Double], steps: [Double], repeats: Int = 5
  ) throws -> [String: Any] {
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms,
      semiangleRadians: model.semiangleRadians)
    func values(_ rows: [[Double]]) throws -> [Double] {
      try model.fifthOrderAgreements(
        rows.map(coordinates.point), packedCoefficients: rows.map(coordinates.pack))
    }
    func exact(_ rows: [[Double]]) throws -> [(agreement: Double, gradient: [Double])] {
      try model.fifthOrderAgreementsAndGradients(
        rows.map(coordinates.point), packedCoefficients: rows.map(coordinates.pack))
    }
    func median(_ f: () throws -> Void) rethrows -> Double {
      var times: [Double] = []
      for _ in 0..<repeats {
        let t = Date()
        try f()
        times.append(Date().timeIntervalSince(t))
      }
      return times.sorted()[times.count / 2]
    }
    _ = try exact([z])
    let objectiveSeconds = try median { _ = try values([z]) }
    let gradientSeconds = try median { _ = try exact([z]) }
    let batch = (0..<28).map { _ in z }
    let objective28Seconds = try median { _ = try values(batch) }
    let gradient28Seconds = try median { _ = try exact(batch) }
    let point = try exact([z])[0]
    let gradient = coordinates.gradient(row: point.gradient)
    var report: [String: Any] = [
      "z": z, "agreement": point.agreement,
      "objective_agreement": try values([z])[0], "exact_gradient": gradient,
      "projected_gradient": SSBFifthOrderGradientSearch.projectedGradient(gradient, at: z),
      "objective_s": objectiveSeconds, "value_and_gradient_s": gradientSeconds,
      "objective_28_s": objective28Seconds, "value_and_gradient_28_s": gradient28Seconds,
    ]
    var differences: [[String: Any]] = []
    for step in steps {
      var rows: [[Double]] = []
      for i in 0..<28 {
        var plus = z
        var minus = z
        plus[i] += step
        minus[i] -= step
        rows.append(plus)
        rows.append(minus)
      }
      let v = try values(rows)
      let central = (0..<28).map { (v[2 * $0] - v[2 * $0 + 1]) / (2 * step) }
      let scale = central.map(abs).max() ?? 0
      let error = (0..<28).map { abs(central[$0] - gradient[$0]) }
      differences.append([
        "step": step, "central": central, "max_abs_error": error.max() ?? 0,
        "max_relative_error": (error.max() ?? 0) / max(scale, 1e-300),
        "per_coordinate_relative_error": (0..<28).map {
          error[$0] / max(abs(gradient[$0]), 1e-300)
        },
      ])
    }
    report["central_differences"] = differences
    return report
  }
}
