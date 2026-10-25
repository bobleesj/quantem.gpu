import Foundation

/// Bounded trust-region quasi-Newton refinement of the fifth-order agreement with exact gradients.
/// The exact full-data gradient defines the optimum; the Hessian and any BF subsample only steer. The initial
/// Hessian is the symmetrized forward difference of exact gradients on every n-th BF pixel at the start (made
/// negative definite), which also fixes the trust-region scaling D = sqrt|diag H0|. Accepted steps update it
/// with dense BFGS. Each solve first holds coordinates on a bound whose gradient points out of the box, then
/// clips and holds coordinates that leave it. A proposal's value is evaluated first and its gradient pass only
/// when the agreement increases. An optional coarse stage climbs the agreement of every m-th BF pixel first.
/// On the full data a step is accepted only when the full objective increases, the result is the best
/// full-data point, and the search stops when the bounded model step predicts a full-data gain below
/// `Settings.gainTolerance` (normalized by the initial agreement).
public enum SSBFifthOrderGradientSearch {
  /// `bfStride` 1 is the full objective; a larger stride the agreement of every n-th BF pixel (steering only).
  public enum Request: Sendable, Equatable {
    case valueAndGradient(bfStride: Int)
    /// The value only (empty gradient); its projections are kept for `gradientOfLastValue`.
    case value(bfStride: Int)
    /// The gradients of the points of the last `value` request, adding only the gradient pass.
    case gradientOfLastValue(bfStride: Int)

    var stride: Int {
      switch self {
      case .valueAndGradient(let s), .value(let s), .gradientOfLastValue(let s): return s
      }
    }
  }

  /// Values and gradients (d value / d z) of candidate coordinates, divided by the initial full agreement.
  public typealias Evaluate = (_ points: [[Double]], _ request: Request) throws -> [(
    value: Double, gradient: [Double]
  )]

  public enum Update: String, Sendable { case bfgs, sr1 }

  public struct Settings: Sendable {
    /// BF subsample of the initial difference Hessian (1: all BF pixels).
    public var hessianBFStride = 64
    /// BF subsample of an optional coarse stage (1: none).
    public var coarseBFStride = 1
    public var coarseTolerance = 1e-7
    /// Trust radii in units of D = sqrt|diag H0|.
    public var initialRadius = 1.0
    public var fineRadius = 0.1
    public var maxRadius = 64.0
    /// The smallest normalized full-data gain a step must predict (one float32 agreement resolves about 1e-7).
    public var gainTolerance = 1e-8
    /// A rejected step that predicted less than this normalized gain ends the stage: the Double-summed float32
    /// powers vary by a few 1e-8 between nearby points, so such a step cannot be resolved.
    public var noiseFloor = 5e-8
    public var update = Update.bfgs
    public var maxObjectives = 400
    public var maxIterations = 200
    public init() {}
  }

  /// - Parameter startValue: the normalized full agreement at `start` when the caller already has it.
  /// - Returns: the best full-data point; `evaluations` and `executedObjectives` count full-data objective
  ///   passes, `gradientEvaluations` full-data gradient passes, `subsampledGradients` BF-subsample evaluations
  ///   (Hessian columns and coarse steps).
  public static func refine(
    _ evaluate: Evaluate, start: [Double], startValue: Double? = nil,
    bounds: [ClosedRange<Double>] = SSBFifthOrderSearch.bounds,
    settings: Settings = .init(),
    progress: (_ completed: Int, _ total: Int) -> Void = { _, _ in },
    log: ((String) -> Void)? = nil,
    isCancelled: () -> Bool = { false }
  ) throws -> SSBFifthOrderSearch.Result {
    let n = start.count
    guard n >= 6, bounds.count == n, settings.maxObjectives > 1, settings.hessianBFStride >= 1,
      settings.coarseBFStride >= 1,
      zip(start, bounds).allSatisfy({ $0.isFinite && $1.contains($0) }),
      bounds.allSatisfy({
        $0.lowerBound.isFinite && $0.upperBound.isFinite && $0.upperBound > $0.lowerBound
      })
    else {
      throw MetalSSBError.invalidGeometry(
        "The initial fit must lie inside the higher-order bounds. Check calibration and the initial fit."
      )
    }
    let lower = bounds.map(\.lowerBound)
    let upper = bounds.map(\.upperBound)
    var objectives = 0
    var gradients = 0
    var subsampled = 0
    var best = -Double.infinity
    var bestPoint = start
    var trace: [SSBFifthOrderSearch.Sample] = []
    let started = Date()
    // A BF subsample is scaled to equal the full normalized agreement at the start.
    var subsampleScale: [Int: Double] = [:]
    func record(_ point: [Double], _ value: Double) {
      objectives += 1
      if value > best {
        best = value
        bestPoint = point
      }
      trace.append(
        .init(evaluation: objectives, agreement: best, elapsedSeconds: Date().timeIntervalSince(started)))
    }
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
      var values = try evaluate(points, request)
      var withGradient = true
      if case .value = request { withGradient = false }
      guard values.count == points.count,
        values.allSatisfy({
          $0.value.isFinite && $0.value > 0
            && (!withGradient || ($0.gradient.count == n && $0.gradient.allSatisfy(\.isFinite)))
        })
      else {
        throw MetalSSBError.invalidGeometry(
          "The higher-order objective returned nonfinite scores. Check the calibration and fitting band."
        )
      }
      let stride = request.stride
      if stride > 1 {
        subsampled += points.count
        let scale = subsampleScale[stride] ?? 1 / values[0].value
        subsampleScale[stride] = scale
        values = values.map { ($0.value * scale, $0.gradient.map { $0 * scale }) }
        return values
      }
      switch request {
      case .gradientOfLastValue:
        gradients += points.count
      case .valueAndGradient:
        gradients += points.count
        for (point, value) in zip(points, values) { record(point, value.value) }
      case .value:
        for (point, value) in zip(points, values) { record(point, value.value) }
      }
      progress(objectives, settings.maxObjectives)
      return values
    }
    func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
    func times(_ m: [[Double]], _ v: [Double]) -> [Double] { m.map { dot($0, v) } }

    if let startValue { record(start, startValue) }
    var center = start
    // Forward differences of exact subsample gradients (the first call fixes the subsample's scale at the start).
    let h = [0.01, 0.01, 0.01, 0.025, 0.025, 0.025] + Array(repeating: 0.01, count: n - 6)
    let offsets = (0..<n).map { i in center[i] + h[i] <= upper[i] ? h[i] : -h[i] }
    let design = (0..<n).map { i -> [Double] in
      var p = center
      p[i] += offsets[i]
      return p
    }
    let columns = try track([center] + design, .valueAndGradient(bfStride: settings.hessianBFStride))
    var hessian = Array(repeating: Array(repeating: 0.0, count: n), count: n)
    for i in 0..<n {
      for j in 0..<n {
        hessian[i][j] =
          0.5
          * ((columns[j + 1].gradient[i] - columns[0].gradient[i]) / offsets[j]
            + (columns[i + 1].gradient[j] - columns[0].gradient[j]) / offsets[i])
      }
    }
    // Negative definite for the ascent model and BFGS.
    let (eigenvalues, eigenvectors) = symmetricEigen(hessian)
    let largest = eigenvalues.map(abs).max() ?? 1
    let clamped = eigenvalues.map { min($0, -1e-6 * largest) }
    for i in 0..<n {
      for j in 0..<n {
        hessian[i][j] = (0..<n).reduce(0.0) {
          $0 + eigenvectors[i][$1] * clamped[$1] * eigenvectors[j][$1]
        }
      }
    }
    let scaling = (0..<n).map { max(abs(hessian[$0][$0]), 1e-12).squareRoot() }
    log?(
      String(
        format:
          "start, Hessian from %d gradients on every %d-th BF pixel (%d positive eigenvalues clamped), %.3f s",
        n, settings.hessianBFStride, eigenvalues.filter { $0 > -1e-6 * largest }.count,
        Date().timeIntervalSince(started)))
    var iterations = 0
    var radius = settings.initialRadius

    /// One trust-region stage on one BF subsample from `center`, with its value and gradient there.
    func stage(stride: Int, tolerance: Double, value startValue: Double, gradient startGradient: [Double])
      throws -> String
    {
      var value = startValue
      var gradient = startGradient
      while iterations < settings.maxIterations {
        if objectives >= settings.maxObjectives { return "evaluation cap" }
        iterations += 1
        let step = boundedStep(
          gradient: gradient, hessian: hessian, radius: radius, at: center, lower: lower,
          upper: upper, scaling: scaling)
        let proposal = (0..<n).map { min(upper[$0], max(lower[$0], center[$0] + step[$0])) }
        let d = (0..<n).map { proposal[$0] - center[$0] }
        if (d.map(abs).max() ?? 0) < 1e-6 {
          log?(String(format: "%3d stride %d: small step", iterations, stride))
          return "converged"
        }
        let predicted = dot(gradient, d) + 0.5 * dot(d, times(hessian, d))
        if predicted < tolerance {
          log?(String(format: "%3d stride %d predicted %+.3e: converged", iterations, stride, predicted))
          return "converged"
        }
        let trial = try track([proposal], .value(bfStride: stride))[0]
        let actual = trial.value - value
        let ratio = actual / predicted
        let scaledNorm = (0..<n).map { scaling[$0] * d[$0] * scaling[$0] * d[$0] }.reduce(0, +)
          .squareRoot()
        log?(
          String(
            format:
              "%3d stride %d radius %8.4f |Dd| %8.4f predicted %+.3e actual %+.3e ratio %+.3f %@ %.3f s",
            iterations, stride, radius, scaledNorm, predicted, actual, ratio,
            actual > 0 ? "accept" : "reject", Date().timeIntervalSince(started)))
        if actual > 0 {
          let next = try track([proposal], .gradientOfLastValue(bfStride: stride))[0]
          let y = (0..<n).map { next.gradient[$0] - gradient[$0] }
          let hs = times(hessian, d)
          switch settings.update {
          case .bfgs:
            let sy = dot(d, y)
            let shs = dot(d, hs)
            if sy < -1e-16 && shs < 0 {
              for i in 0..<n {
                for j in 0..<n { hessian[i][j] += -hs[i] * hs[j] / shs + y[i] * y[j] / sy }
              }
            }
          case .sr1:
            let r = (0..<n).map { y[$0] - hs[$0] }
            let rs = dot(r, d)
            if abs(rs) > 1e-8 * dot(r, r).squareRoot() * dot(d, d).squareRoot() {
              for i in 0..<n {
                for j in 0..<n { hessian[i][j] += r[i] * r[j] / rs }
              }
            }
          }
          center = proposal
          value = next.value
          gradient = next.gradient
          if ratio > 0.75 && scaledNorm > 0.8 * radius {
            radius = min(radius * 2, settings.maxRadius)
          } else if ratio < 0.25 {
            radius *= 0.5
          }
        } else {
          if predicted < settings.noiseFloor {
            log?(String(format: "%3d stride %d: rejected at the noise floor", iterations, stride))
            return "converged"
          }
          radius *= 0.25
          if radius < 1e-7 { return "trust radius exhausted" }
        }
      }
      return "iteration limit"
    }

    if settings.coarseBFStride > 1 {
      let coarse =
        settings.coarseBFStride == settings.hessianBFStride
        ? columns[0] : try track([center], .valueAndGradient(bfStride: settings.coarseBFStride))[0]
      let coarseStop = try stage(
        stride: settings.coarseBFStride, tolerance: settings.coarseTolerance, value: coarse.value,
        gradient: coarse.gradient)
      log?(
        String(
          format: "coarse stage (every %d-th BF pixel) %@ after %d iterations, %.3f s",
          settings.coarseBFStride, coarseStop, iterations, Date().timeIntervalSince(started)))
      radius = settings.fineRadius
    }
    let fine = try track([center], .valueAndGradient(bfStride: 1))[0]
    let stop = try stage(
      stride: 1, tolerance: settings.gainTolerance, value: fine.value, gradient: fine.gradient)

    var result = SSBFifthOrderSearch.Result(
      coordinates: bestPoint, agreement: best, evaluations: objectives,
      executedObjectives: objectives, iterations: iterations, stop: stop, trace: trace)
    result.gradientEvaluations = gradients
    result.subsampledGradients = subsampled
    result.hessianStride = settings.hessianBFStride
    log?(
      String(
        format:
          "%@ after %d iterations: %d objectives, %d gradients, %d subsample evaluations, %.3f s",
        stop, iterations, objectives, gradients, subsampled, Date().timeIntervalSince(started)))
    return result
  }

  /// Maximize g.d + d'Hd / 2 subject to |D d| <= radius inside the box. Coordinates on a bound whose gradient
  /// points out of the box are held first; coordinates the solve moves outside are clipped, held, and the rest
  /// re-solved. The subproblem uses the eigendecomposition and bisection on the shift.
  static func boundedStep(
    gradient g: [Double], hessian: [[Double]], radius: Double, at z: [Double], lower: [Double],
    upper: [Double], scaling: [Double]
  ) -> [Double] {
    let n = g.count
    var free = (0..<n).map { i in
      !((z[i] <= lower[i] + 1e-12 && g[i] < 0) || (z[i] >= upper[i] - 1e-12 && g[i] > 0))
    }
    var held = Array(repeating: 0.0, count: n)
    var d = held
    for _ in 0..<n {
      d = held
      let ids = (0..<n).filter { free[$0] }
      if !ids.isEmpty {
        let gs = ids.map { i in
          (0..<n).reduce(g[i]) { free[$1] ? $0 : $0 + hessian[i][$1] * held[$1] } / scaling[i]
        }
        // A = -D^-1 H D^-1 (positive definite for an ascent model); minimize -gs.u + u'Au/2, |u| <= r.
        let a = ids.map { i in ids.map { j in -hessian[i][j] / (scaling[i] * scaling[j]) } }
        let (w, v) = symmetricEigen(a)
        let m = ids.count
        let b = (0..<m).map { k in (0..<m).reduce(0.0) { $0 + v[$1][k] * gs[$1] } }
        func norm(_ shift: Double) -> Double {
          (0..<m).reduce(0.0) { $0 + pow(b[$1] / (w[$1] + shift), 2) }.squareRoot()
        }
        let smallest = w.min() ?? 0
        var shift = 0.0
        if !(smallest > 0 && norm(0) <= radius) {
          var lo = max(0, -smallest) + 1e-15
          var hi = lo + 1
          while norm(hi) > radius { hi *= 2 }
          for _ in 0..<80 {
            let mid = 0.5 * (lo + hi)
            if norm(mid) > radius { lo = mid } else { hi = mid }
          }
          shift = hi
        }
        for (index, i) in ids.enumerated() {
          let u = (0..<m).reduce(0.0) { $0 + v[index][$1] * b[$1] / (w[$1] + shift) }
          d[i] = u / scaling[i]
        }
      }
      var outside = false
      for i in 0..<n where free[i] && (z[i] + d[i] < lower[i] || z[i] + d[i] > upper[i]) {
        held[i] = min(upper[i], max(lower[i], z[i] + d[i])) - z[i]
        free[i] = false
        outside = true
      }
      if !outside { return d }
    }
    return d
  }

  /// Eigenvalues and column eigenvectors of a symmetric matrix (cyclic Jacobi, Double).
  static func symmetricEigen(_ matrix: [[Double]]) -> (values: [Double], vectors: [[Double]]) {
    let n = matrix.count
    var a = matrix
    var v = (0..<n).map { i in (0..<n).map { $0 == i ? 1.0 : 0.0 } }
    for _ in 0..<64 {
      var off = 0.0
      for p in 0..<n { for q in (p + 1)..<max(p + 1, n) { off += a[p][q] * a[p][q] } }
      var diagonal = 0.0
      for p in 0..<n { diagonal += a[p][p] * a[p][p] }
      if off <= 1e-30 * max(diagonal, 1e-300) { break }
      for p in 0..<n {
        for q in (p + 1)..<max(p + 1, n) where a[p][q] != 0 {
          let tau = (a[q][q] - a[p][p]) / (2 * a[p][q])
          let t = (tau < 0 ? -1.0 : 1.0) / (abs(tau) + (1 + tau * tau).squareRoot())
          let c = 1 / (1 + t * t).squareRoot()
          let s = t * c
          for k in 0..<n {
            let x = a[k][p]
            let y = a[k][q]
            a[k][p] = c * x - s * y
            a[k][q] = s * x + c * y
          }
          for k in 0..<n {
            let x = a[p][k]
            let y = a[q][k]
            a[p][k] = c * x - s * y
            a[q][k] = s * x + c * y
          }
          for k in 0..<n {
            let x = v[k][p]
            let y = v[k][q]
            v[k][p] = c * x - s * y
            v[k][q] = s * x + c * y
          }
        }
      }
    }
    return ((0..<n).map { a[$0][$0] }, v)
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
    /// `SSBFifthOrderGradientSearch` on the full data: exact gradients, trust-region quasi-Newton.
    case exactGradient
    /// The same, first climbing the agreement of every 8th BF pixel, then converging on the full data.
    case exactGradientCoarseToFine
  }

  /// Refine a low-order fit through fifth order on `model` at its current rotation.
  /// - Returns: the refined point, its agreement (the float32 objective score) and the same per-frequency powers
  ///   summed in Double, the initial agreement that normalizes the trace, and the trace. The exact-gradient search
  ///   compares Double-summed agreements.
  public static func refineFifthOrder(
    _ primary: SSBModelPoint, model: MetalSSBJointModel, search: HigherOrderSearch,
    maxEvaluations: Int = 5000,
    gradientSettings: SSBFifthOrderGradientSearch.Settings = .init(),
    log: ((String) -> Void)? = nil,
    progress: (_ completed: Int, _ total: Int) -> Void = { _, _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> (
    point: SSBModelPoint, agreement: Double, agreement64: Double, normalization: Double,
    result: SSBFifthOrderSearch.Result
  ) {
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms,
      semiangleRadians: model.semiangleRadians)
    let start = coordinates.initial(primary)
    func values(_ z: [Double]) throws -> (agreement: Double, agreement64: Double) {
      let value = try model.fifthOrderAgreementsAndGradients(
        [coordinates.point(z)], packedCoefficients: [coordinates.pack(z)], gradients: false,
        isCancelled: isCancelled)[0]
      return (value.agreement, value.agreement64)
    }
    let initial = try values(start)
    // Bit-identical to the production objective `fifthOrderAgreements` (ssb-fifth-agreement v0.3).
    let normalization = initial.agreement
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
    case .exactGradient, .exactGradientCoarseToFine:
      var settings = gradientSettings
      settings.maxObjectives = min(settings.maxObjectives, maxEvaluations)
      if search == .exactGradientCoarseToFine && settings.coarseBFStride == 1 {
        settings.coarseBFStride = 8
      }
      // A BF subsample steers only while its agreement depends on the coordinates (that of a single pixel is
      // constant): keep at least 32 pixels, so a small bright field uses all of them.
      let usable = max(1, model.brightFieldPixelCount / 32)
      settings.hessianBFStride = min(settings.hessianBFStride, usable)
      settings.coarseBFStride = min(settings.coarseBFStride, usable)
      result = try SSBFifthOrderGradientSearch.refine(
        { rows, request in
          let points = rows.map(coordinates.point)
          let packed = rows.map(coordinates.pack)
          let values: [(agreement: Double, agreement64: Double, gradient: [Double])]
          switch request {
          case .valueAndGradient(let stride):
            values = try model.fifthOrderAgreementsAndGradients(
              points, packedCoefficients: packed, kStride: stride, isCancelled: isCancelled)
          case .value(let stride):
            values = try model.fifthOrderAgreementsAndGradients(
              points, packedCoefficients: packed, kStride: stride, gradients: false,
              isCancelled: isCancelled)
          case .gradientOfLastValue(let stride):
            values = try model.fifthOrderRetainedGradients(
              points, packedCoefficients: packed, kStride: stride, isCancelled: isCancelled)
          }
          return values.map {
            (
              value: $0.agreement64 / normalization,
              gradient: $0.gradient.isEmpty
                ? [] : coordinates.gradient(row: $0.gradient).map { $0 / normalization }
            )
          }
        }, start: start, startValue: initial.agreement64 / normalization, settings: settings,
        progress: progress, log: log, isCancelled: isCancelled)
    }
    let final = try values(result.coordinates)
    return (
      coordinates.point(result.coordinates), final.agreement, final.agreement64, normalization,
      result
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
    func exact(_ rows: [[Double]]) throws -> [(agreement: Double, agreement64: Double, gradient: [Double])] {
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

extension SSBModelFit {
  /// Benchmark diagnostics: the fifth-order agreement at dimensionless coordinates `z` with the float32
  /// per-frequency powers summed in Double.
  public static func fifthOrderAgreement64(model: MetalSSBJointModel, z: [Double]) throws -> Double {
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms,
      semiangleRadians: model.semiangleRadians)
    return try model.fifthOrderAgreementsAndGradients(
      [coordinates.point(z)], packedCoefficients: [coordinates.pack(z)], gradients: false)[0]
      .agreement64
  }
}

extension SSBModelFit {
  /// Benchmark diagnostics: the symmetrized central-difference Hessian of exact full-data gradients
  /// (d^2 agreement / dz^2) at `z`, from 56 value-and-gradient evaluations.
  public static func fifthOrderHessian(model: MetalSSBJointModel, z: [Double], step: Double = 2e-3)
    throws -> [[Double]]
  {
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms,
      semiangleRadians: model.semiangleRadians)
    var rows: [[Double]] = []
    for i in 0..<28 {
      for sign in [1.0, -1.0] {
        var p = z
        p[i] += sign * step
        rows.append(p)
      }
    }
    let values = try model.fifthOrderAgreementsAndGradients(
      rows.map(coordinates.point), packedCoefficients: rows.map(coordinates.pack))
    let g = values.map { coordinates.gradient(row: $0.gradient) }
    var hessian = Array(repeating: Array(repeating: 0.0, count: 28), count: 28)
    for i in 0..<28 {
      for j in 0..<28 {
        hessian[i][j] =
          0.5
          * ((g[2 * j][i] - g[2 * j + 1][i]) / (2 * step) + (g[2 * i][j] - g[2 * i + 1][j]) / (2 * step))
      }
    }
    return hessian
  }
}

extension SSBModelFit {
  /// Benchmark diagnostics: the exact full-data gradient d agreement / dz at `z`.
  public static func fifthOrderGradient(model: MetalSSBJointModel, z: [Double]) throws -> [Double] {
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms,
      semiangleRadians: model.semiangleRadians)
    let value = try model.fifthOrderAgreementsAndGradients(
      [coordinates.point(z)], packedCoefficients: [coordinates.pack(z)])[0]
    return coordinates.gradient(row: value.gradient)
  }
}

extension SSBModelFit {
  /// Benchmark diagnostics: GPU (command buffer) and wall time per call of the exact-gradient search's fifth-order
  /// evaluations at `z`, median of `repeats`: the value pass alone, the gradient pass after it, and both together,
  /// on all BF pixels and on every `coarseStride`-th; then the start Hessian batch (`z` and its 28 forward offsets)
  /// on every `hessianStride`-th BF pixel. Each entry records its agreements and the first candidate's d agreement
  /// / dz, so kernels can be compared at equal coordinates.
  public static func fifthOrderKernelTimings(
    model: MetalSSBJointModel, z: [Double], coarseStride: Int = 8, hessianStride: Int = 64,
    repeats: Int = 5
  ) throws -> [[String: Any]] {
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms,
      semiangleRadians: model.semiangleRadians)
    typealias Values = [(agreement: Double, agreement64: Double, gradient: [Double])]
    var entries: [[String: Any]] = []
    func measure(
      _ name: String, _ zs: [[Double]], prepare: ([SSBModelPoint], [[Float]]) throws -> Void = { _, _ in },
      _ call: ([SSBModelPoint], [[Float]]) throws -> Values
    ) throws {
      let points = zs.map(coordinates.point)
      let packed = zs.map(coordinates.pack)
      try prepare(points, packed)
      var values = try call(points, packed)
      var gpu: [Double] = []
      var wall: [Double] = []
      for _ in 0..<max(1, repeats) {
        try prepare(points, packed)
        let started = Date()
        values = try call(points, packed)
        wall.append(Date().timeIntervalSince(started))
        gpu.append(model.lastAgreementGPUSeconds)
      }
      entries.append([
        "name": name, "candidates": zs.count, "gpu_ms": gpu.sorted()[gpu.count / 2] * 1e3,
        "wall_ms": wall.sorted()[wall.count / 2] * 1e3, "gpu_s": gpu, "wall_s": wall,
        "agreements": values.map(\.agreement), "agreements64": values.map(\.agreement64),
        "gradient": values[0].gradient.isEmpty ? [] : coordinates.gradient(row: values[0].gradient),
      ])
    }
    for stride in [1, coarseStride] {
      try measure("value stride \(stride)", [z]) {
        try model.fifthOrderAgreementsAndGradients($0, packedCoefficients: $1, kStride: stride, gradients: false)
      }
      try measure(
        "gradient stride \(stride)", [z],
        prepare: {
          _ = try model.fifthOrderAgreementsAndGradients(
            $0, packedCoefficients: $1, kStride: stride, gradients: false)
        }
      ) { try model.fifthOrderRetainedGradients($0, packedCoefficients: $1, kStride: stride) }
      try measure("value+gradient stride \(stride)", [z]) {
        try model.fifthOrderAgreementsAndGradients($0, packedCoefficients: $1, kStride: stride)
      }
    }
    let h = [0.01, 0.01, 0.01, 0.025, 0.025, 0.025] + Array(repeating: 0.01, count: 22)
    let bounds = SSBFifthOrderSearch.bounds
    let design = (0..<28).map { i -> [Double] in
      var p = z
      p[i] += p[i] + h[i] <= bounds[i].upperBound ? h[i] : -h[i]
      return p
    }
    try measure("hessian batch stride \(hessianStride)", [z] + design) {
      try model.fifthOrderAgreementsAndGradients($0, packedCoefficients: $1, kStride: hessianStride)
    }
    return entries
  }
}
