import Foundation

/// Bounded finite-difference Newton refinement used by the fifth-order Python fitter.
/// The controller uses Double; the caller evaluates batches of scientific objectives on the GPU.
public enum SSBFifthOrderSearch {
  public typealias Evaluate = ([[Double]]) throws -> [Double]

  public struct Sample: Codable, Sendable {
    public let evaluation: Int
    public let agreement: Double
    public let elapsedSeconds: Double
  }

  public struct Result: Codable, Sendable {
    public let coordinates: [Double]
    public let agreement: Double
    public let evaluations: Int
    public let executedObjectives: Int
    public let iterations: Int
    public let stop: String
    public let trace: [Sample]
    /// Exact-gradient search only: full-band gradient passes, gradient evaluations on a band subsample for the
    /// steering Hessian, and the subsample stride. `evaluations` then counts full-band objective passes.
    public var gradientEvaluations: Int? = nil
    public var subsampledGradients: Int? = nil
    public var hessianStride: Int? = nil
  }

  /// The same dimensionless box as `fifth_order_fit.py`: C10, Cartesian C12, tilt, depth,
  /// then Cartesian coefficients scaled to unit phase at the aperture edge.
  public static let bounds: [ClosedRange<Double>] =
    [
      -3...3, -2...2, -2...2, -5...5, -5...5, 0.2...6,
    ] + Array(repeating: -2...2, count: 22)

  private enum Budget: Error { case reached }

  /// Refine the initial point, retaining the best of every stencil and proposal evaluation.
  /// A finite evaluation cap is a safety bound, not a convergence criterion. Identical points reuse
  /// exact scores without changing their place in the evaluation budget or convergence trace.
  public static func refine(
    _ evaluate: Evaluate, start: [Double], bounds: [ClosedRange<Double>] = bounds,
    maxEvaluations: Int = 5000,
    progress: (_ completed: Int, _ total: Int) -> Void = { _, _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> Result {
    let n = start.count
    guard n >= 6, bounds.count == n, maxEvaluations > 0,
      zip(start, bounds).allSatisfy({ $0.isFinite && $1.contains($0) }),
      bounds.allSatisfy({
        $0.lowerBound.isFinite && $0.upperBound.isFinite && $0.upperBound > $0.lowerBound
      })
    else {
      throw MetalSSBError.invalidGeometry(
        "The initial fit must lie inside the higher-order bounds. Check calibration and the initial fit."
      )
    }
    let h = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05] + Array(repeating: 0.02, count: n - 6)
    var best = -Double.infinity
    var bestPoint = start
    var evaluations = 0
    var cache: [[UInt64]: Double] = [:]
    var trace: [Sample] = []
    let started = Date()
    func track(_ points: [[Double]]) throws -> [Double] {
      if isCancelled() { throw CancellationError() }
      guard evaluations < maxEvaluations else { throw Budget.reached }
      let selected = Array(points.prefix(maxEvaluations - evaluations))
      guard
        selected.allSatisfy({ row in
          row.count == n
            && zip(row, bounds).allSatisfy {
              $0.isFinite && $0 >= $1.lowerBound - 1e-9 && $0 <= $1.upperBound + 1e-9
            }
        })
      else { throw MetalSSBError.invalidGeometry("A Newton candidate left its calibrated bounds.") }
      var missing: [[Double]] = []
      var keys: [[UInt64]] = []
      var seen = Set<[UInt64]>()
      for point in selected {
        let key = point.map(\.bitPattern)
        if cache[key] == nil, seen.insert(key).inserted {
          missing.append(point)
          keys.append(key)
        }
      }
      if !missing.isEmpty {
        let scores = try evaluate(missing)
        guard scores.count == missing.count, scores.allSatisfy(\.isFinite) else {
          throw MetalSSBError.invalidGeometry(
            "The higher-order objective returned nonfinite scores. Check the calibration and fitting band."
          )
        }
        for (key, score) in zip(keys, scores) { cache[key] = score }
      }
      var values: [Double] = []
      for point in selected {
        let score = cache[point.map(\.bitPattern)]!
        values.append(score)
        evaluations += 1
        if score > best {
          best = score
          bestPoint = point
        }
        trace.append(
          Sample(
            evaluation: evaluations, agreement: best,
            elapsedSeconds: Date().timeIntervalSince(started)))
      }
      progress(evaluations, maxEvaluations)
      if selected.count < points.count { throw Budget.reached }
      return values
    }
    var center = start
    var radius = 8.0
    var curvature = Array(repeating: Array(repeating: 0.0, count: n), count: n)
    var accepted = 0
    var iterations = 0
    var stop = "iteration limit"
    do {
      _ = try track([start])
      for iteration in 0..<200 {
        iterations = iteration + 1
        let design = try stencil(
          center, h: h, bounds: bounds, full: iteration == 0 || accepted % 4 == 0)
        let values = try track(design.points)
        let local = model(values: values, offsets: design.offsets, old: curvature)
        curvature = local.hessian
        let delta = boundedStep(
          center: center, gradient: local.gradient, hessian: curvature,
          steps: h, radius: radius, bounds: bounds)
        let proposal = (0..<n).map {
          min(bounds[$0].upperBound, max(bounds[$0].lowerBound, center[$0] + delta[$0]))
        }
        if (0..<n).map({ abs(delta[$0]) / h[$0] }).max()! < 0.01 {
          stop = "small bounded Newton step"
          break
        }
        let score = try track([proposal])[0]
        var predicted = 0.0
        for i in 0..<n {
          predicted += local.gradient[i] * delta[i]
          for j in 0..<n { predicted += 0.5 * delta[i] * curvature[i][j] * delta[j] }
        }
        let actual = score - values[0]
        if actual > 0 {
          center = proposal
          accepted += 1
          if predicted > 0 && actual / predicted > 0.75 {
            radius = min(radius * 2, 64)
          } else if predicted <= 0 || actual / predicted < 0.25 {
            radius *= 0.5
          }
        } else {
          radius *= 0.25
        }
        if radius < 0.01 {
          stop = "trust radius exhausted"
          break
        }
      }
    } catch Budget.reached { stop = "evaluation cap" }
    return Result(
      coordinates: bestPoint, agreement: best, evaluations: evaluations,
      executedObjectives: cache.count, iterations: iterations, stop: stop, trace: trace)
  }

  static func stencil(_ center: [Double], h: [Double], bounds: [ClosedRange<Double>], full: Bool)
    throws -> (points: [[Double]], offsets: [(Double, Double)])
  {
    var points = [center]
    var offsets: [(Double, Double)] = []
    for i in center.indices {
      let lo = bounds[i].lowerBound
      let hi = bounds[i].upperBound
      let a: Double
      let b: Double
      if center[i] - h[i] >= lo && center[i] + h[i] <= hi {
        a = h[i]
        b = -h[i]
      } else if hi - center[i] >= center[i] - lo {
        a = min(h[i], (hi - center[i]) / 2)
        b = 2 * a
      } else {
        a = -min(h[i], (center[i] - lo) / 2)
        b = 2 * a
      }
      guard abs(a) > 1e-12 else {
        throw MetalSSBError.invalidGeometry(
          "The Newton search range is too narrow for a finite-difference stencil.")
      }
      offsets.append((a, b))
      for delta in [a, b] {
        var p = center
        p[i] += delta
        points.append(p)
      }
    }
    if full {
      for i in center.indices {
        for j in (i + 1)..<center.count {
          var p = center
          p[i] += offsets[i].0
          p[j] += offsets[j].0
          points.append(p)
        }
      }
    }
    return (points, offsets)
  }

  static func model(values: [Double], offsets: [(Double, Double)], old: [[Double]])
    -> (gradient: [Double], hessian: [[Double]])
  {
    let n = offsets.count
    var gradient = Array(repeating: 0.0, count: n)
    var hessian = old
    for i in 0..<n {
      let (a, b) = offsets[i]
      let fa = values[1 + 2 * i]
      let fb = values[2 + 2 * i]
      let f0 = values[0]
      gradient[i] = -(a + b) / (a * b) * f0 - b / (a * (a - b)) * fa - a / (b * (b - a)) * fb
      hessian[i][i] = 2 / (a * b) * f0 + 2 / (a * (a - b)) * fa + 2 / (b * (b - a)) * fb
    }
    if values.count > 1 + 2 * n {
      var k = 1 + 2 * n
      for i in 0..<n {
        for j in (i + 1)..<n {
          let value =
            (values[k] - values[1 + 2 * i] - values[1 + 2 * j] + values[0])
            / (offsets[i].0 * offsets[j].0)
          hessian[i][j] = value
          hessian[j][i] = value
          k += 1
        }
      }
    }
    return (gradient, hessian)
  }

  static func boundedStep(
    center: [Double], gradient: [Double], hessian: [[Double]],
    steps: [Double], radius: Double, bounds: [ClosedRange<Double>]
  ) -> [Double] {
    let n = center.count
    var free = Array(repeating: true, count: n)
    var held = Array(repeating: 0.0, count: n)
    var move = held
    for _ in 0..<n {
      move = held
      let ids = (0..<n).filter { free[$0] }
      if !ids.isEmpty {
        let g = ids.map { i in
          (0..<n).reduce(gradient[i]) { free[$1] ? $0 : $0 + hessian[i][$1] * held[$1] }
        }
        let h = ids.map { i in ids.map { hessian[i][$0] } }
        let step = SSBAgreementSearch.trustStep(
          gradient: g, hessian: h, steps: ids.map { steps[$0] }, radius: radius,
          floatLiterals: false)
        for (index, i) in ids.enumerated() { move[i] = step[index] }
      }
      var outside = false
      for i in 0..<n where free[i] && !bounds[i].contains(center[i] + move[i]) {
        held[i] =
          min(bounds[i].upperBound, max(bounds[i].lowerBound, center[i] + move[i])) - center[i]
        free[i] = false
        outside = true
      }
      if !outside { break }
    }
    return move
  }
}
