import Foundation

/// The model-based SSB fit: Python's resident multivariate TPE search and trust-region Newton
/// (`quantem.gpu.ssb._resident_search`), ported line by line to host double precision, the arithmetic of its
/// CUDA build. The objective is the least-squares agreement of the transfer model with the measured
/// double-overlap data (`MetalSSBJointModel.agreements`), larger is better.
///
/// Points are (C10, C12, phi12, tilt row, tilt column, depth spread) in engine units: Angstrom, Angstrom, rad,
/// mrad, mrad, Angstrom. The search draws in the unit cube; trials, densities and Newton steps follow the
/// Python kernels, including their float32 literals, so the same scores give the same trials. A joint fit with
/// a search is polished once more from starts in other basins (`_other_basins`).
public enum SSBAgreementSearch {
  /// Evaluates trial points (engine units) and returns their agreements.
  public typealias Evaluate = ([[Double]]) throws -> [Double]

  /// The agreement with its exact gradient and Hessian in the Newton coordinates z = (C10, C12 cos 2 phi12,
  /// C12 sin 2 phi12, tilt row, tilt column, depth spread), at each row of z.
  public struct Derivatives: Sendable {
    public let value: Double
    public let gradient: [Double]
    public let hessian: [[Double]]
    public init(value: Double, gradient: [Double], hessian: [[Double]]) {
      self.value = value
      self.gradient = gradient
      self.hessian = hessian
    }
  }
  public typealias EvaluateDerivatives = ([[Double]]) throws -> [Derivatives]

  /// Python's `DEPTH_SPREAD_RANGE_A`.
  public static let depthSpreadRangeAngstroms: ClosedRange<Double> = 20...600

  public struct Trial: Codable, Sendable {
    public let point: [Double]
    public let agreement: Double
  }

  public struct NewtonStart: Codable, Sendable {
    public let iterations: Int
    /// Objective evaluations: stencil points of the finite-difference Newton, or centers of the exact one (each
    /// with its gradient and Hessian).
    public let evaluations: Int
    public let converged: Bool
  }

  public struct Result: Codable, Sendable {
    /// The fitted point in engine units, phi12 wrapped to [-pi/2, pi/2).
    public let point: [Double]
    public let agreement: Double
    public let trials: [Trial]
    public let newtonStarts: [NewtonStart]
    public let searchSeconds: Double
    public let refinementSeconds: Double
  }

  // MARK: Random streams (MT19937 and the 53-bit uniform of the existing sampler)

  struct MersenneTwister {
    var state = [UInt32](repeating: 0, count: 624)
    var position = 624
    init(seed: UInt32) {
      state[0] = seed
      for i in 1..<624 {
        state[i] = 1_812_433_253 &* (state[i - 1] ^ (state[i - 1] >> 30)) &+ UInt32(i)
      }
    }
    mutating func next() -> UInt32 {
      if position == 624 {
        for i in 0..<624 {
          let y = (state[i] & 0x8000_0000) | (state[(i + 1) % 624] & 0x7fff_ffff)
          state[i] = state[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) != 0 ? 0x9908_b0df : 0)
        }
        position = 0
      }
      var y = state[position]
      position += 1
      y ^= y >> 11
      y ^= (y << 7) & 0x9d2c_5680
      y ^= (y << 15) & 0xefc6_0000
      y ^= y >> 18
      return y
    }
    mutating func uniform() -> Double {
      let a = next() >> 5
      let b = next() >> 6
      return (Double(a) * 67_108_864 + Double(b)) / 9_007_199_254_740_992
    }
  }

  /// `trials` rows of `width` uniforms from one MT19937 stream seeded with `seed` (`_random`).
  static func randoms(seed: Int, trials: Int, width: Int) -> [[Double]] {
    var generator = MersenneTwister(seed: UInt32(truncatingIfNeeded: seed))
    return (0..<trials).map { _ in (0..<width).map { _ in generator.uniform() } }
  }

  // MARK: Real arithmetic helpers (the CUDA Real is a double)

  /// A float32 literal of the kernels, as CUDA's `Real(xf)` holds it.
  @inline(__always) static func f(_ value: Float) -> Double { Double(value) }
  static let pi =
    Double(Float(3.1415927410125732)) + Double(Float(-8.742277657347586e-8))
    + Double(Float(-3.4302490200117637e-15))
  static func cdf(_ x: Double) -> Double { 0.5 * erfc(-x / sqrt(2.0)) }
  static func inverseCDF(_ probability: Double) -> Double {
    var p = probability
    let flip = p > f(0.5)
    if flip { p = 1 - p }
    var x = p < f(0.01) ? -sqrt(max(0, -2 * log(p))) : log(p / (1 - p)) * f(0.62665707)
    let target = log(p)
    let inverseSqrt = f(0.3989422917366028) + f(-1.133517013585105e-8) + f(3.058787842895743e-17)
    for _ in 0..<12 {
      let c = max(cdf(x), f(1e-30))
      let delta = (log(c) - target) * c / (exp(-x * x / 2) * inverseSqrt)
      x -= delta
      if abs(delta) < f(1e-16) { break }
    }
    return flip ? -x : x
  }
  static func atan2r(_ y: Double, _ x: Double) -> Double {
    if abs(x) < f(1e-30) { return y < 0 ? -pi / 2 : pi / 2 }
    var a = atan(y / x)
    if x < 0 { a += y < 0 ? -pi : pi }
    return a
  }
  static func clip(_ x: Double, _ lower: Double, _ upper: Double) -> Double {
    min(max(x, lower), upper)
  }

  /// Descending rank, ties to the earlier trial (`RANK_BODY`).
  static func ranks(_ scores: [Double]) -> [Int] {
    scores.indices.map { i in
      scores.indices.reduce(0) { rank, j in
        rank + ((scores[j] > scores[i] || (!(scores[j] < scores[i]) && j < i)) ? 1 : 0)
      }
    }
  }

  // MARK: The multivariate TPE search (`_search`)

  struct Space {
    let lower: [Double]
    let upper: [Double]
    let fixed: Bool
    let thin: Bool
    var dimensions: Int { fixed || thin ? 3 : 6 }
    func active(_ d: Int) -> Bool { !((fixed && d < 3) || (thin && d >= 3)) }
    func decode(_ unit: [Double]) -> [Double] {
      (0..<6).map { lower[$0] + unit[$0] * (upper[$0] - lower[$0]) }
    }
  }

  static func search(
    _ evaluate: Evaluate, space: Space, trials: Int, seed: Int, isCancelled: () -> Bool
  ) throws -> (points: [[Double]], scores: [Double]) {
    let startup = space.thin ? 10 : max(30, trials / 6)
    let dimensions = space.dimensions
    let uniform = randoms(seed: seed, trials: trials, width: dimensions)
    let tpe = randoms(seed: seed, trials: trials, width: (dimensions + 1) * 24)
    var history: [[Double]] = []
    var scores: [Double] = []
    var offset = 0
    for done in stride(from: 0, to: trials, by: 8) {
      if isCancelled() { throw CancellationError() }
      let batch = min(8, trials - done)
      var points: [[Double]] = []
      if done < startup {
        for b in 0..<batch {
          let row = uniform[done + b]
          points.append(dimensions == 3 ? (space.fixed ? [0, 0, 0] + row : row + [0, 0, 0]) : row)
        }
      } else {
        points = proposals(
          history: history, scores: scores, done: done, batch: batch, offset: offset, random: tpe,
          space: space)
        offset += batch
      }
      let values = try evaluate(points.map(space.decode))
      guard values.count == points.count, values.allSatisfy(\.isFinite) else {
        throw SSBOptimizationError.nonFiniteLoss
      }
      history += points
      scores += values
    }
    return (history.map(space.decode), scores)
  }

  struct Density {
    var good: Bool
    var sigma: Double
    var norm: Double
    var weight: Double
    var order: Int
  }

  /// `DENSITY_BODY` for the `done` trials and the prior (index `done`).
  static func densities(history: [[Double]], scores: [Double], done: Int, space: Space) -> [Density]
  {
    let rank = ranks(scores)
    let goodCount = min(25, (done + 9) / 10)
    return (0...done).map { i in
      var good = i < done ? rank[i] < goodCount : false
      if i == done { good = false }
      var order = 0
      if i < done {
        for j in 0..<i where (rank[j] < goodCount) == good { order += 1 }
      }
      let n = good ? goodCount : done - goodCount
      var sigma = max(
        (1.0 / 5.0) * exp(-log(Double(max(n, 1))) / Double(space.dimensions + 4)),
        1.0 / Double(min(100, n + 2)))
      if i == done { sigma = 1 }
      var norm = 0.0
      for d in 0..<6 where space.active(d) {
        let mu = i < done ? history[i][d] : f(0.5)
        let mass = cdf((1 - mu) / sigma) - cdf(-mu / sigma)
        norm += log(mass) + log(sigma)
      }
      var weight = 1.0
      if n >= 25 && order < n - 25 {
        weight =
          n == 26
          ? 1 / Double(n) : 1 / Double(n) + (1 - 1 / Double(n)) * Double(order) / Double(n - 26)
      }
      return Density(good: good, sigma: sigma, norm: norm, weight: weight, order: order)
    }
  }

  /// One batch of TPE proposals: 24 candidates per trial, the best density ratio kept
  /// (`SAMPLE_BODY`, `COMPONENT_BODY`, the mixture bodies and `SELECT_BODY`).
  static func proposals(
    history: [[Double]], scores: [Double], done: Int, batch: Int, offset: Int, random: [[Double]],
    space: Space
  ) -> [[Double]] {
    let density = densities(history: history, scores: scores, done: done, space: space)
    let below = min(25, (done + 9) / 10)
    var points: [[Double]] = []
    for b in 0..<batch {
      let draws = random[offset + b]
      var bestScore = -Double.infinity
      var best: [Double] = []
      for c in 0..<24 {
        let choose = draws[c] * Double(below + 1)
        let chosen = Int(Float(choose))
        var selected = done
        for i in 0..<done where density[i].good && density[i].order == chosen { selected = i }
        var candidate = [Double](repeating: 0, count: 6)
        for d in 0..<6 where space.active(d) {
          let sigma = selected == done ? 1 : density[selected].sigma
          let mu = selected == done ? f(0.5) : history[selected][d]
          // Optuna's multivariate dimensions are alphabetical after startup.
          var order = d < 3 ? d : (d == 3 ? 5 : (d == 4 ? 4 : 3))
          if space.fixed { order -= 3 }
          let q = draws[24 + order * 24 + c]
          let lo = cdf(-mu / sigma)
          let hi = cdf((1 - mu) / sigma)
          let p = clip(lo + q * (hi - lo), f(1e-18), 1 - f(1e-18))
          candidate[d] = clip(mu + sigma * inverseCDF(p), 0, 1)
        }
        // Log density ratio of the good and bad Parzen mixtures, each with the prior.
        var good = f(-1e30)
        var bad = f(-1e30)
        var goodWeight = 0.0
        var badWeight = 0.0
        var values = [Double](repeating: 0, count: done + 1)
        for i in 0...done {
          let width = density[i].sigma
          var sum = 0.0
          for d in 0..<6 where space.active(d) {
            let mu = i == done ? f(0.5) : history[i][d]
            let delta = candidate[d] - mu
            sum += delta * delta
          }
          let weight = i == done ? 1 : density[i].weight
          values[i] = -sum / (2 * width * width) - density[i].norm + log(weight)
          let prior = i == done
          let w = prior ? 1 : density[i].weight
          if prior || density[i].good {
            good = max(good, values[i])
            goodWeight += w
          }
          if prior || !density[i].good {
            bad = max(bad, values[i])
            badWeight += w
          }
        }
        var goodSum = 0.0
        var badSum = 0.0
        for i in 0...done {
          let prior = i == done
          if prior || density[i].good { goodSum += exp(values[i] - good) }
          if prior || !density[i].good { badSum += exp(values[i] - bad) }
        }
        let score = log(goodSum) + good - log(goodWeight) - log(badSum) - bad + log(badWeight)
        if c == 0 || score > bestScore {
          bestScore = score
          best = candidate
        }
      }
      points.append(best)
    }
    return points
  }

  // MARK: Trust-region Newton (`_newton`)

  static let steps = [2.0, 2.0, 2.0, f(0.25), f(0.25), 5.0]
  static let tolerances = [f(0.05), f(0.05), f(0.05), f(0.005), f(0.005), f(0.5)]
  /// The exact Newton's convergence tolerances: a tenth of the stencil Newton's. Its steps converge
  /// quadratically, so the last center is within about this of the maximum for one or two more centers.
  static let exactTolerances = tolerances.map { $0 / 10 }

  static func encode(_ p: [Double]) -> [Double] {
    var z = p
    z[1] = p[1] * cos(2 * p[2])
    z[2] = p[1] * sin(2 * p[2])
    return z
  }
  static func decode(_ z: [Double]) -> [Double] {
    var p = z
    p[1] = sqrt(max(0, z[1] * z[1] + z[2] * z[2]))
    p[2] = atan2r(z[2], z[1]) / 2
    p[5] = abs(z[5])
    return p
  }

  /// The scaled trust-region step that maximizes the quadratic model (`trust_step`, cyclic Jacobi).
  static func trustStep(
    gradient g: [Double], hessian h: [[Double]], steps: [Double], radius: Double,
    floatLiterals: Bool = true
  )
    -> [Double]
  {
    // Preserve primary resident arithmetic; the fifth-order host controller uses Double literals.
    func literal(_ value: Double) -> Double { floatLiterals ? Double(Float(value)) : value }
    let n = g.count
    var a = (0..<n).map { i in (0..<n).map { j in h[i][j] * steps[i] * steps[j] } }
    var v = (0..<n).map { i in (0..<n).map { j in i == j ? 1.0 : 0.0 } }
    let scaled = (0..<n).map { g[$0] * steps[$0] }
    for _ in 0..<32 {
      var maximum = 0.0
      for p in 0..<n {
        for q in (p + 1)..<max(p + 1, n) {
          let off = a[p][q]
          maximum = max(maximum, abs(off))
          if abs(off) < literal(1e-25) { continue }
          let tau = (a[q][q] - a[p][p]) / (2 * off)
          let t = (tau < 0 ? -1.0 : 1.0) / (abs(tau) + sqrt(1 + tau * tau))
          let c = 1 / sqrt(1 + t * t)
          let s = t * c
          let pp = a[p][p]
          let qq = a[q][q]
          a[p][p] = pp - t * off
          a[q][q] = qq + t * off
          a[p][q] = 0
          a[q][p] = 0
          for k in 0..<n {
            if k != p && k != q {
              let x = a[k][p]
              let y = a[k][q]
              a[k][p] = c * x - s * y
              a[p][k] = a[k][p]
              a[k][q] = s * x + c * y
              a[q][k] = a[k][q]
            }
            let x = v[k][p]
            let y = v[k][q]
            v[k][p] = c * x - s * y
            v[k][q] = s * x + c * y
          }
        }
      }
      if maximum < literal(1e-24) { break }
    }
    var largest = literal(-1e30)
    var gradientNorm = 0.0
    var norm = 0.0
    var along = [Double](repeating: 0, count: n)
    for i in 0..<n {
      largest = max(largest, a[i][i])
      gradientNorm += scaled[i] * scaled[i]
      for j in 0..<n { along[i] += v[j][i] * scaled[j] }
      let denominator = -a[i][i]
      if abs(denominator) > literal(1e-30) {
        let y = along[i] / denominator
        norm += y * y
      } else {
        norm = literal(1e30)
      }
    }
    var mu = 0.0
    if !(largest < 0 && sqrt(norm) <= radius) {
      var low = max(0, largest)
      var high = low + sqrt(gradientNorm) / radius + literal(1e-30)
      low += literal(1e-12) * (abs(low) + 1)
      for _ in 0..<100 {
        let middle = (low + high) / 2
        norm = 0
        for i in 0..<n {
          let y = along[i] / (middle - a[i][i])
          norm += y * y
        }
        if sqrt(norm) > radius { low = middle } else { high = middle }
      }
      mu = high
    }
    return (0..<n).map { i in
      (0..<n).reduce(0.0) { $0 + v[i][$1] * (along[$1] / (mu - a[$1][$1])) } * steps[i]
    }
  }

  /// The trust-region step with coordinates that leave the box held at their bound (`bounded_step`).
  static func boundedStep(
    center: [Double], gradient: [Double], hessian: [[Double]], radius: Double, lower: [Double],
    upper: [Double], free initial: [Bool]
  ) -> [Double] {
    var free = initial
    var held = [Double](repeating: 0, count: 6)
    var move = held
    for _ in 0..<6 {
      move = held
      let ids = (0..<6).filter { free[$0] }
      if !ids.isEmpty {
        let fg = ids.map { d in
          (0..<6).reduce(gradient[d]) { free[$1] ? $0 : $0 + hessian[d][$1] * held[$1] }
        }
        let fh = ids.map { d in ids.map { hessian[d][$0] } }
        let fm = trustStep(gradient: fg, hessian: fh, steps: ids.map { steps[$0] }, radius: radius)
        for (i, d) in ids.enumerated() { move[d] = fm[i] }
      }
      var any = false
      for d in 0..<6
      where free[d] && (center[d] + move[d] < lower[d] || center[d] + move[d] > upper[d]) {
        held[d] = clip(center[d] + move[d], lower[d], upper[d]) - center[d]
        free[d] = false
        any = true
      }
      if !any { break }
    }
    return move
  }

  struct NewtonState {
    var center: [Double]
    var keptCenter = [Double](repeating: 0, count: 6)
    var keptValue = 0.0
    var keptGradient = [Double](repeating: 0, count: 6)
    var keptHessian = [[Double]](repeating: [Double](repeating: 0, count: 6), count: 6)
    var radius = 8.0
    var full = true
    var active = true
    var hasPrevious = false
    var iterations = 0
    var evaluations = 0
    var converged = false
    var heading: [Double]
    /// The start's z.
    var origin: [Double] = []
    /// A start of an earlier polish, present only for the pruning of the others.
    var known = false
  }

  /// `_newton`: up to three starts, the `explicit` points first and then the best distinct trials, 30
  /// iterations. `prune` keeps only the highest start after the second step; without it every start climbs
  /// until it converges or meets a higher one.
  ///
  /// With `derivatives` every iteration evaluates each active center once, with its exact gradient and Hessian,
  /// in place of the finite-difference stencil (2n + 1 points, or n(n + 3)/2 + 1 with the mixed terms), and
  /// converges to `exactTolerances`. The trust region, bounds, acceptance and start pruning are the same.
  static func newton(
    _ evaluate: Evaluate, points: [[Double]], scores: [Double], lower boxLower: [Double],
    upper boxUpper: [Double], explicit: [[Double]], starts requested: Int, fixed: Bool,
    standard: Bool,
    prune: Bool = true, derivatives: EvaluateDerivatives? = nil, known: [NewtonState] = [],
    isCancelled: () -> Bool
  ) throws -> (point: [Double], agreement: Double, starts: [NewtonState]) {
    let starts = min(3, requested)
    let rank = ranks(scores)
    let distinct = [50.0, 20, f(0.3), 2, 2, 50]
    var chosen: [[Double]] = explicit
    for r in 0..<points.count where chosen.count < starts {
      guard let index = rank.firstIndex(of: r) else { continue }
      let p = points[index]
      let different = chosen.allSatisfy { kept in
        (0..<6).reduce(0.0) { max($0, abs(p[$1] - kept[$1]) / distinct[$1]) } > 1
      }
      if different { chosen.append(p) }
    }
    var state =
      known.map { finished -> NewtonState in
        var kept = finished
        kept.active = false
        kept.known = true
        return kept
      }
      + chosen.map { p -> NewtonState in
        let z = encode(p)
        return NewtonState(center: z, heading: z, origin: z)
      }
    let scale = scores.reduce(f(1e-30)) { max($0, abs($1)) }
    var lower = boxLower
    var upper = boxUpper
    lower[1] = -upper[1]
    lower[2] = -upper[1]
    upper[2] = upper[1]
    let first = standard ? 0 : (fixed ? 3 : 0)
    let last = standard ? 3 : 6
    let n = last - first
    let free = (0..<6).map { standard ? $0 < 3 : !(fixed && $0 < 3) }
    let tolerance = derivatives == nil ? tolerances : exactTolerances
    for iteration in 0..<30 {
      if isCancelled() { throw CancellationError() }
      let active = state.indices.filter { state[$0].active }
      if active.isEmpty { break }
      var designs: [[Double]] = []
      var sizes: [Int] = []
      var exact: [Derivatives] = []
      let values: [Double]
      if let derivatives {
        exact = try derivatives(active.map { state[$0].center })
        guard exact.count == active.count,
          exact.allSatisfy({ at in
            at.value.isFinite && at.gradient.count == 6 && at.gradient.allSatisfy(\.isFinite)
              && at.hessian.count == 6
              && at.hessian.allSatisfy { $0.count == 6 && $0.allSatisfy(\.isFinite) }
          })
        else { throw SSBOptimizationError.nonFiniteLoss }
        sizes = active.map { _ in 1 }
        values = exact.map(\.value)
      } else {
        for k in active {
          let size = 1 + 2 * n + (state[k].full ? n * (n - 1) / 2 : 0)
          sizes.append(size)
          for local in 0..<size {
            var z = state[k].center
            if local > 0 && local <= 2 * n {
              let d = first + (local - 1) / 2
              z[d] += steps[d] * (local % 2 == 1 ? 1 : -1)
            }
            if local > 2 * n {
              var pair = 2 * n + 1
              for i in first..<last {
                for j in (i + 1)..<max(i + 1, last) {
                  if pair == local {
                    z[i] += steps[i]
                    z[j] += steps[j]
                  }
                  pair += 1
                }
              }
            }
            designs.append(decode(z))
          }
        }
        values = try evaluate(designs)
        guard values.count == designs.count, values.allSatisfy(\.isFinite) else {
          throw SSBOptimizationError.nonFiniteLoss
        }
      }
      var offset = 0
      for (position, (k, size)) in zip(active, sizes).enumerated() {
        var s = state[k]
        let value = values[offset] / scale
        var center = s.center
        var moved = 0.0
        for d in 0..<6 {
          let delta = (center[d] - s.keptCenter[d]) / steps[d]
          moved += delta * delta
        }
        moved = sqrt(moved)
        var g = [Double](repeating: 0, count: 6)
        var h = s.keptHessian
        if s.hasPrevious && value < s.keptValue {
          center = s.keptCenter
          g = s.keptGradient
          s.radius = min(s.radius, moved) / 4
        } else if !exact.isEmpty {
          let at = exact[position]
          for d in first..<last {
            g[d] = at.gradient[d] / scale
            for e in first..<last { h[d][e] = at.hessian[d][e] / scale }
          }
          if s.hasPrevious && moved >= f(0.99) * s.radius { s.radius *= 2 }
          s.keptCenter = center
          s.keptGradient = g
          s.keptHessian = h
          s.keptValue = value
          s.hasPrevious = true
        } else {
          var plus = [Double](repeating: 0, count: 6)
          for d in first..<last {
            plus[d] = values[offset + 1 + 2 * (d - first)] / scale
            let minus = values[offset + 2 + 2 * (d - first)] / scale
            g[d] = (plus[d] - minus) / (2 * steps[d])
            h[d][d] = (plus[d] - 2 * value + minus) / (steps[d] * steps[d])
          }
          if size > 2 * n + 1 {
            var pair = offset + 2 * n + 1
            for i in first..<last {
              for j in (i + 1)..<max(i + 1, last) {
                let mixed =
                  (values[pair] / scale - plus[i] - plus[j] + value) / (steps[i] * steps[j])
                h[i][j] = mixed
                h[j][i] = mixed
                pair += 1
              }
            }
          }
          if s.hasPrevious && moved >= f(0.99) * s.radius { s.radius *= 2 }
          s.full = s.hasPrevious && moved > 2
          s.keptCenter = center
          s.keptGradient = g
          s.keptHessian = h
          s.keptValue = value
          s.hasPrevious = true
        }
        let move = boundedStep(
          center: center, gradient: g, hessian: h, radius: s.radius, lower: lower, upper: upper,
          free: free)
        var proposal = (0..<6).map { d in
          free[d] ? clip(center[d] + move[d], lower[d], upper[d]) : center[d]
        }
        let norm = sqrt(proposal[1] * proposal[1] + proposal[2] * proposal[2])
        if !fixed && norm > upper[1] {
          proposal[1] = proposal[1] * upper[1] / norm
          proposal[2] = proposal[2] * upper[1] / norm
        }
        let converged = (first..<last).allSatisfy {
          abs(proposal[$0] - center[$0]) < tolerance[$0]
        }
        if converged {
          s.active = false
          s.converged = true
        }
        s.center = converged ? center : proposal
        s.heading = s.center
        s.iterations += 1
        s.evaluations += size
        offset += size
        state[k] = s
      }
      for k in state.indices where state[k].active {
        for j in state.indices where j != k && state[j].hasPrevious {
          var near = true
          var heading = true
          for d in first..<last {
            if abs(state[k].keptCenter[d] - state[j].keptCenter[d]) >= 4 * steps[d] { near = false }
            if abs(state[k].heading[d] - state[j].heading[d]) >= 2 * steps[d] { heading = false }
          }
          let worse =
            state[k].keptValue < state[j].keptValue
            || (!(state[k].keptValue > state[j].keptValue) && k > j)
          if (near || heading) && worse {
            state[k].active = false
            break
          }
        }
      }
      if prune && iteration == 1 {
        var best = -1
        for k in state.indices where state[k].active {
          if best < 0 || state[k].keptValue > state[best].keptValue { best = k }
        }
        for k in state.indices where k != best { state[k].active = false }
      }
    }
    var best = 0
    for k in 1..<max(1, state.count)
    where state[k].hasPrevious && state[k].keptValue > state[best].keptValue {
      best = k
    }
    return (
      decode(state[best].keptCenter), state[best].keptValue * scale, state.filter { !$0.known }
    )
  }

  /// The second polish's relative margin (`BASIN_MARGIN`): a start that climbs back to the same maximum leaves
  /// the first polish in place.
  static let basinMargin = 1e-6

  /// Starts in other basins (`ALTERNATE_STARTS_BODY`): the best trial with phi12 mirrored, and the best trial
  /// whose (phi12, tilt, depth spread) differs from the polished point and that mirror by more than one Newton
  /// start scale (0.3 rad, 2 mrad, 2 mrad, 50 A; phi12 periodic in pi).
  static func alternateStarts(points: [[Double]], scores: [Double], polished: [Double])
    -> [[Double]]
  {
    let scale = [50.0, 20, f(0.3), 2, 2, 50]
    let rank = ranks(scores)
    let best = points[rank.firstIndex(of: 0)!]
    var mirror = best
    mirror[2] = -best[2]
    let halfPi = pi * f(0.5)
    var chosen: [[Double]] = []
    func different(_ p: [Double]) -> Bool {
      for q in [polished] + chosen {
        var phi = abs(p[2] - q[2])
        if phi > halfPi { phi = pi - phi }
        var distance = phi / scale[2]
        for d in 3..<6 { distance = max(distance, abs(p[d] - q[d]) / scale[d]) }
        if distance <= 1 { return false }
      }
      return true
    }
    if different(mirror) { chosen.append(mirror) }
    for r in 0..<points.count where chosen.count < 2 {
      let p = points[rank.firstIndex(of: r)!]
      if different(p) { chosen.append(p) }
    }
    return chosen
  }

  /// `_other_basins`: polish the alternate starts, each to its own maximum, and keep that result only when its
  /// agreement is higher by more than `basinMargin`.
  ///
  /// The exact Newton (`derivatives`) climbs only new basins: an alternate that is one of the first polish's
  /// starts would retrace that start's deterministic climb and is not polished again, and the polished maximum
  /// takes part in the start pruning, so an alternate that comes near it, or heads where it is, while lower,
  /// stops.
  static func otherBasins(
    _ evaluate: Evaluate, points: [[Double]], scores: [Double], lower: [Double], upper: [Double],
    first: (point: [Double], agreement: Double, starts: [NewtonState]),
    derivatives: EvaluateDerivatives? = nil, isCancelled: () -> Bool
  ) throws -> (point: [Double], agreement: Double, starts: [NewtonState]) {
    var alternates = alternateStarts(points: points, scores: scores, polished: first.point)
    var known: [NewtonState] = []
    if derivatives != nil {
      let origins = first.starts.map(\.origin)
      alternates = alternates.filter { !origins.contains(encode($0)) }
      if let polished = first.starts.filter(\.hasPrevious).max(by: { $0.keptValue < $1.keptValue }) {
        var marker = polished
        marker.heading = polished.keptCenter
        known = [marker]
      }
    }
    if alternates.isEmpty { return first }
    let second = try newton(
      evaluate, points: points, scores: scores, lower: lower, upper: upper, explicit: alternates,
      starts: alternates.count, fixed: false, standard: false, prune: false,
      derivatives: derivatives, known: known, isCancelled: isCancelled)
    let starts = first.starts + second.starts
    if second.agreement > first.agreement * (1 + basinMargin) {
      return (second.point, second.agreement, starts)
    }
    return (first.point, first.agreement, starts)
  }

  static func wrapPhi(_ phi: Double) -> Double {
    var wrapped = (phi + .pi / 2).truncatingRemainder(dividingBy: .pi)
    if wrapped < 0 { wrapped += .pi }
    return wrapped - .pi / 2
  }

  // MARK: Public fits

  /// The tilt-corrected fit (`fit_resident`): C10, C12, phi12, sample tilt and depth spread together.
  /// `trials = 0` refines from `start` only.
  public static func fitTilt(
    _ evaluate: Evaluate, trials: Int = 200, seed: Int = 42, tiltLimitMrad: Double = 25,
    c10RangeAngstroms: ClosedRange<Double> = -300...300, c12MaximumAngstroms: Double = 200,
    start: [Double]? = nil, refine: Bool = true, derivatives: EvaluateDerivatives? = nil,
    isCancelled: () -> Bool = { false }
  ) throws -> Result {
    guard trials >= 0, trials > 0 || start != nil else {
      throw MetalSSBError.invalidGeometry(
        "Use at least one trial, or refine from a previous fit's start.")
    }
    let lower = [
      c10RangeAngstroms.lowerBound, 0, -Double.pi / 2, -tiltLimitMrad, -tiltLimitMrad,
      depthSpreadRangeAngstroms.lowerBound,
    ]
    let upper = [
      c10RangeAngstroms.upperBound, c12MaximumAngstroms, Double.pi / 2, tiltLimitMrad,
      tiltLimitMrad,
      depthSpreadRangeAngstroms.upperBound,
    ]
    return try fit(
      evaluate, lower: lower, upper: upper, trials: trials, seed: seed, thin: false, start: start,
      refine: refine, derivatives: derivatives, isCancelled: isCancelled)
  }

  /// The model-based fit at depth spread 0 (`fit_standard_resident`, Newton): C10, C12 and phi12 with the
  /// sample tilt and depth spread held at 0. The default box is MPS's `STANDARD_RANGES`.
  public static func fitStandard(
    _ evaluate: Evaluate, trials: Int = 200, seed: Int = 42,
    c10RangeAngstroms: ClosedRange<Double> = -400...400, c12MaximumAngstroms: Double = 100,
    refine: Bool = true, derivatives: EvaluateDerivatives? = nil,
    isCancelled: () -> Bool = { false }
  ) throws -> Result {
    guard trials > 0 else { throw MetalSSBError.invalidGeometry("Use at least one trial.") }
    let lower = [c10RangeAngstroms.lowerBound, 0, -Double.pi / 2, 0, 0, 0]
    let upper = [c10RangeAngstroms.upperBound, c12MaximumAngstroms, Double.pi / 2, 0, 0, 0]
    return try fit(
      evaluate, lower: lower, upper: upper, trials: trials, seed: seed, thin: true, start: nil,
      refine: refine, derivatives: derivatives, isCancelled: isCancelled)
  }

  /// The tilt fit's standard comparison: the best of a 200-trial search at depth spread 0, unrefined.
  public static func standardComparison(
    _ evaluate: Evaluate, seed: Int = 42, c10RangeAngstroms: ClosedRange<Double> = -300...300,
    c12MaximumAngstroms: Double = 200, isCancelled: () -> Bool = { false }
  ) throws -> Result {
    let lower = [c10RangeAngstroms.lowerBound, 0, -Double.pi / 2, 0, 0, 0]
    let upper = [c10RangeAngstroms.upperBound, c12MaximumAngstroms, Double.pi / 2, 0, 0, 0]
    return try fit(
      evaluate, lower: lower, upper: upper, trials: 200, seed: seed, thin: true, start: nil,
      refine: false, isCancelled: isCancelled)
  }

  static func fit(
    _ evaluate: Evaluate, lower: [Double], upper: [Double], trials: Int, seed: Int, thin: Bool,
    start: [Double]?, refine: Bool, derivatives: EvaluateDerivatives? = nil,
    isCancelled: () -> Bool
  ) throws -> Result {
    let space = Space(lower: lower, upper: upper, fixed: false, thin: thin)
    let begin = Date()
    var points: [[Double]]
    var scores: [Double]
    if trials > 0 {
      (points, scores) = try search(
        evaluate, space: space, trials: trials, seed: seed, isCancelled: isCancelled)
    } else {
      points = [start!]
      scores = try evaluate(points)
    }
    let searchSeconds = Date().timeIntervalSince(begin)
    let refineBegin = Date()
    var point: [Double]
    var agreement: Double
    var newtonStarts: [NewtonStart] = []
    if refine {
      var fitted = try newton(
        evaluate, points: points, scores: scores, lower: lower, upper: upper,
        explicit: thin ? [] : (start.map { [$0] } ?? []), starts: 3, fixed: false, standard: thin,
        derivatives: derivatives, isCancelled: isCancelled)
      if trials > 0 && !thin {
        fitted = try otherBasins(
          evaluate, points: points, scores: scores, lower: lower, upper: upper, first: fitted,
          derivatives: derivatives, isCancelled: isCancelled)
      }
      point = fitted.point
      agreement = fitted.agreement
      if derivatives != nil {
        // Report the fitted point on the search's own objective.
        let values = try evaluate([point])
        guard values.count == 1, values[0].isFinite else { throw SSBOptimizationError.nonFiniteLoss }
        agreement = values[0]
      }
      newtonStarts = fitted.starts.filter(\.hasPrevious).map {
        NewtonStart(iterations: $0.iterations, evaluations: $0.evaluations, converged: $0.converged)
      }
    } else {
      let best = ranks(scores).firstIndex(of: 0)!
      point = points[best]
      agreement = scores[best]
    }
    guard agreement.isFinite, agreement > 0 else {
      throw MetalSSBError.invalidGeometry(
        "The SSB model fit found no positive finite agreement. Check the scan sampling and the signal in the fitting band."
      )
    }
    point[1] = abs(point[1])
    point[2] = wrapPhi(point[2])
    return Result(
      point: point, agreement: agreement,
      trials: trials > 0 ? zip(points, scores).map { Trial(point: $0, agreement: $1) } : [],
      newtonStarts: newtonStarts, searchSeconds: searchSeconds,
      refinementSeconds: Date().timeIntervalSince(refineBegin))
  }
}
