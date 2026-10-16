import Foundation

public struct SSBJointTrial: Codable, Sendable {
  public let parameters: SSBJointParameters
  public let loss: Double
  public let stage: String
}

public struct SSBJointOptimizationResult: Codable, Sendable {
  public let best: SSBJointParameters
  public let loss: Double
  public let trials: [SSBJointTrial]
  public let elapsedSeconds: Double
  public let seed: UInt64
  public let globalTrials: Int
  public let refinementIterationsPerStart: Int
}

/// Six-dimensional seeded Parzen search followed by bounded Nelder–Mead.
/// Native trial proposals are independent of Optuna; fixed-parameter losses
/// follow the same joint transfer model. Fitting never changes scan rotation.
public struct SSBJointOptimizer: Sendable {
  public let globalTrials: Int
  public let refinementIterations: Int
  public let seed: UInt64
  private let bounds: [ClosedRange<Double>] = [
    -300...300, 0...200, (-Double.pi / 2)...(Double.pi / 2), -25...25, -25...25, 20...600]

  public init(globalTrials: Int = 200, refinementIterations: Int = 200, seed: UInt64 = 42) {
    self.globalTrials = globalTrials; self.refinementIterations = refinementIterations; self.seed = seed
  }

  public func run(start: SSBJointParameters,
                  evaluate: (SSBJointParameters) throws -> Double,
                  progress: (String, Int, Int, Double) -> Void = { _, _, _, _ in },
                  isCancelled: () -> Bool = { false }) throws -> SSBJointOptimizationResult {
    guard globalTrials > 0, refinementIterations > 0 else {
      throw MetalSSBError.invalidGeometry("Use at least one search trial and refinement iteration.")
    }
    let started = Date()
    var random = JointRandom(state: seed), history: [SSBJointTrial] = []
    var best = start, bestLoss = Double.infinity
    func loss(_ values: [Double], stage: String) throws -> Double {
      if isCancelled() { throw CancellationError() }
      let parameters = point(values)
      let value = try evaluate(parameters)
      guard value.isFinite else { throw SSBOptimizationError.nonFiniteLoss }
      history.append(SSBJointTrial(parameters: parameters, loss: value, stage: stage))
      if value < bestLoss { bestLoss = value; best = parameters }
      return value
    }
    _ = try loss(vector(start), stage: "initial")
    for trial in 0..<globalTrials {
      var candidate = bounds.map { random.uniform($0) }
      if trial >= max(30, globalTrials / 6) {
        let sorted = history.sorted { $0.loss < $1.loss }.map { vector($0.parameters) }
        let goodCount = max(1, min(25, Int(ceil(Double(sorted.count) * 0.1))))
        let good = Array(sorted.prefix(goodCount)), bad = Array(sorted.dropFirst(goodCount))
        var score = -Double.infinity
        for _ in 0..<24 {
          let sample = bounds.indices.map { d in sampleParzen(good.map { $0[d] }, bounds: bounds[d], random: &random) }
          let value = bounds.indices.reduce(0.0) { sum, d in
            sum + logDensity(sample[d], good: good.map { $0[d] }, bad: bad.map { $0[d] }, bounds: bounds[d])
          }
          if value > score { score = value; candidate = sample }
        }
      }
      _ = try loss(candidate, stage: "search")
      progress("Search trials", trial + 1, globalTrials, bestLoss)
    }
    // Refine three separated basins. Report iteration count independently of
    // evaluations: a simplex iteration can evaluate more than one candidate.
    var starts: [[Double]] = []
    let separation: [Double] = [50, 20, 0.3, 2, 2, 50]
    for trial in history.sorted(by: { $0.loss < $1.loss }) {
      let v = vector(trial.parameters)
      if starts.allSatisfy({ existing in zip(zip(v, existing), separation).map { abs($0.0.0 - $0.0.1) / $0.1 }.max()! > 1 }) {
        starts.append(v)
      }
      if starts.count == 3 { break }
    }
    let scale = max(abs(bestLoss), 1e-30)
    for (run, initial) in starts.enumerated() {
      var simplex = [initial]
      let minimumSteps = [2.0, 2.0, 0.04, 0.2, 0.2, 2.0]
      for d in bounds.indices {
        var v = initial
        let step = max(abs(v[d]) * 0.05, minimumSteps[d])
        v[d] += v[d] + step <= bounds[d].upperBound ? step : -step
        simplex.append(clamp(v))
      }
      var losses = try simplex.map { try loss($0, stage: "refinement") / scale }
      for iteration in 0..<refinementIterations {
        let order = losses.indices.sorted { losses[$0] < losses[$1] }
        simplex = order.map { simplex[$0] }; losses = order.map { losses[$0] }
        progress("Refinement \(run + 1)/\(starts.count)", iteration + 1, refinementIterations, bestLoss)
        let spread = simplex.dropFirst().flatMap { v in zip(v, simplex[0]).map { abs($0 - $1) } }.max() ?? 0
        if spread < 0.05 && losses.last! - losses[0] < 1e-6 { break }
        let centroid = bounds.indices.map { d in simplex.prefix(6).reduce(0) { $0 + $1[d] } / 6 }
        let reflected = clamp(zip(centroid, simplex[6]).map { 2 * $0 - $1 })
        let r = try loss(reflected, stage: "refinement") / scale
        if r < losses[0] {
          let expanded = clamp(zip(centroid, reflected).map { $0 + 2 * ($1 - $0) })
          let e = try loss(expanded, stage: "refinement") / scale
          simplex[6] = e < r ? expanded : reflected; losses[6] = min(e, r)
        } else if r < losses[5] {
          simplex[6] = reflected; losses[6] = r
        } else {
          let target = r < losses[6] ? reflected : simplex[6]
          let contracted = clamp(zip(centroid, target).map { $0 + 0.5 * ($1 - $0) })
          let c = try loss(contracted, stage: "refinement") / scale
          if c < min(r, losses[6]) { simplex[6] = contracted; losses[6] = c }
          else {
            for i in 1...6 {
              simplex[i] = clamp(zip(simplex[0], simplex[i]).map { $0 + 0.5 * ($1 - $0) })
              losses[i] = try loss(simplex[i], stage: "refinement") / scale
            }
          }
        }
      }
    }
    return SSBJointOptimizationResult(best: best, loss: bestLoss, trials: history,
      elapsedSeconds: Date().timeIntervalSince(started), seed: seed, globalTrials: globalTrials,
      refinementIterationsPerStart: refinementIterations)
  }

  private func vector(_ p: SSBJointParameters) -> [Double] {
    [Double(p.aberrations.c10Angstroms), Double(p.aberrations.c12Angstroms), Double(p.aberrations.phi12Radians),
     Double(p.tiltRowMrad), Double(p.tiltColumnMrad), Double(p.depthAngstroms)]
  }
  private func clamp(_ v: [Double]) -> [Double] {
    v.enumerated().map { d, x in min(bounds[d].upperBound, max(bounds[d].lowerBound, x)) }
  }
  private func point(_ v: [Double]) -> SSBJointParameters {
    let v = clamp(v)
    return SSBJointParameters(aberrations: .init(c10Angstroms: Float(v[0]), c12Angstroms: Float(v[1]), phi12Radians: Float(v[2])),
      depthAngstroms: Float(v[5]), tiltRowMrad: Float(v[3]), tiltColumnMrad: Float(v[4]))
  }
  private func sampleParzen(
    _ values: [Double],
    bounds: ClosedRange<Double>,
    random: inout JointRandom
  ) -> Double {
    guard !values.isEmpty else { return random.uniform(bounds) }
    let center = values[random.index(values.count)]
    let span = bounds.upperBound - bounds.lowerBound
    let neighbors = values.map { abs($0 - center) }.filter { $0 > 0 }.sorted()
    let sigma = max(span * 0.01, neighbors.first ?? span * 0.20)
    return min(
      bounds.upperBound,
      max(bounds.lowerBound, center + sigma * random.normal())
    )
  }

  private func logDensity(
    _ value: Double,
    good: [Double],
    bad: [Double],
    bounds: ClosedRange<Double>
  ) -> Double {
    log(max(parzenDensity(value, values: good, bounds: bounds), 1e-300))
      - log(max(parzenDensity(value, values: bad, bounds: bounds), 1e-300))
  }

  private func parzenDensity(
    _ value: Double,
    values: [Double],
    bounds: ClosedRange<Double>
  ) -> Double {
    let span = bounds.upperBound - bounds.lowerBound
    guard !values.isEmpty else { return 1 / span }
    let sorted = values.sorted()
    var total = 0.0
    for (index, center) in sorted.enumerated() {
      let left = index > 0 ? center - sorted[index - 1] : span * 0.20
      let right =
        index + 1 < sorted.count
        ? sorted[index + 1] - center : span * 0.20
      let sigma = max(span * 0.01, max(left, right))
      let z = (value - center) / sigma
      total += exp(-0.5 * z * z) / sigma
    }
    return total / Double(sorted.count)
  }

}

private struct JointRandom {
  var state: UInt64

  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var value = state
    value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
    value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
    return value ^ (value >> 31)
  }

  mutating func unit() -> Double {
    Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
  }

  mutating func uniform(_ bounds: ClosedRange<Double>) -> Double {
    uniform(bounds.lowerBound, bounds.upperBound)
  }

  mutating func uniform(_ lower: Double, _ upper: Double) -> Double {
    lower + (upper - lower) * unit()
  }

  mutating func index(_ count: Int) -> Int {
    min(count - 1, Int(unit() * Double(count)))
  }

  mutating func normal() -> Double {
    let first = max(unit(), Double.leastNonzeroMagnitude)
    return sqrt(-2 * log(first)) * cos(2 * .pi * unit())
  }
}
