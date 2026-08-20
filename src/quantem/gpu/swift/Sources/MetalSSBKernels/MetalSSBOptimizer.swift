import Foundation

public struct SSBOptimizationPoint: Equatable, Sendable {
  public var c10Nanometers: Double
  public var c12Nanometers: Double
  public var phi12Radians: Double

  public var phi12Degrees: Double { phi12Radians * 180 / .pi }

  public init(
    c10Nanometers: Double,
    c12Nanometers: Double,
    phi12Radians: Double
  ) {
    self.c10Nanometers = c10Nanometers
    self.c12Nanometers = c12Nanometers
    self.phi12Radians = phi12Radians
  }
}

public struct SSBOptimizationTrial: Sendable {
  public let point: SSBOptimizationPoint
  public let loss: Double
  public let stage: String
}

public struct SSBOptimizationProgress: Sendable {
  public let stage: String
  public let completed: Int
  public let total: Int
  public let best: SSBOptimizationPoint
  public let bestLoss: Double
}

public struct SSBOptimizationResult: Sendable {
  public let best: SSBOptimizationPoint
  public let loss: Double
  public let trials: [SSBOptimizationTrial]
  public let globalTrials: Int
  public let refinementEvaluations: Int
  public let elapsedSeconds: Double
}

public enum SSBOptimizationError: LocalizedError {
  case nonFiniteLoss
  case invalidBatchResult(expected: Int, actual: Int)

  public var errorDescription: String? {
    switch self {
    case .nonFiniteLoss:
      "The exact SSB objective returned a non-finite loss."
    case .invalidBatchResult(let expected, let actual):
      "The exact SSB batch objective returned \(actual) losses for \(expected) candidates."
    }
  }
}

/// Deterministic full-objective TPE search followed by QuantEM-compatible
/// Nelder–Mead refinement. The evaluator owns the Metal synchronization and
/// must return the native phase-variance loss for the caller's explicit BF
/// selection contract.
public struct SSBOptimizer: Sendable {
  public let globalTrials: Int
  public let seed: UInt64

  public init(globalTrials: Int = 200, seed: UInt64 = 42) {
    self.globalTrials = globalTrials
    self.seed = seed
  }

  public func run(
    start: SSBOptimizationPoint,
    evaluate: (SSBOptimizationPoint) throws -> Double,
    evaluateBatch: (([SSBOptimizationPoint]) throws -> [Double])? = nil,
    progress: (SSBOptimizationProgress) -> Void = { _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> SSBOptimizationResult {
    let started = Date()
    var random = SplitMix64(state: seed)
    var history: [SSBOptimizationTrial] = []
    var best = start
    var bestLoss = try checkedEvaluate(
      start, evaluate: evaluate, isCancelled: isCancelled
    )
    history.append(
      SSBOptimizationTrial(
        point: start, loss: bestLoss, stage: "initial"
      ))
    progress(
      SSBOptimizationProgress(
        stage: "initial", completed: 0, total: globalTrials,
        best: best, bestLoss: bestLoss
      ))

    var trial = 0
    while trial < globalTrials {
      try checkCancellation(isCancelled)
      let batchCount =
        evaluateBatch == nil
        ? 1 : min(2, globalTrials - trial)
      var candidates = [SSBOptimizationPoint]()
      candidates.reserveCapacity(batchCount)
      for offset in 0..<batchCount {
        candidates.append(
          trial + offset < 10
            ? uniformCandidate(random: &random)
            : tpeCandidate(history: history, random: &random))
      }
      let losses: [Double]
      if let evaluateBatch, candidates.count > 1 {
        losses = try checkedEvaluateBatch(
          candidates, evaluateBatch: evaluateBatch,
          isCancelled: isCancelled
        )
      } else {
        losses = [
          try checkedEvaluate(
            candidates[0], evaluate: evaluate,
            isCancelled: isCancelled
          )
        ]
      }
      for (candidate, loss) in zip(candidates, losses) {
        history.append(
          SSBOptimizationTrial(
            point: candidate, loss: loss, stage: "tpe"
          ))
        if loss < bestLoss {
          best = candidate
          bestLoss = loss
        }
        trial += 1
        progress(
          SSBOptimizationProgress(
            stage: "tpe", completed: trial, total: globalTrials,
            best: best, bestLoss: bestLoss
          ))
      }
    }

    let refined = try nelderMead(
      start: best,
      startLoss: bestLoss,
      evaluate: evaluate,
      history: &history,
      progress: progress,
      isCancelled: isCancelled
    )
    return SSBOptimizationResult(
      best: refined.point,
      loss: refined.loss,
      trials: history,
      globalTrials: globalTrials,
      refinementEvaluations: refined.evaluations,
      elapsedSeconds: Date().timeIntervalSince(started)
    )
  }

  private func uniformCandidate(random: inout SplitMix64) -> SSBOptimizationPoint {
    SSBOptimizationPoint(
      c10Nanometers: random.uniform(-400, 400),
      c12Nanometers: random.uniform(0, 100),
      phi12Radians: random.uniform(-.pi / 2, .pi / 2)
    )
  }

  private func tpeCandidate(
    history: [SSBOptimizationTrial],
    random: inout SplitMix64
  ) -> SSBOptimizationPoint {
    let ordered = history.enumerated().sorted { first, second in
      if first.element.loss == second.element.loss {
        return first.offset < second.offset
      }
      return first.element.loss < second.element.loss
    }.map(\.element)
    let goodCount = max(1, min(25, Int(ceil(Double(ordered.count) * 0.10))))
    let good = Array(ordered.prefix(goodCount).map(\.point))
    let bad = Array(ordered.dropFirst(goodCount).map(\.point))
    var winner = uniformCandidate(random: &random)
    var winnerScore = -Double.infinity
    for _ in 0..<24 {
      let sampled = SSBOptimizationPoint(
        c10Nanometers: sampleParzen(
          good.map(\.c10Nanometers), bounds: -400...400,
          random: &random
        ),
        c12Nanometers: sampleParzen(
          good.map(\.c12Nanometers), bounds: 0...100,
          random: &random
        ),
        phi12Radians: sampleParzen(
          good.map(\.phi12Radians), bounds: (-Double.pi / 2)...(.pi / 2),
          random: &random
        )
      )
      let score =
        logDensity(
          sampled.c10Nanometers,
          good: good.map(\.c10Nanometers), bad: bad.map(\.c10Nanometers),
          bounds: -400...400
        )
        + logDensity(
          sampled.c12Nanometers,
          good: good.map(\.c12Nanometers), bad: bad.map(\.c12Nanometers),
          bounds: 0...100
        )
        + logDensity(
          sampled.phi12Radians,
          good: good.map(\.phi12Radians), bad: bad.map(\.phi12Radians),
          bounds: (-Double.pi / 2)...(.pi / 2)
        )
      if score > winnerScore {
        winner = sampled
        winnerScore = score
      }
    }
    return winner
  }

  private func sampleParzen(
    _ values: [Double],
    bounds: ClosedRange<Double>,
    random: inout SplitMix64
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

  private func nelderMead(
    start: SSBOptimizationPoint,
    startLoss: Double,
    evaluate: (SSBOptimizationPoint) throws -> Double,
    history: inout [SSBOptimizationTrial],
    progress: (SSBOptimizationProgress) -> Void,
    isCancelled: () -> Bool
  ) throws -> (point: SSBOptimizationPoint, loss: Double, evaluations: Int) {
    var simplex = [vector(start)]
    for index in 0..<3 {
      var point = simplex[0]
      let floor = index == 1 ? 2.0 : (index == 2 ? 0.04 : 0.00025)
      let step = max(abs(point[index]) * 0.05, floor)
      point[index] += (step * 100).rounded() / 100
      simplex.append(point)
    }
    var losses = [startLoss]
    var evaluations = 0
    var cache: [Float32ObjectiveKey: Double] = [:]

    func evaluateVector(_ values: [Double]) throws -> Double {
      try checkCancellation(isCancelled)
      let point = point(values)
      let key = Float32ObjectiveKey(point)
      let loss: Double
      if let cached = cache[key] {
        loss = cached
      } else {
        loss = try checkedEvaluate(
          point, evaluate: evaluate, isCancelled: isCancelled
        )
        cache[key] = loss
        evaluations += 1
      }
      history.append(
        SSBOptimizationTrial(
          point: point, loss: loss, stage: "nelder-mead"
        ))
      return loss
    }
    for index in 1..<simplex.count {
      losses.append(try evaluateVector(simplex[index]))
    }

    for iteration in 0..<80 {
      let order = losses.indices.sorted { first, second in
        if losses[first] == losses[second] {
          return first < second
        }
        return losses[first] < losses[second]
      }
      simplex = order.map { simplex[$0] }
      losses = order.map { losses[$0] }
      let coordinateSpread =
        zip(simplex.last!, simplex[0])
        .map { abs($0 - $1) }.max() ?? 0
      let lossSpread = abs(losses.last! - losses[0])
      progress(
        SSBOptimizationProgress(
          stage: "nelder-mead", completed: iteration, total: 80,
          best: point(simplex[0]), bestLoss: losses[0]
        ))
      if coordinateSpread < 0.1 && lossSpread < 3e-6 { break }

      let centroid = (0..<3).map { dimension in
        simplex[0..<3].reduce(0) { $0 + $1[dimension] } / 3
      }
      let reflected = zip(centroid, simplex[3]).map { $0 + ($0 - $1) }
      let reflectedLoss = try evaluateVector(reflected)
      if losses[0] <= reflectedLoss && reflectedLoss < losses[2] {
        simplex[3] = reflected
        losses[3] = reflectedLoss
        continue
      }
      if reflectedLoss < losses[0] {
        let expanded = zip(centroid, reflected).map { $0 + 2 * ($1 - $0) }
        let expandedLoss = try evaluateVector(expanded)
        if expandedLoss < reflectedLoss {
          simplex[3] = expanded
          losses[3] = expandedLoss
        } else {
          simplex[3] = reflected
          losses[3] = reflectedLoss
        }
        continue
      }

      let contracted: [Double]
      if reflectedLoss < losses[3] {
        contracted = zip(centroid, reflected).map { $0 + 0.5 * ($1 - $0) }
      } else {
        contracted = zip(centroid, simplex[3]).map { $0 - 0.5 * ($0 - $1) }
      }
      let contractedLoss = try evaluateVector(contracted)
      if contractedLoss < min(reflectedLoss, losses[3]) {
        simplex[3] = contracted
        losses[3] = contractedLoss
        continue
      }
      for index in 1..<4 {
        simplex[index] = zip(simplex[0], simplex[index]).map {
          $0 + 0.5 * ($1 - $0)
        }
        losses[index] = try evaluateVector(simplex[index])
      }
    }
    let bestIndex = losses.indices.min { losses[$0] < losses[$1] }!
    return (point(simplex[bestIndex]), losses[bestIndex], evaluations)
  }

  private func vector(_ point: SSBOptimizationPoint) -> [Double] {
    [point.c10Nanometers, point.c12Nanometers, point.phi12Radians]
  }

  private func point(_ values: [Double]) -> SSBOptimizationPoint {
    SSBOptimizationPoint(
      c10Nanometers: values[0],
      c12Nanometers: max(0, values[1]),
      phi12Radians: values[2]
    )
  }

  private func checkedEvaluate(
    _ point: SSBOptimizationPoint,
    evaluate: (SSBOptimizationPoint) throws -> Double,
    isCancelled: () -> Bool
  ) throws -> Double {
    try checkCancellation(isCancelled)
    let loss = try evaluate(point)
    guard loss.isFinite else {
      throw SSBOptimizationError.nonFiniteLoss
    }
    return loss
  }

  private func checkedEvaluateBatch(
    _ points: [SSBOptimizationPoint],
    evaluateBatch: ([SSBOptimizationPoint]) throws -> [Double],
    isCancelled: () -> Bool
  ) throws -> [Double] {
    try checkCancellation(isCancelled)
    let losses = try evaluateBatch(points)
    guard losses.count == points.count else {
      throw SSBOptimizationError.invalidBatchResult(
        expected: points.count,
        actual: losses.count
      )
    }
    guard losses.allSatisfy(\.isFinite) else {
      throw SSBOptimizationError.nonFiniteLoss
    }
    return losses
  }

  private func checkCancellation(_ isCancelled: () -> Bool) throws {
    if isCancelled() { throw CancellationError() }
  }
}

private struct Float32ObjectiveKey: Hashable {
  let c10: UInt32
  let c12: UInt32
  let cosine: UInt32
  let sine: UInt32

  init(_ point: SSBOptimizationPoint) {
    let c10 = Float(point.c10Nanometers)
    let c12 = Float(point.c12Nanometers)
    let phi = Float(point.phi12Degrees) * .pi / 180
    self.c10 = c10.bitPattern
    self.c12 = c12.bitPattern
    cosine = cos(2 * phi).bitPattern
    sine = sin(2 * phi).bitPattern
  }
}

private struct SplitMix64 {
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
