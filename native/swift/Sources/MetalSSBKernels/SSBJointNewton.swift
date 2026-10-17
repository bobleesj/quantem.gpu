import Accelerate
import Foundation

/// The bounded six-parameter Newton control used by the Python workflow.
/// All objective values come from the caller's GPU batch evaluator. Only the
/// small optimizer state and six-by-six trust-region system are host scalars.
enum SSBJointNewton {
  static let step = [2.0, 2.0, 2.0, 0.25, 0.25, 5.0]
  static let tolerance = [0.05, 0.05, 0.05, 0.005, 0.005, 0.5]

  struct Kept {
    var center: [Double]
    var value: Double
    var gradient: [Double]
    var hessian: [[Double]]
  }

  static func coordinates(_ point: [Double]) -> [Double] {
    [
      point[0], point[1] * cos(2 * point[2]), point[1] * sin(2 * point[2]),
      point[3], point[4], point[5],
    ]
  }

  static func parameters(_ z: [Double]) -> [Double] {
    [z[0], hypot(z[1], z[2]), 0.5 * atan2(z[2], z[1]), z[3], z[4], abs(z[5])]
  }

  static func design(_ center: [Double], full: Bool) -> [[Double]] {
    var points = [center]
    for i in step.indices {
      for sign in [1.0, -1.0] {
        var point = center
        point[i] += sign * step[i]
        points.append(point)
      }
    }
    if full {
      for i in 0..<step.count {
        for j in (i + 1)..<step.count {
          var point = center
          point[i] += step[i]
          point[j] += step[j]
          points.append(point)
        }
      }
    }
    return points
  }

  static func model(_ values: [Double], curvature: [[Double]]) -> ([Double], [[Double]]) {
    var gradient = [Double](repeating: 0, count: step.count)
    var hessian = curvature
    for i in step.indices {
      gradient[i] = (values[1 + 2 * i] - values[2 + 2 * i]) / (2 * step[i])
      hessian[i][i] = (values[1 + 2 * i] - 2 * values[0] + values[2 + 2 * i]) / (step[i] * step[i])
    }
    if values.count > 13 {
      var index = 13
      for i in step.indices {
        for j in (i + 1)..<step.count {
          let mixed =
            (values[index] - values[1 + 2 * i] - values[1 + 2 * j] + values[0])
            / (step[i] * step[j])
          hessian[i][j] = mixed
          hessian[j][i] = mixed
          index += 1
        }
      }
    }
    return (gradient, hessian)
  }

  static func norm(_ values: [Double]) -> Double {
    sqrt(values.reduce(0) { $0 + $1 * $1 })
  }

  /// Maximize a quadratic model inside a scaled trust-region ball.
  static func trustStep(
    gradient: [Double], hessian: [[Double]], steps: [Double], radius: Double
  ) throws -> [Double] {
    let count = gradient.count
    let scaledGradient = zip(gradient, steps).map(*)
    // LAPACK returns the eigenvectors in columns, matching NumPy's eigh.
    var vectors = (0..<count).flatMap { column in
      (0..<count).map { row in hessian[row][column] * steps[row] * steps[column] }
    }
    var eigenvalues = [Double](repeating: 0, count: count)
    var size = __CLPK_integer(count)
    var leading = size
    var info: __CLPK_integer = 0
    var workSize = __CLPK_integer(max(1, 3 * count - 1))
    var work = [Double](repeating: 0, count: Int(workSize))
    var job = Int8(86)
    var triangle = Int8(76)  // V, L
    dsyev_(&job, &triangle, &size, &vectors, &leading, &eigenvalues, &work, &workSize, &info)
    guard info == 0, eigenvalues.allSatisfy(\.isFinite) else {
      throw SSBOptimizationError.nonFiniteLoss
    }
    let along = (0..<count).map { column in
      (0..<count).reduce(0.0) { $0 + vectors[column * count + $1] * scaledGradient[$1] }
    }
    func scaled(_ mu: Double) -> [Double] {
      (0..<count).map { row in
        (0..<count).reduce(0.0) { sum, column in
          sum + vectors[column * count + row] * (along[column] / (mu - eigenvalues[column]))
        }
      }
    }
    let maximum = eigenvalues.last!
    if maximum < 0 {
      let unconstrained = scaled(0)
      if norm(unconstrained) <= radius { return zip(unconstrained, steps).map(*) }
    }
    var low = max(0, maximum)
    var high = low + norm(scaledGradient) / radius + 1e-30
    low += 1e-12 * (abs(low) + 1)
    for _ in 0..<100 {
      let middle = 0.5 * (low + high)
      if norm(scaled(middle)) > radius { low = middle } else { high = middle }
    }
    return zip(scaled(high), steps).map(*)
  }

  static func boundedStep(
    center: [Double], gradient: [Double], hessian: [[Double]], radius: Double,
    lower: [Double], upper: [Double]
  ) throws -> [Double] {
    var free = [Bool](repeating: true, count: step.count)
    var held = [Double](repeating: 0, count: step.count)
    var move = held
    for _ in step.indices {
      move = held
      let indices = step.indices.filter { free[$0] }
      if !indices.isEmpty {
        let fixed = step.indices.filter { !free[$0] }
        let onFace = indices.map { i in
          gradient[i] + fixed.reduce(0.0) { $0 + hessian[i][$1] * held[$1] }
        }
        let reduced = try trustStep(
          gradient: onFace,
          hessian: indices.map { i in indices.map { hessian[i][$0] } },
          steps: indices.map { step[$0] }, radius: radius)
        for (offset, index) in indices.enumerated() { move[index] = reduced[offset] }
      }
      let outside = indices.filter {
        center[$0] + move[$0] < lower[$0] || center[$0] + move[$0] > upper[$0]
      }
      if outside.isEmpty { break }
      for index in outside {
        held[index] =
          min(upper[index], max(lower[index], center[index] + move[index])) - center[index]
        free[index] = false
      }
    }
    return move
  }

  static func run(
    starts: [[Double]], iterations: Int, evaluate: ([[Double]]) throws -> [Double],
    progress: (Int, Double) -> Void, isCancelled: () -> Bool
  ) throws -> (point: [Double], value: Double) {
    let lower = [-300.0, -200, -200, -25, -25, 20]
    let upper = [300.0, 200, 200, 25, 25, 600]
    var centers = starts.map(coordinates)
    var active = Array(starts.indices)
    var curvature = starts.map { _ in Array(repeating: Array(repeating: 0.0, count: 6), count: 6) }
    var full = [Bool](repeating: true, count: starts.count)
    var radius = [Double](repeating: 8, count: starts.count)
    var previous = [Kept?](repeating: nil, count: starts.count)
    var heading = centers
    func clip(_ z: [Double]) -> [Double] {
      var clipped = z.indices.map { min(upper[$0], max(lower[$0], z[$0])) }
      let magnitude = hypot(clipped[1], clipped[2])
      if magnitude > 200 {
        clipped[1] *= 200 / magnitude
        clipped[2] *= 200 / magnitude
      }
      return clipped
    }
    for iteration in 0..<iterations {
      if active.isEmpty { break }
      if isCancelled() { throw CancellationError() }
      let designs = active.map { design(centers[$0], full: full[$0]) }
      let values = try evaluate(designs.flatMap { $0.map(parameters) })
      guard values.count == designs.reduce(0, { $0 + $1.count }), values.allSatisfy(\.isFinite)
      else {
        throw SSBOptimizationError.nonFiniteLoss
      }
      var offset = 0
      for (index, points) in zip(active, designs) {
        let current = Array(values[offset..<(offset + points.count)])
        offset += points.count
        let gradient: [Double]
        let hessian: [[Double]]
        if let kept = previous[index], current[0] < kept.value {
          let rejected = norm(
            step.indices.map { (centers[index][$0] - kept.center[$0]) / step[$0] })
          centers[index] = kept.center
          gradient = kept.gradient
          hessian = kept.hessian
          radius[index] = min(radius[index], rejected) / 4
        } else {
          (gradient, hessian) = model(current, curvature: curvature[index])
          let moved =
            previous[index].map { kept in
              norm(step.indices.map { (centers[index][$0] - kept.center[$0]) / step[$0] })
            } ?? .infinity
          if previous[index] != nil && moved >= 0.99 * radius[index] { radius[index] *= 2 }
          curvature[index] = hessian
          full[index] = previous[index] != nil && moved > 2
          previous[index] = Kept(
            center: centers[index], value: current[0], gradient: gradient, hessian: hessian)
        }
        let move = try boundedStep(
          center: centers[index], gradient: gradient, hessian: hessian,
          radius: radius[index], lower: lower, upper: upper)
        let proposal = clip(zip(centers[index], move).map(+))
        if step.indices.allSatisfy({ abs(proposal[$0] - centers[index][$0]) < tolerance[$0] }) {
          active.removeAll { $0 == index }
          heading[index] = centers[index]
        } else {
          centers[index] = proposal
          heading[index] = proposal
        }
      }
      for index in active {
        for other in starts.indices where other != index {
          guard let mine = previous[index], let theirs = previous[other] else { continue }
          let near =
            step.indices.allSatisfy { abs(mine.center[$0] - theirs.center[$0]) < 4 * step[$0] }
            || step.indices.allSatisfy {
              abs(heading[index][$0] - heading[other][$0]) < 2 * step[$0]
            }
          if near && (mine.value < theirs.value || (mine.value == theirs.value && index > other)) {
            active.removeAll { $0 == index }
            break
          }
        }
      }
      if iteration == 1 && active.count > 1 {
        active = [
          active.max { a, b in
            let x = previous[a]!.value
            let y = previous[b]!.value
            return x == y ? a > b : x < y
          }!
        ]
      }
      progress(iteration + 1, previous.compactMap { $0?.value }.max() ?? -.infinity)
    }
    guard let best = previous.compactMap({ $0 }).max(by: { $0.value < $1.value }) else {
      throw SSBOptimizationError.nonFiniteLoss
    }
    return (parameters(best.center), best.value)
  }
}
