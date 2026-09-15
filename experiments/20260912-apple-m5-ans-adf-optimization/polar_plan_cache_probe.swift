import Foundation

private enum ProbeFailure: Error { case failed(String) }
private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  if !condition() { throw ProbeFailure.failed(message) }
}

private struct Fixture {
  let delta: [Int32]
  let validity: [UInt8]
  let rows: Int
  let columns: Int
  let leaf: Int
  let layout: String
}

private func signature(_ plan: PairedRuntimeTANSPolarPlan) -> String {
  var parts = [
    String(plan.detectorRows), String(plan.detectorColumns), String(plan.usedIndex),
    String(plan.leafPixelCount), plan.layoutKind,
  ]
  parts.append(plan.selectedFields.map { String($0) }.joined(separator: ","))
  parts.append(plan.fieldCoefficients.map { String($0) }.joined(separator: ","))
  parts.append(plan.residualPixels.map { String($0) }.joined(separator: ","))
  parts.append(plan.residualCoefficients.map { String($0) }.joined(separator: ","))
  parts.append(plan.reconstructedDelta().map { String($0) }.joined(separator: ","))
  return parts.joined(separator: "|")
}

private func make(_ fixture: Fixture) -> PairedRuntimeTANSPolarPlan {
  PairedRuntimeTANSPolarPlan.make(
    delta: fixture.delta, validPixels: fixture.validity,
    detectorRows: fixture.rows, detectorColumns: fixture.columns,
    leafPixels: fixture.leaf, layoutKind: fixture.layout)
}

@main enum PolarPlanCacheProbe {
  static func main() throws {
    let count = 192 * 192
    var fixtures = [Fixture]()
    for (fixtureIndex, layout) in ["polar", "radial1", "radialhalf"].enumerated() {
      for leaf in [16, 32, 64] {
        var delta = [Int32](repeating: fixtureIndex == 1 ? -1 : 1, count: count)
        var validity = [UInt8](repeating: 1, count: count)
        for pixel in stride(from: fixtureIndex + leaf / 8, to: count, by: 79 + leaf) {
          delta[pixel] = Int32((pixel + leaf) % 5) - 2
        }
        for pixel in stride(from: leaf, to: count, by: 211 + fixtureIndex) {
          validity[pixel] = 0
          delta[pixel] = 7
        }
        fixtures.append(Fixture(
          delta: delta, validity: validity, rows: 192, columns: 192,
          leaf: leaf, layout: layout))
      }
    }
    fixtures.append(Fixture(
      delta: (0..<35).map { Int32($0 % 4) - 1 },
      validity: (0..<35).map { $0 % 7 == 0 ? 0 : 1 },
      rows: 5, columns: 7, leaf: 16, layout: "polar"))

    setenv("QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN", "0", 1)
    let expected = fixtures.map { signature(make($0)) }
    setenv("QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN", "1", 1)
    for index in fixtures.indices {
      try require(signature(make(fixtures[index])) == expected[index],
        "cache-on result differs for fixture \(index)")
      // Force a miss with another key, then require an exact hit after replacement.
      _ = make(fixtures[(index + 1) % fixtures.count])
      try require(signature(make(fixtures[index])) == expected[index],
        "cache replacement differs for fixture \(index)")
    }

    let lock = NSLock()
    var failures: [String] = []
    DispatchQueue.concurrentPerform(iterations: 160) { iteration in
      let index = (iteration * 17) % fixtures.count
      let actual = signature(make(fixtures[index]))
      if actual != expected[index] {
        lock.lock(); failures.append("iteration \(iteration), fixture \(index)"); lock.unlock()
      }
    }
    try require(failures.isEmpty,
      "concurrent cache mismatch: \(failures.prefix(3).joined(separator: "; "))")
    print("polar plan cache CPU parity passed: \(fixtures.count) fixtures, 160 concurrent calls")
  }
}
