import Foundation
import Metal
import MetalSSBKernels

func require(_ condition: Bool, _ message: String) {
  guard condition else { fatalError(message) }
}

@main enum SSBBFSamplingCheck {
  static func main() throws {
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    let device = MTLCreateSystemDefaultDevice()!
    for size in [128, 256, 512] {
      let folder = root.appendingPathComponent(String(size))
      let geometryData = try Data(contentsOf: folder.appendingPathComponent("geometry.json"))
      let geometry = try JSONDecoder().decode(MetalSSBGeometry.self, from: geometryData)
      let counts = try Data(contentsOf: folder.appendingPathComponent("counts.uint8"))
      let source = counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
      }
      let aberrations = MetalSSBAberrations(c10Nanometers: 18, c12Nanometers: 7, phi12Radians: 0.3)
      for budget: Int? in [nil, 0, 32 * size * (size / 2 + 1) * 8] {
        let engine = try MetalSSBEngine(device: device, geometry: geometry, cacheBudgetBytes: budget)
        try engine.prepare(brightfield: source)
        let before = try engine.reconstruct(aberrations: aberrations)
        let beforeBytes = Data(bytes: before.object.contents(), count: size * size * 8)
        for fraction in [0.25, 0.5, 1.0] {
          let sampling = try SSBBrightfieldSampling(totalCount: geometry.logicalBrightfieldCount, fraction: fraction)
          require(Set(sampling.indices).count == sampling.indices.count, "No duplicate sampled BF pixels")
          require(sampling == (try SSBBrightfieldSampling(totalCount: geometry.logicalBrightfieldCount, fraction: fraction)), "Deterministic selection")
          // Independent explicit reduced input, not a second use of the selection-aware evaluator.
          var dictionary = try JSONSerialization.jsonObject(with: geometryData) as! [String: Any]
          for key in ["brightfieldKX", "brightfieldKY", "brightfieldAlphaSquared", "brightfieldAperture", "brightfieldCos2Phi", "brightfieldSin2Phi"] {
            let values = dictionary[key] as! [Any]
            dictionary[key] = sampling.indices.map { values[$0] }
          }
          let reducedGeometry = try JSONDecoder().decode(MetalSSBGeometry.self,
            from: JSONSerialization.data(withJSONObject: dictionary))
          var selectedCounts = Data()
          for index in sampling.indices {
            selectedCounts.append(counts[(index * size * size)..<((index + 1) * size * size)])
          }
          let reducedSource = selectedCounts.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
          }
          let reference = try MetalSSBEngine(device: device, geometry: reducedGeometry, cacheBudgetBytes: 0)
          try reference.prepare(brightfield: reducedSource)
          let expected = try reference.phaseVariance(aberrations: aberrations, rotationDegrees: 23)
          let actual = try engine.phaseVariance(aberrations: aberrations, rotationDegrees: 23,
            brightfieldFraction: fraction)
          let error = abs(actual.loss - expected.loss)
          require(error <= 1e-6 + abs(expected.loss) * 1e-5, "Subset loss differs from explicit input: \(error)")
          require(actual.provenance.logicalBrightfieldCount == sampling.indices.count, "Subset provenance")
          let after = try engine.reconstruct(aberrations: aberrations)
          require(after.provenance.logicalBrightfieldCount == geometry.logicalBrightfieldCount, "Final phase must use full BF")
          require(Data(bytes: after.object.contents(), count: size * size * 8) == beforeBytes,
            "Optimization sampling must not change full reconstruction")
          print("PASS size=\(size) budget=\(String(describing: budget)) fraction=\(fraction) selected=\(sampling.indices.count) error=\(error)")
        }
        if budget == nil {
          let fit = try engine.optimize(start: aberrations, globalTrials: 3, brightfieldFraction: 0.25)
          let restored = try JSONDecoder().decode(SSBOptimizationResult.self, from: JSONEncoder().encode(fit))
          require(restored.brightfieldSampling == fit.brightfieldSampling, "Saved fit retains exact subset")
          for fraction in [1.0, 0.25] {
            var samples = [Double]()
            for trial in 0..<12 {
              let started = CFAbsoluteTimeGetCurrent()
              _ = try engine.phaseVariance(aberrations: aberrations, brightfieldFraction: fraction)
              if trial >= 2 { samples.append(CFAbsoluteTimeGetCurrent() - started) }
            }
            samples.sort()
            print("TIMING synthetic size=\(size) fraction=\(fraction) n=10 p50_ms=\(samples[5] * 1000) max_ms=\(samples.last! * 1000)")
          }
        }
      }
    }
    print("PASS BF optimization sampling; full reconstruction unchanged")
  }
}
