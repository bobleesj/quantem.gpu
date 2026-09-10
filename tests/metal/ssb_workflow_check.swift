import Foundation
import Metal
import MetalSSBKernels

@main enum SSBWorkflowCheck {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      fatalError("An Apple GPU is required")
    }
    let n = 512
    let q = (0..<n).map { Float($0 < n / 2 ? $0 : $0 - n) * 0.001 }
    func makeGeometry(dc: SIMD2<Float>) -> MetalSSBGeometry {
      MetalSSBGeometry(
        brightfieldKX: [0.1, 0.2, 0.3], brightfieldKY: [0.2, -0.1, 0.1],
        brightfieldAlphaSquared: [0.00003125, 0.00003125, 0.0000625],
        brightfieldAperture: [1, 1, 1],
        brightfieldCos2Phi: [-0.6, 0.6, 0.8], brightfieldSin2Phi: [0.8, -0.8, 0.6],
        qxByRow: q, qyByColumn: q, wavelengthAngstroms: 0.025,
        semiangleRadians: 0.02, angularSamplingYRadians: 0.001,
        angularSamplingXRadians: 0.001, dcValue: dc, referenceRotationDegrees: 0)
    }
    let geometry = makeGeometry(dc: .zero)
    func buffer<T>(_ values: [T]) -> MTLBuffer {
      values.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
      }
    }
    func read(_ value: MTLBuffer) -> [SIMD2<Float>] {
      Array(
        UnsafeBufferPointer(
          start: value.contents().assumingMemoryBound(to: SIMD2<Float>.self), count: n * n))
    }
    func error(_ actual: [SIMD2<Float>], _ reference: [SIMD2<Float>], scale: Float = 1) -> Double {
      var numerator = 0.0
      var denominator = 0.0
      for (a, b) in zip(actual, reference) {
        let expected = b * scale
        let difference = a - expected
        numerator +=
          Double(difference.x) * Double(difference.x) + Double(difference.y) * Double(difference.y)
        denominator +=
          Double(expected.x) * Double(expected.x) + Double(expected.y) * Double(expected.y)
      }
      return sqrt(numerator / max(denominator, 1e-30))
    }
    let values: [UInt8] = (0..<(3 * n * n)).map { index in
      let mixed = index * 17 + index / 113 + 3
      return UInt8(mixed % 251)
    }
    let aberrations = MetalSSBAberrations(c10Nanometers: 55, c12Nanometers: 13, phi12Radians: 0.23)
    let engine = try MetalSSBEngine(device: device, geometry: geometry)
    try engine.prepare(brightfield: buffer(values))
    let reference = read(try engine.reconstruct(aberrations: aberrations).object)
    let referenceLoss = try engine.phaseVariance(aberrations: aberrations).loss
    for (type, source) in [
      (MetalSSBCountType.uint16, buffer(values.map(UInt16.init))),
      (.uint32, buffer(values.map(UInt32.init))),
    ] {
      try engine.prepare(brightfield: source, countType: type)
      let result = try engine.reconstruct(aberrations: aberrations)
      precondition(
        error(read(result.object), reference) == 0, "Equal counts must give identical output")
      let loss = try engine.phaseVariance(aberrations: aberrations).loss
      precondition(loss == referenceLoss)
      precondition(result.provenance.sourceDType == String(describing: type))
    }
    // Powers of two preserve exact float conversion and FFT scaling. These
    // values exceed uint8/uint16 and catch silent narrowing at the source boundary.
    for (type, source, scale) in [
      (MetalSSBCountType.uint16, buffer(values.map { UInt16($0) * 256 }), Float(256)),
      (.uint32, buffer(values.map { UInt32($0) * 131072 }), Float(131072)),
    ] {
      for budget in [Int?.none, Int?(0)] {
        let candidate = try MetalSSBEngine(
          device: device, geometry: geometry, cacheBudgetBytes: budget)
        try candidate.prepare(brightfield: source, countType: type)
        let result = try candidate.reconstruct(aberrations: aberrations)
        let relativeError = error(read(result.object), reference, scale: scale)
        precondition(relativeError < 1e-4, "Source count scaling parity failed: \(relativeError)")
        print(
          "PASS type=\(type) cache=\(budget == nil ? "full" : "streamed") relative_l2=\(relativeError)"
        )
      }
    }
    // Fitting needs physical positive DC; zero DC above isolates linear count scaling.
    let fittingGeometry = makeGeometry(dc: SIMD2<Float>(37, 0))
    let fittingEngine = try MetalSSBEngine(device: device, geometry: fittingGeometry)
    try fittingEngine.prepare(brightfield: buffer(values))
    let fit = try fittingEngine.optimize(start: aberrations, globalTrials: 200, seed: 42)
    let fitted = MetalSSBAberrations(
      c10Nanometers: Float(fit.best.c10Nanometers),
      c12Nanometers: Float(fit.best.c12Nanometers), phi12Radians: Float(fit.best.phi12Radians))
    let fittedResult = try fittingEngine.reconstruct(aberrations: fitted)
    let run = try MetalSSBSavedRun(
      result: fittedResult, sourceIdentity: "synthetic-count-fixture-v1",
      backendRevision: "count-parity-test",
      geometry: fittingGeometry, aberrations: fitted, rotationDegrees: 0, optimization: fit)
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("result.ssb")
    try run.save(to: url)
    let loaded = try MetalSSBSavedRun.load(from: url, matchingSourceIdentity: run.sourceIdentity)
    let restored = try loaded.reconstruction(device: device)
    precondition(read(restored.object) == read(fittedResult.object))
    precondition(read(restored.fourierSum) == read(fittedResult.fourierSum))
    precondition(loaded.optimization?.trials.map(\.loss) == fit.trials.map(\.loss))
    precondition(loaded.aberrations == fitted && loaded.provenance == fittedResult.provenance)
    var rejected = false
    do {
      _ = try MetalSSBSavedRun.load(from: url, matchingSourceIdentity: "different-acquisition")
    } catch { rejected = true }
    precondition(rejected, "A saved run must not attach to a different acquisition")
    print(
      "PASS 200-trial fit and saved-run exact round trip; refinement_evaluations=\(fit.refinementEvaluations)"
    )
    print("PASS native Metal SSB uint8/uint16/uint32 count parity; device=\(device.name)")
  }
}
