import Foundation
import Metal
import MetalSSBKernels

private func require(
  _ condition: @autoclosure () -> Bool,
  _ message: @autoclosure () -> String = "Scientific parity check failed",
  line: UInt = #line
) {
  guard condition() else {
    FileHandle.standardError.write(Data("SSB workflow line \(line): \(message())\n".utf8))
    exit(1)
  }
}

@main enum SSBWorkflowCheck {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      fatalError("An Apple GPU is required")
    }
    let n = 512
    let calibration = MetalSSBCalibration(beamEnergyKeV: 300, semiangleMrad: 30,
      scanStepRowAngstroms: 2, scanStepColumnAngstroms: 3,
      detectorStepRowMrad: 10, detectorStepColumnMrad: 10, centerRow: 4, centerColumn: 4)
    let calibrated = try calibration.geometry(detectorRows: 9, detectorColumns: 9,
      detectorSum: Array(repeating: 100, count: 81), excludedPixels: [40])
    require(abs(calibrated.geometry.wavelengthAngstroms - 0.01968749) < 1e-7)
    require(!calibrated.pixels.contains(40) && calibrated.pixels == calibrated.pixels.sorted())
    require(abs(calibrated.geometry.qxByRow[1] - 1 / 1024) < 1e-8)
    require(abs(calibrated.geometry.qyByColumn[1] - 1 / 1536) < 1e-8)
    var diskCalibration = calibration
    diskCalibration.brightfieldRadiusPixels = 4
    let disk = try diskCalibration.geometry(detectorRows: 9, detectorColumns: 9,
      detectorSum: Array(repeating: 100, count: 81), excludedPixels: [40])
    require(disk.pixels.count == 48)
    require(disk.geometry.brightfieldAperture.contains(0))
    require(disk.geometry.dcValue.x == 100)
    diskCalibration.excludedDetectorPixels = [4]
    let maskedDisk = try diskCalibration.geometry(detectorRows: 9, detectorColumns: 9,
      detectorSum: Array(repeating: 100, count: 81), excludedPixels: [40])
    require(maskedDisk.pixels.count == 47 && !maskedDisk.pixels.contains(4))
    var fullDisk = MetalSSBCalibration(beamEnergyKeV: 300, semiangleMrad: 30,
      scanStepRowAngstroms: 0.264, scanStepColumnAngstroms: 0.264,
      detectorStepRowMrad: 1.090909090909091, detectorStepColumnMrad: 1.090909090909091,
      centerRow: 94.88451385498047, centerColumn: 96.35952758789062,
      brightfieldRadiusPixels: 53.35992814757164, excludedDetectorPixels: [78 * 192 + 74])
    let sums = Array(repeating: UInt64(100), count: 192 * 192)
    let historical = try fullDisk.geometry(detectorRows: 192, detectorColumns: 192, detectorSum: sums)
    require(historical.pixels.count == 8937)
    require(historical.geometry.brightfieldAperture.filter { $0 > 0 }.count == 2464)
    try fullDisk.matchApertureToBrightfieldDisk()
    require(abs(fullDisk.detectorStepRowMrad - 0.5622196476170719) < 1e-12)
    let complete = try fullDisk.geometry(detectorRows: 192, detectorColumns: 192, detectorSum: sums)
    require(complete.pixels == historical.pixels)
    require(complete.geometry.brightfieldAperture.allSatisfy { $0 > 0 })
    require(complete.geometry.dcValue == historical.geometry.dcValue)
    // Choosing fewer BF pixels must not silently change angular calibration.
    fullDisk.brightfieldRadiusPixels = 25
    let selected = try fullDisk.geometry(detectorRows: 192, detectorColumns: 192, detectorSum: sums)
    require(selected.pixels.count < complete.pixels.count)
    require(abs(fullDisk.detectorStepRowMrad - 0.5622196476170719) < 1e-12)
    require(selected.geometry.brightfieldAperture.allSatisfy { $0 > 0 })
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
      require(
        error(read(result.object), reference) == 0, "Equal counts must give identical output")
      let loss = try engine.phaseVariance(aberrations: aberrations).loss
      require(loss == referenceLoss)
      require(result.provenance.sourceDType == String(describing: type))
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
        require(relativeError < 1e-4, "Source count scaling parity failed: \(relativeError)")
        print(
          "PASS type=\(type) cache=\(budget == nil ? "full" : "streamed") relative_l2=\(relativeError)"
        )
      }
    }
    // Resident callbacks must preserve requested column order and widths, and
    // retain exact streaming behavior when no Fourier cache is admitted.
    let source32 = buffer(values.map(UInt32.init))
    for budget in [Int?.none, Int?(0)] {
      let callbackEngine = try MetalSSBEngine(device: device, geometry: geometry, cacheBudgetBytes: budget)
      try callbackEngine.prepare(countType: .uint32) { indices, output, command in
        let blit = command.makeBlitCommandEncoder()!
        for (local, logical) in indices.enumerated() {
          blit.copy(from: source32, sourceOffset: logical * n * n * 4,
            to: output, destinationOffset: local * n * n * 4, size: n * n * 4)
        }
        blit.endEncoding()
      }
      let callbackResult = try callbackEngine.reconstruct(aberrations: aberrations)
      require(error(read(callbackResult.object), reference) < 1e-4)
    }
    let streamedEngine = try MetalSSBEngine(device: device, geometry: geometry, cacheBudgetBytes: 0)
    try streamedEngine.prepare(brightfield: source32, countType: .uint32)
    try engine.prepare(brightfield: source32, countType: .uint32)
    for term in MetalSSBHigherOrder.supported {
      var adjusted = aberrations
      adjusted.higherOrder = [.init(order: term.order, symmetry: term.symmetry,
        magnitudeNanometers: pow(10, Float(term.order + 1)), angleRadians: 0.21)]
      let cached = try engine.reconstruct(aberrations: adjusted)
      let streamed = try streamedEngine.reconstruct(aberrations: adjusted)
      let difference = error(read(cached.object), read(streamed.object))
      require(difference < 1e-4, "Higher-order cache/stream parity: \(term.name) \(difference)")
      require(error(read(cached.object), reference) > 1e-6, "Control must change the scientific result")
      let phase = try engine.phase(of: cached).contents().assumingMemoryBound(to: Float.self)
      let object = read(cached.object)
      for i in 0..<(n * n) {
        require(abs(phase[i] - atan2(object[i].y, object[i].x)) < 1e-5)
      }
      print("PASS live \(term.name) cache/stream relative_l2=\(difference)")
    }
    let reset = try engine.reconstruct(aberrations: aberrations)
    require(error(read(reset.object), reference) == 0,
      "Resetting higher orders must restore the original reconstruction exactly")
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
    require(read(restored.object) == read(fittedResult.object))
    require(read(restored.fourierSum) == read(fittedResult.fourierSum))
    require(loaded.optimization?.trials.map(\.loss) == fit.trials.map(\.loss))
    require(loaded.aberrations == fitted && loaded.provenance == fittedResult.provenance)
    var manual = fitted
    manual.higherOrder = [.init(order: 2, symmetry: 1, magnitudeNanometers: 50, angleRadians: 0.3)]
    let manualResult = try fittingEngine.reconstruct(aberrations: manual)
    let manualRun = try MetalSSBSavedRun(result: manualResult, sourceIdentity: run.sourceIdentity,
      backendRevision: "manual-calibration-test", geometry: fittingGeometry,
      aberrations: manual, rotationDegrees: 12, optimization: fit,
      calibration: calibration, calibrationProvenance: ["semiangle": "Assumed"], optimizedRotationDegrees: 0)
    try manualRun.save(to: url)
    let manualLoaded = try MetalSSBSavedRun.load(from: url, matchingSourceIdentity: run.sourceIdentity)
    require(manualLoaded.calibration == calibration)
    require(manualLoaded.calibrationProvenance == ["semiangle": "Assumed"])
    require(manualLoaded.aberrations == manual && manualLoaded.rotationDegrees == 12)
    require(manualLoaded.optimizedRotationDegrees == 0)
    let manualRestored = try manualLoaded.reconstruction(device: device)
    require(read(manualRestored.object) == read(manualResult.object))
    var rejected = false
    do {
      _ = try MetalSSBSavedRun.load(from: url, matchingSourceIdentity: "different-acquisition")
    } catch { rejected = true }
    require(rejected, "A saved run must not attach to a different acquisition")
    print(
      "PASS 200-trial fit and saved-run exact round trip; refinement_evaluations=\(fit.refinementEvaluations)"
    )
    print("PASS native Metal SSB uint8/uint16/uint32 count parity; device=\(device.name)")
  }
}
