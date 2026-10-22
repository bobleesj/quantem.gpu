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
    let calibration = MetalSSBCalibration(voltageKV: 300, semiangleMrad: 30,
      scanSamplingRowAngstroms: 2, scanSamplingColumnAngstroms: 3,
      detectorSamplingRowMrad: 10, detectorSamplingColumnMrad: 10,
      brightfieldCenterRowPixels: 4, brightfieldCenterColumnPixels: 4)
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
    var fullDisk = MetalSSBCalibration(voltageKV: 300, semiangleMrad: 30,
      scanSamplingRowAngstroms: 0.264, scanSamplingColumnAngstroms: 0.264,
      detectorSamplingRowMrad: 1.090909090909091, detectorSamplingColumnMrad: 1.090909090909091,
      brightfieldCenterRowPixels: 94.88451385498047, brightfieldCenterColumnPixels: 96.35952758789062,
      brightfieldRadiusPixels: 53.35992814757164, excludedDetectorPixels: [78 * 192 + 74])
    let sums = Array(repeating: UInt64(100), count: 192 * 192)
    let historical = try fullDisk.geometry(detectorRows: 192, detectorColumns: 192, detectorSum: sums)
    require(historical.pixels.count == 8937)
    require(historical.geometry.brightfieldAperture.filter { $0 > 0 }.count == 2464)
    try fullDisk.matchApertureToBrightfieldDisk()
    require(abs(fullDisk.detectorSamplingRowMrad - 0.5622196476170719) < 1e-12)
    let complete = try fullDisk.geometry(detectorRows: 192, detectorColumns: 192, detectorSum: sums)
    require(complete.pixels == historical.pixels)
    require(complete.geometry.brightfieldAperture.allSatisfy { $0 > 0 })
    require(complete.geometry.dcValue == historical.geometry.dcValue)
    // Choosing fewer BF pixels must not silently change angular calibration.
    fullDisk.brightfieldRadiusPixels = 25
    let selected = try fullDisk.geometry(detectorRows: 192, detectorColumns: 192, detectorSum: sums)
    require(selected.pixels.count < complete.pixels.count)
    require(abs(fullDisk.detectorSamplingRowMrad - 0.5622196476170719) < 1e-12)
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
    let aberrations = MetalSSBAberrations(c10Angstroms: 55, c12Angstroms: 13, phi12Radians: 0.23)
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
        let candidateLoss = try candidate.phaseVariance(aberrations: aberrations).loss
        let lossRelativeError = abs(candidateLoss - referenceLoss)
          / max(abs(referenceLoss), 1e-12)
        if lossRelativeError >= 5e-5 {
          FileHandle.standardError.write(Data(
            "Loss parity failed: type=\(type), cacheBudget=\(String(describing: budget)), reference=\(referenceLoss), actual=\(candidateLoss), relative_error=\(lossRelativeError), limit=5e-5. Investigate cached/streamed objective semantics; do not rebase the tolerance.\n".utf8))
        }
        // Match the existing cached/streamed XCTest tolerance; do not rebase it.
        require(lossRelativeError < 5e-5,
          "Source count scaling loss parity failed: \(lossRelativeError)")
        print(
          "PASS type=\(type) cache=\(budget == nil ? "full" : "streamed") relative_l2=\(relativeError) loss_relative_error=\(lossRelativeError)"
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
        magnitudeAngstroms: pow(10, Float(term.order + 1)), angleRadians: 0.21)]
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
    try checkOddAberrationSign(device: device)
    // Fitting needs physical positive DC; zero DC above isolates linear count scaling.
    let fittingGeometry = makeGeometry(dc: SIMD2<Float>(37, 0))
    let fittingEngine = try MetalSSBEngine(device: device, geometry: fittingGeometry)
    try fittingEngine.prepare(brightfield: buffer(values))
    let fit = try fittingEngine.optimize(start: aberrations, globalTrials: 200, seed: 42)
    let fitted = MetalSSBAberrations(
      c10Angstroms: Float(fit.best.c10Angstroms),
      c12Angstroms: Float(fit.best.c12Angstroms), phi12Radians: Float(fit.best.phi12Radians))
    let fittedResult = try fittingEngine.reconstruct(aberrations: fitted)
    let run = try MetalSSBSavedRun(
      result: fittedResult, sourceIdentity: "synthetic-count-fixture-v1",
      backendRevision: "count-parity-test",
      geometry: fittingGeometry, aberrations: fitted, scanDetectorRotationDegrees: 0, optimization: fit)
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
    manual.higherOrder = [.init(order: 2, symmetry: 1, magnitudeAngstroms: 50, angleRadians: 0.3)]
    let manualResult = try fittingEngine.reconstruct(aberrations: manual)
    let manualRun = try MetalSSBSavedRun(result: manualResult, sourceIdentity: run.sourceIdentity,
      backendRevision: "manual-calibration-test", geometry: fittingGeometry,
      aberrations: manual, scanDetectorRotationDegrees: 12, optimization: fit,
      calibration: calibration, calibrationProvenance: ["semiangle": "Assumed"], optimizedScanDetectorRotationDegrees: 0)
    try manualRun.save(to: url)
    let manualLoaded = try MetalSSBSavedRun.load(from: url, matchingSourceIdentity: run.sourceIdentity)
    require(manualLoaded.calibration == calibration)
    require(manualLoaded.calibrationProvenance == ["semiangle": "Assumed"])
    require(manualLoaded.aberrations == manual && manualLoaded.scanDetectorRotationDegrees == 12)
    require(manualLoaded.optimizedScanDetectorRotationDegrees == 0)
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

  /// Coma and trefoil must be corrected with the probe phase of the first sideband at k - q.
  ///
  /// For a thin object G(q, k) ~ P(k - q) conj P(k) - P(k) conj P(k + q). ssb.metal once evaluated the
  /// first term at q - k, which equals k - q for C10, C12 and C30 but reverses the odd orders on that
  /// sideband. Three weak phase gratings eps cos(2 pi q.r + theta) under a probe with C21, C23 and C30
  /// give counts |Psi(k)|^2 of the exit wave p(r - r0) (1 + i phi), from the shift theorem alone. With
  /// the probe's own coefficients every BF pixel's corrected G at a grating's q has the phase
  /// pi / 2 + theta, so their mean is e^{i (pi/2 + theta)} mean_k |G_k(q)|. With the probe phase at q - k
  /// the BF pixels disagree: the mean misses by 0.14 to 0.18 of its size. Reversing the odd orders, or
  /// leaving out every term above C12, must miss by more than 0.1: the gratings sit where they matter.
  static func checkOddAberrationSign(device: MTLDevice) throws {
    let n = 128, step = 0.25                                       // scan pixels and Angstrom
    let volts = 300e3
    let wavelength = 12.2643 / (volts * (1 + 0.978476e-6 * volts)).squareRoot()
    let semiangle = 0.030, sampling = 0.003                        // rad, and rad per detector pixel
    let c10 = -50.0, c12 = 20.0, phi12 = Double.pi / 6
    // Krivanek order, symmetry, magnitude (Angstrom) and azimuth: coma and trefoil are odd, C30 even
    let truth: [(order: Int, symmetry: Int, magnitude: Double, angle: Double)] = [
      (2, 1, 400, .pi / 3), (2, 3, 300, -.pi / 9), (3, 0, 3e4, 0),
    ]
    // Scan Fourier row and column of each grating, and its phase; no sum or difference of two is a third
    let gratings: [(row: Int, column: Int, theta: Double)] = [(24, 9, 0.4), (-11, 30, -1.1), (-30, -12, 2.0)]
    let epsilon = 0.05
    func times(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> SIMD2<Double> {
      SIMD2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
    }
    func aperture(_ kr: Double, _ kc: Double) -> Double {      // the engines' soft edge, one pixel wide
      min(max((semiangle - wavelength * (kr * kr + kc * kc).squareRoot()) / sampling + 0.5, 0), 1)
    }
    func probe(_ kr: Double, _ kc: Double) -> SIMD2<Double> {
      let alpha = wavelength * (kr * kr + kc * kc).squareRoot()
      let azimuth = atan2(kc, kr)
      var chi = alpha * alpha / 2 * (c10 + c12 * cos(2 * (azimuth - phi12)))
      for term in truth {
        chi += term.magnitude * pow(alpha, Double(term.order + 1)) / Double(term.order + 1)
          * cos(Double(term.symmetry) * (azimuth - term.angle))
      }
      chi *= 2 * .pi / wavelength
      return aperture(kr, kc) * SIMD2(cos(chi), -sin(chi))
    }
    var kx: [Double] = [], ky: [Double] = []
    let radius = Int((semiangle / sampling).rounded(.up)) + 1
    for i in -radius...radius {
      for j in -radius...radius where aperture(Double(i) * sampling / wavelength, Double(j) * sampling / wavelength) > 0 {
        kx.append(Double(i) * sampling / wavelength)
        ky.append(Double(j) * sampling / wavelength)
      }
    }
    let plane = n * n
    let dq = 1 / (Double(n) * step)
    let twiddle = (0..<n).map { SIMD2(cos(2 * Double.pi * Double($0) / Double(n)), sin(2 * Double.pi * Double($0) / Double(n))) }
    func shift(_ grating: Int, _ row: Int, _ column: Int) -> SIMD2<Double> {   // e^{2 pi i q.r0}
      twiddle[(((gratings[grating].row * row + gratings[grating].column * column) % n) + n) % n]
    }
    var counts = [UInt16](repeating: 0, count: kx.count * plane)
    var spectrum: [[SIMD2<Double>]] = []                              // G_k(q) = sum_r0 I e^{-2 pi i q.r0}
    var dc = 0.0
    for bf in 0..<kx.count {
      let centre = probe(kx[bf], ky[bf])
      // i eps / 2 e^{+-i theta} P(k -+ q): the grating's two orders, moved by the scan
      let minus = gratings.map { g in
        times(SIMD2(-sin(g.theta), cos(g.theta)) * (epsilon / 2), probe(kx[bf] - Double(g.row) * dq, ky[bf] - Double(g.column) * dq))
      }
      let plus = gratings.map { g in
        times(SIMD2(sin(g.theta), cos(g.theta)) * (epsilon / 2), probe(kx[bf] + Double(g.row) * dq, ky[bf] + Double(g.column) * dq))
      }
      var g = [SIMD2<Double>](repeating: .zero, count: gratings.count)
      for row in 0..<n {
        for column in 0..<n {
          var wave = centre
          for index in gratings.indices {
            let w = shift(index, row, column)
            wave += times(minus[index], w) + times(plus[index], SIMD2(w.x, -w.y))
          }
          let value = ((wave * wave).sum() * 40000).rounded()
          require(value <= 65535, "Grating counts exceed uint16")
          counts[bf * plane + row * n + column] = UInt16(value)
          dc += value
          for index in gratings.indices {
            let w = shift(index, row, column)
            g[index] += value * SIMD2(w.x, -w.y)
          }
        }
      }
      spectrum.append(g)
    }
    let q = (0..<n).map { Float(Double($0 < n / 2 ? $0 : $0 - n) * dq) }
    let k2 = zip(kx, ky).map { $0 * $0 + $1 * $1 }
    let geometry = MetalSSBGeometry(
      brightfieldKX: kx.map(Float.init), brightfieldKY: ky.map(Float.init),
      brightfieldAlphaSquared: k2.map { Float(wavelength * wavelength * $0) },
      brightfieldAperture: zip(kx, ky).map { Float(aperture($0, $1)) },
      brightfieldCos2Phi: zip(zip(kx, ky), k2).map { $1 > 0 ? Float(($0.0 * $0.0 - $0.1 * $0.1) / $1) : 1 },
      brightfieldSin2Phi: zip(zip(kx, ky), k2).map { $1 > 0 ? Float(2 * $0.0 * $0.1 / $1) : 0 },
      qxByRow: q, qyByColumn: q, wavelengthAngstroms: Float(wavelength), semiangleRadians: Float(semiangle),
      angularSamplingYRadians: Float(sampling), angularSamplingXRadians: Float(sampling),
      dcValue: SIMD2(Float(dc / Double(kx.count)), 0), referenceRotationDegrees: 0,
      phaseFactor: Float(Double.pi / wavelength))
    let source = counts.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    let expected = gratings.indices.map { index in
      spectrum.map { ($0[index] * $0[index]).sum().squareRoot() }.reduce(0, +) / Double(kx.count)
        * SIMD2(cos(Double.pi / 2 + gratings[index].theta), sin(Double.pi / 2 + gratings[index].theta))
    }
    func higherOrder(reversingOddBy turns: (Double, Double)) -> [MetalSSBHigherOrder] {
      zip(truth, [turns.0, turns.1, 0]).map {
        MetalSSBHigherOrder(order: $0.order, symmetry: $0.symmetry, magnitudeAngstroms: Float($0.magnitude),
          angleRadians: Float($0.angle + $1))
      }
    }
    let cases: [(name: String, higherOrder: [MetalSSBHigherOrder]?)] = [
      ("truth", higherOrder(reversingOddBy: (0, 0))), ("reversed", higherOrder(reversingOddBy: (.pi, .pi / 3))),
      ("none", nil),
    ]
    var report: [String] = []
    for budget in [Int?.none, Int?(0)] {
      let engine = try MetalSSBEngine(device: device, geometry: geometry, cacheBudgetBytes: budget)
      try engine.prepare(brightfield: source, countType: .uint16)
      for item in cases {
        let result = try engine.reconstruct(aberrations: MetalSSBAberrations(
          c10Angstroms: Float(c10), c12Angstroms: Float(c12), phi12Radians: Float(phi12), higherOrder: item.higherOrder))
        let sum = result.fourierSum.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let misses = gratings.indices.map { index -> Double in
          let value = sum[((gratings[index].row + n) % n) * n + (gratings[index].column + n) % n]
          let difference = SIMD2(Double(value.x), Double(value.y)) - expected[index]
          return ((difference * difference).sum() / (expected[index] * expected[index]).sum()).squareRoot()
        }
        let label = "\(budget == nil ? "cached" : "streamed") \(item.name)"
        if item.name == "truth" {
          require(misses.allSatisfy { $0 < 1e-4 },
            "\(label): the corrected Fourier sum at the gratings misses e^{i (pi/2 + theta)} mean |G| by \(misses) of its size; the first sideband's probe phase must be at k - q")
        } else {
          require(misses.allSatisfy { $0 > 0.1 }, "\(label): the gratings must sit where the odd orders matter, \(misses)")
        }
        report.append("\(label) \(misses.map { String(format: "%.2g", $0) }.joined(separator: "/"))")
      }
    }
    print("PASS odd aberrations corrected at k - q, relative miss per grating: \(report.joined(separator: ", "))")
  }
}
