import Foundation
import Metal
import XCTest

@testable @_spi(SSBBenchmark) import MetalSSBKernels

/// The regularized direct reconstructions (`.phase`, `.complex`) on the synthetic scene of
/// `SSBModelReconstructionTests`, against an independent double-precision evaluation of their definition (the CUDA
/// reference `PreparedDirect`): per scan frequency the supported output aliases are fitted jointly by the damped
/// normal equations of the weak-object transfer, the aliases fill the output spectrum and one inverse DFT gives
/// eta + i phi. Also the API contract, the regularization checks and the cached systems.
///
/// Stated tolerance: phase within 5e-7 rad rms and 2e-6 rad maximum, complex amplitude within 5e-7 of its mean (rms):
/// the float32 transfer, batch sums and FFT of the native path against float64 here. Observed: 4e-9 to 3e-8 rad rms,
/// at most 1.1e-7 rad; amplitude 8e-8.
final class SSBDirectReconstructionTests: XCTestCase {
  typealias Scene = SSBModelReconstructionTests.Scene

  private struct Complex {
    var re = 0.0
    var im = 0.0
    static func + (a: Complex, b: Complex) -> Complex { Complex(re: a.re + b.re, im: a.im + b.im) }
    static func - (a: Complex, b: Complex) -> Complex { Complex(re: a.re - b.re, im: a.im - b.im) }
    static func * (a: Complex, b: Complex) -> Complex {
      Complex(re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re)
    }
    static func * (a: Double, b: Complex) -> Complex { Complex(re: a * b.re, im: a * b.im) }
    var conjugate: Complex { Complex(re: re, im: -im) }
    static func phase(_ angle: Double) -> Complex { Complex(re: cos(angle), im: sin(angle)) }
  }

  /// G_k(Q) of every BF pixel on the source half plane (columns 0 ... n/2), by a double-precision DFT of its counts.
  private static func spectra(_ s: Scene) -> [[Complex]] {
    let n = s.scan
    let half = n / 2 + 1
    let twiddle = (0..<n).map { Complex.phase(-2 * Double.pi * Double($0) / Double(n)) }
    return (0..<s.pixels.count).map { b in
      let plane = s.counts[(b * n * n)..<((b + 1) * n * n)].map(Double.init)
      var rows = [Complex](repeating: Complex(), count: n * half)
      for x in 0..<n {
        for c in 0..<half {
          var sum = Complex()
          for y in 0..<n { sum = sum + plane[x * n + y] * twiddle[(c * y) % n] }
          rows[x * half + c] = sum
        }
      }
      var out = [Complex](repeating: Complex(), count: n * half)
      for r in 0..<n {
        for c in 0..<half {
          var sum = Complex()
          for x in 0..<n { sum = sum + rows[x * half + c] * twiddle[(r * x) % n] }
          out[r * half + c] = sum
        }
      }
      return out
    }
  }

  /// The double-precision definition, from the same float32 inputs the kernels read (rotated BF frequencies,
  /// wavelength, angles, packed aberration terms, tilt and depth).
  private static func reference(
    _ s: Scene, spectra: [[Complex]], model: MetalSSBJointModel, point: SSBModelPoint, factor: Int,
    complex: Bool, regularization r: SSBDirectRegularization
  ) throws -> (phase: [Double], amplitude: [Double]) {
    let g = s.geometry
    let n = s.scan
    let half = n / 2 + 1
    let outputN = factor * n
    let k = try model.objectiveArrays().k.map { (Double($0.x), Double($0.y)) }
    precondition(k.count == s.pixels.count, "every BF pixel of the scene is inside the aperture")
    let wl = Double(g.wavelengthAngstroms)
    let semi = Double(g.semiangleRadians)
    let angular = Double(g.angularSamplingXRadians)
    let factorPi = Double(g.phaseFactor!)
    let depth = Double(Float(point.depthSpreadAngstroms))
    let tilt = (
      Double(Float(point.sampleTiltMrad[0] * 1e-3)), Double(Float(point.sampleTiltMrad[1] * 1e-3))
    )
    let coefficients = point.reconstructionCoefficients.map(Double.init)
    func aperture(_ x: Double, _ y: Double) -> Double {
      let r2 = x * x + y * y
      guard r2 > 1e-30 else { return 1 }
      return min(1, max(0, (semi - r2.squareRoot() * wl) / angular + 0.5))
    }
    func chi(_ row: Double, _ col: Double) -> Double {
      let x = row * wl
      let y = col * wl
      let r2 = x * x + y * y
      let radial = [1, r2, r2 * r2, r2 * r2 * r2]
      var re = [1.0]
      var im = [0.0]
      for m in 1...6 {
        re.append(re[m - 1] * x - im[m - 1] * y)
        im.append(im[m - 1] * x + re[m - 1] * y)
      }
      let azimuth = [0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6]
      let power = [1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0]
      var sum = 0.0
      for i in 0..<14 {
        sum +=
          radial[power[i]]
          * (re[azimuth[i]] * coefficients[2 * i] + im[azimuth[i]] * coefficients[2 * i + 1])
      }
      return 2 * Double.pi / wl * sum
    }
    func sinc(_ x: Double) -> Double { abs(x) < 1e-6 ? 1 : sin(x) / x }
    let pixel = k.map { (kr, kc) in (aperture(kr, kc), (kr * kr + kc * kc) * wl * wl, chi(kr, kc)) }
    // Dose, weights and damping scale over every BF pixel.
    let dc = spectra.map { $0[0].re }
    let apertures = g.brightfieldAperture.map(Double.init)
    let dose = dc.reduce(0, +) / (Double(n * n) * apertures.map { $0 * $0 }.reduce(0, +))
    var weights = [Double](repeating: 1, count: k.count)
    if r.detectorWeighting == .inverseMean {
      let mean = dc.map { max($0 / Double(n * n), 0) }
      let sorted = mean.sorted()
      let middle = sorted.count / 2
      let median =
        sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
      weights = mean.map { 1 / max($0, 0.05 * median) }
      let average = weights.reduce(0, +) / Double(weights.count)
      weights = weights.map { $0 / average }
    }
    let scale = zip(weights, apertures).map { $0 * pow($1, 4) }.reduce(0, +)
    // The supported aliases, as the float32 layout decides them.
    let cutoff =
      (2 * (g.modelSemiangleRadians ?? semi) + angular) / (g.modelWavelengthAngstroms ?? wl)
    let cutoff2 = Float(cutoff * cutoff)
    func signed(_ index: Int) -> Int { index <= (outputN - 1) / 2 ? index : index - outputN }
    var eta = [Complex](repeating: Complex(), count: outputN * outputN)
    var phi = eta
    for row in 0..<n {
      for column in 0..<half {
        var keep = column > 0 || (row > 0 && row <= (n - 1) / 2)
        if n % 2 == 0 { keep = keep && row != n / 2 && column != n / 2 }
        guard keep else { continue }
        var aliases: [(q: (Double, Double), index: Int, prior: Double)] = []
        for ar in 0..<factor {
          for ac in 0..<factor {
            let outputRow = row + ar * n
            let outputColumn = column + ac * n
            let qr = Float(signed(outputRow)) * g.qxByRow[1]
            let qc = Float(signed(outputColumn)) * g.qyByColumn[1]
            guard qr * qr + qc * qc < cutoff2 else { continue }
            let q2 = Double(qr) * Double(qr) + Double(qc) * Double(qc)
            let s2 = r.smoothnessInverseAngstroms * r.smoothnessInverseAngstroms
            aliases.append(
              (
                (Double(qr), Double(qc)), outputRow * outputN + outputColumn,
                r.smoothnessInverseAngstroms == 0 ? 1 : pow(1 + q2 / s2, 2)
              ))
          }
        }
        let count = aliases.count
        guard count > 0 else { continue }
        let dim = complex ? 2 * count : count
        var a = [Complex](repeating: Complex(), count: dim * dim)
        var b = [Complex](repeating: Complex(), count: dim)
        for (index, (kr, kc)) in k.enumerated() {
          let (ak, alpha2k, chik) = pixel[index]
          var v = [Complex](repeating: Complex(), count: dim)
          for (alias, entry) in aliases.enumerated() {
            let (qr, qc) = entry.q
            let shift = 2 * Double.pi * (qr * tilt.0 + qc * tilt.1)
            let mr = kr - qr
            let mc = kc - qc
            let pr = kr + qr
            let pc = kc + qc
            let t1 =
              aperture(mr, mc) * ak
              * sinc(0.5 * depth * (-factorPi * ((mr * mr + mc * mc) * wl * wl - alpha2k) - shift))
              * Complex.phase(-(chi(mr, mc) - chik))
            let t2 =
              aperture(pr, pc) * ak
              * sinc(0.5 * depth * (factorPi * ((pr * pr + pc * pc) * wl * wl - alpha2k) - shift))
              * Complex.phase(chi(pr, pc) - chik)
            v[dim - count + alias] = Complex(re: 0, im: 1) * (t1 - t2)
            if complex { v[alias] = t1 + t2 }
          }
          let gq = spectra[index][row * half + column]
          for i in 0..<dim {
            let left = weights[index] * v[i].conjugate
            for j in 0..<dim { a[i * dim + j] = a[i * dim + j] + left * v[j] }
            b[i] = b[i] + (1 / dose) * (left * gq)
          }
        }
        for i in 0..<dim {
          let damping = complex && i < count ? r.amplitudeDamping : r.phaseDamping
          a[i * dim + i].re += damping * scale * aliases[i % count].prior
        }
        let x = solve(a, b, dim)
        let gain = Double(factor * factor)
        for (alias, entry) in aliases.enumerated() {
          let p = entry.index
          let mirror =
            ((outputN - p / outputN) % outputN) * outputN + (outputN - p % outputN) % outputN
          phi[p] = gain * x[dim - count + alias]
          phi[mirror] = phi[p].conjugate
          if complex {
            eta[p] = gain * x[alias]
            eta[mirror] = eta[p].conjugate
          }
        }
      }
    }
    let etaImage = inverseDFT(eta, outputN)
    let phiImage = inverseDFT(phi, outputN)
    return (
      zip(etaImage, phiImage).map { atan2(exp($0.0) * sin($0.1), exp($0.0) * cos($0.1)) },
      etaImage.map(exp)
    )
  }

  /// Gaussian elimination with partial pivoting (an independent solver; production uses Cholesky).
  private static func solve(_ matrix: [Complex], _ rhs: [Complex], _ n: Int) -> [Complex] {
    var a = matrix
    var b = rhs
    for column in 0..<n {
      let pivot = (column..<n).max {
        hypot(a[$0 * n + column].re, a[$0 * n + column].im)
          < hypot(a[$1 * n + column].re, a[$1 * n + column].im)
      }!
      if pivot != column {
        for j in 0..<n { a.swapAt(pivot * n + j, column * n + j) }
        b.swapAt(pivot, column)
      }
      let p = a[column * n + column]
      let inverse = (1 / (p.re * p.re + p.im * p.im)) * p.conjugate
      for row in (column + 1)..<n {
        let f = a[row * n + column] * inverse
        for j in column..<n { a[row * n + j] = a[row * n + j] - f * a[column * n + j] }
        b[row] = b[row] - f * b[column]
      }
    }
    var x = [Complex](repeating: Complex(), count: n)
    for row in stride(from: n - 1, through: 0, by: -1) {
      var sum = b[row]
      for j in (row + 1)..<n { sum = sum - a[row * n + j] * x[j] }
      let p = a[row * n + row]
      x[row] = (1 / (p.re * p.re + p.im * p.im)) * (sum * p.conjugate)
    }
    return x
  }

  /// The real part of the normalized inverse 2D DFT (the spectra are Hermitian).
  private static func inverseDFT(_ spectrum: [Complex], _ n: Int) -> [Double] {
    let twiddle = (0..<n).map { Complex.phase(2 * Double.pi * Double($0) / Double(n)) }
    var rows = [Complex](repeating: Complex(), count: n * n)
    for u in 0..<n {
      for y in 0..<n {
        var sum = Complex()
        for v in 0..<n where spectrum[u * n + v].re != 0 || spectrum[u * n + v].im != 0 {
          sum = sum + spectrum[u * n + v] * twiddle[(v * y) % n]
        }
        rows[u * n + y] = sum
      }
    }
    var out = [Double](repeating: 0, count: n * n)
    for x in 0..<n {
      for y in 0..<n {
        var sum = 0.0
        for u in 0..<n { sum += (rows[u * n + y] * twiddle[(u * x) % n]).re }
        out[x * n + y] = sum / Double(n * n)
      }
    }
    return out
  }

  private func model(_ s: Scene, device: MTLDevice, cache: Int? = nil) throws -> MetalSSBJointModel
  {
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(device: device, geometry: s.geometry, cacheBudgetBytes: cache)
    try engine.prepare(brightfield: input, countType: .uint16)
    return try engine.jointModel(scanDetectorRotationDegrees: 17.25)
  }

  private func values(_ buffer: MTLBuffer, _ count: Int) -> [Double] {
    let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
    return (0..<count).map { Double(pointer[$0]) }
  }

  func testDirectSolvesMatchDoublePrecisionDefinition() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try SSBModelReconstructionTests.scene()
    let spectra = Self.spectra(s)
    let low = SSBModelPoint(c10Angstroms: 61, c12Angstroms: 14, phi12Radians: 0.7)
    let thick = SSBModelPoint(
      c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -0.4, sampleTiltMrad: [2.5, -4],
      depthSpreadAngstroms: 140,
      higherOrder: [
        .init(order: 2, symmetry: 1, magnitudeAngstroms: 900, angleRadians: 0.3),
        .init(order: 3, symmetry: 0, magnitudeAngstroms: -4e4),
      ])
    let cases: [(SSBModelPoint, Int, SSBReconstructionMethod, SSBDirectRegularization, Int?)] = [
      (low, 1, .phase, .init(), nil),
      (low, 1, .phase, .init(), 0),  // streamed (uncached) spectra
      (thick, 2, .phase, .init(phaseDamping: 0.1, smoothnessInverseAngstroms: 0), nil),
      (thick, 2, .complex, .init(), nil),
      (
        low, 2, .complex,
        .init(phaseDamping: 0.01, amplitudeDamping: 0.1, detectorWeighting: .inverseMean), nil
      ),
    ]
    for (point, factor, method, regularization, cache) in cases {
      let model = try model(s, device: device, cache: cache)
      let result = try model.reconstruct(
        point, upsamplingFactor: factor, method: method, regularization: regularization)
      XCTAssertEqual(result.rows, s.scan * factor)
      XCTAssertEqual(result.method, method)
      XCTAssertEqual(result.regularization, regularization)
      let expected = try Self.reference(
        s, spectra: spectra, model: model, point: point, factor: factor,
        complex: method == .complex,
        regularization: regularization)
      let count = result.rows * result.columns
      let phase = values(result.phase, count)
      var squared = 0.0
      var maximum = 0.0
      var power = 0.0
      for i in 0..<count {
        var d = phase[i] - expected.phase[i]
        d -= 2 * .pi * (d / (2 * .pi)).rounded()
        squared += d * d
        maximum = max(maximum, abs(d))
        power += expected.phase[i] * expected.phase[i]
      }
      let rms = (squared / Double(count)).squareRoot()
      var line =
        "DIRECT \(method.rawValue) \(factor)x cache \(cache.map(String.init) ?? "full"): phase rms \(rms) max \(maximum) (std \((power / Double(count)).squareRoot()))"
      XCTAssertLessThan(rms, 5e-7)
      XCTAssertLessThan(maximum, 2e-6)
      if method == .complex {
        let amplitude = values(try XCTUnwrap(result.amplitude), count)
        let mean = expected.amplitude.reduce(0, +) / Double(count)
        let error =
          (zip(amplitude, expected.amplitude).map { ($0 - $1) * ($0 - $1) }.reduce(0, +)
          / Double(count))
          .squareRoot() / mean
        line += " amplitude rms/mean \(error)"
        XCTAssertLessThan(error, 5e-7)
      } else {
        XCTAssertNil(result.amplitude)
      }
      print(line)
    }
  }

  func testMethodContractAndRegularizationChecks() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try SSBModelReconstructionTests.scene()
    let model = try model(s, device: device)
    let point = SSBModelPoint(c10Angstroms: 61, c12Angstroms: 14, phi12Radians: 0.7)
    // .ssb is the phase_of_mean path; its regularization is ignored and not recorded.
    let ssb = try model.reconstruct(
      point, upsamplingFactor: 2, method: .ssb, regularization: .init(phaseDamping: -1))
    let plain = try model.reconstruct(point, upsamplingFactor: 2)
    XCTAssertEqual(ssb.method, .ssb)
    XCTAssertNil(ssb.regularization)
    XCTAssertNil(ssb.amplitude)
    XCTAssertEqual(
      Data(bytes: ssb.phase.contents(), count: ssb.rows * ssb.columns * 4),
      Data(bytes: plain.phase.contents(), count: plain.rows * plain.columns * 4))
    let phase = try model.reconstruct(point, upsamplingFactor: 1, method: .phase)
    XCTAssertEqual(phase.kernel?.name, "ssb-direct")
    XCTAssertEqual(phase.regularization, SSBDirectRegularization())
    XCTAssertEqual(phase.point, point)
    // Python's settings() checks, with its messages.
    func message(_ regularization: SSBDirectRegularization, _ method: SSBReconstructionMethod)
      -> String?
    {
      do {
        _ = try model.reconstruct(
          point, upsamplingFactor: 1, method: method, regularization: regularization)
        return nil
      } catch let error as SSBDirectError {
        return error.errorDescription
      } catch {
        return "\(error)"
      }
    }
    for value in [0, -0.1, .nan, .infinity] {
      XCTAssertEqual(
        message(.init(phaseDamping: value), .phase),
        "phase_damping must be finite and positive; got \(value).")
      XCTAssertEqual(
        message(.init(amplitudeDamping: value), .complex),
        "amplitude_damping must be finite and positive; got \(value).")
    }
    // amplitude damping belongs to complex only.
    XCTAssertNil(message(.init(amplitudeDamping: 0), .phase))
    XCTAssertEqual(
      message(.init(smoothnessInverseAngstroms: -1), .phase),
      "smoothness_inv_A must be nonnegative and finite; use 0 for a flat prior, got -1.0.")
    XCTAssertNil(message(.init(smoothnessInverseAngstroms: 0), .phase))
    for factor in [0, 5, 8] {
      XCTAssertThrowsError(try model.reconstruct(point, upsamplingFactor: factor, method: .complex))
      {
        XCTAssertEqual($0 as? SSBDirectError, .unsupportedSampling(factor))
      }
    }
    XCTAssertThrowsError(
      try model.reconstruct(point, upsamplingFactor: 1, method: .phase, isCancelled: { true })
    ) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertEqual(SSBReconstructionMethod.allCases, [.ssb, .phase, .complex])
    XCTAssertEqual(SSBDetectorWeighting.inverseMean.rawValue, "inverse_mean")
    let decoded = try JSONDecoder().decode(
      SSBDirectRegularization.self,
      from: JSONEncoder().encode(SSBDirectRegularization(phaseDamping: 0.1)))
    XCTAssertEqual(decoded, SSBDirectRegularization(phaseDamping: 0.1))
  }

  func testPreparedSystemsAreReusedAcrossDampingAndMethods() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try SSBModelReconstructionTests.scene()
    let model = try model(s, device: device)
    let point = SSBModelPoint(
      c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -0.4, sampleTiltMrad: [2.5, -4],
      depthSpreadAngstroms: 140)
    func image(_ result: MetalSSBJointPhase) -> Data {
      Data(bytes: result.phase.contents(), count: result.rows * result.columns * 4)
    }
    let phase = try model.reconstruct(point, upsamplingFactor: 2, method: .phase)
    XCTAssertEqual(model.lastDirectTiming?.preparedFromCache, false)
    // A damping change solves the kept system again.
    let damped = try model.reconstruct(
      point, upsamplingFactor: 2, method: .phase, regularization: .init(phaseDamping: 0.3))
    XCTAssertEqual(model.lastDirectTiming?.preparedFromCache, true)
    XCTAssertNotEqual(image(phase), image(damped))
    XCTAssertEqual(
      image(phase), image(try model.reconstruct(point, upsamplingFactor: 2, method: .phase)))
    // A complex system serves the phase method with the same normal equations.
    _ = try model.reconstruct(point, upsamplingFactor: 2, method: .complex)
    XCTAssertEqual(model.lastDirectTiming?.preparedFromCache, false)
    let fresh = try model.reconstruct(point, upsamplingFactor: 1, method: .complex)
    XCTAssertEqual(fresh.rows, s.scan)
    _ = try model.reconstruct(point, upsamplingFactor: 2, method: .complex)
    let served = try model.reconstruct(point, upsamplingFactor: 2, method: .phase)
    XCTAssertEqual(model.lastDirectTiming?.preparedFromCache, true)
    XCTAssertEqual(image(phase), image(served))
    // New optics, a new prior or a new rotation prepare again.
    var moved = point
    moved.c10Angstroms += 5
    _ = try model.reconstruct(moved, upsamplingFactor: 2, method: .phase)
    XCTAssertEqual(model.lastDirectTiming?.preparedFromCache, false)
    _ = try model.reconstruct(
      moved, upsamplingFactor: 2, method: .phase,
      regularization: .init(smoothnessInverseAngstroms: 0.5))
    XCTAssertEqual(model.lastDirectTiming?.preparedFromCache, false)
    try model.retarget(scanDetectorRotationDegrees: 197.25)
    _ = try model.reconstruct(
      moved, upsamplingFactor: 2, method: .phase,
      regularization: .init(smoothnessInverseAngstroms: 0.5))
    XCTAssertEqual(model.lastDirectTiming?.preparedFromCache, false)
  }
}
