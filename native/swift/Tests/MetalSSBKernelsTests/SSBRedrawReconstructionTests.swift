import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

/// The interactive redraw kernel (`ssb_redraw.metal`) against the per-term coherent kernel it replaces at 1x-4x,
/// and against an independent double-precision definition with every higher-order term.
///
/// Stated tolerance: the redraw evaluates chi(k +- q) - chi(k) as row-collapsed polynomials expanded about each BF
/// pixel, mirrors the anti-Hermitian half, and uses fast trigonometry, so it differs from the per-term kernel by
/// float32 rounding only. Phases (radians, wrapped) agree within 1e-5 maximum and 1e-6 RMS (observed: at most
/// 1.4e-7 and 5.7e-8); against the double-precision definition within 2e-4, the bound the existing coherent test
/// uses (observed: at most 1.2e-7), and within 1e-6 maximum and 2e-7 RMS where the sidebands nearly cancel.
final class SSBRedrawReconstructionTests: XCTestCase {
  static let fifthOrder: [SSBModelHigherOrder] = zip(
    SSBModelHigherOrder.supported,
    [900.0, 600, 2.4e4, 1.6e4, 9e3, 3e5, 2e5, 1.5e5, 4e6, 3e6, 2e6, 1.5e6]
  ).enumerated().map { index, pair in
    var term = pair.0
    term.magnitudeAngstroms = pair.1
    term.angleRadians = 0.35 + 0.53 * Double(index)
    return term
  }

  static let points = [
    SSBModelPoint(
      c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -0.4, sampleTiltMrad: [2.5, -4],
      depthSpreadAngstroms: 140),
    SSBModelPoint(
      c10Angstroms: 61, c12Angstroms: 14, phi12Radians: 0.7, sampleTiltMrad: [-1.5, 3],
      depthSpreadAngstroms: 0, higherOrder: fifthOrder),
    SSBModelPoint(
      c10Angstroms: -25, c12Angstroms: 11, phi12Radians: 1.1, sampleTiltMrad: [3, 1.2],
      depthSpreadAngstroms: 180, higherOrder: fifthOrder),
  ]

  private func wrappedDifference(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, count: Int)
    -> (maximum: Double, rms: Double)
  {
    var maximum = 0.0
    var squares = 0.0
    for i in 0..<count {
      var d = Double(a[i]) - Double(b[i])
      d -= 2 * .pi * (d / (2 * .pi)).rounded()
      maximum = max(maximum, abs(d))
      squares += d * d
    }
    return (maximum, sqrt(squares / Double(count)))
  }

  func testRedrawMatchesPerTermKernel() throws {
    try redrawMatchesPerTermKernel(scan: 128, caches: [nil, 0])
  }

  /// A 256 x 256 scan: two column tiles per threadgroup row and its own Nyquist geometry.
  func testRedrawMatchesPerTermKernelAt256() throws {
    try redrawMatchesPerTermKernel(scan: 256, caches: [nil])
  }

  private func redrawMatchesPerTermKernel(scan: Int, caches: [Int?]) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try SSBModelReconstructionTests.scene(scan: scan)
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    // Cached spectra (half spectra) and a streamed (uncached) engine.
    for cache in caches {
      let engine = try MetalSSBEngine(device: device, geometry: s.geometry, cacheBudgetBytes: cache)
      try engine.prepare(brightfield: input, countType: .uint16)
      let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
      for point in Self.points {
        for factor in [1, 2, 3, 4] {
          model.redrawReconstruction = false
          let reference = try model.reconstruct(point, upsamplingFactor: factor)
          model.redrawReconstruction = true
          let redraw = try model.reconstruct(point, upsamplingFactor: factor)
          XCTAssertEqual(redraw.rows, s.scan * factor)
          XCTAssertEqual(redraw.columns, s.scan * factor)
          let d = wrappedDifference(
            redraw.phase.contents().assumingMemoryBound(to: Float.self),
            reference.phase.contents().assumingMemoryBound(to: Float.self),
            count: redraw.rows * redraw.columns)
          print(
            "REDRAW scan \(scan) factor \(factor) higher \(point.hasHigherOrder) depth \(point.depthSpreadAngstroms) cache \(cache.map(String.init) ?? "full") max \(d.maximum) rms \(d.rms)"
          )
          XCTAssertLessThan(d.maximum, 1e-5)
          XCTAssertLessThan(d.rms, 1e-6)
        }
      }
    }
  }

  func testRedrawMatchesDoublePrecisionFifthOrderDefinition() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    for (scan, factors) in [(128, [1, 3]), (256, [2])] {
      let s = try SSBModelReconstructionTests.scene(scan: scan)
      let input = try XCTUnwrap(
        s.counts.withUnsafeBytes {
          device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
      let engine = try MetalSSBEngine(device: device, geometry: s.geometry)
      try engine.prepare(brightfield: input, countType: .uint16)
      let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
      for point in Self.points.dropFirst() {
        for factor in factors {
          let expected = Self.referencePhase(s, point: point, rotation: 17.25, factor: factor)
          let phase = try model.reconstruct(point, upsamplingFactor: factor)
          let actual = phase.phase.contents().assumingMemoryBound(to: Float.self)
          var maximum = 0.0
          for i in expected.indices {
            var d = Double(actual[i]) - expected[i]
            d -= 2 * .pi * (d / (2 * .pi)).rounded()
            maximum = max(maximum, abs(d))
          }
          print(
            "REDRAW DEFINITION scan \(scan) factor \(factor) depth \(point.depthSpreadAngstroms) max \(maximum)")
          XCTAssertLessThan(maximum, 2e-4)
        }
      }
    }
  }

  /// The synthetic scene's geometry with strong contrast at the first scan frequencies (one period across the
  /// scan, a different phase per detector pixel), where the two sidebands nearly coincide.
  static func slowScene() throws -> SSBModelReconstructionTests.Scene {
    let base = try SSBModelReconstructionTests.scene(scan: 128)
    let scan = base.scan
    let detector = base.detector
    var sums = [UInt64](repeating: 0, count: detector * detector)
    var planes: [[UInt16]] = []
    for pixel in 0..<(detector * detector) {
      let dr = Double(pixel / detector) - 5.4
      let dc = Double(pixel % detector) - 5.7
      var plane = [UInt16](repeating: 0, count: scan * scan)
      for r in 0..<scan {
        for c in 0..<scan {
          let x = 2 * Double.pi * Double(r) / Double(scan)
          let y = 2 * Double.pi * Double(c) / Double(scan)
          let value =
            400 + 60 * cos(x + 0.4 * dr - 0.3 * dc) + 45 * sin(y + 0.2 * dr + 0.5 * dc)
            + 25 * cos(x + y + 0.1 * dr * dc) + Double((r * 31 + c * 17 + pixel * 7) % 13)
          plane[r * scan + c] = UInt16(value.rounded())
          sums[pixel] += UInt64(plane[r * scan + c])
        }
      }
      planes.append(plane)
    }
    let setup = try base.calibration.geometry(
      detectorRows: detector, detectorColumns: detector, detectorSum: sums, scanRows: scan, scanColumns: scan)
    return SSBModelReconstructionTests.Scene(
      calibration: base.calibration, scan: scan, counts: setup.pixels.flatMap { planes[$0] },
      pixels: setup.pixels, geometry: setup.geometry)
  }

  /// Nearly cancelling sidebands at the lowest scan frequency. C10 and C12 balance along the scan rows (phi12 =
  /// 90 degrees, C12 = C10 - 0.5 A), so at the first row frequency the second difference
  /// chi(k + q) + chi(k - q) - 2 chi(k) is about 2e-5 rad while chi itself reaches about 100 rad; the sign of that
  /// second difference decides each pair's correction, and |gamma| is about 2e-5. Differences of separately rounded
  /// float32 chi values cannot resolve it (ssb-redraw v0.2: 4.5e-3 / 5.3e-3 rad RMS against the definition); the
  /// expansion of chi about each BF pixel and gamma in half-sum / half-difference form (v0.3) do (observed: 3.7e-8 /
  /// 2.3e-8 rad RMS, 1.2e-7 / 8.1e-8 maximum, without / with small fifth-order terms).
  func testRedrawResolvesNearlyCancellingSidebands() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try Self.slowScene()
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(device: device, geometry: s.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
    let small = Self.fifthOrder.map { term -> SSBModelHigherOrder in
      var term = term
      term.magnitudeAngstroms *= 1e-3
      return term
    }
    for higherOrder in [[], small] {
      let point = SSBModelPoint(
        c10Angstroms: 400, c12Angstroms: 399.5, phi12Radians: .pi / 2, higherOrder: higherOrder)
      let expected = Self.referencePhase(s, point: point, rotation: 17.25, factor: 1)
      let phase = try model.reconstruct(point, upsamplingFactor: 1)
      let actual = phase.phase.contents().assumingMemoryBound(to: Float.self)
      var maximum = 0.0
      var squares = 0.0
      for i in expected.indices {
        var d = Double(actual[i]) - expected[i]
        d -= 2 * .pi * (d / (2 * .pi)).rounded()
        maximum = max(maximum, abs(d))
        squares += d * d
      }
      let rms = sqrt(squares / Double(expected.count))
      print("REDRAW CANCELLING higher \(point.hasHigherOrder) max \(maximum) rms \(rms)")
      XCTAssertLessThan(rms, 2e-7)
      XCTAssertLessThan(maximum, 1e-6)
    }
  }

  /// The coherent definition in double precision with the full 14-term chi: the minus sideband uses chi(k - q),
  /// the plus sideband chi(q + k), each pair G conj(gamma) / max(|gamma|, 1e-8), the scan-mean replicas zero,
  /// the common DC at the origin, one inverse DFT, one phase.
  static func referencePhase(
    _ s: SSBModelReconstructionTests.Scene, point: SSBModelPoint, rotation: Double, factor: Int
  ) -> [Double] {
    let g = s.geometry
    let n = s.scan
    let m = n * factor
    let wavelength = Double(g.wavelengthAngstroms)
    let semiangle = Double(g.semiangleRadians)
    let (ay, ax) = (Double(g.angularSamplingYRadians), Double(g.angularSamplingXRadians))
    let factorChi = Double(g.phaseFactor ?? Float(Double.pi / wavelength))
    let coefficients = point.reconstructionCoefficients.map(Double.init)
    let azimuth = [0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6]
    let power = [1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0]
    func chi(_ row: Double, _ col: Double) -> Double {
      let x = row * wavelength
      let y = col * wavelength
      let r2 = x * x + y * y
      var re = [1.0]
      var im = [0.0]
      for j in 1...6 {
        re.append(re[j - 1] * x - im[j - 1] * y)
        im.append(im[j - 1] * x + re[j - 1] * y)
      }
      var value = 0.0
      for term in 0..<14 {
        value +=
          pow(r2, Double(power[term]))
          * (re[azimuth[term]] * coefficients[2 * term] + im[azimuth[term]] * coefficients[2 * term + 1])
      }
      return 2 * Double.pi / wavelength * value
    }
    let angle = -rotation * .pi / 180
    let k = g.brightfieldKX.indices.map { i -> (Double, Double) in
      let x = Double(g.brightfieldKX[i])
      let y = Double(g.brightfieldKY[i])
      return (x * cos(angle) + y * sin(angle), -x * sin(angle) + y * cos(angle))
    }
    func aperture(_ x: Double, _ y: Double) -> (aperture: Double, alpha2: Double) {
      let r2 = x * x + y * y
      let r = sqrt(r2)
      let alpha = r * wavelength
      let denominator = r > 0 ? sqrt((x * ay) * (x * ay) + (y * ax) * (y * ax)) / r : 0
      let weight = denominator > 0 ? min(1, max(0, (semiangle - alpha) / denominator + 0.5)) : 1
      return (weight, alpha * alpha)
    }
    func sinc(_ x: Double) -> Double { abs(x) < 1e-12 ? 1 : sin(x) / x }
    let spectra = s.pixels.indices.map { index in
      dft(s.counts[(index * n * n)..<((index + 1) * n * n)].map(Double.init), n: n)
    }
    let dq = (Double(g.qxByRow[1]), Double(g.qyByColumn[1]))
    var total = [(Double, Double)](repeating: (0, 0), count: m * m)
    for row in 0..<m {
      for col in 0..<m {
        if row % n == 0 && col % n == 0 { continue }
        let fr = row < (m - 1) / 2 + 1 ? row : row - m
        let fc = col < (m - 1) / 2 + 1 ? col : col - m
        let q = (Double(fr) * dq.0, Double(fc) * dq.1)
        let shift =
          2 * Double.pi
          * (q.0 * point.sampleTiltMrad[0] * 1e-3 + q.1 * point.sampleTiltMrad[1] * 1e-3)
        var re = 0.0
        var im = 0.0
        for (index, kk) in k.enumerated() {
          let pk = aperture(kk.0, kk.1)
          let pm = aperture(q.0 - kk.0, q.1 - kk.1)
          let pp = aperture(q.0 + kk.0, q.1 + kk.1)
          let w1 = sinc(
            0.5 * point.depthSpreadAngstroms * (-factorChi * (pm.alpha2 - pk.alpha2) - shift))
          let w2 = sinc(
            0.5 * point.depthSpreadAngstroms * (factorChi * (pp.alpha2 - pk.alpha2) - shift))
          let a = w1 * pm.aperture * pk.aperture
          let b = w2 * pp.aperture * pk.aperture
          if a == 0 && b == 0 { continue }
          let chiK = chi(kk.0, kk.1)
          let dm = chi(kk.0 - q.0, kk.1 - q.1) - chiK
          let dp = chi(q.0 + kk.0, q.1 + kk.1) - chiK
          let gr = a * cos(dm) - b * cos(dp)
          let gi = -a * sin(dm) - b * sin(dp)
          let magnitude = max(sqrt(gr * gr + gi * gi), 1e-8)
          let (vr, vi) = spectra[index][((fr % n + n) % n) * n + (fc % n + n) % n]
          re += (vr * gr + vi * gi) / magnitude
          im += (vi * gr - vr * gi) / magnitude
        }
        total[row * m + col] = (re, im)
      }
    }
    total[0] = (Double(g.logicalBrightfieldCount) * Double(g.dcValue.x), 0)
    return inverseDFT(total, m: m).map { atan2($0.1, $0.0) }
  }

  /// Two-dimensional forward DFT of an n x n real image.
  static func dft(_ values: [Double], n: Int) -> [(Double, Double)] {
    let w = (0..<n).map {
      (cos(-2 * Double.pi * Double($0) / Double(n)), sin(-2 * Double.pi * Double($0) / Double(n)))
    }
    var rows = [(Double, Double)](repeating: (0, 0), count: n * n)
    for r in 0..<n {
      for v in 0..<n {
        var re = 0.0
        var im = 0.0
        for c in 0..<n {
          let (wr, wi) = w[(v * c) % n]
          re += values[r * n + c] * wr
          im += values[r * n + c] * wi
        }
        rows[r * n + v] = (re, im)
      }
    }
    var out = rows
    for u in 0..<n {
      for v in 0..<n {
        var re = 0.0
        var im = 0.0
        for r in 0..<n {
          let (wr, wi) = w[(u * r) % n]
          let (a, b) = rows[r * n + v]
          re += a * wr - b * wi
          im += a * wi + b * wr
        }
        out[u * n + v] = (re, im)
      }
    }
    return out
  }

  /// Unscaled two-dimensional inverse DFT (the scale does not change the phase).
  static func inverseDFT(_ values: [(Double, Double)], m: Int) -> [(Double, Double)] {
    let w = (0..<m).map {
      (cos(2 * Double.pi * Double($0) / Double(m)), sin(2 * Double.pi * Double($0) / Double(m)))
    }
    var rows = values
    for r in 0..<m {
      for c in 0..<m {
        var re = 0.0
        var im = 0.0
        for v in 0..<m {
          let (wr, wi) = w[(v * c) % m]
          let (a, b) = values[r * m + v]
          re += a * wr - b * wi
          im += a * wi + b * wr
        }
        rows[r * m + c] = (re, im)
      }
    }
    var out = rows
    for c in 0..<m {
      for r in 0..<m {
        var re = 0.0
        var im = 0.0
        for u in 0..<m {
          let (wr, wi) = w[(u * r) % m]
          let (a, b) = rows[u * m + c]
          re += a * wr - b * wi
          im += a * wi + b * wr
        }
        out[r * m + c] = (re, im)
      }
    }
    return out
  }
}
