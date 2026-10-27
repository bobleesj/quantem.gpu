import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

/// Synthetic checks that always run: the coherent (phase_of_mean) reconstruction against an independent
/// double-precision evaluation of its definition, the centre-of-mass rotation and the branch test.
final class SSBModelReconstructionTests: XCTestCase {
  struct Scene {
    let calibration: MetalSSBCalibration
    let detector = 12
    let scan: Int
    let counts: [UInt16]
    let pixels: [Int]
    let geometry: MetalSSBGeometry
  }

  /// `scan` is 128 (the default) or another supported size; the pattern is the same function of position.
  /// `radius` is the BF selection radius in detector pixels; the aperture's support ends at 4.5 pixels, so a
  /// larger radius selects BF pixels without transfer.
  static func scene(scan: Int = 128, radius: Double = 3.6) throws -> Scene {
    let calibration = MetalSSBCalibration(
      voltageKV: 300, semiangleMrad: 30, scanSamplingRowAngstroms: 0.4,
      scanSamplingColumnAngstroms: 0.4,
      detectorSamplingRowMrad: 7.5, detectorSamplingColumnMrad: 7.5,
      brightfieldCenterRowPixels: 5.4,
      brightfieldCenterColumnPixels: 5.7, brightfieldRadiusPixels: radius)
    let detector = 12
    // A weak lattice with a slow gradient, so the phase has structure at several frequencies.
    var sums = [UInt64](repeating: 0, count: detector * detector)
    var planes: [[UInt16]] = []
    for pixel in 0..<(detector * detector) {
      let dr = Double(pixel / detector) - 5.4
      let dc = Double(pixel % detector) - 5.7
      var plane = [UInt16](repeating: 0, count: scan * scan)
      for r in 0..<scan {
        for c in 0..<scan {
          let x = 2 * Double.pi * Double(r) / 6.4
          let y = 2 * Double.pi * Double(c) / 5.12
          let value =
            400 + 40 * cos(x + 0.3 * dr) * cos(y - 0.2 * dc) + 25 * sin(x + y + 0.1 * dr * dc)
            + Double((r * 31 + c * 17 + pixel * 7) % 13)
          plane[r * scan + c] = UInt16(value.rounded())
          sums[pixel] += UInt64(plane[r * scan + c])
        }
      }
      planes.append(plane)
    }
    let setup = try calibration.geometry(
      detectorRows: detector, detectorColumns: detector, detectorSum: sums, scanRows: scan,
      scanColumns: scan)
    return Scene(
      calibration: calibration, scan: scan, counts: setup.pixels.flatMap { planes[$0] },
      pixels: setup.pixels,
      geometry: setup.geometry)
  }

  /// The definition in double precision: per output frequency the corrected waves of every BF pixel summed,
  /// G conj(gamma) / |gamma| with gamma = w1 P(q-k) P*(k) - w2 P*(q+k) P(k), the scan-mean replicas zero,
  /// the common DC at the origin, one inverse DFT, one phase.
  static func referencePhase(_ s: Scene, point: SSBModelPoint, rotation: Double, factor: Int)
    -> [Double]
  {
    let g = s.geometry
    let n = s.scan
    let m = n * factor
    let wavelength = Double(g.wavelengthAngstroms)
    let semiangle = Double(g.semiangleRadians)
    let (ay, ax) = (Double(g.angularSamplingYRadians), Double(g.angularSamplingXRadians))
    let factorChi = Double.pi / wavelength
    let angle = -rotation * .pi / 180
    let k = g.brightfieldKX.indices.map { i -> (Double, Double) in
      let x = Double(g.brightfieldKX[i])
      let y = Double(g.brightfieldKY[i])
      return (x * cos(angle) + y * sin(angle), -x * sin(angle) + y * cos(angle))
    }
    // G_k(q) by a separable DFT of each BF image.
    let forward = (0..<n).map {
      (cos(-2 * Double.pi * Double($0) / Double(n)), sin(-2 * Double.pi * Double($0) / Double(n)))
    }
    func dft(_ values: [Double]) -> [(Double, Double)] {
      var rows = [(Double, Double)](repeating: (0, 0), count: n * n)
      for r in 0..<n {
        for v in 0..<n {
          var re = 0.0
          var im = 0.0
          for c in 0..<n {
            let (wr, wi) = forward[(v * c) % n]
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
            let (wr, wi) = forward[(u * r) % n]
            let (a, b) = rows[r * n + v]
            re += a * wr - b * wi
            im += a * wi + b * wr
          }
          out[u * n + v] = (re, im)
        }
      }
      return out
    }
    let spectra = s.pixels.indices.map { index in
      dft(s.counts[(index * n * n)..<((index + 1) * n * n)].map(Double.init))
    }
    func probe(_ x: Double, _ y: Double) -> (chi: Double, aperture: Double, alpha2: Double) {
      let r2 = x * x + y * y
      let r = sqrt(r2)
      let alpha = r * wavelength
      let cos2 = r2 > 0 ? (x * x - y * y) / r2 : 0
      let sin2 = r2 > 0 ? 2 * x * y / r2 : 0
      let denominator = r > 0 ? sqrt((x * ay) * (x * ay) + (y * ax) * (y * ax)) / r : 0
      let aperture = denominator > 0 ? min(1, max(0, (semiangle - alpha) / denominator + 0.5)) : 1
      let chi =
        factorChi * alpha * alpha
        * (point.c10Angstroms + point.c12Angstroms
          * (cos2 * cos(2 * point.phi12Radians) + sin2 * sin(2 * point.phi12Radians)))
      return (chi, aperture, alpha * alpha)
    }
    func sinc(_ x: Double) -> Double { abs(x) < 1e-12 ? 1 : sin(x) / x }
    let dq = (Double(g.qxByRow[1]), Double(g.qyByColumn[1]))
    let dc = Double(g.dcValue.x)
    var total = [(Double, Double)](repeating: (0, 0), count: m * m)
    for row in 0..<m {
      for col in 0..<m {
        if row % n == 0 && col % n == 0 { continue }
        let fr = row < m / 2 ? row : row - m
        let fc = col < m / 2 ? col : col - m
        let q = (Double(fr) * dq.0, Double(fc) * dq.1)
        let shift =
          2 * Double.pi
          * (q.0 * point.sampleTiltMrad[0] * 1e-3 + q.1 * point.sampleTiltMrad[1] * 1e-3)
        var re = 0.0
        var im = 0.0
        for (index, kk) in k.enumerated() {
          let pk = probe(kk.0, kk.1)
          let pm = probe(q.0 - kk.0, q.1 - kk.1)
          let pp = probe(q.0 + kk.0, q.1 + kk.1)
          let w1 = sinc(
            0.5 * point.depthSpreadAngstroms * (-factorChi * (pm.alpha2 - pk.alpha2) - shift))
          let w2 = sinc(
            0.5 * point.depthSpreadAngstroms * (factorChi * (pp.alpha2 - pk.alpha2) - shift))
          let a = w1 * pm.aperture * pk.aperture
          let b = w2 * pp.aperture * pk.aperture
          if a == 0 && b == 0 { continue }
          let gr = a * cos(pm.chi - pk.chi) - b * cos(pp.chi - pk.chi)
          let gi = -a * sin(pm.chi - pk.chi) - b * sin(pp.chi - pk.chi)
          let magnitude = max(sqrt(gr * gr + gi * gi), 1e-8)
          let (vr, vi) = spectra[index][((fr % n + n) % n) * n + (fc % n + n) % n]
          re += (vr * gr + vi * gi) / magnitude
          im += (vi * gr - vr * gi) / magnitude
        }
        total[row * m + col] = (re, im)
      }
    }
    total[0] = (Double(g.logicalBrightfieldCount) * dc, 0)
    // Inverse DFT (the scale does not change the phase).
    let inverse = (0..<m).map {
      (cos(2 * Double.pi * Double($0) / Double(m)), sin(2 * Double.pi * Double($0) / Double(m)))
    }
    var rows = total
    for r in 0..<m {
      for c in 0..<m {
        var re = 0.0
        var im = 0.0
        for v in 0..<m {
          let (wr, wi) = inverse[(v * c) % m]
          let (a, b) = total[r * m + v]
          re += a * wr - b * wi
          im += a * wi + b * wr
        }
        rows[r * m + c] = (re, im)
      }
    }
    var phase = [Double](repeating: 0, count: m * m)
    for c in 0..<m {
      for r in 0..<m {
        var re = 0.0
        var im = 0.0
        for u in 0..<m {
          let (wr, wi) = inverse[(u * r) % m]
          let (a, b) = rows[u * m + c]
          re += a * wr - b * wi
          im += a * wi + b * wr
        }
        phase[r * m + c] = atan2(im, re)
      }
    }
    return phase
  }

  func testCoherentPhaseMatchesDoublePrecisionDefinition() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try Self.scene()
    XCTAssertGreaterThan(s.pixels.count, 20)
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    for cache in [nil, 0] as [Int?] {
      let engine = try MetalSSBEngine(device: device, geometry: s.geometry, cacheBudgetBytes: cache)
      try engine.prepare(brightfield: input, countType: .uint16)
      let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
      let points = [
        SSBModelPoint(c10Angstroms: 61, c12Angstroms: 14, phi12Radians: 0.7),
        SSBModelPoint(
          c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -0.4, sampleTiltMrad: [2.5, -4],
          depthSpreadAngstroms: 140),
      ]
      for point in points {
        for factor in [1, 3] where cache == nil || factor == 1 {
          let expected = Self.referencePhase(s, point: point, rotation: 17.25, factor: factor)
          let phase = try model.reconstruct(point, upsamplingFactor: factor)
          XCTAssertEqual(phase.rows, s.scan * factor)
          let actual = phase.phase.contents().assumingMemoryBound(to: Float.self)
          var maximum = 0.0
          for i in expected.indices { maximum = max(maximum, abs(Double(actual[i]) - expected[i])) }
          print(
            "COHERENT factor \(factor) depth \(point.depthSpreadAngstroms) cache \(cache.map(String.init) ?? "full") max |Δ| \(maximum)"
          )
          XCTAssertLessThan(maximum, 2e-4)
        }
      }
    }
  }

  func testAgreementRetargetEqualsAFreshModel() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try Self.scene()
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(device: device, geometry: s.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let point = SSBModelPoint(
      c10Angstroms: 20, c12Angstroms: 5, phi12Radians: 0.1, sampleTiltMrad: [1, 1],
      depthSpreadAngstroms: 90)
    let moved = try engine.jointModel(
      scanDetectorRotationDegrees: 10, bandInverseAngstroms: 0.2...0.9)
    try moved.retarget(scanDetectorRotationDegrees: 190.5)
    let fresh = try engine.jointModel(
      scanDetectorRotationDegrees: 190.5, bandInverseAngstroms: 0.2...0.9)
    XCTAssertEqual(try moved.agreements([point]), try fresh.agreements([point]))
    let a = try moved.reconstruct(point)
    let b = try fresh.reconstruct(point)
    XCTAssertEqual(
      Data(bytes: a.phase.contents(), count: a.rows * a.columns * 4),
      Data(bytes: b.phase.contents(), count: b.rows * b.columns * 4))
    // One evaluation of eight candidates equals eight evaluations of one.
    let points = (0..<9).map {
      SSBModelPoint(c10Angstroms: Double($0) * 7 - 30, c12Angstroms: 4, phi12Radians: 0.2)
    }
    XCTAssertEqual(try fresh.agreements(points), try points.flatMap { try fresh.agreements([$0]) })
  }

  /// The 180-degree map: a point and its opposite-branch image have the same agreement, for the low-order
  /// model with tilt and depth and for the fifth-order model, the image of the image is the point, and the
  /// branched fit gives the opposite branch the mapped fit of the first.
  func testOppositeBranchPointHasTheSameAgreement() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try Self.scene()
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(device: device, geometry: s.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let rotation = 17.25
    let opposite = SSBRotationBranches.opposite(rotation)
    let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)
    let higher = [
      SSBModelHigherOrder(order: 2, symmetry: 1, magnitudeAngstroms: 900, angleRadians: 0.4),
      SSBModelHigherOrder(order: 2, symmetry: 3, magnitudeAngstroms: 600, angleRadians: -0.9),
      SSBModelHigherOrder(order: 3, symmetry: 0, magnitudeAngstroms: -30_000),
      SSBModelHigherOrder(order: 3, symmetry: 2, magnitudeAngstroms: 20_000, angleRadians: 1.2),
      SSBModelHigherOrder(order: 3, symmetry: 4, magnitudeAngstroms: 9_000, angleRadians: -0.3),
      SSBModelHigherOrder(order: 4, symmetry: 3, magnitudeAngstroms: 400_000, angleRadians: 0.2),
      SSBModelHigherOrder(order: 5, symmetry: 0, magnitudeAngstroms: 2e6),
      SSBModelHigherOrder(order: 5, symmetry: 6, magnitudeAngstroms: 5e6, angleRadians: 0.45),
    ]
    let points = [
      SSBModelPoint(c10Angstroms: 61, c12Angstroms: 14, phi12Radians: 0.7),
      SSBModelPoint(
        c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -1.4, sampleTiltMrad: [2.5, -4],
        depthSpreadAngstroms: 140),
      SSBModelPoint(
        c10Angstroms: 45, c12Angstroms: 12, phi12Radians: 0.3, sampleTiltMrad: [-1.5, 3],
        depthSpreadAngstroms: 90, higherOrder: higher),
    ]
    for point in points {
      let image = SSBRotationBranches.oppositeBranch(point)
      XCTAssertEqual(image.c10Angstroms, -point.c10Angstroms)
      XCTAssertEqual(image.c12Angstroms, point.c12Angstroms)
      XCTAssertEqual(image.sampleTiltMrad, point.sampleTiltMrad.map { -$0 })
      XCTAssertEqual(image.depthSpreadAngstroms, point.depthSpreadAngstroms)
      XCTAssertLessThan(image.phi12Radians, Double.pi / 2)
      XCTAssertGreaterThanOrEqual(image.phi12Radians, -Double.pi / 2)
      let back = SSBRotationBranches.oppositeBranch(image)
      XCTAssertEqual(back.c10Angstroms, point.c10Angstroms)
      XCTAssertEqual(back.phi12Radians, point.phi12Radians, accuracy: 1e-14)
      for (a, b) in zip(back.higherOrder, point.higherOrder) {
        XCTAssertEqual(a.magnitudeAngstroms, b.magnitudeAngstroms)
        XCTAssertEqual(a.angleRadians, b.angleRadians, accuracy: 1e-14)
      }
      try model.retarget(scanDetectorRotationDegrees: rotation)
      let here = try model.agreements([point])[0]
      let phase = try model.reconstruct(point)
      let forward = Array(
        UnsafeBufferPointer(
          start: phase.phase.contents().assumingMemoryBound(to: Float.self),
          count: phase.rows * phase.columns))
      try model.retarget(scanDetectorRotationDegrees: opposite)
      let there = try model.agreements([image])[0]
      // On the other branch the same point is not an image: the symmetry is a property of the pair.
      let unmapped = try model.agreements([point])[0]
      print(
        String(
          format: "OPPOSITE_BRANCH higher %d agreement %.9e image %.9e (rel %.2e) unmapped %.9e",
          point.hasHigherOrder ? 1 : 0, here, there, (there - here) / here, unmapped))
      XCTAssertEqual(there, here, accuracy: 1e-6 * here)
      XCTAssertGreaterThan(abs(unmapped - here), 1e-3 * here)
      // The image's phase has the opposite contrast, which is what the branch test reads.
      let mirrored = try model.reconstruct(image)
      let backward = Array(
        UnsafeBufferPointer(
          start: mirrored.phase.contents().assumingMemoryBound(to: Float.self),
          count: mirrored.rows * mirrored.columns))
      func centred(_ values: [Float]) -> [Double] {
        let mean = values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
        return values.map { Double($0) - mean }
      }
      let a = centred(forward)
      let b = centred(backward)
      let correlation =
        zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
        / sqrt(a.reduce(0.0) { $0 + $1 * $1 } * b.reduce(0.0) { $0 + $1 * $1 })
      print(String(format: "OPPOSITE_BRANCH phase correlation %.6f", correlation))
      XCTAssertLessThan(correlation, -0.99)
    }
    // The branched fit searches the first branch only and maps it.
    let fitted = try SSBModelFit.findAberrations(
      model: model, rotationDegrees: rotation, darkField: nil,
      scanSampling: (0.4, 0.4),
      options: SSBFitOptions(trials: 40, compareStandard: false, maximumOrder: 1))
    XCTAssertNil(fitted.fits[0].mappedFromBranch)
    XCTAssertEqual(fitted.fits[1].mappedFromBranch, 0)
    XCTAssertEqual(fitted.fits[1].scanDetectorRotationDegrees, opposite)
    XCTAssertEqual(fitted.fits[1].point, SSBRotationBranches.oppositeBranch(fitted.fits[0].point))
    let first = try XCTUnwrap(fitted.fits[0].sampleAgreement)
    XCTAssertEqual(try XCTUnwrap(fitted.fits[1].sampleAgreement), first, accuracy: 1e-6 * first)
    XCTAssertEqual(
      model.scanDetectorRotationDegrees, fitted.fits[fitted.kept].scanDetectorRotationDegrees)
  }

  /// The tilt fit and the tilt-0 fit climb on the exact derivatives to a maximum: no small step in any
  /// fitted coordinate raises the agreement beyond float32 noise.
  func testExactNewtonFitsEndAtAMaximum() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try Self.scene()
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(device: device, geometry: s.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
    for tilt in [true, false] {
      let fit = try SSBModelFit.fit(
        model: model,
        options: SSBFitOptions(tilt: tilt, trials: 40, compareStandard: false, maximumOrder: 1))
      let agreement = try XCTUnwrap(fit.sampleAgreement)
      let p = fit.point
      let z = SSBAgreementSearch.encode([
        p.c10Angstroms, p.c12Angstroms, p.phi12Radians, p.sampleTiltMrad[0], p.sampleTiltMrad[1],
        p.depthSpreadAngstroms,
      ])
      let dimensions = tilt ? 6 : 3
      var neighbours: [SSBModelPoint] = []
      for d in 0..<dimensions {
        for sign in [1.0, -1.0] {
          var shifted = z
          shifted[d] += sign * [0.2, 0.2, 0.2, 0.02, 0.02, 0.5][d]
          let q = SSBAgreementSearch.decode(shifted)
          neighbours.append(
            SSBModelPoint(
              c10Angstroms: q[0], c12Angstroms: q[1], phi12Radians: q[2],
              sampleTiltMrad: [q[3], q[4]], depthSpreadAngstroms: q[5]))
        }
      }
      let best = try model.agreements(neighbours).max()!
      print(
        String(
          format: "EXACT_FIT tilt %d agreement %.9e best neighbour %.9e (rel %.2e) centers %d",
          tilt ? 1 : 0,
          agreement, best, (best - agreement) / agreement,
          fit.newtonStarts?.reduce(0) { $0 + $1.evaluations } ?? -1))
      XCTAssertLessThan(best, agreement * (1 + 1e-6))
      if !tilt {
        XCTAssertEqual(p.sampleTiltMrad, [0, 0])
        XCTAssertEqual(p.depthSpreadAngstroms, 0)
      }
    }
  }

  /// The exact low-order gradient and Hessian against Richardson-extrapolated central differences of the
  /// value and of the gradient, at points with and without tilt and depth spread; the value against the
  /// agreement objective.
  func testLowOrderDerivativesMatchFiniteDifferences() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try Self.scene()
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(device: device, geometry: s.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
    let points = [
      SSBModelPoint(c10Angstroms: 61, c12Angstroms: 14, phi12Radians: 0.7),
      SSBModelPoint(
        c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -1.4, sampleTiltMrad: [2.5, -4],
        depthSpreadAngstroms: 140),
      SSBModelPoint(
        c10Angstroms: 12, c12Angstroms: 30, phi12Radians: 0.2, sampleTiltMrad: [-7, 1.5],
        depthSpreadAngstroms: 420),
    ]
    let steps = [2.0, 2.0, 2.0, 0.25, 0.25, 8.0]
    for point in points {
      let z = SSBAgreementSearch.encode([
        point.c10Angstroms, point.c12Angstroms, point.phi12Radians, point.sampleTiltMrad[0],
        point.sampleTiltMrad[1], point.depthSpreadAngstroms,
      ])
      var rows = [z]
      for d in 0..<6 {
        for scale in [1.0, 0.5] {
          for sign in [1.0, -1.0] {
            var shifted = z
            shifted[d] += sign * scale * steps[d]
            rows.append(shifted)
          }
        }
      }
      let results = try model.lowOrderDerivatives(rows)
      let exact = results[0]
      let agreement = try model.agreements([point])[0]
      XCTAssertEqual(exact.value, agreement, accuracy: 2e-6 * agreement)
      func richardson(_ d: Int, _ f: (SSBLowOrderDerivatives.Result) -> Double) -> Double {
        let r = results[1 + 4 * d..<5 + 4 * d].map(f)
        let coarse = (r[0] - r[1]) / (2 * steps[d])
        let fine = (r[2] - r[3]) / steps[d]
        return (4 * fine - coarse) / 3
      }
      // Scales: the change of the value and of each gradient component over one step.
      let valueScale = (0..<6).map { abs(exact.gradient[$0]) * steps[$0] }.max()! + 1e-6 * agreement
      var worstGradient = 0.0
      var worstHessian = 0.0
      for d in 0..<6 {
        let numeric = richardson(d) { $0.value }
        worstGradient = max(worstGradient, abs(numeric - exact.gradient[d]) * steps[d] / valueScale)
        for e in 0..<6 {
          let mixed = richardson(d) { $0.gradient[e] }
          let scale =
            (0..<6).map { abs(exact.hessian[e][$0]) * steps[$0] * steps[e] }.max()! + 1e-6
            * valueScale
          worstHessian = max(
            worstHessian, abs(mixed - exact.hessian[d][e]) * steps[d] * steps[e] / scale)
        }
      }
      print(
        String(
          format:
            "LOW_ORDER_DERIVATIVES depth %.0f value rel %.2e gradient rel %.2e hessian rel %.2e",
          point.depthSpreadAngstroms, (exact.value - agreement) / agreement, worstGradient,
          worstHessian))
      XCTAssertLessThan(worstGradient, 2e-3)
      XCTAssertLessThan(worstHessian, 2e-3)
      // The fused pass (above) and the value pass followed by the derivative pass at the same rows: the same
      // value bitwise, the derivatives to float32 summation order.
      let twoStage = try model.lowOrderDerivatives(rows, twoStage: true)
      func worst(_ a: [SSBLowOrderDerivatives.Result], _ b: [SSBLowOrderDerivatives.Result]) -> (
        value: Double, derivatives: Double
      ) {
        var value = 0.0
        var derivatives = 0.0
        for (x, y) in zip(a, b) {
          value = max(value, abs(x.value - y.value) / x.value)
          let gradientScale =
            (0..<6).map { abs(x.gradient[$0]) * steps[$0] }.max()! + 1e-6 * x.value
          for d in 0..<6 {
            derivatives = max(
              derivatives, abs(x.gradient[d] - y.gradient[d]) * steps[d] / gradientScale)
            let hessianScale =
              (0..<6).map { abs(x.hessian[d][$0]) * steps[$0] * steps[d] }.max()! + 1e-6
              * gradientScale
            for e in 0..<6 {
              derivatives = max(
                derivatives,
                abs(x.hessian[d][e] - y.hessian[d][e]) * steps[d] * steps[e] / hessianScale)
            }
          }
        }
        return (value, derivatives)
      }
      let stages = worst(results, twoStage)
      print(
        String(
          format: "LOW_ORDER_DERIVATIVES depth %.0f two-stage value %.1e derivatives %.1e",
          point.depthSpreadAngstroms, stages.value, stages.derivatives))
      XCTAssertEqual(stages.value, 0)
      XCTAssertLessThan(stages.derivatives, 1e-4)
    }
  }

  func testCurlRotationRecoversAKnownAngle() {
    let (rows, columns) = (64, 80)
    func potential(_ r: Int, _ c: Int) -> Double {
      sin(Double(r) * 0.21) * cos(Double(c) * 0.17) + 0.3 * sin(Double(r + 2 * c) * 0.11)
    }
    // The field a rotated detector measures: the gradient (central differences) rotated by -theta.
    let theta = 37.5 * Double.pi / 180
    var row = [Float](repeating: 0, count: rows * columns)
    var column = row
    for r in 0..<rows {
      for c in 0..<columns {
        let gr = 0.5 * (potential(r + 1, c) - potential(r - 1, c))
        let gc = 0.5 * (potential(r, c + 1) - potential(r, c - 1))
        row[r * columns + c] = Float(cos(theta) * gr + sin(theta) * gc + 96)
        column[r * columns + c] = Float(-sin(theta) * gr + cos(theta) * gc + 98)
      }
    }
    let estimate = SSBRotationBranches.curlRotationDegrees(
      row: row, column: column, rows: rows, columns: columns)
    XCTAssertEqual(estimate, 37.5, accuracy: 0.05)
    XCTAssertEqual(SSBRotationBranches.opposite(165.25), 345.25)
    XCTAssertEqual(SSBRotationBranches.opposite(345.25), 165.25)
    XCTAssertEqual(SSBRotationBranches.wrap(-20), 340)
    let com = SSBRotationBranches.centerOfMass(
      total: [4, 0], rowMoment: [10, 0], columnMoment: [6, 0])
    XCTAssertEqual(com.row, [2.5, 0])
    XCTAssertEqual(com.column, [1.5, 0])
  }

  func testFiltersMatchSciPy() {
    var image: [Double] = []
    for r in 0..<6 {
      for c in 0..<7 {
        let base = Double((r * 7 + c * 3) % 11) * 0.5
        image.append(base + Double(r) * 0.1)
      }
    }
    let smoothed = SSBRotationBranches.gaussianFilter(image, rows: 6, columns: 7, sigma: (1.0, 1.7))
    for (index, value) in zip(
      [0, 5, 17, 41],
      [2.0513797971355063, 2.3636615048879412, 2.6821982078592734, 3.47217591601569])
    {
      XCTAssertEqual(smoothed[index], value, accuracy: 1e-13)
    }
    let maximum = SSBRotationBranches.rankFilter(
      image, rows: 6, columns: 7, size: (3, 5), maximum: true)
    let minimum = SSBRotationBranches.rankFilter(
      image, rows: 6, columns: 7, size: (3, 5), maximum: false)
    XCTAssertEqual([0, 9, 23, 40].map { maximum[$0] }, [5.1, 5.1, 5.3, 5.4])
    XCTAssertEqual([0, 9, 23, 40].map { minimum[$0] }, [0.0, 0.0, 0.30000000000000004, 0.5])
    XCTAssertEqual(SSBRotationBranches.quantile([3, 1, 4, 1, 5, 9, 2, 6], 0.8), 5.6000000000000005)
    XCTAssertEqual(SSBRotationBranches.columnFilter(scanSampling: (0.373, 0.264)).window.0, 11)
    XCTAssertEqual(SSBRotationBranches.columnFilter(scanSampling: (0.373, 0.264)).window.1, 17)
  }

  func testBranchTestKeepsPositiveColumns() {
    let size = 96
    let step = 0.4
    var dark = [Float](repeating: 0, count: size * size)
    var lattice = [Float](repeating: 0, count: size * size)
    for r in 0..<size {
      for c in 0..<size {
        // Columns every 4 A; the dark field and the true phase peak at them.
        let x = Double(r) * step
        let y = Double(c) * step
        let value = Float(
          exp(
            -((x.truncatingRemainder(dividingBy: 4) - 2)
              * (x.truncatingRemainder(dividingBy: 4) - 2)
              + (y.truncatingRemainder(dividingBy: 4) - 2)
                * (y.truncatingRemainder(dividingBy: 4) - 2))
              / 0.5))
        dark[r * size + c] = 100 + 50 * value
        lattice[r * size + c] = 0.2 * value
      }
    }
    let negative = lattice.map { -$0 }
    let result = SSBRotationBranches.compare(
      phases: [negative, lattice], rotations: [10, 190], rows: size, columns: size, darkField: dark,
      scanSampling: (step, step))
    XCTAssertEqual(result.decidedBy, .columns)
    XCTAssertEqual(result.kept, 1)
    XCTAssertGreaterThan(result.columns, 10)
    XCTAssertGreaterThan(result.branches[1].column?.z ?? 0, 3)
    let tail = SSBRotationBranches.compare(
      phases: [negative, lattice], rotations: [10, 190], rows: size, columns: size, darkField: nil,
      scanSampling: (step, step))
    XCTAssertEqual(tail.decidedBy, .phaseTail)
    XCTAssertEqual(tail.kept, 1)
    XCTAssertEqual(
      SSBRotationBranches.phaseSkewness([0, 0, 0, 1]), 1.1547005383792515, accuracy: 1e-15)
  }
}
