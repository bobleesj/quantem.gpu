import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

final class SSBFifthOrderGradientTests: XCTestCase {
  /// The fifth-order agreement in Double from the model's exact float32 inputs, at Double coefficients.
  struct Reference {
    let k: [SIMD2<Float>]
    let q: [SIMD2<Float>]
    let weights: [Float]
    let probe: [SIMD4<Float>]
    let evidence: [SIMD2<Float>]
    let wl, semiangle, angularRow, angularColumn, factor: Double
    let coordinates: SSBFifthOrderCoordinates

    init(model: MetalSSBJointModel) throws {
      let arrays = try model.objectiveArrays()
      k = arrays.k
      q = arrays.q
      weights = arrays.weights
      probe = arrays.probe
      evidence = arrays.evidence
      wl = arrays.constants["wavelength_A"]!
      semiangle = arrays.constants["semiangle_rad"]!
      angularRow = arrays.constants["angular_row_rad"]!
      angularColumn = arrays.constants["angular_column_rad"]!
      factor = arrays.constants["factor"]!
      coordinates = SSBFifthOrderCoordinates(
        wavelengthAngstroms: model.wavelengthAngstroms, semiangleRadians: model.semiangleRadians)
    }

    func geometry(_ dx: Double, _ dy: Double) -> SIMD4<Double> {
      let r2 = dx * dx + dy * dy
      let r = r2.squareRoot()
      let alpha = r * wl
      let invR2 = r2 > 1e-30 ? 1 / r2 : 0
      let invR = r > 1e-15 ? 1 / r : 0
      let denom = ((dx * angularRow) * (dx * angularRow) + (dy * angularColumn) * (dy * angularColumn))
        .squareRoot() * invR
      let edge = denom > 1e-15 ? (semiangle - alpha) / denom + 0.5 : 1
      return SIMD4(
        alpha * alpha, (dx * dx - dy * dy) * invR2, 2 * dx * dy * invR2, min(max(edge, 0), 1))
    }

    static let azimuth = [0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6]
    static let power = [1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0]

    func chi(_ row: Double, _ col: Double, _ c: [Double]) -> Double {
      let x = row * wl
      let y = col * wl
      let r2 = x * x + y * y
      let radial = SIMD4<Double>(1, r2, r2 * r2, r2 * r2 * r2)
      var re = SIMD8<Double>(repeating: 0)
      var im = SIMD8<Double>(repeating: 0)
      re[0] = 1
      for m in 1...6 {
        re[m] = re[m - 1] * x - im[m - 1] * y
        im[m] = im[m - 1] * x + re[m - 1] * y
      }
      var value = 0.0
      for i in 0..<14 {
        let m = Self.azimuth[i]
        value += radial[Self.power[i]] * (re[m] * c[2 * i] + im[m] * c[2 * i + 1])
      }
      return 2 * Double.pi / wl * value
    }

    func sinc(_ x: Double) -> Double { abs(x) < 1e-12 ? 1 : sin(x) / x }

    /// The Double pack of `SSBFifthOrderCoordinates.pack`.
    func packed(_ z: [Double]) -> [Double] {
      var c = Array(repeating: 0.0, count: 28)
      c[0] = z[0] * 50
      c[2] = z[1] * 50
      c[3] = z[2] * 50
      var j = 6
      for (i, (n, m)) in SSBFifthOrderCoordinates.orders.enumerated() where n >= 2 {
        let scale = coordinates.wavelengthAngstroms
          / (2 * Double.pi * pow(coordinates.semiangleRadians, Double(n + 1)))
        c[2 * i] = z[j] * scale
        j += 1
        if m != 0 {
          c[2 * i + 1] = z[j] * scale
          j += 1
        }
      }
      return c
    }

    /// The agreement over every `kStride`-th BF pixel.
    func agreement(_ z: [Double], kStride: Int = 1) -> Double {
      let c = packed(z)
      let t = z[5] * 100
      let thr = z[3] * 5e-3
      let thc = z[4] * 5e-3
      let chiK = k.map { chi(Double($0.x), Double($0.y), c) }
      var total = 0.0
      for (qi, qv) in q.enumerated() {
        let qx = Double(qv.x)
        let qy = Double(qv.y)
        let shift = 2 * Double.pi * (qx * thr + qy * thc)
        var nr = 0.0
        var ni = 0.0
        var dd = 0.0
        for (kk, kv) in k.enumerated() where kk % kStride == 0 {
          let kx = Double(kv.x)
          let ky = Double(kv.y)
          let gk = SIMD4<Double>(probe[kk])
          let gm = geometry(qx - kx, qy - ky)
          let gp = geometry(qx + kx, qy + ky)
          let a1 = gm.w * gk.w
          let a2 = gp.w * gk.w
          if a1 == 0 && a2 == 0 { continue }
          let phi1 = chi(kx - qx, ky - qy, c) - chiK[kk]
          let phi2 = chi(kx + qx, ky + qy, c) - chiK[kk]
          let b1 = sinc(0.5 * t * (-factor * (gm.x - gk.x) - shift)) * a1
          let b2 = sinc(0.5 * t * (factor * (gp.x - gk.x) - shift)) * a2
          let gr = b1 * cos(phi1) - b2 * cos(phi2)
          let gi = -b1 * sin(phi1) - b2 * sin(phi2)
          let g = evidence[kk * q.count + qi]
          let gre = Double(g.x)
          let gim = Double(g.y)
          nr += gre * gr + gim * gi
          ni += gim * gr - gre * gi
          dd += gr * gr + gi * gi
        }
        if dd > 0 { total += Double(weights[qi]) * (nr * nr + ni * ni) / dd }
      }
      return total
    }
  }

  static let z: [Double] = [
    -0.38, 0.12, -0.07, 0.5, -0.36, 1.4,
    0.31, -0.42, 0.18, 0.27, -0.55, 0.62, -0.21, 0.09, 0.44, -0.33, 0.15,
    -0.48, 0.26, 0.35, -0.12, 0.58, -0.29, 0.07, 0.41, -0.17, 0.22, -0.38,
  ]

  func testExactGradientMatchesDoubleCentralDifferences() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try Self.model(device)
    let reference = try Reference(model: model)
    let coordinates = reference.coordinates
    let z = Self.z
    let exact = try model.fifthOrderAgreementsAndGradients(
      [coordinates.point(z)], packedCoefficients: [coordinates.pack(z)])[0]
    // The value pass is the production objective (ssb-fifth-agreement) bit for bit.
    XCTAssertEqual(
      exact.agreement,
      try model.fifthOrderAgreements([coordinates.point(z)], packedCoefficients: [coordinates.pack(z)])[0])
    let value = reference.agreement(z)
    XCTAssertEqual(exact.agreement, value, accuracy: 1e-5 * value)
    XCTAssertEqual(exact.agreement64, value, accuracy: 1e-5 * value)
    let gradient = coordinates.gradient(row: exact.gradient)
    var central = Array(repeating: 0.0, count: 28)
    for i in 0..<28 {
      let h = 1e-5
      var plus = z
      var minus = z
      plus[i] += h
      minus[i] -= h
      central[i] = (reference.agreement(plus) - reference.agreement(minus)) / (2 * h)
    }
    let scale = central.map(abs).max()!
    XCTAssertGreaterThan(scale, 0)
    var worst = 0.0
    for i in 0..<28 {
      worst = max(worst, abs(gradient[i] - central[i]) / scale)
      XCTAssertEqual(gradient[i], central[i], accuracy: 1e-4 * scale, "coordinate \(i)")
    }
    print("fifth-order gradient: worst |exact - central| / max|central| = \(worst)")

    // The value-only pass with the retained gradient pass equals the joint evaluation bit for bit.
    let points = [coordinates.point(z)]
    let packed = [coordinates.pack(z)]
    let valueOnly = try model.fifthOrderAgreementsAndGradients(
      points, packedCoefficients: packed, gradients: false)[0]
    XCTAssertEqual(valueOnly.agreement, exact.agreement)
    XCTAssertTrue(valueOnly.gradient.isEmpty)
    let retained = try model.fifthOrderRetainedGradients(points, packedCoefficients: packed)[0]
    XCTAssertEqual(retained.agreement, exact.agreement)
    XCTAssertEqual(retained.gradient, exact.gradient)
    // A BF subsample (steering only): every 3rd BF pixel, unscaled; its value and gradient pass, value-only with the
    // retained gradient pass, and the joint call agree.
    let bf = try model.fifthOrderAgreementsAndGradients(
      points, packedCoefficients: packed, kStride: 3)[0]
    let bfValue = reference.agreement(z, kStride: 3)
    XCTAssertEqual(bf.agreement, bfValue, accuracy: 1e-5 * bfValue)
    XCTAssertEqual(bf.agreement64, bfValue, accuracy: 1e-5 * bfValue)
    let bfValueOnly = try model.fifthOrderAgreementsAndGradients(
      points, packedCoefficients: packed, kStride: 3, gradients: false)[0]
    XCTAssertEqual(bfValueOnly.agreement, bf.agreement)
    let bfRetained = try model.fifthOrderRetainedGradients(
      points, packedCoefficients: packed, kStride: 3)[0]
    XCTAssertEqual(bfRetained.gradient, bf.gradient)
    let bfGradient = coordinates.gradient(row: bf.gradient)
    var worstBF = 0.0
    for i in 0..<28 {
      let h = 1e-5
      var plus = z
      var minus = z
      plus[i] += h
      minus[i] -= h
      let expected =
        (reference.agreement(plus, kStride: 3) - reference.agreement(minus, kStride: 3)) / (2 * h)
      worstBF = max(worstBF, abs(bfGradient[i] - expected) / scale)
      XCTAssertEqual(bfGradient[i], expected, accuracy: 1e-4 * scale, "BF-subsampled coordinate \(i)")
    }
    print("fifth-order gradient, every 3rd BF pixel: worst |exact - central| / max|central| = \(worstBF)")
  }

  static func model(_ device: MTLDevice) throws -> MetalSSBJointModel {
    let scene = try SSBModelReconstructionTests.scene()
    let input = try SSBCoherentReconstruction.buffer(device, scene.counts)
    let engine = try MetalSSBEngine(device: device, geometry: scene.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    return try engine.jointModel(scanDetectorRotationDegrees: 17.25)
  }

  /// From the same low-order fit, the exact-gradient search reaches at least the agreement of the
  /// finite-difference Newton reference with far fewer objective passes.
  func testExactGradientSearchMatchesFiniteDifferenceNewton() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try Self.model(device)
    let primary = SSBModelPoint(
      c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -0.4,
      sampleTiltMrad: [2.5, -4], depthSpreadAngstroms: 140)
    let newton = try SSBModelFit.refineFifthOrder(
      primary, model: model, search: .finiteDifferenceNewton)
    for search in [SSBModelFit.HigherOrderSearch.exactGradient, .exactGradientCoarseToFine] {
      let gradient = try SSBModelFit.refineFifthOrder(primary, model: model, search: search)
      XCTAssertEqual(gradient.normalization, newton.normalization)
      XCTAssertGreaterThanOrEqual(gradient.agreement64, newton.agreement64 * (1 - 1e-7))
      XCTAssertLessThan(gradient.result.evaluations, newton.result.executedObjectives / 10)
      XCTAssertTrue(["converged", "trust radius exhausted"].contains(gradient.result.stop))
      for (bound, value) in zip(SSBFifthOrderSearch.bounds, gradient.result.coordinates) {
        XCTAssertTrue(bound.contains(value))
      }
      print(
        "fifth-order search \(search): Newton \(newton.agreement64) in \(newton.result.executedObjectives) objectives, gradient \(gradient.agreement64) in \(gradient.result.evaluations) objectives, \(gradient.result.gradientEvaluations ?? 0) gradients"
      )
    }
  }

  /// Refining a low-order fit gives exactly the fifth-order fit that `findAberrations` computes in one call.
  func testRefiningALowOrderFitEqualsTheFullFit() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let scene = try SSBModelReconstructionTests.scene()
    let input = try SSBCoherentReconstruction.buffer(device, scene.counts)
    let engine = try MetalSSBEngine(device: device, geometry: scene.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    func fit(_ order: Int) throws -> (SSBBranchedFit, MetalSSBJointModel) {
      let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
      let result = try SSBModelFit.findAberrations(
        model: model, rotationDegrees: 17.25, darkField: nil,
        scanSampling: (0.5, 0.5), options: SSBFitOptions(trials: 40, maximumOrder: order),
        checkRotation: false)
      return (result, model)
    }
    let (full, _) = try fit(5)
    let (low, model) = try fit(1)
    XCTAssertNil(low.refinedFit)
    let refined = try SSBModelFit.refineToFifthOrder(
      low, model: model, options: SSBFitOptions(trials: 40, maximumOrder: 5))
    XCTAssertEqual(refined.fit.point, full.fit.point)
    XCTAssertEqual(refined.fit.sampleAgreement, full.fit.sampleAgreement)
    XCTAssertEqual(refined.fits[refined.kept].point, low.fits[low.kept].point)
  }
}
