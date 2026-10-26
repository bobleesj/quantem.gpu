import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

/// The higher-order box (`SSBFitOptions.higherOrderLimit`) and the opt-in guard of a fifth-order fit
/// (`SSBFitOptions.computeHigherOrderGuard`, `SSBHigherOrderGuard`, Python `higher_order_fit.assess`).
final class SSBHigherOrderLimitTests: XCTestCase {
  // MARK: Options

  func testLimitDefaultsToWideAndRefusesOtherValues() throws {
    XCTAssertEqual(SSBFitOptions.defaultHigherOrderLimit, SSBFitOptions.wideHigherOrderLimit)
    XCTAssertEqual(SSBFitOptions().higherOrderLimit, 20)
    XCTAssertTrue(SSBFitOptions().checkHigherOrderConsistency)
    XCTAssertEqual(SSBFitOptions.higherOrderLimits, [2, 20])
    XCTAssertNoThrow(try SSBFitOptions(higherOrderLimit: 20).validate())
    for limit in [0, 3, 14, 19.5, -2, .infinity, .nan] {
      XCTAssertThrowsError(try SSBFitOptions(higherOrderLimit: limit).validate(), "limit \(limit)")
      { error in
        XCTAssertTrue(
          error.localizedDescription.contains("higher-order limit of 2 (standard) or 20 (wide)"),
          error.localizedDescription)
      }
      // Also for a primary-only fit: the setting is checked wherever it is given.
      XCTAssertThrowsError(try SSBFitOptions(maximumOrder: 1, higherOrderLimit: limit).validate())
    }
    // Saved options record the limit; options saved before it decode as the standard box, unchecked.
    let wide = SSBFitOptions(higherOrderLimit: 20)
    let encoded = try JSONEncoder().encode(wide)
    XCTAssertEqual(try JSONDecoder().decode(SSBFitOptions.self, from: encoded), wide)
    var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    XCTAssertEqual(legacy["higherOrderLimit"] as? Double, 20)
    legacy.removeValue(forKey: "higherOrderLimit")
    let restored = try JSONDecoder().decode(
      SSBFitOptions.self, from: JSONSerialization.data(withJSONObject: legacy))
    XCTAssertEqual(restored.higherOrderLimit, 2)
    legacy.removeValue(forKey: "checkHigherOrderConsistency")
    XCTAssertFalse(
      try JSONDecoder().decode(
        SSBFitOptions.self, from: JSONSerialization.data(withJSONObject: legacy)
      )
      .checkHigherOrderConsistency)
    // The guard is opt-in; options saved without the key do not compute it.
    XCTAssertFalse(SSBFitOptions().computeHigherOrderGuard)
    let guarded = SSBFitOptions(computeHigherOrderGuard: true)
    XCTAssertEqual(
      try JSONDecoder().decode(SSBFitOptions.self, from: JSONEncoder().encode(guarded)), guarded)
    XCTAssertFalse(restored.computeHigherOrderGuard)
  }

  func testBoundsKeepTheLowOrderBoxAndScaleTheHigherOrders() {
    for limit in SSBFitOptions.higherOrderLimits {
      let bounds = SSBModelFit.fifthOrderBounds(higherOrderLimit: limit)
      XCTAssertEqual(bounds.count, 28)
      XCTAssertEqual(Array(bounds.prefix(6)), [-3...3, -2...2, -2...2, -5...5, -5...5, 0.2...6])
      XCTAssertTrue(bounds.dropFirst(6).allSatisfy { $0 == -limit...limit })
    }
  }

  /// The aperture-phase disagreement of the validation's `common.py` (`rms_piston_free(chi(a) - chi(b))`) on known
  /// differences: C10 alone gives k sqrt(1/12) with k = (2 pi / lambda) alpha^2 / 2 * 100 A, a C21 component
  /// r^3 cos(phi) sqrt(1/8), a round C30 r^4 sqrt(4/45); the grid samples the disk to about 1%.
  func testApertureDisagreementAndRingSplit() {
    let wavelength = 0.019687
    let semiangle = 0.03
    let zero = [Double](repeating: 0, count: 28)
    func shifted(_ i: Int, _ value: Double) -> [Double] {
      var z = zero
      z[i] = value
      return z
    }
    let k = 2 * Double.pi / wavelength * semiangle * semiangle / 2 * 100
    let c10 = SSBHigherOrderConsistency.apertureDisagreement(
      shifted(0, 0.1), zero, wavelengthAngstroms: wavelength, semiangleRadians: semiangle)
    XCTAssertEqual(c10, 0.1 * k * (1.0 / 12).squareRoot(), accuracy: 0.01 * c10)
    let c21 = SSBHigherOrderConsistency.apertureDisagreement(
      shifted(6, 1), zero, wavelengthAngstroms: wavelength, semiangleRadians: semiangle)
    XCTAssertEqual(c21, (1.0 / 8).squareRoot(), accuracy: 0.01 * c21)
    let c30 = SSBHigherOrderConsistency.apertureDisagreement(
      shifted(10, 1), zero, wavelengthAngstroms: wavelength, semiangleRadians: semiangle)
    XCTAssertEqual(c30, (1.0 / 5 - 1.0 / 9).squareRoot(), accuracy: 0.01 * c30)
    // Tilt and depth do not enter; the difference is symmetric.
    XCTAssertEqual(
      SSBHigherOrderConsistency.apertureDisagreement(
        shifted(3, 2), shifted(5, 3), wavelengthAngstroms: wavelength, semiangleRadians: semiangle),
      0)
    XCTAssertEqual(
      SSBHigherOrderConsistency.apertureDisagreement(
        zero, shifted(6, 1), wavelengthAngstroms: wavelength, semiangleRadians: semiangle), c21)
    // Rings of 0.05 1/A alternate.
    XCTAssertEqual(SSBHigherOrderConsistency.ringParity(SIMD2(0.21, 0)), 0)
    XCTAssertEqual(SSBHigherOrderConsistency.ringParity(SIMD2(0.18, 0.18)), 1)
    XCTAssertEqual(SSBHigherOrderConsistency.ringParity(SIMD2(-0.3, 0.001)), 0)
    XCTAssertEqual(SSBHigherOrderConsistency.ringParity(SIMD2(0, -0.36)), 1)
  }

  func testRefineRefusesAnInvalidLimit() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try SSBFifthOrderGradientTests.model(device)
    let low = try SSBModelFit.findAberrations(
      model: model, rotationDegrees: 17.25, darkField: nil, scanSampling: (0.5, 0.5),
      options: SSBFitOptions(trials: 20, maximumOrder: 1), checkRotation: false)
    XCTAssertThrowsError(
      try SSBModelFit.refineToFifthOrder(
        low, model: model, options: SSBFitOptions(trials: 20, higherOrderLimit: 10))
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("not 10.0"), error.localizedDescription)
    }
    XCTAssertThrowsError(
      try SSBModelFit.fit(model: model, options: SSBFitOptions(trials: 20, higherOrderLimit: 4)))
  }

  // MARK: Guard arithmetic

  /// The guard's arithmetic against hand-computed values: a diagonal Hessian gives variances 2 sigma^2 / |H_ii|.
  func testAssessComputesCoherenceAndEdgePhaseErrors() {
    let band = SSBBandPower(
      power: 1_000, bandWeight: 10, noiseVariance: 2, brightFieldPixelCount: 30)
    // signal = 1000 - 10 * 2 * 30 = 400; coherence = (A - 20) / 400.
    var hessian = (0..<28).map { i in (0..<28).map { j in i == j ? -4.0 * Double(i + 1) : 0 } }
    let wavelength = 0.019687
    let semiangle = 0.03
    let check = SSBHigherOrderGuard.assess(
      agreement: 380, band: band, hessian: hessian, wavelengthAngstroms: wavelength,
      semiangleRadians: semiangle)
    XCTAssertEqual(check.coherence, 0.9, accuracy: 1e-15)
    XCTAssertEqual(
      check.terms.map(\.name),
      [
        "C21", "C23", "C30", "C32", "C34", "C41", "C43", "C45", "C50", "C52", "C54", "C56",
      ])
    var column = 6
    for term in check.terms {
      let width = term.symmetry == 0 ? 1 : 2
      // var_i = 2 sigma^2 / (4 (i + 1)) = 1 / (i + 1)
      let expected = (column..<(column + width)).reduce(0.0) { $0 + 1 / Double($1 + 1) }
        .squareRoot()
      XCTAssertEqual(term.edgePhaseErrorRadians!, expected, accuracy: 1e-12 * expected, term.name)
      let n = Double(term.order)
      let nanometers =
        expected * (n + 1) * wavelength / (2 * Double.pi * pow(semiangle, n + 1)) / 10
      XCTAssertEqual(
        term.standardErrorNanometers!, nanometers, accuracy: 1e-12 * nanometers, term.name)
      // The definition (2 pi / lambda) sigma_C alpha^(n+1) / (n+1) returns the edge phase.
      let edge =
        2 * Double.pi / wavelength * term.standardErrorNanometers! * 10 * pow(semiangle, n + 1)
        / (n + 1)
      XCTAssertEqual(edge, expected, accuracy: 1e-12 * expected)
      column += width
    }
    XCTAssertFalse(check.trusted)
    XCTAssertEqual(check.reasons.count, 1)
    XCTAssertTrue(
      check.reasons[0].hasPrefix("C21 is uncertain by 0.52 rad at the aperture edge (above 0.1)"))
    XCTAssertEqual(check.largestEdgePhaseErrorRadians, check.terms[0].edgePhaseErrorRadians)

    // Below the coherence minimum, and a curvature that is not a maximum: both reasons, errors not determined.
    hessian[10][10] = 1
    let failed = SSBHigherOrderGuard.assess(
      agreement: 300, band: band, hessian: hessian, wavelengthAngstroms: wavelength,
      semiangleRadians: semiangle)
    XCTAssertEqual(failed.coherence, 0.7, accuracy: 1e-15)
    XCTAssertTrue(
      failed.terms.allSatisfy {
        $0.edgePhaseErrorRadians == nil && $0.standardErrorNanometers == nil
      })
    XCTAssertEqual(failed.largestEdgePhaseErrorRadians, .infinity)
    XCTAssertEqual(
      failed.reasons,
      [
        "the model explains 0.70 of the bright-field signal (below 0.9): scattering it does not describe, as in a thick or strongly scattering crystal",
        "C21 is uncertain by inf rad at the aperture edge (above 0.1): too little signal for its precision",
      ])
    // The record survives JSON with undetermined errors.
    let decoded = try? JSONDecoder().decode(
      SSBHigherOrderGuard.self, from: JSONEncoder().encode(failed))
    XCTAssertEqual(decoded, failed)

    // Precise and coherent: trusted. No signal above the noise floors: coherence 0.
    let precise = (0..<28).map { i in (0..<28).map { j in i == j ? -1e6 : 0.0 } }
    XCTAssertTrue(
      SSBHigherOrderGuard.assess(
        agreement: 390, band: band, hessian: precise, wavelengthAngstroms: wavelength,
        semiangleRadians: semiangle
      ).trusted)
    let noise = SSBBandPower(
      power: 600, bandWeight: 10, noiseVariance: 2, brightFieldPixelCount: 30)
    XCTAssertEqual(
      SSBHigherOrderGuard.assess(
        agreement: 390, band: noise, hessian: precise, wavelengthAngstroms: wavelength,
        semiangleRadians: semiangle
      ).coherence, 0)
  }

  // MARK: Synthetic weak-phase scene

  /// A weak-phase object (0.05 rad rms, one slice) probed with coma, trefoil and a C30 of 5 um, whose phase at the
  /// aperture edge (3.2 rad) lies outside the standard box. The wide box recovers it, the standard box holds it at 2.
  static let truth: [(order: Int, symmetry: Int, nanometers: Double, degrees: Double)] = [
    (1, 0, -5, 0), (1, 2, 2, 30), (2, 1, 40, 60), (2, 3, 30, -20), (3, 0, 5000, 0),
  ]

  func testWideLimitReproducesTheFreeBoxAndTheGuardTrustsAWeakPhaseFit() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let scene = try WeakPhaseScene(aberrations: Self.truth, dose: 2e5, seed: 7)
    let engine = try MetalSSBEngine(device: device, geometry: scene.geometry)
    let input = try SSBCoherentReconstruction.buffer(device, scene.counts)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 0)

    // The band summary against the evidence and the integer counts.
    let band = try model.bandPowerSummary()
    let arrays = try model.objectiveArrays()
    var power = 0.0
    for (i, w) in arrays.weights.enumerated() {
      var sum = 0.0
      for k in 0..<band.brightFieldPixelCount {
        let g = arrays.evidence[k * arrays.weights.count + i]
        sum += Double(g.x) * Double(g.x) + Double(g.y) * Double(g.y)
      }
      power += Double(w) * sum
    }
    XCTAssertEqual(band.power, power, accuracy: 1e-12 * power)
    XCTAssertEqual(band.bandWeight, arrays.weights.reduce(0.0) { $0 + Double($1) })
    XCTAssertEqual(band.brightFieldPixelCount, scene.pixels.count)
    let counts =
      Double(scene.pixels.reduce(UInt64(0)) { $0 + scene.detectorSum[$1] })
      / Double(scene.pixels.count)
    XCTAssertEqual(band.noiseVariance, counts, accuracy: 1e-5 * counts)

    let low = try SSBModelFit.findAberrations(
      model: model, rotationDegrees: 0, darkField: nil,
      scanSampling: (WeakPhaseScene.step, WeakPhaseScene.step),
      options: SSBFitOptions(trials: 60, maximumOrder: 1), checkRotation: false)
    XCTAssertNil(low.fit.higherOrderGuard)
    // Off by default: the same refinement without the guard.
    let unguarded = try SSBModelFit.refineToFifthOrder(
      low, model: model, options: SSBFitOptions(trials: 60, higherOrderLimit: 20))
    XCTAssertNil(unguarded.fit.higherOrderGuard)
    XCTAssertNil(unguarded.fit.timings["higher_order_guard_s"])
    var fits: [Double: SSBFit] = [:]
    for limit in SSBFitOptions.higherOrderLimits {
      let refined = try SSBModelFit.refineToFifthOrder(
        low, model: model,
        options: SSBFitOptions(trials: 60, higherOrderLimit: limit, computeHigherOrderGuard: true))
      XCTAssertEqual(refined.fit.options.higherOrderLimit, limit)
      fits[limit] = refined.fit
    }
    let standard = try XCTUnwrap(fits[2])
    let wide = try XCTUnwrap(fits[20])
    // The standard box is not checked; the wide fit's ring halves agree on this weak-phase object, so it is shown.
    XCTAssertNil(standard.higherOrderConsistency)
    let consistency = try XCTUnwrap(wide.higherOrderConsistency)
    XCTAssertTrue(consistency.trusted)
    XCTAssertNil(consistency.reason)
    XCTAssertEqual(consistency.shownLimit, 20)
    XCTAssertEqual(wide.options.higherOrderLimit, 20)
    XCTAssertLessThan(consistency.halfBandRMSRadians, 0.5)
    XCTAssertEqual(consistency.halves.map(\.ringParity), [0, 1])
    XCTAssertEqual(consistency.halves.reduce(0) { $0 + $1.frequencies }, model.bandCount)
    XCTAssertEqual(consistency.checkedCoordinates, wide.higherOrderRefinement?.coordinates)
    XCTAssertNotNil(wide.timings["higher_order_consistency_s"])
    // Each half is the same refinement on its ring half of the band, from the same low-order fit.
    let parities = model.bandFrequencies.map { SSBHigherOrderConsistency.ringParity($0) }
    for half in consistency.halves {
      let alone = try model.withBandMask(parities.map { $0 == half.ringParity }) {
        try SSBModelFit.refineFifthOrder(
          low.fit.point, model: model, search: .exactGradientCoarseToFine,
          bounds: SSBModelFit.fifthOrderBounds(higherOrderLimit: 20))
      }
      XCTAssertEqual(alone.result.coordinates, half.coordinates)
    }
    XCTAssertEqual(
      consistency.halfBandRMSRadians,
      SSBHigherOrderConsistency.apertureDisagreement(
        consistency.halves[0].coordinates, consistency.halves[1].coordinates,
        wavelengthAngstroms: model.wavelengthAngstroms, semiangleRadians: model.semiangleRadians))

    // 20 is a free box here: no coordinate reaches it, and a box of 1000 gives the same search bit for bit.
    let start = low.fit.point
    let free = try SSBModelFit.refineFifthOrder(
      start, model: model, search: .exactGradientCoarseToFine,
      bounds: SSBModelFit.fifthOrderBounds(higherOrderLimit: 1000))
    let boxed = try SSBModelFit.refineFifthOrder(
      start, model: model, search: .exactGradientCoarseToFine,
      bounds: SSBModelFit.fifthOrderBounds(higherOrderLimit: 20))
    XCTAssertEqual(boxed.result.coordinates, free.result.coordinates)
    XCTAssertEqual(boxed.agreement64, free.agreement64)
    XCTAssertEqual(wide.higherOrderRefinement?.coordinates, free.result.coordinates)
    XCTAssertEqual(unguarded.fit.point, wide.point)
    XCTAssertTrue(free.result.coordinates.dropFirst(6).allSatisfy { abs($0) < 20 })
    // The standard box holds C30 (the 9th coordinate, z 3.2 rad) at its bound.
    let c30 = 6 + 4
    XCTAssertEqual(
      try XCTUnwrap(standard.higherOrderRefinement?.coordinates[c30]), 2, accuracy: 1e-12)
    XCTAssertGreaterThan(free.result.coordinates[c30], 2.5)

    func term(_ fit: SSBFit, _ order: Int, _ symmetry: Int) -> SSBModelHigherOrder? {
      fit.point.higherOrder.first { $0.order == order && $0.symmetry == symmetry }
    }
    // The wide box recovers the probe.
    XCTAssertEqual(wide.point.c10Nanometers, -5, accuracy: 0.3)
    XCTAssertEqual(wide.point.c12Nanometers, 2, accuracy: 0.3)
    XCTAssertEqual(try XCTUnwrap(term(wide, 2, 1)).magnitudeAngstroms / 10, 40, accuracy: 4)
    XCTAssertEqual(try XCTUnwrap(term(wide, 2, 3)).magnitudeAngstroms / 10, 30, accuracy: 3)
    XCTAssertEqual(try XCTUnwrap(term(wide, 3, 0)).magnitudeAngstroms / 10, 5000, accuracy: 500)
    XCTAssertGreaterThan(wide.sampleAgreement!, standard.sampleAgreement!)

    // The guard: a weak-phase object is the SSB model, so the fit explains its signal and is precise.
    let check = try XCTUnwrap(wide.higherOrderGuard)
    XCTAssertTrue(check.trusted, "\(check.reasons)")
    XCTAssertGreaterThan(check.coherence, 0.95)
    XCTAssertLessThanOrEqual(check.coherence, 1.02)
    XCTAssertEqual(check.terms.count, 12)
    XCTAssertTrue(check.terms.allSatisfy { ($0.edgePhaseErrorRadians ?? .infinity) < 0.1 })
    XCTAssertEqual(check.band, band)
    XCTAssertNotNil(wide.timings["higher_order_guard_s"])
    // Its coherence is the definition on the fitted agreement, and its precision the covariance of the diagnostic
    // central-difference Hessian, inverted independently.
    let z = wide.higherOrderRefinement!.coordinates
    let agreement = try SSBModelFit.fifthOrderAgreement64(model: model, z: z)
    XCTAssertEqual(check.agreement, agreement, accuracy: 1e-12 * agreement)
    let floor = band.bandWeight * band.noiseVariance
    XCTAssertEqual(
      check.coherence,
      (agreement - floor) / (band.power - floor * Double(band.brightFieldPixelCount)),
      accuracy: 1e-12)
    let hessian = try SSBModelFit.fifthOrderHessian(model: model, z: z)
    let covariance = try Self.inverse(hessian.map { $0.map { -$0 } }).map {
      $0.map { 2 * band.noiseVariance * $0 }
    }
    var column = 6
    for term in check.terms {
      let width = term.symmetry == 0 ? 1 : 2
      let expected = (column..<(column + width)).reduce(0.0) { $0 + covariance[$1][$1] }
        .squareRoot()
      XCTAssertEqual(term.edgePhaseErrorRadians!, expected, accuracy: 0.02 * expected, term.name)
      column += width
    }
    print("weak-phase halves disagree by \(consistency.halfBandRMSRadians) rad")
    print(
      "weak-phase guard: coherence standard \(standard.higherOrderGuard!.coherence), wide \(check.coherence); "
        + "largest edge-phase error \(check.largestEdgePhaseErrorRadians) rad; C30 wide "
        + "\(term(wide, 3, 0)!.magnitudeAngstroms / 10) nm, standard \(term(standard, 3, 0)!.magnitudeAngstroms / 10) nm"
    )
  }

  /// The lattice of `SSBModelReconstructionTests.scene` is not a probe-object measurement: the model explains little
  /// of its signal (the opt-in coherence guard) and the ring halves of a wide fit disagree (the default check).
  func testGuardsRejectASceneTheModelDoesNotDescribe() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try SSBFifthOrderGradientTests.model(device)
    var primary: SSBModelPoint?
    let fitted = try SSBModelFit.findAberrations(
      model: model, rotationDegrees: 17.25, darkField: nil, scanSampling: (0.5, 0.5),
      options: SSBFitOptions(trials: 40, computeHigherOrderGuard: true), checkRotation: false,
      onInitialFit: { primary = $0.point })
    let check = try XCTUnwrap(fitted.fit.higherOrderGuard)
    XCTAssertFalse(check.trusted)
    XCTAssertLessThan(check.coherence, 0.9)
    XCTAssertTrue(check.reasons[0].hasPrefix("the model explains"))
    // Its ring halves disagree (8.2 rad), so the fit falls back to the standard box, flagged, and keeps the checked
    // wide fit in the record.
    let consistency = try XCTUnwrap(fitted.fit.higherOrderConsistency)
    print("lattice halves disagree by \(consistency.halfBandRMSRadians) rad")
    XCTAssertFalse(consistency.trusted)
    XCTAssertGreaterThan(consistency.halfBandRMSRadians, 0.5)
    XCTAssertEqual(consistency.shownLimit, 2)
    XCTAssertEqual(consistency.limit, 20)
    XCTAssertEqual(fitted.fit.options.higherOrderLimit, 2)
    XCTAssertEqual(
      consistency.reason,
      String(format: "halves disagree by %.2f rad", consistency.halfBandRMSRadians))
    XCTAssertNotNil(fitted.fit.timings["higher_order_fallback_s"])
    let low = try XCTUnwrap(primary)
    for (limit, expected) in [
      (2.0, fitted.fit.higherOrderRefinement?.coordinates), (20, consistency.checkedCoordinates),
    ] {
      let direct = try SSBModelFit.refineFifthOrder(
        low, model: model, search: .exactGradientCoarseToFine,
        bounds: SSBModelFit.fifthOrderBounds(higherOrderLimit: limit))
      XCTAssertEqual(direct.result.coordinates, expected, "limit \(limit)")
    }
    // The record carries the names Python uses.
    let record = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(consistency)) as? [String: Any])
    XCTAssertEqual(record["half_band_rms_rad"] as? Double, consistency.halfBandRMSRadians)
    XCTAssertEqual(record["trusted"] as? Bool, false)
    XCTAssertEqual(record["shown_limit"] as? Double, 2)
    XCTAssertEqual(
      try JSONDecoder().decode(
        SSBHigherOrderConsistency.self, from: JSONEncoder().encode(consistency)), consistency)
  }

  /// Gauss-Jordan inverse with partial pivoting.
  static func inverse(_ matrix: [[Double]]) throws -> [[Double]] {
    let n = matrix.count
    var a = matrix.enumerated().map { i, row in row + (0..<n).map { $0 == i ? 1.0 : 0.0 } }
    for column in 0..<n {
      let pivot = (column..<n).max { abs(a[$0][column]) < abs(a[$1][column]) }!
      guard abs(a[pivot][column]) > 0 else { throw XCTSkip("singular") }
      a.swapAt(column, pivot)
      let p = a[column][column]
      a[column] = a[column].map { $0 / p }
      for row in 0..<n where row != column {
        let f = a[row][column]
        if f != 0 { a[row] = zip(a[row], a[column]).map { $0 - f * $1 } }
      }
    }
    return a.map { Array($0[n...]) }
  }
}

/// Python `tests/hardware/thin_object.py` on a 32 x 32 probe window at 0.25 A, tiled to the 128 x 128 scan the native
/// engine accepts, with Poisson counts: one weak-phase slice (a smooth random phase, 0.05 rad rms) under a probe with
/// the given Krivanek coefficients (nm, degrees), 300 kV, 30 mrad, detector = the full 32 x 32 diffraction plane.
struct WeakPhaseScene {
  static let side = 32
  static let tile = 4
  static let step = 0.25
  let calibration: MetalSSBCalibration
  let geometry: MetalSSBGeometry
  let pixels: [Int]
  let detectorSum: [UInt64]
  let counts: [UInt16]

  init(
    aberrations: [(order: Int, symmetry: Int, nanometers: Double, degrees: Double)], dose: Double,
    seed: UInt64
  ) throws {
    let n = Self.side
    let scan = n * Self.tile
    let wavelength = MetalSSBCalibration.wavelengthAngstroms(voltageKV: 300)
    let k = (0..<n).map { Double($0 < n / 2 ? $0 : $0 - n) / (Double(n) * Self.step) }
    var random = SplitMix(seed: seed)
    // The probe in Fourier space: a soft 30 mrad aperture and exp(-i chi).
    var probe = Complex2D(n)
    for r in 0..<n {
      for c in 0..<n {
        let alpha = wavelength * hypot(k[r], k[c])
        let phi = atan2(k[c], k[r])
        let aperture = min(1, max(0, (30e-3 - alpha) / (wavelength * k[1]) + 0.5))
        var chi = 0.0
        for term in aberrations {
          chi +=
            term.nanometers * 10 * pow(alpha, Double(term.order + 1)) / Double(term.order + 1)
            * cos(Double(term.symmetry) * (phi - term.degrees * .pi / 180))
        }
        chi *= 2 * Double.pi / wavelength
        probe.re[r * n + c] = aperture * cos(chi)
        probe.im[r * n + c] = -aperture * sin(chi)
      }
    }
    probe.fft(inverse: true)
    // A smooth random phase of 0.05 rad rms.
    var phase = Complex2D(n)
    for i in 0..<(n * n) { phase.re[i] = random.normal() }
    phase.fft(inverse: false)
    for r in 0..<n {
      for c in 0..<n {
        let s = exp(-0.5 * (k[r] * k[r] + k[c] * k[c]) / (0.8 * 0.8))
        phase.re[r * n + c] *= s
        phase.im[r * n + c] *= s
      }
    }
    phase.fft(inverse: true)
    let mean = phase.re.reduce(0, +) / Double(n * n)
    let rms = (phase.re.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(n * n)).squareRoot()
    let object = phase.re.map { 0.05 * $0 / rms }
    // One diffraction pattern per position of the 32 x 32 period, k = 0 at detector pixel (16, 16).
    var patterns = [[Double]](repeating: [], count: n * n)
    for r0 in 0..<n {
      for c0 in 0..<n {
        var wave = Complex2D(n)
        for r in 0..<n {
          for c in 0..<n {
            let p = ((r - r0 + n) % n) * n + (c - c0 + n) % n
            let t = object[r * n + c]
            wave.re[r * n + c] = probe.re[p] * cos(t) - probe.im[p] * sin(t)
            wave.im[r * n + c] = probe.re[p] * sin(t) + probe.im[p] * cos(t)
          }
        }
        wave.fft(inverse: false)
        var intensity = [Double](repeating: 0, count: n * n)
        for r in 0..<n {
          for c in 0..<n {
            let i = r * n + c
            intensity[((r + n / 2) % n) * n + (c + n / 2) % n] =
              wave.re[i] * wave.re[i] + wave.im[i] * wave.im[i]
          }
        }
        patterns[r0 * n + c0] = intensity
      }
    }
    let total = patterns[0].reduce(0, +)
    let detector = wavelength * k[1] * 1e3
    let radius = 30 / detector
    calibration = MetalSSBCalibration(
      voltageKV: 300, semiangleMrad: 30, scanSamplingRowAngstroms: Self.step,
      scanSamplingColumnAngstroms: Self.step,
      detectorSamplingRowMrad: detector, detectorSamplingColumnMrad: detector,
      brightfieldCenterRowPixels: Double(n / 2), brightfieldCenterColumnPixels: Double(n / 2),
      brightfieldRadiusPixels: radius)
    // Poisson counts of every scan position of the tiled scan, independent draws, detector-pixel-major. Pixels
    // outside the calibrated bright field are not read by the fit and stay 0.
    let bright = (0..<(n * n)).filter {
      let dr = Double($0 / n - n / 2)
      let dc = Double($0 % n - n / 2)
      return dr * dr + dc * dc <= radius * radius
    }
    var planes = [[UInt16]](repeating: [UInt16](repeating: 0, count: scan * scan), count: n * n)
    var sums = [UInt64](repeating: 0, count: n * n)
    for r in 0..<scan {
      for c in 0..<scan {
        let pattern = patterns[(r % n) * n + c % n]
        for pixel in bright {
          let value = random.poisson(dose * pattern[pixel] / total)
          planes[pixel][r * scan + c] = UInt16(min(value, Int(UInt16.max)))
          sums[pixel] += UInt64(planes[pixel][r * scan + c])
        }
      }
    }
    let setup = try calibration.geometry(
      detectorRows: n, detectorColumns: n, detectorSum: sums, scanRows: scan, scanColumns: scan)
    geometry = setup.geometry
    pixels = setup.pixels
    detectorSum = sums
    counts = setup.pixels.flatMap { planes[$0] }
  }
}

/// A square complex array with an in-place radix-2 2-D FFT (NumPy's sign and 1/N^2 inverse scaling).
struct Complex2D {
  let n: Int
  var re: [Double]
  var im: [Double]

  init(_ n: Int) {
    self.n = n
    re = [Double](repeating: 0, count: n * n)
    im = re
  }

  mutating func fft(inverse: Bool) {
    var rowRe = [Double](repeating: 0, count: n)
    var rowIm = rowRe
    for pass in 0..<2 {
      for line in 0..<n {
        for i in 0..<n {
          let index = pass == 0 ? line * n + i : i * n + line
          rowRe[i] = re[index]
          rowIm[i] = im[index]
        }
        Self.transform(&rowRe, &rowIm, inverse: inverse)
        for i in 0..<n {
          let index = pass == 0 ? line * n + i : i * n + line
          re[index] = rowRe[i]
          im[index] = rowIm[i]
        }
      }
    }
    if inverse {
      let scale = 1 / Double(n * n)
      for i in 0..<(n * n) {
        re[i] *= scale
        im[i] *= scale
      }
    }
  }

  static func transform(_ re: inout [Double], _ im: inout [Double], inverse: Bool) {
    let n = re.count
    var j = 0
    for i in 1..<n {
      var bit = n >> 1
      while j & bit != 0 {
        j ^= bit
        bit >>= 1
      }
      j |= bit
      if i < j {
        re.swapAt(i, j)
        im.swapAt(i, j)
      }
    }
    var length = 2
    while length <= n {
      let angle = (inverse ? 2 : -2) * Double.pi / Double(length)
      for start in stride(from: 0, to: n, by: length) {
        for m in 0..<(length / 2) {
          let wr = cos(angle * Double(m))
          let wi = sin(angle * Double(m))
          let a = start + m
          let b = a + length / 2
          let tr = re[b] * wr - im[b] * wi
          let ti = re[b] * wi + im[b] * wr
          re[b] = re[a] - tr
          im[b] = im[a] - ti
          re[a] += tr
          im[a] += ti
        }
      }
      length <<= 1
    }
  }
}

/// A seeded generator (SplitMix64) with normal and Poisson draws, so the scene is identical on every run.
struct SplitMix {
  var state: UInt64
  init(seed: UInt64) { state = seed }

  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }

  /// Uniform in (0, 1).
  mutating func uniform() -> Double { (Double(next() >> 11) + 0.5) / Double(1 << 53) }

  mutating func normal() -> Double {
    (-2 * log(uniform())).squareRoot() * cos(2 * Double.pi * uniform())
  }

  /// Knuth's product method below a mean of 30, the rounded normal approximation above it.
  mutating func poisson(_ mean: Double) -> Int {
    guard mean > 0 else { return 0 }
    if mean >= 30 { return max(0, Int((mean + mean.squareRoot() * normal()).rounded())) }
    let limit = exp(-mean)
    var product = uniform()
    var count = 0
    while product > limit {
      product *= uniform()
      count += 1
    }
    return count
  }
}
