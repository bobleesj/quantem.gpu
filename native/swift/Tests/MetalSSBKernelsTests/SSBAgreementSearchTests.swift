import CryptoKit
import Foundation
import XCTest

@testable import MetalSSBKernels

/// The Swift port of the resident TPE + Newton fit against the Python MPS implementation
/// (`ssb._resident_search.fit_resident` and `fit_standard_resident`, seed 42, 200 trials) on a synthetic
/// float32 objective that both sides evaluate from the same float32 trial rows. The digests pin every trial
/// row Python evaluated (search and Newton), in order; the points are Python's fits. Python keeps the Newton
/// state in a three-float expansion and this port in double precision (the CUDA build's arithmetic), so the
/// fitted points agree to about 1e-8 while every float32 trial row is identical.
final class SSBAgreementSearchTests: XCTestCase {
  /// A smooth float32 objective with its maximum at C10 37.25 A, C12 (cos, sin) (30, -12) A, tilt (3.5, -1.25)
  /// mrad and depth 151 A, written as the Python generator writes it.
  static func score(_ row: [Float]) -> Double {
    let a = row[1] * row[2] - 30
    let b = row[1] * row[3] + 12
    let d10 = row[0] - 37.25
    let dr = row[5] * 1000 - 3.5
    let dc = row[6] * 1000 + 1.25
    let dt = row[4] - 151
    var s: Float = 1.0e6 - d10 * d10 * 0.25 - a * a - b * b
    s = s - dr * dr * 400 - dc * dc * 400
    s = s - dt * dt * 0.01
    return Double(s)
  }

  final class Recorder {
    var rows: [[Float]] = []
    func evaluate(_ points: [[Double]]) -> [Double] {
      points.map { point in
        let row = SSBModelPoint(
          c10Angstroms: point[0], c12Angstroms: point[1], phi12Radians: point[2],
          sampleTiltMrad: [point[3], point[4]], depthSpreadAngstroms: point[5]
        ).trialRow
        rows.append(row)
        return SSBAgreementSearchTests.score(row)
      }
    }
    var digest: String {
      var bytes = Data()
      for row in rows { row.withUnsafeBytes { bytes.append(contentsOf: $0) } }
      return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
  }

  /// `scripts/ssb_agreement_gate.py` runs the current Python search and passes its records here, so a change on
  /// either side fails without editing the pins below.
  struct Reference: Decodable {
    struct Fit: Decodable {
      let point: [Double]
      let agreement: Double
      let rows: Int
      let sha256: String
    }
    let tilt: Fit
    let standard: Fit
  }

  func testCurrentPythonSearchWhenGiven() throws {
    guard let path = ProcessInfo.processInfo.environment["SSB_AGREEMENT_REFERENCE"] else {
      throw XCTSkip(
        "scripts/ssb_agreement_gate.py sets SSB_AGREEMENT_REFERENCE to the current Python records.")
    }
    let reference = try JSONDecoder().decode(
      Reference.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    for (name, expected) in [("tilt", reference.tilt), ("standard", reference.standard)] {
      let recorder = Recorder()
      let fit =
        name == "tilt"
        ? try SSBAgreementSearch.fitTilt(recorder.evaluate)
        : try SSBAgreementSearch.fitStandard(recorder.evaluate)
      XCTAssertEqual(recorder.rows.count, expected.rows, name)
      XCTAssertEqual(
        recorder.digest, expected.sha256, "\(name): a trial row differs from the Python search")
      for (actual, value) in zip(fit.point, expected.point) {
        XCTAssertEqual(actual, value, accuracy: 1e-6, name)
      }
      XCTAssertEqual(fit.agreement, expected.agreement, name)
    }
  }

  func testTiltFitMatchesPythonTrialByTrial() throws {
    let recorder = Recorder()
    let fit = try SSBAgreementSearch.fitTilt(recorder.evaluate)
    XCTAssertEqual(recorder.rows.count, 573)
    XCTAssertEqual(
      recorder.digest, "d78c322600a4aa4da277fdb39c3c9eee456bca3b547de85ef05fdb5f0a26312c",
      "a trial row differs from the Python resident search")
    let expected = [
      37.267501144443685, 32.30775752459612, -0.19035700659647992, 3.49993612504556,
      -1.2500771160326833, 150.91333765686667,
    ]
    for (actual, value) in zip(fit.point, expected) {
      XCTAssertEqual(actual, value, accuracy: 1e-6)
    }
    XCTAssertEqual(fit.agreement, 1.0e6)
    XCTAssertEqual(fit.newtonStarts.map(\.evaluations), [125, 41, 41, 97, 69])
    XCTAssertEqual(fit.newtonStarts.map(\.converged), [true, false, false, true, false])
  }

  func testStandardFitMatchesPythonTrialByTrial() throws {
    let recorder = Recorder()
    let fit = try SSBAgreementSearch.fitStandard(recorder.evaluate)
    XCTAssertEqual(recorder.rows.count, 244)
    XCTAssertEqual(
      recorder.digest, "a645708f9348b0c2c60eb0609f91a2c377bba8c1af084a748cd85511e8b12ca6",
      "a trial row differs from the Python resident search")
    let expected = [37.22132425139795, 32.306412714345484, -0.19023146368751864]
    for (actual, value) in zip(fit.point, expected) {
      XCTAssertEqual(actual, value, accuracy: 1e-6)
    }
    XCTAssertEqual(Array(fit.point[3...]), [0, 0, 0])
    XCTAssertEqual(fit.agreement, 994247)
  }

  /// The exact-derivative Newton on the same objective: the same 200 search trials, then a few centers to the
  /// maximum, which it reaches more closely than the stencil Newton.
  func testExactNewtonReachesTheMaximumWithTheSameSearch() throws {
    let stencil = Recorder()
    let reference = try SSBAgreementSearch.fitTilt(stencil.evaluate)
    let recorder = Recorder()
    var centers = 0
    // The objective in z in double precision: an exact maximum and analytic derivatives.
    let target = [37.25, 30, -12, 3.5, -1.25, 151]
    let curvature = [0.25, 1, 1, 400, 400, 0.01]
    let fit = try SSBAgreementSearch.fitTilt(
      recorder.evaluate,
      derivatives: { rows in
        centers += rows.count
        return rows.map { z in
          var value = 1.0e6
          var gradient = [Double](repeating: 0, count: 6)
          var hessian = [[Double]](repeating: [Double](repeating: 0, count: 6), count: 6)
          for d in 0..<6 {
            let x = d == 5 ? abs(z[d]) : z[d]
            value -= curvature[d] * (x - target[d]) * (x - target[d])
            gradient[d] = -2 * curvature[d] * (x - target[d]) * (d == 5 && z[d] < 0 ? -1 : 1)
            hessian[d][d] = -2 * curvature[d]
          }
          return SSBAgreementSearch.Derivatives(value: value, gradient: gradient, hessian: hessian)
        }
      })
    XCTAssertEqual(Array(recorder.rows.prefix(200)), Array(stencil.rows.prefix(200)))
    // One more row: the fitted point on the search's own objective.
    XCTAssertEqual(recorder.rows.count, 201)
    XCTAssertEqual(fit.newtonStarts.map(\.evaluations).reduce(0, +), centers)
    XCTAssertLessThan(centers, 30)
    let point = SSBAgreementSearch.encode(fit.point)
    for d in 0..<6 {
      XCTAssertEqual(point[d], target[d], accuracy: [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-2][d])
    }
    print(
      "EXACT_NEWTON centers \(centers) (stencil rows \(stencil.rows.count - 200)) agreement \(fit.agreement) (stencil \(reference.agreement))"
    )
    XCTAssertGreaterThanOrEqual(fit.agreement, reference.agreement)
  }

  func testRandomStreamsAndRanks() {
    // The first MT19937 outputs for seed 5489 (the reference generator's default seed).
    var generator = SSBAgreementSearch.MersenneTwister(seed: 5489)
    XCTAssertEqual(
      (0..<3).map { _ in generator.next() }, [3_499_211_612, 581_869_302, 3_890_346_734])
    // NumPy's RandomState(42).random_sample() uses the same 53-bit construction.
    var seeded = SSBAgreementSearch.MersenneTwister(seed: 42)
    XCTAssertEqual(seeded.uniform(), 0.3745401188473625, accuracy: 0)
    XCTAssertEqual(SSBAgreementSearch.ranks([3, 5, 5, 1]), [2, 0, 1, 3])
    XCTAssertEqual(SSBAgreementSearch.inverseCDF(SSBAgreementSearch.cdf(0.7)), 0.7, accuracy: 1e-12)
    XCTAssertEqual(SSBAgreementSearch.wrapPhi(.pi / 2), -.pi / 2, accuracy: 1e-15)
  }

  func testRefinementFromAStartNeedsNoTrials() throws {
    let recorder = Recorder()
    let start = [37.0, 30.0, -0.2, 3.0, -1.0, 140.0]
    let fit = try SSBAgreementSearch.fitTilt(recorder.evaluate, trials: 0, start: start)
    XCTAssertTrue(fit.trials.isEmpty)
    XCTAssertEqual(fit.point[0], 37.25, accuracy: 0.5)
    XCTAssertEqual(fit.point[3], 3.5, accuracy: 0.05)
    XCTAssertThrowsError(try SSBAgreementSearch.fitTilt(recorder.evaluate, trials: 0))
    XCTAssertThrowsError(try SSBAgreementSearch.fitTilt(recorder.evaluate, isCancelled: { true }))
  }
}
