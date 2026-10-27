import CryptoKit
import Foundation
import XCTest

@testable import MetalSSBKernels

/// The Swift port of the resident TPE search against the Python implementation
/// (`direct_ptycho._resident_search.fit_resident` and `fit_standard_resident`, seed 42, 200 trials) on a synthetic float32
/// objective that both sides evaluate from the same float32 trial rows: the digests pin the 200 search rows Python
/// evaluated, in order. The trust-region Newton after the search climbs on exact derivatives (Python
/// `direct_ptycho._exact_newton`); here they are the analytic derivatives of the same objective in double precision, whose
/// maximum is known exactly.
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

  /// `score` in the Newton coordinates z (C10, C12 cos 2 phi12, C12 sin 2 phi12, tilt row, tilt column, depth
  /// spread) in double precision, with its analytic gradient and Hessian; `quartic` adds a C10 term that makes some
  /// steps overshoot and get rejected.
  static let target = [37.25, 30, -12, 3.5, -1.25, 151]
  static let curvature = [0.25, 1, 1, 400, 400, 0.01]
  static func exact(_ z: [Double], quartic: Double = 0) -> SSBAgreementSearch.Derivatives {
    var value = 1.0e6
    var gradient = [Double](repeating: 0, count: 6)
    var hessian = [[Double]](repeating: [Double](repeating: 0, count: 6), count: 6)
    for d in 0..<6 {
      let x = d == 5 ? abs(z[d]) : z[d]
      let r = x - target[d]
      let q = d == 0 ? quartic : 0
      value -= curvature[d] * r * r + q * r * r * r * r
      let slope = -2 * curvature[d] * r - 4 * q * r * r * r
      gradient[d] = slope * (d == 5 && z[d] < 0 ? -1 : 1)
      hessian[d][d] = -2 * curvature[d] - 12 * q * r * r
    }
    return SSBAgreementSearch.Derivatives(value: value, gradient: gradient, hessian: hessian)
  }
  static var derivatives: SSBAgreementSearch.StagedDerivatives {
    SSBAgreementSearch.staged { rows in rows.map { exact($0) } }
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
    func digest(_ count: Int) -> String {
      var bytes = Data()
      for row in rows.prefix(count) { row.withUnsafeBytes { bytes.append(contentsOf: $0) } }
      return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
  }

  /// `scripts/ssb_agreement_gate.py` runs the current Python fits (`fit_resident`, `fit_standard_resident`, with
  /// `exact` as their derivatives) and passes their records here, so a change on either side fails without editing
  /// the pins below.
  struct Reference: Decodable {
    struct Fit: Decodable {
      let searchRows: Int
      let searchSHA256: String
      let point: [Double]
      enum CodingKeys: String, CodingKey {
        case searchRows = "search_rows"
        case searchSHA256 = "search_sha256"
        case point
      }
    }
    let tilt: Fit
    let standard: Fit
  }

  func testCurrentPythonFitsWhenGiven() throws {
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
        ? try SSBAgreementSearch.fitTilt(recorder.evaluate, staged: Self.derivatives)
        : try SSBAgreementSearch.fitStandard(recorder.evaluate, staged: Self.derivatives)
      XCTAssertGreaterThan(recorder.rows.count, expected.searchRows, name)
      XCTAssertEqual(
        recorder.digest(expected.searchRows), expected.searchSHA256,
        "\(name): a search trial differs from the Python search")
      XCTAssertEqual(fit.point.count, 6, name)
      for (actual, value) in zip(fit.point, expected.point) {
        XCTAssertEqual(actual, value, accuracy: 1e-6, name)
      }
    }
  }

  /// The 200 search trials are Python's, then the exact Newton reaches the analytic maximum in a few centers.
  func testTiltFitSearchesAsPythonAndReachesTheMaximum() throws {
    let recorder = Recorder()
    var centers = 0
    let fit = try SSBAgreementSearch.fitTilt(
      recorder.evaluate,
      staged: SSBAgreementSearch.staged { rows in
        centers += rows.count
        return rows.map { Self.exact($0) }
      })
    XCTAssertEqual(
      recorder.digest(200), "d4e7f707fef64321c05c91cd93f89cec09de3df9f9e19ed3a3ea5ab652b100eb",
      "a search trial differs from the Python search")
    // One more row: the fitted point on the search's own objective.
    XCTAssertEqual(recorder.rows.count, 201)
    XCTAssertEqual(fit.newtonStarts.map(\.evaluations).reduce(0, +), centers)
    XCTAssertLessThan(centers, 30)
    let point = SSBAgreementSearch.encode(fit.point)
    for d in 0..<6 {
      XCTAssertEqual(point[d], Self.target[d], accuracy: [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-2][d])
    }
    XCTAssertEqual(fit.agreement, Self.score(recorder.rows[200]))
    XCTAssertGreaterThanOrEqual(fit.agreement, fit.trials.map(\.agreement).max()!)
    print("EXACT_NEWTON tilt centers \(centers) point \(fit.point) agreement \(fit.agreement)")
  }

  /// The tilt=False fit: the same search, tilt and depth spread held at 0, the maximum of C10 and C12 there.
  func testStandardFitSearchesAsPythonAndReachesTheMaximum() throws {
    let recorder = Recorder()
    let fit = try SSBAgreementSearch.fitStandard(recorder.evaluate, staged: Self.derivatives)
    XCTAssertEqual(
      recorder.digest(200), "98411fa3934f9832762e50d9881d6209412a7b87bd49b4956ecb75981816f83f",
      "a search trial differs from the Python search")
    XCTAssertEqual(recorder.rows.count, 201)
    XCTAssertEqual(Array(fit.point[3...]), [0, 0, 0])
    let point = SSBAgreementSearch.encode(fit.point)
    for d in 0..<3 { XCTAssertEqual(point[d], Self.target[d], accuracy: 1e-3) }
    XCTAssertEqual(fit.agreement, Self.score(recorder.rows[200]))
    XCTAssertGreaterThanOrEqual(fit.agreement, fit.trials.map(\.agreement).max()!)
    print("EXACT_NEWTON standard point \(fit.point) agreement \(fit.agreement)")
  }

  /// The staged exact Newton (values first, derivatives only where a step uses them) reaches the same point as
  /// the one-stage exact Newton and asks for fewer derivatives: none at a rejected step or at the starts that the
  /// second iteration's pruning stops.
  func testStagedExactNewtonMatchesOneStage() throws {
    func exact(_ z: [Double]) -> SSBAgreementSearch.Derivatives { Self.exact(z, quartic: 0.002) }
    let oneStage = Recorder()
    var centers = 0
    let reference = try SSBAgreementSearch.fitTilt(
      oneStage.evaluate,
      staged: SSBAgreementSearch.staged { rows in
        centers += rows.count
        return rows.map(exact)
      })
    let staged = Recorder()
    var latest: [[Double]] = []
    var latestFused = false
    var valueRows = 0
    var fusedRows = 0
    var derivativeOnlyRows = 0
    let fit = try SSBAgreementSearch.fitTilt(
      staged.evaluate,
      staged: SSBAgreementSearch.StagedDerivatives(
        values: { rows, withDerivatives in
          latest = rows
          latestFused = withDerivatives
          valueRows += rows.count
          if withDerivatives { fusedRows += rows.count }
          return rows.map { exact($0).value }
        },
        derivatives: { indices in
          if !latestFused { derivativeOnlyRows += indices.count }
          return indices.map { exact(latest[$0]) }
        }))
    XCTAssertEqual(staged.rows, oneStage.rows)
    XCTAssertEqual(fit.point, reference.point)
    XCTAssertEqual(fit.agreement, reference.agreement)
    XCTAssertEqual(valueRows, centers)
    XCTAssertEqual(fit.newtonStarts.map(\.evaluations), reference.newtonStarts.map(\.evaluations))
    print(
      "STAGED_NEWTON centers \(centers) fused \(fusedRows) derivative-only \(derivativeOnlyRows)")
    XCTAssertLessThan(fusedRows + derivativeOnlyRows, centers)
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
    let fit = try SSBAgreementSearch.fitTilt(
      recorder.evaluate, trials: 0, start: start, staged: Self.derivatives)
    XCTAssertTrue(fit.trials.isEmpty)
    XCTAssertEqual(fit.point[0], 37.25, accuracy: 1e-3)
    XCTAssertEqual(fit.point[3], 3.5, accuracy: 1e-4)
    XCTAssertThrowsError(
      try SSBAgreementSearch.fitTilt(recorder.evaluate, trials: 0, staged: Self.derivatives))
    XCTAssertThrowsError(
      try SSBAgreementSearch.fitTilt(
        recorder.evaluate, staged: Self.derivatives, isCancelled: { true }))
  }
}
