import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

final class SSBFifthOrderNumericsTests: XCTestCase {
  func testPointPreservesDoubleTermsAndReadsLegacyPoints() throws {
    let term = SSBModelHigherOrder(
      order: 5, symmetry: 6, magnitudeAngstroms: 123456789.12345679,
      angleRadians: 0.123456789012345)
    let point = SSBModelPoint(
      c10Angstroms: 23.45678901234, c12Angstroms: 4.5678901234, phi12Radians: 0.123456789,
      sampleTiltMrad: [1.234567890123, -2.34567890123], depthSpreadAngstroms: 123.456789,
      higherOrder: [term])
    let encoded = try JSONEncoder().encode(point)
    XCTAssertEqual(try JSONDecoder().decode(SSBModelPoint.self, from: encoded), point)
    XCTAssertNotEqual(Double(Float(term.magnitudeAngstroms)), term.magnitudeAngstroms)
    var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    legacy.removeValue(forKey: "higherOrder")
    let decoded = try JSONDecoder().decode(
      SSBModelPoint.self, from: JSONSerialization.data(withJSONObject: legacy))
    XCTAssertTrue(decoded.higherOrder.isEmpty)
    XCTAssertEqual(decoded.c10Angstroms, point.c10Angstroms)
    try point.validate()
    var duplicate = point
    duplicate.higherOrder.append(term)
    XCTAssertThrowsError(try duplicate.validate())
  }

  func testZeroHigherOrderKeepsPhaseAndObjectiveBitIdentical() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let scene = try SSBModelReconstructionTests.scene()
    let input = try SSBCoherentReconstruction.buffer(device, scene.counts)
    let engine = try MetalSSBEngine(device: device, geometry: scene.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
    let primary = SSBModelPoint(
      c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -0.4,
      sampleTiltMrad: [2.5, -4], depthSpreadAngstroms: 140)
    var zeros = primary
    zeros.higherOrder = SSBModelHigherOrder.supported
    XCTAssertEqual(try model.agreements([primary]), try model.agreements([zeros]))
    for factor in [1, 3] {
      let before = try model.reconstruct(primary, upsamplingFactor: factor)
      let after = try model.reconstruct(zeros, upsamplingFactor: factor)
      XCTAssertEqual(
        Data(bytes: before.phase.contents(), count: before.rows * before.columns * 4),
        Data(bytes: after.phase.contents(), count: after.rows * after.columns * 4))
      XCTAssertEqual(after.point, zeros)
    }
    var higher = primary
    higher.higherOrder = [.init(order: 2, symmetry: 1, magnitudeAngstroms: 1200, angleRadians: 0.2)]
    let all = try model.fifthOrderAgreements([primary, higher])
    XCTAssertTrue(all.allSatisfy(\.isFinite))
    XCTAssertNotEqual(all[0], all[1])
    XCTAssertEqual(all, try [primary, higher].flatMap { try model.fifthOrderAgreements([$0]) })
    try model.retarget(scanDetectorRotationDegrees: 197.25)
    let fresh = try engine.jointModel(scanDetectorRotationDegrees: 197.25)
    XCTAssertEqual(
      try model.fifthOrderAgreements([higher]), try fresh.fifthOrderAgreements([higher]))
  }

  /// ssb-fifth-agreement against the objective in double precision (`doubleAgreements`) at fixed candidates: shared
  /// and distinct depth/tilt (the shared depth-sinc path), zero depth, every remainder of the four-candidate groups,
  /// and batch invariance.
  func testRowsKernelMatchesDoublePrecisionDefinition() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let scene = try SSBModelReconstructionTests.scene()
    let input = try SSBCoherentReconstruction.buffer(device, scene.counts)
    let engine = try MetalSSBEngine(device: device, geometry: scene.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
    // About one radian at the aperture edge per term: lambda (n + 1) / (2 pi alpha^(n + 1)).
    let points = (0..<11).map { i -> SSBModelPoint in
      var point = SSBModelPoint(
        c10Angstroms: -38 + 3 * Double(i), c12Angstroms: 9, phi12Radians: -0.4,
        sampleTiltMrad: [2.5, i < 6 ? -4 : -3], depthSpreadAngstroms: i % 5 == 4 ? 0 : 140)
      point.higherOrder = SSBModelHigherOrder.supported.enumerated().map { j, term in
        let n = Double(term.order)
        let scale = 0.0197 * (n + 1) / (2 * Double.pi * pow(0.03, n + 1))
        return SSBModelHigherOrder(
          order: term.order, symmetry: term.symmetry,
          magnitudeAngstroms: scale * (0.3 + 0.07 * Double((i + j) % 9)),
          angleRadians: 0.37 * Double(i) - 0.21 * Double(j))
      }
      return point
    }
    let exact = try Self.doubleAgreements(model: model, points: points)
    let all = try model.fifthOrderAgreements(points)
    // A float32 objective about 1e-6 from the float64 one on this 40-pixel scene.
    for i in points.indices {
      let error = abs(all[i] / exact[i] - 1)
      print(String(format: "candidate %2d: %.3e from float64", i, error))
      XCTAssertLessThan(error, 2e-6)
    }
    for count in [1, 2, 3, 4, 5] {
      XCTAssertEqual(
        Array(all.prefix(count)), try model.fifthOrderAgreements(Array(points.prefix(count))))
    }
    XCTAssertEqual(all, try points.map { try model.fifthOrderAgreements([$0])[0] })
    XCTAssertEqual(all, try model.fifthOrderAgreements(points.reversed()).reversed())
  }

  /// ssb-joint-agreement (`joint_agreement_angle_sums`) against the objective in double precision
  /// (`doubleAgreements`, the fifth-order definition at zero higher orders): thick and thin candidates, every
  /// remainder of the four-candidate groups, more than one group of eight, and batch invariance.
  func testLowOrderAngleSumsMatchDoublePrecisionDefinition() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let scene = try SSBModelReconstructionTests.scene()
    let input = try SSBCoherentReconstruction.buffer(device, scene.counts)
    let engine = try MetalSSBEngine(device: device, geometry: scene.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
    let points = (0..<11).map { (i: Int) -> SSBModelPoint in
      let x = Double(i)
      let tilt: [Double] = [2.5 - 0.4 * x, -4 + 0.6 * x]
      let depth: Double = i % 5 == 4 ? 0 : 140 + 20 * x
      return SSBModelPoint(
        c10Angstroms: -38 + 7 * x, c12Angstroms: 9 - 0.5 * x, phi12Radians: -0.4 + 0.3 * x,
        sampleTiltMrad: tilt, depthSpreadAngstroms: depth)
    }
    let exact = try Self.doubleAgreements(model: model, points: points)
    let all = try model.agreements(points)
    // The float32 phasors advance along detector rows between anchors, so this kernel sits farther from the float64
    // objective than the per-pair fifth-order one: at most 4.7e-6 on these candidates (2026-10-25).
    let errors = zip(all, exact).map { abs($0 / $1 - 1) }
    print("ssb-joint-agreement maximum relative difference from float64", errors.max()!)
    for error in errors { XCTAssertLessThan(error, 1e-5) }
    for count in [1, 2, 3, 5, 8] {
      XCTAssertEqual(Array(all.prefix(count)), try model.agreements(Array(points.prefix(count))))
    }
    XCTAssertEqual(all, try points.map { try model.agreements([$0])[0] })
  }

  /// The fifth-order agreement objective with every term and sum in double precision, from the model's
  /// float32 inputs (k, q, weights, probe geometry, evidence) and the float32 packed coefficients.
  static func doubleAgreements(model: MetalSSBJointModel, points: [SSBModelPoint]) throws
    -> [Double]
  {
    let arrays = try model.objectiveArrays()
    let wl = arrays.constants["wavelength_A"]!
    let semi = arrays.constants["semiangle_rad"]!
    let ay = arrays.constants["angular_row_rad"]!
    let ax = arrays.constants["angular_column_rad"]!
    let factor = arrays.constants["factor"]!
    let azimuth = [0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6]
    let power = [1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0]
    func chi(_ row: Double, _ col: Double, _ c: [Double]) -> Double {
      let x = row * wl
      let y = col * wl
      let r2 = x * x + y * y
      var re = [1.0]
      var im = [0.0]
      for m in 1...6 {
        re.append(re[m - 1] * x - im[m - 1] * y)
        im.append(im[m - 1] * x + re[m - 1] * y)
      }
      var total = 0.0
      for i in 0..<14 {
        total +=
          pow(r2, Double(power[i])) * (re[azimuth[i]] * c[2 * i] + im[azimuth[i]] * c[2 * i + 1])
      }
      return 2 * Double.pi / wl * total
    }
    func geometry(_ dx: Double, _ dy: Double) -> (Double, Double) {
      let r = (dx * dx + dy * dy).squareRoot()
      let alpha = r * wl
      let denom = r > 1e-15 ? (dx * ay * dx * ay + dy * ax * dy * ax).squareRoot() / r : 0
      let edge = denom > 1e-15 ? (semi - alpha) / denom + 0.5 : 1
      return (alpha * alpha, min(max(edge, 0), 1))
    }
    func sinc(_ x: Double) -> Double { abs(x) < 1e-6 ? 1 : sin(x) / x }
    return points.map { point in
      let c = point.reconstructionCoefficients.map(Double.init)
      let t = Double(Float(point.depthSpreadAngstroms))
      let thr = Double(Float(point.sampleTiltMrad[0] * 1e-3))
      let thc = Double(Float(point.sampleTiltMrad[1] * 1e-3))
      var total = 0.0
      for (qi, q) in arrays.q.enumerated() {
        let qx = Double(q.x)
        let qy = Double(q.y)
        let shift = 2 * Double.pi * (qx * thr + qy * thc)
        var nr = 0.0
        var ni = 0.0
        var dd = 0.0
        for (ki, k) in arrays.k.enumerated() {
          let kx = Double(k.x)
          let ky = Double(k.y)
          let gk = arrays.probe[ki]
          let gm = geometry(qx - kx, qy - ky)
          let gp = geometry(qx + kx, qy + ky)
          let a1 = gm.1 * Double(gk.w)
          let a2 = gp.1 * Double(gk.w)
          if a1 == 0 && a2 == 0 { continue }
          let chiK = chi(kx, ky, c)
          var gr = 0.0
          var gi = 0.0
          if a1 != 0 {
            let phase = chi(kx - qx, ky - qy, c) - chiK
            let b1 = t != 0 ? sinc(0.5 * t * (-factor * (gm.0 - Double(gk.x)) - shift)) * a1 : a1
            gr += b1 * cos(phase)
            gi -= b1 * sin(phase)
          }
          if a2 != 0 {
            let phase = chi(qx + kx, qy + ky, c) - chiK
            let b2 = t != 0 ? sinc(0.5 * t * (factor * (gp.0 - Double(gk.x)) - shift)) * a2 : a2
            gr -= b2 * cos(phase)
            gi -= b2 * sin(phase)
          }
          let g = arrays.evidence[ki * arrays.q.count + qi]
          let gre = Double(g.x)
          let gim = Double(g.y)
          nr += gre * gr + gim * gi
          ni += gim * gr - gre * gi
          dd += gr * gr + gi * gi
        }
        if dd > 0 { total += Double(arrays.weights[qi]) * (nr * nr + ni * ni) / dd }
      }
      return total
    }
  }

  func testSavedPhaseRetainsExactHigherOrderPointAndReadsLegacyRun() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let scene = try SSBModelReconstructionTests.scene()
    let input = try SSBCoherentReconstruction.buffer(device, scene.counts)
    let engine = try MetalSSBEngine(device: device, geometry: scene.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
    let point = SSBModelPoint(
      c10Angstroms: 23.45678901234, c12Angstroms: 4.5678901234, phi12Radians: 0.123456789,
      sampleTiltMrad: [1.234567890123, -2.34567890123], depthSpreadAngstroms: 123.456789,
      higherOrder: [
        .init(order: 2, symmetry: 1, magnitudeAngstroms: 1234.5678901234, angleRadians: 0.23),
        .init(
          order: 5, symmetry: 6, magnitudeAngstroms: 123456789.12345679,
          angleRadians: 0.123456789012345),
      ])
    var result = try model.reconstruct(point, upsamplingFactor: 3)
    let calibration = MetalSSBCalibration(
      voltageKV: 300, semiangleMrad: 30,
      scanSamplingRowAngstroms: 0.4, scanSamplingColumnAngstroms: 0.4,
      detectorSamplingRowMrad: 5, detectorSamplingColumnMrad: 5,
      brightfieldCenterRowPixels: 6, brightfieldCenterColumnPixels: 6,
      brightfieldRadiusPixels: 6)
    func saved(_ phase: MetalSSBJointPhase) throws -> MetalSSBJointSavedRun {
      try MetalSSBJointSavedRun(
        result: phase, sourceIdentity: "fixture", backendRevision: "test",
        sourceRows: 512, sourceColumns: 512, region: .init(row: 27, column: 63, size: 128),
        calibration: calibration, optimization: nil, provenance: ["source": "generated"])
    }
    let run = try saved(result)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".jointssb")
    defer { try? FileManager.default.removeItem(at: url) }
    try run.save(to: url)
    let restored = try MetalSSBJointSavedRun.load(from: url, sourceIdentity: "fixture")
    XCTAssertEqual(restored.point, point)
    XCTAssertEqual(restored.parameters, point.jointParameters)
    XCTAssertEqual(restored.phase, run.phase)
    XCTAssertEqual(restored.phaseSHA256, run.phaseSHA256)
    XCTAssertEqual(restored.region, run.region)
    let repeated = try model.reconstruct(try XCTUnwrap(restored.point), upsamplingFactor: 3)
    XCTAssertEqual(Data(bytes: repeated.phase.contents(), count: run.phase.count), run.phase)

    var legacy = try XCTUnwrap(
      PropertyListSerialization.propertyList(from: Data(contentsOf: url), format: nil)
        as? [String: Any])
    legacy.removeValue(forKey: "point")
    try PropertyListSerialization.data(fromPropertyList: legacy, format: .binary, options: 0)
      .write(to: url)
    let oldRun = try MetalSSBJointSavedRun.load(from: url, sourceIdentity: "fixture")
    XCTAssertNil(oldRun.point)
    XCTAssertEqual(oldRun.parameters, run.parameters)
    XCTAssertEqual(oldRun.phase, run.phase)

    result.point?.higherOrder[0].magnitudeAngstroms += 100
    XCTAssertThrowsError(try saved(result))
  }

  struct Reference: Decodable {
    struct Phase: Decodable {
      let factor: Int
      let file: String
    }
    struct Case: Decodable {
      let point: SSBModelPoint
      let packed: [Float]
      let agreement: Double
      let phases: [Phase]
    }
    let geometry: MetalSSBGeometry
    let countType: MetalSSBCountType?
    let cases: [Case]
  }

  /// A Python MPS run writes the complete synthetic input, candidate packs and full Auto/native arrays.
  /// The same reader accepts private matched real-data evidence without putting it in the repository.
  func testCompleteArraysAgainstPythonMPSWhenGiven() throws {
    guard let path = ProcessInfo.processInfo.environment["SSB_FIFTH_ORDER_REFERENCE"] else {
      throw XCTSkip("Set SSB_FIFTH_ORDER_REFERENCE to matched Python MPS fixture.json.")
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let root = URL(fileURLWithPath: path).deletingLastPathComponent()
    let reference = try JSONDecoder().decode(
      Reference.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    let counts = try Data(contentsOf: root.appendingPathComponent("counts.bin"))
    let input = try XCTUnwrap(
      counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let caches: [Int?] = reference.geometry.qxByRow.count > 128 ? [nil] : [nil, 0]
    for cache in caches {
      let engine = try MetalSSBEngine(
        device: device, geometry: reference.geometry, cacheBudgetBytes: cache)
      try engine.prepare(brightfield: input, countType: reference.countType ?? .uint16)
      let model = try engine.jointModel(scanDetectorRotationDegrees: 0)
      let scores = try model.fifthOrderAgreements(
        reference.cases.map(\.point), packedCoefficients: reference.cases.map(\.packed))
      for (index, candidate) in reference.cases.enumerated() {
        let relative = abs(scores[index] / candidate.agreement - 1)
        print(
          "FIFTH agreement case \(index) relative \(relative) cache \(cache.map(String.init) ?? "full")"
        )
        XCTAssertLessThan(relative, 1e-5)
        for phase in candidate.phases {
          let result = try model.reconstruct(candidate.point, upsamplingFactor: phase.factor)
          let expected = try Data(contentsOf: root.appendingPathComponent(phase.file))
          XCTAssertEqual(expected.count, result.rows * result.columns * 4)
          let actual = result.phase.contents().assumingMemoryBound(to: Float.self)
          let maximum: Float = expected.withUnsafeBytes { bytes in
            let values = bytes.bindMemory(to: Float.self)
            return values.indices.reduce(Float(0)) { max($0, abs(actual[$1] - values[$1])) }
          }
          print("FIFTH phase case \(index) factor \(phase.factor) maximum \(maximum)")
          XCTAssertLessThan(maximum, 2e-4)
          XCTAssertEqual(result.point, candidate.point)
        }
      }
    }
  }
}
