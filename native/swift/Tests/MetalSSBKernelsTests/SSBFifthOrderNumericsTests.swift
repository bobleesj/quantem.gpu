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
    XCTAssertThrowsError(try model.reconstruct(higher, estimator: .meanPhase))
    let all = try model.fifthOrderAgreements([primary, higher])
    XCTAssertTrue(all.allSatisfy(\.isFinite))
    XCTAssertNotEqual(all[0], all[1])
    XCTAssertEqual(all, try [primary, higher].flatMap { try model.fifthOrderAgreements([$0]) })
    try model.retarget(scanDetectorRotationDegrees: 197.25)
    let fresh = try engine.jointModel(scanDetectorRotationDegrees: 197.25)
    XCTAssertEqual(
      try model.fifthOrderAgreements([higher]), try fresh.fifthOrderAgreements([higher]))
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
