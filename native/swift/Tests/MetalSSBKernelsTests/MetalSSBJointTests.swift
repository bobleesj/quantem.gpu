import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

final class MetalSSBJointTests: XCTestCase {
  struct CalibratedFixture: Decodable {
    let calibration: MetalSSBCalibration
    let parameters: SSBJointParameters
    let rotationDegrees: Float
    let upsamplingFactor: Int
    let scanSize: Int
    let detectorShape: [Int]
    let pixels: [Int]
    let loss: Double
  }

  /// Exercises native calibration and nonzero rotation as well as the kernel.
  /// Fixture counts are independently selected from the original acquisition.
  func testCalibratedReferenceObjective() throws {
    guard let directory = ProcessInfo.processInfo.environment["SSB_JOINT_CALIBRATED_REFERENCE"]
    else {
      throw XCTSkip("Set SSB_JOINT_CALIBRATED_REFERENCE to an independent calibrated fixture.")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let root = URL(fileURLWithPath: directory)
    let fixture = try JSONDecoder().decode(
      CalibratedFixture.self,
      from: Data(contentsOf: root.appendingPathComponent("real-reference.json")))
    let counts = try Data(contentsOf: root.appendingPathComponent("counts.u32"))
    let plane = fixture.scanSize * fixture.scanSize
    XCTAssertEqual(counts.count, fixture.pixels.count * plane * 4)
    var sums = [UInt64](repeating: 0, count: fixture.detectorShape[0] * fixture.detectorShape[1])
    counts.withUnsafeBytes { bytes in
      let values = bytes.bindMemory(to: UInt32.self)
      for (k, pixel) in fixture.pixels.enumerated() {
        for i in 0..<plane { sums[pixel] += UInt64(values[k * plane + i]) }
      }
    }
    let setup = try fixture.calibration.geometry(
      detectorRows: fixture.detectorShape[0],
      detectorColumns: fixture.detectorShape[1], detectorSum: sums,
      scanRows: fixture.scanSize, scanColumns: fixture.scanSize)
    XCTAssertEqual(setup.pixels, fixture.pixels)
    let input = try XCTUnwrap(
      counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(
      device: device, geometry: setup.geometry, cacheBudgetBytes: 2 << 30)
    try engine.prepare(brightfield: input, countType: .uint32)
    let model = try engine.jointModel(scanDetectorRotationDegrees: Double(fixture.rotationDegrees))
    let loss = -(try model.agreements([SSBModelPoint(fixture.parameters)])[0])
    let lossError = abs(loss - fixture.loss) / abs(fixture.loss)
    print("CALIBRATED_JOINT_PARITY loss=\(lossError)")
    XCTAssertLessThan(lossError, 5e-5)
    // The fixture's mean_phase reference image belongs to the retired mean_phase estimator; only the objective
    // is compared.
  }
  struct FixtureCase: Decodable {
    let name: String
    let parameters: SSBJointParameters
    let loss: Double
  }

  /// Generated from the independent Python MPS joint implementation, outside
  /// the repository. No identifying input or output is embedded in this test.
  func testPythonReferenceObjective() throws {
    guard let directory = ProcessInfo.processInfo.environment["SSB_JOINT_REFERENCE"] else {
      throw XCTSkip("Set SSB_JOINT_REFERENCE to a generated reference fixture folder.")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device required")
    }
    let root = URL(fileURLWithPath: directory)
    let decoder = JSONDecoder()
    let geometry = try decoder.decode(
      MetalSSBGeometry.self, from: Data(contentsOf: root.appendingPathComponent("geometry.json")))
    let cases = try decoder.decode(
      [FixtureCase].self, from: Data(contentsOf: root.appendingPathComponent("cases.json")))
    let counts = try Data(contentsOf: root.appendingPathComponent("counts.u16"))
    let input = try XCTUnwrap(
      counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    var invalid = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(geometry)) as? [String: Any])
    for factor in [Float(0), -1] {
      invalid["phaseFactor"] = factor
      let decoded = try decoder.decode(
        MetalSSBGeometry.self, from: JSONSerialization.data(withJSONObject: invalid))
      XCTAssertThrowsError(try MetalSSBEngine(device: device, geometry: decoded))
    }
    for cacheBudget in [0, 1 << 30] {
      let engine = try MetalSSBEngine(
        device: device, geometry: geometry, cacheBudgetBytes: cacheBudget)
      try engine.prepare(brightfield: input, countType: .uint16)
      let model = try engine.jointModel(scanDetectorRotationDegrees: 0)
      let batchCases = (0..<17).map { SSBModelPoint(cases[$0 % cases.count].parameters) }
      let scalarAgreements = try batchCases.map { try model.agreements([$0])[0] }
      for count in [1, 7, 8, 9, 17] {
        let batched = try model.agreements(Array(batchCases.prefix(count)))
        XCTAssertEqual(
          batched, Array(scalarAgreements.prefix(count)), "GPU batching changed a scalar objective")
      }
      XCTAssertEqual(try model.agreements([]), [])
      XCTAssertThrowsError(try model.agreements(batchCases, isCancelled: { true }))
      for fixture in cases {
        // The fixture's loss is the negative agreement.
        let measured = -(try model.agreements([SSBModelPoint(fixture.parameters)])[0])
        let relative = abs(measured - fixture.loss) / max(abs(fixture.loss), 1)
        XCTAssertLessThan(relative, 5e-5, "\(fixture.name) objective relative error \(relative)")
        print("JOINT_PARITY \(fixture.name) cache=\(cacheBudget) objective=\(relative)")
      }
      if cacheBudget > 0 {
        let tilted = cases.first { $0.name == "tilted" }!
        for factor in [2, 3, 4] {
          // Fixtures generated before 2026-10-25 also hold tilted-<factor>x images of the retired mean_phase
          // estimator, which are not compared; the SSB result is saved and restored here.
          let result = try model.reconstruct(tilted.parameters, upsamplingFactor: factor)
          XCTAssertEqual(result.rows, 128 * factor)
          let calibration = MetalSSBCalibration(
            voltageKV: 300, semiangleMrad: 30,
            scanSamplingRowAngstroms: 0.4, scanSamplingColumnAngstroms: 0.4,
            detectorSamplingRowMrad: 5, detectorSamplingColumnMrad: 5,
            brightfieldCenterRowPixels: 6, brightfieldCenterColumnPixels: 6,
            brightfieldRadiusPixels: 6)
          let run = try MetalSSBJointSavedRun(
            result: result, sourceIdentity: "fixture", backendRevision: "test",
            sourceRows: 512, sourceColumns: 512, region: .init(row: 27, column: 63, size: 128),
            calibration: calibration, optimization: nil, provenance: ["source": "generated"])
          let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString + ".jointssb")
          defer { try? FileManager.default.removeItem(at: url) }
          try run.save(to: url)
          let restored = try MetalSSBJointSavedRun.load(from: url, sourceIdentity: "fixture")
          XCTAssertEqual(run.phase, restored.phase)
          XCTAssertEqual(restored.region?.row, 27)
          XCTAssertEqual(restored.parameters, tilted.parameters)
          XCTAssertThrowsError(try MetalSSBJointSavedRun.load(from: url, sourceIdentity: "other"))
        }
      }
      XCTAssertThrowsError(try model.reconstruct(cases[0].parameters, isCancelled: { true }))
      XCTAssertThrowsError(try model.reconstruct(cases[0].parameters, upsamplingFactor: 5))
      try engine.prepare(brightfield: input, countType: .uint16)
      XCTAssertThrowsError(try model.agreements([SSBModelPoint(cases[0].parameters)]))
      XCTAssertThrowsError(try model.reconstruct(cases[0].parameters))

    }
  }

  func testExactRegionExtractionAndSampling() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
      let command = queue.makeCommandBuffer()
    else { throw XCTSkip("Metal required") }
    let region = SSBScanRegion(row: 19, column: 71, size: 128)
    let values = (0..<(2 * 256 * 256)).map { UInt32($0) + 65536 }
    let input = try XCTUnwrap(
      values.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let output = try XCTUnwrap(
      device.makeBuffer(length: 2 * 128 * 128 * 4, options: .storageModeShared))
    try region.encodeColumns(
      source: input, destination: output, count: 2, sourceRows: 256, sourceColumns: 256,
      commands: command)
    command.commit()
    command.waitUntilCompleted()
    XCTAssertNil(command.error)
    let actual = output.contents().assumingMemoryBound(to: UInt32.self)
    for k in 0..<2 {
      for r in 0..<128 {
        for c in 0..<128 {
          XCTAssertEqual(actual[(k * 128 + r) * 128 + c], values[(k * 256 + r + 19) * 256 + c + 71])
        }
      }
    }
    XCTAssertThrowsError(
      try SSBScanRegion(row: 200, column: 0, size: 128).validate(
        sourceRows: 256, sourceColumns: 256))
    var cal = MetalSSBCalibration(
      voltageKV: 300, semiangleMrad: 30,
      scanSamplingRowAngstroms: 0.4, scanSamplingColumnAngstroms: 0.4,
      detectorSamplingRowMrad: 5, detectorSamplingColumnMrad: 5,
      brightfieldCenterRowPixels: 6, brightfieldCenterColumnPixels: 6)
    XCTAssertEqual(try cal.ssbUpsamplingFactor(), 3)
    cal.scanSamplingRowAngstroms = 0.6
    XCTAssertEqual(try cal.ssbUpsamplingFactor(), 4)
    cal.scanSamplingRowAngstroms = 0
    XCTAssertThrowsError(try cal.ssbUpsamplingFactor())
    XCTAssertThrowsError(try SSBOutputSampling.auto.resolve(cal))
  }

  /// Python `test_ssb_auto_sampling`: Auto covers nominal support up to the 4x limit and is capped there.
  func testAutoSamplingIsCappedAtFourTimes() throws {
    let wavelength = MetalSSBCalibration.wavelengthAngstroms(voltageKV: 300)
    let spacing = wavelength / (4 * 20e-3)
    for (required, expected) in [
      (0.5, 1), (1, 1), (1.1, 2), (2.1, 3), (3.1, 4), (4.1, 4), (8, 4), (12, 4),
    ] {
      let cal = MetalSSBCalibration(
        voltageKV: 300, semiangleMrad: 20,
        scanSamplingRowAngstroms: 0.5 * spacing, scanSamplingColumnAngstroms: required * spacing,
        detectorSamplingRowMrad: 1, detectorSamplingColumnMrad: 1,
        brightfieldCenterRowPixels: 6, brightfieldCenterColumnPixels: 6)
      XCTAssertEqual(try cal.ssbUpsamplingFactor(), expected, "requirement \(required)")
      XCTAssertEqual(try cal.ssbUpsamplingRequirement(), required, accuracy: 1e-9)
      XCTAssertEqual(try SSBOutputSampling.auto.resolve(cal), expected)
      let choice = try cal.ssbUpsamplingChoice()
      XCTAssertEqual(choice.factor, expected)
      XCTAssertEqual(choice.uncappedFactor, required > 4 ? 8 : expected, "requirement \(required)")
      XCTAssertEqual(choice.capped, required > 4, "requirement \(required)")
    }
    let thick = MetalSSBCalibration(
      voltageKV: 300, semiangleMrad: 30, scanSamplingRowAngstroms: 0.373,
      scanSamplingColumnAngstroms: 0.373,
      detectorSamplingRowMrad: 0.57, detectorSamplingColumnMrad: 0.57,
      brightfieldCenterRowPixels: 96, brightfieldCenterColumnPixels: 96)
    XCTAssertEqual(try thick.ssbUpsamplingFactor(), 3)
    for value in 1...4 { XCTAssertEqual(try SSBOutputSampling.factor(value).resolve(thick), value) }
    XCTAssertEqual(try SSBOutputSampling.factor(8).resolve(thick), 8)
    XCTAssertThrowsError(try SSBOutputSampling.factor(5).resolve(thick))
    XCTAssertEqual(SSBOutputSampling.choices.map(\.label), ["Auto", "1×", "2×", "3×", "4×"])
    let coded = try JSONEncoder().encode(SSBOutputSampling.choices)
    XCTAssertEqual(
      try JSONDecoder().decode([SSBOutputSampling].self, from: coded), SSBOutputSampling.choices)
  }

  /// A measured 512 x 512 calibration (300 kV, 1.0855 A step, 33.145 mrad from the BF edge) needs about 7.3x:
  /// Auto is capped at 4x, an explicit 8x stays a recorded manual override.
  func testCoarseScanAutoIsCappedAndOverrideIsRecorded() throws {
    let cal = MetalSSBCalibration(
      voltageKV: 300, semiangleMrad: 33.145, scanSamplingRowAngstroms: 1.0855,
      scanSamplingColumnAngstroms: 1.0855, detectorSamplingRowMrad: 0.6044,
      detectorSamplingColumnMrad: 0.6044, brightfieldCenterRowPixels: 96,
      brightfieldCenterColumnPixels: 96)
    XCTAssertEqual(try cal.ssbUpsamplingRequirement(), 7.31, accuracy: 0.01)
    XCTAssertEqual(try cal.ssbUpsamplingFactor(), 4)
    XCTAssertEqual(try cal.ssbUpsamplingChoice(), SSBUpsamplingChoice(factor: 4, uncappedFactor: 8))
    XCTAssertTrue(try cal.ssbUpsamplingChoice().capped)
    XCTAssertEqual(try SSBOutputSampling.auto.resolve(cal), 4)
    XCTAssertEqual(try SSBOutputSampling.factor(8).resolve(cal), 8)
    for invalid in [0, 5, 6, 7, 16] {
      XCTAssertThrowsError(try SSBOutputSampling.factor(invalid).resolve(cal))
    }

    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try SSBModelReconstructionTests.scene()
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(device: device, geometry: s.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
    let point = SSBModelPoint(c10Angstroms: 40, c12Angstroms: 8, phi12Radians: 0.3)
    let auto = try model.reconstruct(point, outputSampling: .auto, calibration: s.calibration)
    XCTAssertEqual(auto.upsamplingFactor, try s.calibration.ssbUpsamplingFactor())
    XCTAssertEqual(auto.outputSampling, .auto)
    let manual = try model.reconstruct(
      point, outputSampling: .factor(2), calibration: s.calibration)
    XCTAssertEqual(manual.upsamplingFactor, 2)
    XCTAssertEqual(manual.outputSampling, .factor(2))
    XCTAssertNil(try model.reconstruct(point, upsamplingFactor: 2).outputSampling)
    // The saved run keeps the choice through a round trip.
    let run = try MetalSSBJointSavedRun(
      result: manual, sourceIdentity: String(repeating: "a", count: 64), backendRevision: "test",
      sourceRows: s.scan, sourceColumns: s.scan, region: nil, calibration: s.calibration,
      optimization: nil, provenance: [:])
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".jointssb")
    defer { try? FileManager.default.removeItem(at: url) }
    try run.save(to: url)
    let reopened = try MetalSSBJointSavedRun.load(from: url, sourceIdentity: run.sourceIdentity)
    XCTAssertEqual(reopened.outputSampling, .factor(2))
  }
}
