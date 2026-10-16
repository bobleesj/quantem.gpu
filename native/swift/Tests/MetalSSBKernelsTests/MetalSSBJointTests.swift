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
  func testCalibratedReferencePhaseAndObjective() throws {
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
    let model = try engine.jointModel(scanDetectorRotationDegrees: fixture.rotationDegrees)
    let loss = try model.loss(fixture.parameters)
    let result = try model.reconstruct(
      fixture.parameters, upsamplingFactor: fixture.upsamplingFactor)
    let expected = try Data(contentsOf: root.appendingPathComponent("reference.f32"))
      .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let actual = result.phase.contents().assumingMemoryBound(to: Float.self)
    var error = 0.0
    var norm = 0.0
    var maximum: Float = 0
    for i in expected.indices {
      let delta = actual[i] - expected[i]
      error += Double(delta) * Double(delta)
      norm += Double(expected[i]) * Double(expected[i])
      maximum = max(maximum, abs(delta))
    }
    let relative = sqrt(error / norm)
    let lossError = abs(loss - fixture.loss) / abs(fixture.loss)
    print("CALIBRATED_JOINT_PARITY loss=\(lossError) phaseL2=\(relative) phaseMax=\(maximum)")
    XCTAssertLessThan(lossError, 5e-5)
    XCTAssertLessThan(relative, 2e-4)
    XCTAssertLessThan(maximum, 2e-4)
  }
  struct FixtureCase: Decodable {
    let name: String
    let parameters: SSBJointParameters
    let loss: Double
  }

  /// Generated from the independent Python MPS joint implementation, outside
  /// the repository. No identifying input or output is embedded in this test.
  func testPythonReferencePhaseAndObjective() throws {
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
      for fixture in cases {
        let measured = try model.loss(fixture.parameters)
        let relative = abs(measured - fixture.loss) / max(abs(fixture.loss), 1)
        XCTAssertLessThan(relative, 5e-5, "\(fixture.name) objective relative error \(relative)")
        let phase = try model.reconstruct(fixture.parameters)
        let reference = try Data(contentsOf: root.appendingPathComponent("\(fixture.name).f32"))
        let expected = reference.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let actual = phase.phase.contents().assumingMemoryBound(to: Float.self)
        var maxError: Float = 0
        var error = 0.0
        var norm = 0.0
        for i in expected.indices {
          let delta = Double(actual[i] - expected[i])
          error += delta * delta
          norm += Double(expected[i]) * Double(expected[i])
          maxError = max(maxError, abs(Float(delta)))
        }
        print(
          "JOINT_PARITY \(fixture.name) cache=\(cacheBudget) objective=\(relative) phaseL2=\(sqrt(error / max(norm, 1e-30))) phaseMax=\(maxError)"
        )
        XCTAssertLessThan(maxError, 2e-4)
        XCTAssertLessThan(sqrt(error / max(norm, 1e-30)), 2e-4)
      }
      if cacheBudget > 0 {
        let tilted = cases.first { $0.name == "tilted" }!
        for factor in [2, 3, 4] {
          let result = try model.reconstruct(tilted.parameters, upsamplingFactor: factor)
          let bytes = try Data(contentsOf: root.appendingPathComponent("tilted-\(factor)x.f32"))
          let expected = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
          let actual = result.phase.contents().assumingMemoryBound(to: Float.self)
          var maxError: Float = 0
          var error = 0.0
          var norm = 0.0
          for i in expected.indices {
            let delta = Double(actual[i] - expected[i])
            error += delta * delta
            norm += Double(expected[i]) * Double(expected[i])
            maxError = max(maxError, abs(Float(delta)))
          }
          print("JOINT_UPSAMPLE factor=\(factor) L2=\(sqrt(error / norm)) max=\(maxError)")
          XCTAssertLessThan(maxError, 2e-4)
          XCTAssertLessThan(sqrt(error / norm), 2e-4)
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
      XCTAssertThrowsError(try model.loss(cases[0].parameters))
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
  }

  func testSixParameterSearchAndCancellation() throws {
    let initial = SSBJointParameters(
      aberrations: .init(c10Angstroms: 40, c12Angstroms: 15, phi12Radians: 0.2),
      depthAngstroms: 180, tiltRowMrad: 2, tiltColumnMrad: -4)
    func loss(_ p: SSBJointParameters) -> Double {
      let values = [
        p.aberrations.c10Angstroms / 10, p.aberrations.c12Angstroms / 10,
        p.aberrations.phi12Radians, p.tiltRowMrad, p.tiltColumnMrad, p.depthAngstroms / 50,
      ]
      let target: [Float] = [1, 0.5, -0.2, 4, -2, 5]
      return zip(values, target).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) }
    }
    let optimizer = SSBJointOptimizer(globalTrials: 40, refinementIterations: 200)
    let first = try optimizer.run(start: initial, evaluate: loss)
    let second = try optimizer.run(start: initial, evaluate: loss)
    XCTAssertEqual(first.best, second.best)
    XCTAssertLessThan(first.loss, 0.01)
    XCTAssertThrowsError(try optimizer.run(start: initial, evaluate: loss, isCancelled: { true }))
  }
}
