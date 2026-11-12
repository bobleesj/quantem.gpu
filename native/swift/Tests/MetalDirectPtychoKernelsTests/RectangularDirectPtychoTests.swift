import Foundation
import Metal
import XCTest
@testable import MetalDirectPtychoKernels

final class RectangularDirectPtychoTests: XCTestCase {
  private func scene(_ rows: Int, _ columns: Int, background: Double = 0) throws -> (MTLDevice, MetalDirectPtychoGeometry, MTLBuffer) {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let calibration = MetalDirectPtychoCalibration(voltageKV: 80, semiangleMrad: 21.4,
      scanSamplingRowAngstroms: 0.59, scanSamplingColumnAngstroms: 0.43,
      detectorSamplingRowMrad: 7, detectorSamplingColumnMrad: 7,
      brightfieldCenterRowPixels: 4.2, brightfieldCenterColumnPixels: 4.3, brightfieldRadiusPixels: 3.2)
    try calibration.validateFittingBand(scanRows: rows, scanColumns: columns)
    let setup = try calibration.geometry(detectorRows: 9, detectorColumns: 9,
      detectorSignalSum: Array(repeating: Double(rows * columns) * background + 0.375, count: 81), scanRows: rows, scanColumns: columns)
    let count = setup.pixels.count * rows * columns
    let source = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModeShared))
    let values = source.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<count { values[i] = Float(background) }
    for bf in setup.pixels.indices {
      values[bf * rows * columns] = Float(background) + 0.5
      values[bf * rows * columns + columns + 2] = Float(background) - 0.125
    }
    return (device, setup.geometry, source)
  }

  /// Exact two-impulse DFT checks independent axes, odd bins and both cache layouts.
  func testNativeRectangularFourierEvidence() throws {
    for (rows, columns) in [(3, 5), (5, 3), (7, 11), (11, 7), (8, 9), (9, 8), (90, 68), (68, 90), (100, 100), (128, 256), (512, 192), (3, 512), (512, 3), (127, 129), (255, 257), (511, 509), (509, 511), (457, 456), (464, 510)] {
      let (device, geometry, source) = try scene(rows, columns)
      for budget: Int? in [nil, 0] {
        let engine = try MetalDirectPtychoEngine(device: device, geometry: geometry, cacheBudgetBytes: budget)
        try engine.prepare(brightfield: source, countType: .float32)
        let model = try engine.jointModel(scanDetectorRotationDegrees: 0)
        let arrays = try model.objectiveArrays()
        var error: Double = 0
        for i in arrays.evidence.indices {
          let q = arrays.q[i % arrays.q.count]
          let angle = 2 * Double.pi * (Double(q.x) * 0.59 + Double(q.y) * 0.43 * 2)
          error = max(error, abs(Double(arrays.evidence[i].x) - (0.5 - 0.125 * cos(angle))),
            abs(Double(arrays.evidence[i].y) - 0.125 * sin(angle)))
        }
        XCTAssertLessThan(error, 2e-6, "\(rows)x\(columns), cache \(String(describing: budget))")
      }
    }
  }

  func testRectangularReconstructionCacheParity() throws {
    for (rows, columns) in [(3, 5), (5, 3), (7, 11), (90, 68), (68, 90), (127, 129), (257, 263), (511, 509)] {
      let (device, geometry, source) = try scene(rows, columns, background: 10)
      var references: [String: [Float]] = [:]
      for budget: Int? in [nil, 0] {
        let engine = try MetalDirectPtychoEngine(device: device, geometry: geometry, cacheBudgetBytes: budget)
        try engine.prepare(brightfield: source, countType: .float32)
        let model = try engine.jointModel(scanDetectorRotationDegrees: 17)
        let point = DirectPtychoModelPoint(c10Angstroms: 12, c12Angstroms: 3, phi12Radians: 0.2,
          sampleTiltMrad: [1, -2], depthSpreadAngstroms: 30)
        for method: DirectPtychoReconstructionMethod in [.phaseOfMean, .phase, .complex] {
          for factor in [1, 2] {
            let result = try model.reconstruct(point, upsamplingFactor: factor, method: method)
            XCTAssertEqual(result.rows, rows * factor)
            XCTAssertEqual(result.columns, columns * factor)
            XCTAssertEqual(result.provenance.scanRows, rows)
            XCTAssertEqual(result.provenance.scanColumns, columns)
            let values = Array(UnsafeBufferPointer(start: result.phase.contents().assumingMemoryBound(to: Float.self), count: result.rows * result.columns))
            XCTAssertTrue(values.allSatisfy(\.isFinite))
            let key = "\(method)-\(factor)"
            if let reference = references[key] {
              let difference = zip(reference, values).map { abs($0 - $1) }.max()!
              XCTAssertLessThan(difference, 2e-5, key)
            } else { references[key] = values }
          }
        }
      }
    }
  }

  func testLargeRectangularSavedResultRetainsFourierIdentity() throws {
    let (device, geometry, source) = try scene(511, 509, background: 10)
    let engine = try MetalDirectPtychoEngine(device: device, geometry: geometry)
    try engine.prepare(brightfield: source, countType: .float32)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 0)
    let result = try model.reconstruct(DirectPtychoModelPoint(
      c10Angstroms: 12, c12Angstroms: 3, phi12Radians: 0.2,
      sampleTiltMrad: [1, -2], depthSpreadAngstroms: 30), upsamplingFactor: 1, method: .phase)
    let kernel = try XCTUnwrap(result.provenance.fourierKernel)
    XCTAssertEqual(kernel.name, "direct-ptycho-rectangular-fft")
    XCTAssertEqual(kernel.sourceSHA256, DirectPtychoKernelRegistry.rectangularFourier.registeredSourceSHA256)
    let calibration = MetalDirectPtychoCalibration(voltageKV: 80, semiangleMrad: 21.4,
      scanSamplingRowAngstroms: 0.59, scanSamplingColumnAngstroms: 0.43,
      detectorSamplingRowMrad: 7, detectorSamplingColumnMrad: 7,
      brightfieldCenterRowPixels: 4.2, brightfieldCenterColumnPixels: 4.3, brightfieldRadiusPixels: 3.2)
    let run = try MetalDirectPtychoJointSavedRun(result: result, sourceIdentity: "synthetic-rectangular",
      backendRevision: "test", sourceRows: 511, sourceColumns: 509, region: nil,
      calibration: calibration, optimization: nil, provenance: [:])
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".jointdirect")
    defer { try? FileManager.default.removeItem(at: url) }
    try run.save(to: url)
    let reopened = try MetalDirectPtychoJointSavedRun.load(from: url, sourceIdentity: "synthetic-rectangular")
    XCTAssertEqual(reopened.fourierKernel, kernel)
    XCTAssertEqual(reopened.phaseSHA256, run.phaseSHA256)
    XCTAssertEqual(reopened.calibration, calibration)
  }

  func testRotationBranchesKeepRectangularDarkFieldGeometry() throws {
    for (rows, columns) in [(32, 48), (48, 32)] {
      let (device, geometry, source) = try scene(rows, columns, background: 10)
      let engine = try MetalDirectPtychoEngine(device: device, geometry: geometry)
      try engine.prepare(brightfield: source, countType: .float32)
      let model = try engine.jointModel(scanDetectorRotationDegrees: 17)
      var darkField = Array(repeating: Float(0), count: rows * columns)
      for row in stride(from: 4, to: rows - 3, by: 6) {
        for column in stride(from: 4, to: columns - 3, by: 6) {
          darkField[row * columns + column] = Float(10 + (row + column) % 5)
        }
      }
      let expected = DirectPtychoRotationBranches.locateAtomicColumns(
        darkField: darkField, rows: rows, columns: columns, scanSampling: (0.59, 0.43))
      XCTAssertGreaterThan(expected.count, 0)
      let result = try DirectPtychoModelFit.findAberrations(
        model: model, rotationDegrees: 17, darkField: darkField, scanSampling: (0.59, 0.43),
        options: DirectPtychoFitOptions(compareStandard: false, maximumOrder: 1))
      XCTAssertEqual(result.columnCount, expected.count)
      XCTAssertEqual(result.branches.count, 2)
      for branch in result.branches { XCTAssertEqual(branch.column?.count, expected.count) }
    }
  }

}
