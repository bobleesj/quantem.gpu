import Foundation
import Metal
import XCTest

@testable import MetalDirectPtychoKernels

final class DirectPtychoWorkspaceLifetimeTests: XCTestCase {
  func testTrimmedWorkspacePreservesStreamingAndRepreparation() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let flag = "QUANTEM_METAL_TRIM_FFT_WORKSPACE"
    let original = ProcessInfo.processInfo.environment[flag]
    defer {
      if let original { setenv(flag, original, 1) } else { unsetenv(flag) }
    }
    let calibration = MetalDirectPtychoCalibration(
      voltageKV: 300, semiangleMrad: 30, scanSamplingRowAngstroms: 0.4,
      scanSamplingColumnAngstroms: 0.4, detectorSamplingRowMrad: 7.5,
      detectorSamplingColumnMrad: 7.5, brightfieldCenterRowPixels: 5.4,
      brightfieldCenterColumnPixels: 5.7, brightfieldRadiusPixels: 3.6)
    for (rows, columns) in [(128, 128), (256, 256), (512, 512), (90, 68), (68, 90), (511, 509)] {
      let setup = try calibration.geometry(
        detectorRows: 12, detectorColumns: 12,
        detectorSignalSum: Array(repeating: 1.0, count: 144),
        scanRows: rows, scanColumns: columns)
      let plane = rows * columns
      let count = setup.pixels.count * plane
      let source = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModeShared))
      let budgets: [Int?] = [nil, 32 * rows * (columns / 2 + 1) * 8, 0]
      for budget in budgets {
        let control = try MetalDirectPtychoEngine(
          device: device, geometry: setup.geometry, cacheBudgetBytes: budget)
        let candidate = try MetalDirectPtychoEngine(
          device: device, geometry: setup.geometry, cacheBudgetBytes: budget)
        // Reuse both engines while changing dtype and counts. This exercises regrowth
        // for preparation and the eight-plane streaming tail after each shrink.
        for (generation, type) in [MetalDirectPtychoCountType.uint8, .uint16, .uint32, .float32].enumerated() {
          for i in 0..<count {
            let value = UInt32(truncatingIfNeeded: (i &* 7919) ^ (i / columns) ^ generation)
            switch type {
            case .uint8:
              source.contents().assumingMemoryBound(to: UInt8.self)[i] = UInt8(truncatingIfNeeded: value)
            case .uint16:
              source.contents().assumingMemoryBound(to: UInt16.self)[i] = UInt16(truncatingIfNeeded: value)
            case .uint32:
              source.contents().assumingMemoryBound(to: UInt32.self)[i] = value
            case .float32:
              source.contents().assumingMemoryBound(to: Float.self)[i] = Float(Int(value % 257) - 128) / 16
            }
          }
          setenv(flag, "0", 1)
          try control.prepare(brightfield: source, countType: type)
          setenv(flag, "1", 1)
          try candidate.prepare(brightfield: source, countType: type)
          XCTAssertEqual(control.preparationWorkspaceBytes, 32 * plane * (type.byteWidth + 16))
          XCTAssertEqual(candidate.preparationWorkspaceBytes, 8 * plane * (type.byteWidth + 16))
          let originalModel = try control.jointModel(scanDetectorRotationDegrees: 17.25)
          let trimmedModel = try candidate.jointModel(scanDetectorRotationDegrees: 17.25)
          let expected = try originalModel.objectiveArrays()
          let actual = try trimmedModel.objectiveArrays()
          XCTAssertTrue(actual.evidence == expected.evidence,
            "Changed Fourier evidence for \(rows)x\(columns), \(type), cache \(String(describing: budget))")
          let point = DirectPtychoModelPoint(c10Angstroms: 47, c12Angstroms: 9,
            phi12Radians: 0.3, sampleTiltMrad: [2, -3], depthSpreadAngstroms: 120)
          XCTAssertEqual(try trimmedModel.agreements([point]), try originalModel.agreements([point]))
        }
      }
    }
  }
}
