import Foundation
import Metal
import XCTest

@testable import MetalDirectPtychoKernels

/// Compare full detector-column preparation, including the partial final BF batch,
/// against the unfused GPU path. No CPU Fourier reconstruction is used.
final class DirectPtychoPreparationParityTests: XCTestCase {
  func testFusedPreparationPreservesEveryEvidenceValue() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let names = ["QUANTEM_METAL_FUSED_HALF", "QUANTEM_METAL_FUSED_COUNTS"]
    let original = names.map { ProcessInfo.processInfo.environment[$0] }
    defer {
      for (name, value) in zip(names, original) {
        if let value { setenv(name, value, 1) } else { unsetenv(name) }
      }
    }
    let calibration = MetalDirectPtychoCalibration(
      voltageKV: 300, semiangleMrad: 30, scanSamplingRowAngstroms: 0.4,
      scanSamplingColumnAngstroms: 0.4, detectorSamplingRowMrad: 7.5,
      detectorSamplingColumnMrad: 7.5, brightfieldCenterRowPixels: 5.4,
      brightfieldCenterColumnPixels: 5.7, brightfieldRadiusPixels: 3.6)
    for size in [128, 256, 512] {
      let setup = try calibration.geometry(
        detectorRows: 12, detectorColumns: 12,
        detectorSum: Array(repeating: 1, count: 144), scanRows: size, scanColumns: size)
      let count = setup.pixels.count * size * size
      for width in [1, 2, 4] {
        let source = try XCTUnwrap(
          device.makeBuffer(length: count * width, options: .storageModeShared))
        for i in 0..<count {
          let value = UInt32(truncatingIfNeeded: (i &* 7919) ^ (i / size))
          switch width {
          case 1:
            source.contents().assumingMemoryBound(to: UInt8.self)[i] = UInt8(
              truncatingIfNeeded: value)
          case 2:
            source.contents().assumingMemoryBound(to: UInt16.self)[i] = UInt16(
              truncatingIfNeeded: value)
          default: source.contents().assumingMemoryBound(to: UInt32.self)[i] = value
          }
        }
        let type: MetalDirectPtychoCountType = width == 1 ? .uint8 : width == 2 ? .uint16 : .uint32
        let budgets: [Int?] = size == 512 ? [nil, 32 * size * (size / 2 + 1) * 8, 0] : [nil]
        for budget in budgets {
          var reference: [SIMD2<Float>]?
          var scores: [Double]?
          for enabled in ["0", "1"] {
            for name in names { setenv(name, enabled, 1) }
            let engine = try MetalDirectPtychoEngine(
              device: device, geometry: setup.geometry, cacheBudgetBytes: budget)
            try engine.prepare(brightfield: source, countType: type)
            let model = try engine.jointModel(scanDetectorRotationDegrees: 17)
            let evidence = try model.objectiveArrays().evidence
            let measured = try model.agreements([
              DirectPtychoModelPoint(
                c10Angstroms: 47, c12Angstroms: 9, phi12Radians: 0.3,
                sampleTiltMrad: [2, -3], depthSpreadAngstroms: 120)
            ])
            if let reference, let scores {
              XCTAssertTrue(
                evidence == reference, "Evidence changed at size \(size), count width \(width)")
              XCTAssertEqual(measured, scores)
            } else {
              reference = evidence
              scores = measured
            }
          }
        }
      }
    }
  }
}
