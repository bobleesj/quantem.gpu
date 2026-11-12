import Foundation
import Metal
import XCTest

@testable import MetalDirectPtychoKernels

final class DirectPtychoIndexedBandTests: XCTestCase {
  /// Cross a resident chunk boundary, retain signed fractional evidence and exercise
  /// both rectangular orientations and the partial-cache fallback.
  func testResidentEvidenceMatchesGatheredAcrossChunksAndCacheBudgets() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    guard device.argumentBuffersSupport == .tier2 else { throw XCTSkip("Tier 2 Metal required") }
    let key = "QUANTEM_METAL_INDEXED_BAND"
    let original = ProcessInfo.processInfo.environment[key]
    defer {
      if let original { setenv(key, original, 1) } else { unsetenv(key) }
    }
    let calibration = MetalDirectPtychoCalibration(
      voltageKV: 300, semiangleMrad: 30, scanSamplingRowAngstroms: 0.4,
      scanSamplingColumnAngstroms: 0.47, detectorSamplingRowMrad: 2.2,
      detectorSamplingColumnMrad: 2.2, brightfieldCenterRowPixels: 15.4,
      brightfieldCenterColumnPixels: 15.7, brightfieldRadiusPixels: 13.4)
    for (rows, columns) in [(32, 48), (48, 32), (64, 80)] {
      let setup = try calibration.geometry(
        detectorRows: 32, detectorColumns: 32,
        detectorSignalSum: Array(repeating: 100, count: 32 * 32),
        scanRows: rows, scanColumns: columns)
      XCTAssertGreaterThan(setup.pixels.count, 512)
      let count = setup.pixels.count * rows * columns
      let source = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModeShared))
      let values = source.contents().assumingMemoryBound(to: Float.self)
      for i in 0..<count { values[i] = Float((i &* 7919 ^ (i / columns)) % 257 - 32) / 8 }
      let halfPlaneBytes = rows * (columns / 2 + 1) * 8
      for budget: Int? in [nil, 32 * halfPlaneBytes, 0] {
        let engine = try MetalDirectPtychoEngine(
          device: device, geometry: setup.geometry, cacheBudgetBytes: budget)
        try engine.prepare(brightfield: source, countType: .float32)
        setenv(key, "0", 1)
        let reference = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
        setenv(key, "1", 1)
        let candidate = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
        XCTAssertEqual(candidate.bandPower, reference.bandPower)
        XCTAssertEqual(try candidate.objectiveArrays().evidence, try reference.objectiveArrays().evidence)
        if budget == nil {
          XCTAssertLessThan(candidate.gatheredEvidenceBytes, reference.gatheredEvidenceBytes)
        } else {
          XCTAssertEqual(candidate.gatheredEvidenceBytes, reference.gatheredEvidenceBytes)
        }
        let coordinates = DirectPtychoFifthOrderCoordinates(
          wavelengthAngstroms: candidate.wavelengthAngstroms,
          semiangleRadians: candidate.semiangleRadians)
        let points = [coordinates.point(DirectFifthOrderGradientTests.z)]
        let packed = [coordinates.pack(DirectFifthOrderGradientTests.z)]
        for rotation in [17.25, 197.25] {
          try reference.retarget(scanDetectorRotationDegrees: rotation)
          try candidate.retarget(scanDetectorRotationDegrees: rotation)
          XCTAssertEqual(try candidate.agreements(points), try reference.agreements(points))
          for stride in [1, 8, 64] {
            let expected = try reference.fifthOrderAgreementsAndGradients(
              points, packedCoefficients: packed, kStride: stride)[0]
            let measured = try candidate.fifthOrderAgreementsAndGradients(
              points, packedCoefficients: packed, kStride: stride)[0]
            XCTAssertEqual(measured.agreement, expected.agreement)
            XCTAssertEqual(measured.agreement64, expected.agreement64)
            XCTAssertEqual(measured.gradient, expected.gradient)
          }
          for method: DirectPtychoReconstructionMethod in [.phaseOfMean, .phase, .complex] {
            let expected: MetalDirectPtychoJointPhase
            do {
              expected = try reference.reconstruct(points[0], upsamplingFactor: 2, method: method)
            } catch DirectPtychoSolveError.diverged {
              // This signed synthetic spectrum is not a weak object. Its opposite-branch
              // Complex amplitude can overflow; the indexed path must reject it identically.
              XCTAssertEqual(method, .complex)
              XCTAssertThrowsError(try candidate.reconstruct(
                points[0], upsamplingFactor: 2, method: method)) { error in
                XCTAssertEqual(error as? DirectPtychoSolveError, .diverged)
              }
              continue
            }
            let measured = try candidate.reconstruct(points[0], upsamplingFactor: 2, method: method)
            let n = rows * columns * 4
            let a = UnsafeBufferPointer(start: expected.phase.contents().assumingMemoryBound(to: Float.self), count: n)
            let b = UnsafeBufferPointer(start: measured.phase.contents().assumingMemoryBound(to: Float.self), count: n)
            XCTAssertEqual(Array(a), Array(b))
          }
        }
        XCTAssertThrowsError(try candidate.agreements(points, isCancelled: { true }))
        try engine.prepare(brightfield: source, countType: .float32)
        XCTAssertThrowsError(try candidate.agreements(points))
      }
    }
  }
}
