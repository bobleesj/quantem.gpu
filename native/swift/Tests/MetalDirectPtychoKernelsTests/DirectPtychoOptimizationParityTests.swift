import Foundation
import Metal
import XCTest

@testable import MetalDirectPtychoKernels

final class DirectPtychoOptimizationParityTests: XCTestCase {
  private func preservingEnvironment(_ names: [String], _ body: () throws -> Void) rethrows {
    let originals = names.map { ProcessInfo.processInfo.environment[$0] }
    defer {
      for (name, value) in zip(names, originals) {
        if let value { setenv(name, value, 1) } else { unsetenv(name) }
      }
    }
    try body()
  }

  /// The half-band consistency check must have the same values and derivatives,
  /// including after cancellation, without leaving the next fit masked.
  func testMaskedBandMatchesUnskippedObjectiveAndRestoresWeights() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    try preservingEnvironment(["QUANTEM_METAL_MASKED_BAND"]) {
      var reference: [[Double]] = []
      for enabled in ["0", "1"] {
        setenv("QUANTEM_METAL_MASKED_BAND", enabled, 1)
        let model = try DirectFifthOrderGradientTests.model(device)
        let coordinates = DirectPtychoFifthOrderCoordinates(
          wavelengthAngstroms: model.wavelengthAngstroms, semiangleRadians: model.semiangleRadians)
        let rows = (0..<7).map { i -> [Double] in
          var z = DirectFifthOrderGradientTests.z
          z[0] += 0.03 * Double(i)
          z[3] -= 0.02 * Double(i)
          z[5] = i == 6 ? 0 : z[5]
          return z
        }
        let points = rows.map { coordinates.point($0) }
        let packed = rows.map { coordinates.pack($0) }
        let original = try model.objectiveArrays().weights
        let masks = [
          Array(repeating: true, count: model.bandCount),
          (0..<model.bandCount).map { $0 % 2 == 0 },
          (0..<model.bandCount).map { $0 % 3 == 1 },
          Array(repeating: false, count: model.bandCount),
        ]
        var measured: [[Double]] = []
        for mask in masks {
          try model.withBandMask(mask) {
            measured.append(try model.fifthOrderAgreements(points, packedCoefficients: packed))
            for stride in [1, 3] {
              let values = try model.fifthOrderAgreementsAndGradients(
                points, packedCoefficients: packed, kStride: stride)
              for value in values {
                measured.append([value.agreement, value.agreement64] + value.gradient)
              }
              let retained = try model.fifthOrderAgreementsAndGradients(
                points, packedCoefficients: packed, kStride: stride, gradients: false)
              let gradients = try model.fifthOrderRetainedGradients(
                points, packedCoefficients: packed, kStride: stride)
              for i in values.indices {
                XCTAssertEqual(retained[i].agreement, values[i].agreement)
                XCTAssertEqual(gradients[i].gradient, values[i].gradient)
              }
            }
          }
          XCTAssertEqual(try model.objectiveArrays().weights, original)
        }
        XCTAssertThrowsError(
          try model.withBandMask(masks[1]) {
            try model.fifthOrderAgreements(
              points, packedCoefficients: packed, isCancelled: { true })
          })
        XCTAssertEqual(try model.objectiveArrays().weights, original)
        XCTAssertThrowsError(try model.withBandMask([false]) { () })
        XCTAssertEqual(try model.objectiveArrays().weights, original)
        measured.append(try model.fifthOrderAgreements(points, packedCoefficients: packed))
        if enabled == "0" {
          reference = measured
        } else {
          XCTAssertTrue(measured == reference, "Masked objectives or derivatives changed")
        }
      }
    }
  }

  /// Gathering fewer, larger batches must preserve every Fourier sample and the
  /// Double normalization, including a partially cached and fully streamed source.
  func testBatchedGatherPreservesEvidenceAndNormalization() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let scene = try DirectModelReconstructionTests.scene()
    let input = try DirectCoherentReconstruction.buffer(device, scene.counts)
    let halfPlaneBytes = scene.scan * (scene.scan / 2 + 1) * 8
    try preservingEnvironment(["QUANTEM_METAL_GATHER_BATCH", "QUANTEM_METAL_BAND_POWER_ROW_MAJOR"])
    {
      for budget in [nil, 32 * halfPlaneBytes, 0] as [Int?] {
        let engine = try MetalDirectPtychoEngine(
          device: device, geometry: scene.geometry, cacheBudgetBytes: budget)
        try engine.prepare(brightfield: input, countType: .uint16)
        var reference: [SIMD2<Float>]?
        var normalization: Double?
        var scores: [Double]?
        for (batch, rowMajor) in [(8, 0), (128, 0), (128, 1)] {
          setenv("QUANTEM_METAL_GATHER_BATCH", String(batch), 1)
          setenv("QUANTEM_METAL_BAND_POWER_ROW_MAJOR", String(rowMajor), 1)
          let model = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
          let evidence = try model.objectiveArrays().evidence
          let measured = try model.agreements([
            DirectPtychoModelPoint(
              c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -0.4,
              sampleTiltMrad: [2.5, -4], depthSpreadAngstroms: 140)
          ])
          if let reference, let normalization, let scores {
            XCTAssertTrue(evidence == reference)
            XCTAssertEqual(model.bandPower, normalization)
            XCTAssertEqual(measured, scores)
          } else {
            reference = evidence
            normalization = model.bandPower
            scores = measured
          }
        }
        XCTAssertThrowsError(
          try engine.jointModel(scanDetectorRotationDegrees: 17.25, isCancelled: { true }))
        // Cancellation must leave the prepared source usable.
        let restored = try engine.jointModel(scanDetectorRotationDegrees: 17.25)
        XCTAssertEqual(restored.bandPower, normalization)
      }
    }
  }
}
