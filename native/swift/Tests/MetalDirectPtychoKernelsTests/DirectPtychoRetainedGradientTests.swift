import Metal
import XCTest

@testable import MetalDirectPtychoKernels

final class DirectPtychoRetainedGradientTests: XCTestCase {
  func testRetainedGradientDoesNotCrossRotationChanges() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let model = try DirectFifthOrderGradientTests.model(device)
    let fresh = try DirectFifthOrderGradientTests.model(device)
    let coordinates = DirectPtychoFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms, semiangleRadians: model.semiangleRadians)
    let points = [coordinates.point(DirectFifthOrderGradientTests.z)]
    let packed = [coordinates.pack(DirectFifthOrderGradientTests.z)]
    _ = try model.fifthOrderAgreementsAndGradients(points, packedCoefficients: packed, gradients: false)
    try model.retarget(scanDetectorRotationDegrees: 197.25)
    try fresh.retarget(scanDetectorRotationDegrees: 197.25)
    let measured = try model.fifthOrderRetainedGradients(points, packedCoefficients: packed)[0]
    let expected = try fresh.fifthOrderAgreementsAndGradients(points, packedCoefficients: packed)[0]
    XCTAssertEqual(measured.agreement, expected.agreement)
    XCTAssertEqual(measured.agreement64, expected.agreement64)
    XCTAssertEqual(measured.gradient, expected.gradient)
  }

  func testRetainedGradientDoesNotCrossBandMasksOrTheirRestoration() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let model = try DirectFifthOrderGradientTests.model(device)
    let fresh = try DirectFifthOrderGradientTests.model(device)
    let coordinates = DirectPtychoFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms, semiangleRadians: model.semiangleRadians)
    let points = [coordinates.point(DirectFifthOrderGradientTests.z)]
    let packed = [coordinates.pack(DirectFifthOrderGradientTests.z)]
    let mask = (0..<model.bandCount).map { $0 % 2 == 0 }
    _ = try model.fifthOrderAgreementsAndGradients(points, packedCoefficients: packed, gradients: false)
    let measured = try model.withBandMask(mask) {
      try model.fifthOrderRetainedGradients(points, packedCoefficients: packed)[0]
    }
    let expected = try fresh.withBandMask(mask) {
      try fresh.fifthOrderAgreementsAndGradients(points, packedCoefficients: packed)[0]
    }
    XCTAssertEqual(measured.agreement, expected.agreement)
    XCTAssertEqual(measured.agreement64, expected.agreement64)
    XCTAssertEqual(measured.gradient, expected.gradient)
    _ = try model.withBandMask(mask) {
      try model.fifthOrderAgreementsAndGradients(points, packedCoefficients: packed, gradients: false)
    }
    let restored = try model.fifthOrderRetainedGradients(points, packedCoefficients: packed)[0]
    let full = try fresh.fifthOrderAgreementsAndGradients(points, packedCoefficients: packed)[0]
    XCTAssertEqual(restored.agreement, full.agreement)
    XCTAssertEqual(restored.agreement64, full.agreement64)
    XCTAssertEqual(restored.gradient, full.gradient)
  }
  func testLowOrderStagingCannotCrossRotationOrBandChanges() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let model = try DirectFifthOrderGradientTests.model(device)
    let rows: [[Double]] = [[20, 4, -2, 1, -1, 80]]
    let mask = (0..<model.bandCount).map { $0 % 2 == 0 }
    for fused in [false, true] {
      _ = try model.lowOrderValues(rows, withDerivatives: fused)
      try model.retarget(scanDetectorRotationDegrees: model.scanDetectorRotationDegrees + 180)
      XCTAssertThrowsError(try model.lowOrderDerivatives(at: [0]))
      _ = try model.lowOrderValues(rows, withDerivatives: fused)
      try model.withBandMask(mask) {
        XCTAssertThrowsError(try model.lowOrderDerivatives(at: [0]))
        _ = try model.lowOrderValues(rows, withDerivatives: fused)
        XCTAssertNoThrow(try model.lowOrderDerivatives(at: [0]))
      }
      XCTAssertThrowsError(try model.lowOrderDerivatives(at: [0]))
      _ = try model.lowOrderValues(rows, withDerivatives: fused)
      XCTAssertNoThrow(try model.lowOrderDerivatives(at: [0]))
    }
  }

}
