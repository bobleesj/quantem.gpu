import XCTest
@testable import MetalDirectPtychoKernels

final class DirectPtychoFittingBandTests: XCTestCase {
  private func calibration(_ row: Double, _ column: Double) -> MetalDirectPtychoCalibration {
    .init(voltageKV: 300, semiangleMrad: 30,
      scanSamplingRowAngstroms: row, scanSamplingColumnAngstroms: column,
      detectorSamplingRowMrad: 0.5, detectorSamplingColumnMrad: 0.5,
      brightfieldCenterRowPixels: 96, brightfieldCenterColumnPixels: 96)
  }
  func testAtomicScansRetainTheirDefaultBandAndCalibration() throws {
    for size in [128, 256, 512] {
      for step in [0.373, 0.49492961, 0.7662431372549019] {
        let c = calibration(step, step), original = c
        try c.validateFittingBand(scanSize: size)
        XCTAssertEqual(c, original)
      }
    }
  }
  func testCoarseScanRejectsBeforePreparationWithoutChangingMetadata() {
    let c = calibration(4.152888888888889, 4.152888888888889), original = c
    XCTAssertThrowsError(try c.validateFittingBand(scanSize: 512)) { error in
      XCTAssertTrue(error.localizedDescription.contains("0.170"))
      XCTAssertTrue(error.localizedDescription.contains("finer-sampled"))
    }
    XCTAssertEqual(c, original)
  }
  func testAnisotropicScanUsesBothAxes() throws {
    try calibration(4.153, 0.5).validateFittingBand(scanSize: 512)
  }
  func testInvalidSamplingAndEmptyStrictBandAreRejected() {
    for step in [0, -1, Double.nan, Double.infinity] {
      XCTAssertThrowsError(try calibration(step, 0.5).validateFittingBand(scanSize: 512))
    }
    XCTAssertThrowsError(try calibration(0.5, 0.5).validateFittingBand(scanSize: 512, band: 0...0))
  }
}
