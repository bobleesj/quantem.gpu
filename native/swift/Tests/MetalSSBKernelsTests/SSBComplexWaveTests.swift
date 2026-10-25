import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

/// The native `complexWave` estimator against Python's `phase_estimator="complex_wave"` on the MPS backend
/// (`scripts/export_ssb_complex_wave_fixture.py`, the synthetic scene of `SSBModelReconstructionTests`).
///
/// Stated tolerance: phase within 1e-6 rad (wrapped) and amplitude within 3e-6 of the mean amplitude, the float32
/// rounding of two independent implementations (Python sums the scaled spectrum with fast-math trigonometry,
/// the native redraw evaluates chi as a polynomial and mirrors the anti-Hermitian half). Observed: 2.5e-8 rad
/// and 3.1e-7.
final class SSBComplexWaveTests: XCTestCase {
  struct Fixture: Decodable {
    let shape: [Int]
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let bright_field_pixels: Int
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let scan_detector_rotation_deg: Double
    let aberrations: [String: Double]
  }

  func testComplexWaveMatchesPythonMPS() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let folder = try XCTUnwrap(Bundle.module.resourceURL).appendingPathComponent("Fixtures")
    let fixture = try JSONDecoder().decode(
      Fixture.self, from: Data(contentsOf: folder.appendingPathComponent("ssb_complex_wave_mps.json")))
    let reference = try Data(contentsOf: folder.appendingPathComponent("ssb_complex_wave_mps.f32"))
      .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let s = try SSBModelReconstructionTests.scene()
    XCTAssertEqual(s.pixels.count, fixture.bright_field_pixels)
    XCTAssertEqual(fixture.shape, [s.scan, s.scan])
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalSSBEngine(device: device, geometry: s.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: fixture.scan_detector_rotation_deg)
    let point = SSBModelPoint(
      c10Nanometers: try XCTUnwrap(fixture.aberrations["C10_nm"]),
      c12Nanometers: try XCTUnwrap(fixture.aberrations["C12_nm"]),
      phi12Degrees: try XCTUnwrap(fixture.aberrations["phi12_deg"]))
    for redraw in [true, false] {
      model.redrawReconstruction = redraw
      let result = try model.reconstruct(point, estimator: .complexWave)
      XCTAssertEqual(result.estimator, .complexWave)
      let amplitude = try XCTUnwrap(result.amplitude).contents().assumingMemoryBound(to: Float.self)
      let phase = result.phase.contents().assumingMemoryBound(to: Float.self)
      let count = s.scan * s.scan
      var mean = 0.0
      for i in 0..<count { mean += hypot(Double(reference[2 * i]), Double(reference[2 * i + 1])) }
      mean /= Double(count)
      var phaseError = 0.0
      var amplitudeError = 0.0
      for i in 0..<count {
        let re = Double(reference[2 * i]), im = Double(reference[2 * i + 1])
        var d = Double(phase[i]) - atan2(im, re)
        d -= 2 * .pi * (d / (2 * .pi)).rounded()
        phaseError = max(phaseError, abs(d))
        amplitudeError = max(amplitudeError, abs(Double(amplitude[i]) - hypot(re, im)) / mean)
      }
      print("COMPLEX WAVE redraw \(redraw) phase max \(phaseError) amplitude max/mean \(amplitudeError)")
      XCTAssertLessThan(phaseError, 1e-6)
      XCTAssertLessThan(amplitudeError, 3e-6)
    }
    // Python's restrictions: native sampling, C10/C12, no tilt or depth spread.
    XCTAssertThrowsError(try model.reconstruct(point, upsamplingFactor: 2, estimator: .complexWave))
    var tilted = point
    tilted.sampleTiltMrad = [1, 0]
    XCTAssertThrowsError(try model.reconstruct(tilted, estimator: .complexWave))
    var higher = point
    higher.higherOrder = [.init(order: 3, symmetry: 0, magnitudeAngstroms: 1e4)]
    XCTAssertThrowsError(try model.reconstruct(higher, estimator: .complexWave))
    // The phase estimators return no amplitude.
    XCTAssertNil(try model.reconstruct(point).amplitude)
  }
}
