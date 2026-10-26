import Foundation
import Metal
import XCTest

@testable import MetalDirectPtychoKernels

/// The three reconstructions of a measured acquisition against CUDA references of the same optics, outside the
/// repository.
///
/// `SSB_DIRECT_FIXTURE` names a folder with `case.json` (calibration, scan and detector shape, selected detector
/// pixels, count type), `counts.bin` (selected BF counts, plane-major `[pixel, scan row, scan column]`, after the
/// loader's flagged-pixel median fill), `detector_sum.u64`, `optics.json` (`optics_sets`: name, `aberrations` with
/// `C10_nm`, `C12_nm`, `phi12_deg`, `higher_order` [{order, symmetry, magnitude_A, angle_rad}], `sample_tilt_mrad`,
/// `depth_spread_nm`, `rotation_deg`) and `references.json` (`references`: file, `amplitude_file`, method,
/// `upsampling`, `factor` "1", "2" or "auto", setting, `requested` regularization, optics) with the CUDA `.npy` arrays.
/// The default presets run unless `SSB_DIRECT_ALL=1`; `SSB_DIRECT_OPTICS` picks one optics set. Without the fixture
/// these tests skip.
///
/// Gates (per reconstruction): phase correlation >= 0.99999 and rms difference <= 1e-4 rad and <= 1 % of the
/// reference's phase std; complex amplitude rms <= 1e-4 of its mean. `ssb` is held to the same phase gates.
final class SSBDirectMeasuredParityTests: XCTestCase {
  struct Case: Decodable {
    let calibration: MetalDirectPtychoCalibration
    let scanRows: Int
    let scanColumns: Int
    let detectorRows: Int
    let detectorColumns: Int
    let pixels: [Int]
    let countType: Int
    let excludedPixels: [Int]?
  }
  struct Optics: Decodable {
    struct Term: Decodable {
      let order: Int
      let symmetry: Int
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let magnitude_A: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let angle_rad: Double
    }
    struct Set: Decodable {
      let name: String
      let aberrations: [String: Double]
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let higher_order: [Term]
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let sample_tilt_mrad: [Double]
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let depth_spread_nm: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let rotation_deg: Double
    }
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let optics_sets: [Set]
  }
  struct References: Decodable {
    struct Entry: Decodable {
      let file: String
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let amplitude_file: String?
      let method: String
      let upsampling: Int
      let factor: String
      let setting: String?
      let optics: String
      let requested: [String: Requested]?
    }
    enum Requested: Decodable {
      case number(Double)
      case text(String)
      init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let number = try? value.decode(Double.self) {
          self = .number(number)
        } else {
          self = .text(try value.decode(String.self))
        }
      }
    }
    let references: [Entry]
  }

  func testThreeMethodsMatchCUDA() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let folder = environment["SSB_DIRECT_FIXTURE"] else {
      throw XCTSkip(
        "Set SSB_DIRECT_FIXTURE to a measured fixture with CUDA ssb/phase/complex references.")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let root = URL(fileURLWithPath: folder)
    let fixture = try JSONDecoder().decode(
      Case.self, from: Data(contentsOf: root.appendingPathComponent("case.json")))
    let optics = try JSONDecoder().decode(
      Optics.self, from: Data(contentsOf: root.appendingPathComponent("optics.json")))
    let references = try JSONDecoder().decode(
      References.self, from: Data(contentsOf: root.appendingPathComponent("references.json")))
    let sums = try Data(contentsOf: root.appendingPathComponent("detector_sum.u64"))
      .withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
    let setup = try fixture.calibration.geometry(
      detectorRows: fixture.detectorRows, detectorColumns: fixture.detectorColumns,
      detectorSum: sums,
      excludedPixels: Set(fixture.excludedPixels ?? []), scanRows: fixture.scanRows,
      scanColumns: fixture.scanColumns)
    XCTAssertEqual(setup.pixels, fixture.pixels, "native BF selection differs from the fixture's")
    let counts = try MeasuredSSBFixture.mapBuffer(
      device: device, url: root.appendingPathComponent("counts.bin"))
    let engine = try MetalDirectPtychoEngine(device: device, geometry: setup.geometry)
    try engine.prepare(
      brightfield: counts,
      countType: try XCTUnwrap(MetalDirectPtychoCountType(rawValue: fixture.countType)))
    let auto = try fixture.calibration.ssbUpsamplingFactor()
    let all = environment["SSB_DIRECT_ALL"] == "1"
    var checked = 0
    for set in optics.optics_sets
    where environment["SSB_DIRECT_OPTICS"].map({ $0 == set.name }) ?? true {
      let a = set.aberrations
      let point = DirectPtychoModelPoint(
        c10Nanometers: try XCTUnwrap(a["C10_nm"]), c12Nanometers: try XCTUnwrap(a["C12_nm"]),
        phi12Degrees: try XCTUnwrap(a["phi12_deg"]), sampleTiltMrad: set.sample_tilt_mrad,
        depthSpreadNanometers: set.depth_spread_nm,
        higherOrder: set.higher_order.map {
          DirectPtychoModelHigherOrder(
            order: $0.order, symmetry: $0.symmetry, magnitudeAngstroms: $0.magnitude_A,
            angleRadians: $0.angle_rad)
        })
      let model = try engine.jointModel(scanDetectorRotationDegrees: set.rotation_deg)
      for entry in references.references where entry.optics == set.name {
        guard all || entry.setting == nil || entry.setting == "default" else { continue }
        let method = try XCTUnwrap(DirectPtychoReconstructionMethod(rawValue: entry.method))
        if entry.factor == "auto" {
          XCTAssertEqual(entry.upsampling, auto, "Auto factor differs from CUDA's")
        }
        func number(_ key: String, _ fallback: Double) -> Double {
          if case .number(let value) = entry.requested?[key] { return value }
          return fallback
        }
        var weighting = DirectPtychoDetectorWeighting.uniform
        if case .text(let value) = entry.requested?["detector_weighting"] {
          weighting = try XCTUnwrap(DirectPtychoDetectorWeighting(rawValue: value))
        }
        let regularization = DirectPtychoRegularization(
          phaseDamping: number("phase_damping", 0.03),
          amplitudeDamping: number("amplitude_damping", 0.03),
          smoothnessInverseAngstroms: number("smoothness_inv_A", 1), detectorWeighting: weighting)
        let result = try model.reconstruct(
          point, upsamplingFactor: entry.upsampling, method: method, regularization: regularization)
        let count = result.rows * result.columns
        let phase = try npy(root.appendingPathComponent(entry.file))
        XCTAssertEqual(phase.count, count, entry.file)
        guard phase.count == count else { continue }
        let p = Self.compare(result.phase, phase, wrap: true)
        var line =
          "DIRECT_PARITY \(set.name) \(entry.file): phase corr \(p.correlation) rms \(p.rms) (std \(p.std))"
        XCTAssertGreaterThanOrEqual(p.correlation, 0.99999, entry.file)
        XCTAssertLessThanOrEqual(p.rms, 1e-4, entry.file)
        XCTAssertLessThanOrEqual(p.rms, 0.01 * p.std, entry.file)
        if let file = entry.amplitude_file {
          let reference = try npy(root.appendingPathComponent(file))
          let amplitude = try XCTUnwrap(result.amplitude, "complex result without amplitude")
          let q = Self.compare(amplitude, reference, wrap: false)
          line += " amplitude rms/mean \(q.rms / q.mean)"
          XCTAssertLessThanOrEqual(q.rms, 1e-4 * q.mean, file)
        } else {
          XCTAssertNil(result.amplitude)
        }
        print(line)
        checked += 1
      }
    }
    XCTAssertGreaterThan(checked, 0, "no reference matched")
  }

  /// Correlation, rms difference (wrapped for phases), the reference's std and mean.
  static func compare(_ buffer: MTLBuffer, _ reference: [Float], wrap: Bool) -> (
    correlation: Double, rms: Double, std: Double, mean: Double
  ) {
    let actual = buffer.contents().assumingMemoryBound(to: Float.self)
    let n = Double(reference.count)
    var meanA = 0.0
    var meanB = 0.0
    for i in reference.indices {
      meanA += Double(actual[i])
      meanB += Double(reference[i])
    }
    meanA /= n
    meanB /= n
    var ab = 0.0
    var aa = 0.0
    var bb = 0.0
    var dd = 0.0
    for i in reference.indices {
      let a = Double(actual[i]) - meanA
      let b = Double(reference[i]) - meanB
      ab += a * b
      aa += a * a
      bb += b * b
      var d = Double(actual[i]) - Double(reference[i])
      if wrap { d -= 2 * .pi * (d / (2 * .pi)).rounded() }
      dd += d * d
    }
    return (ab / (aa * bb).squareRoot(), (dd / n).squareRoot(), (bb / n).squareRoot(), meanB)
  }
}
