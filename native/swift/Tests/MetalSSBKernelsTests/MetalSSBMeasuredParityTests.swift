import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

/// A measured acquisition exported by the Python MPS path, outside the repository.
///
/// `SSB_PARITY_FIXTURE` names a folder with `case.json` (calibration, selected detector pixels, rotation estimate),
/// `counts.u8` (selected BF counts, plane-major `[pixel, scan row, scan column]`, after the loader's flagged-pixel
/// median fill), `detector_sum.u64` and `dark_field.f32`. `SSB_PARITY_REFERENCE` names the MPS reference folder:
/// `reference.json` and the `fixed_<name>_<factor>x.npy` phases. Without them these tests skip.
// Decodable properties retain the external Python fixture keys verbatim.
struct MeasuredSSBFixture {
  struct Case: Decodable {
    struct Calibration: Decodable {
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let voltage_kV: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let semiangle_mrad: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let scan_sampling_A: [Double]
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let detector_sampling_mrad: [Double]
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let bright_field_center_px: [Double]
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let bright_field_radius_px: Double
    }
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let scan_shape: [Int]
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let detector_shape: [Int]
    let calibration: Calibration
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let edge_radius_px: Double
    let pixels: [Int]
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let dc_value: Double
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let rotation_estimate_deg: Double
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let fixed_agreement: [String: Double]
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let band_count: Int
  }
  struct Reference: Decodable {
    struct Fixed: Decodable {
      struct Aberrations: Decodable {
        // swift-format-ignore: AlwaysUseLowerCamelCase
        let C10_nm: Double
        // swift-format-ignore: AlwaysUseLowerCamelCase
        let C12_nm: Double
        // swift-format-ignore: AlwaysUseLowerCamelCase
        let phi12_deg: Double
      }
      let rotation: Double
      let aberrations: Aberrations
      let tilt: [Double]
      let depth: Double
    }
    struct Branch: Decodable {
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let rotation_deg: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let C10_nm: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let C12_nm: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let phi12_deg: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let column_z: Double?
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let column_effect: Double?
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let phase_skewness: Double
    }
    struct Default: Decodable {
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let decided_by: String
      let branches: [Branch]
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let rotation_deg: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let C10_nm: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let C12_nm: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let phi12_deg: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let sample_tilt_mrad: [Double]
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let depth_spread_nm: Double
      let kept: Int
    }
    struct Standard: Decodable {
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let C10_nm: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let C12_nm: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let phi12_deg: Double
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let sample_agreement: Double?
    }
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let fixed_parameters: [String: Fixed]
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let fixed_loss: [String: Double]
    let `default`: Default
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let model_tilt0: Standard
    // swift-format-ignore: AlwaysUseLowerCamelCase
    let rotation_estimate_deg: Double
  }

  let root: URL
  let reference: URL
  let fixture: Case
  let expected: Reference
  let calibration: MetalSSBCalibration
  let detectorSum: [UInt64]
  let counts: MTLBuffer

  static func load(device: MTLDevice) throws -> MeasuredSSBFixture {
    let environment = ProcessInfo.processInfo.environment
    guard let folder = environment["SSB_PARITY_FIXTURE"],
      let reference = environment["SSB_PARITY_REFERENCE"]
    else {
      throw XCTSkip("Set SSB_PARITY_FIXTURE and SSB_PARITY_REFERENCE to a measured MPS export.")
    }
    let root = URL(fileURLWithPath: folder)
    let referenceRoot = URL(fileURLWithPath: reference)
    let fixture = try JSONDecoder().decode(
      Case.self, from: Data(contentsOf: root.appendingPathComponent("case.json")))
    let expected = try JSONDecoder().decode(
      Reference.self, from: Data(contentsOf: referenceRoot.appendingPathComponent("reference.json"))
    )
    let c = fixture.calibration
    let calibration = MetalSSBCalibration(
      voltageKV: c.voltage_kV, semiangleMrad: c.semiangle_mrad,
      scanSamplingRowAngstroms: c.scan_sampling_A[0],
      scanSamplingColumnAngstroms: c.scan_sampling_A[1],
      detectorSamplingRowMrad: c.detector_sampling_mrad[0],
      detectorSamplingColumnMrad: c.detector_sampling_mrad[1],
      brightfieldCenterRowPixels: c.bright_field_center_px[0],
      brightfieldCenterColumnPixels: c.bright_field_center_px[1],
      brightfieldRadiusPixels: c.bright_field_radius_px)
    let sums = try Data(contentsOf: root.appendingPathComponent("detector_sum.u64"))
      .withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
    let counts = try mapBuffer(device: device, url: root.appendingPathComponent("counts.u8"))
    return MeasuredSSBFixture(
      root: root, reference: referenceRoot, fixture: fixture, expected: expected,
      calibration: calibration, detectorSum: sums, counts: counts)
  }

  /// The prepared native engine of this acquisition, in the order the Python selection lists its pixels.
  func engine(device: MTLDevice) throws -> (engine: MetalSSBEngine, geometry: MetalSSBGeometry) {
    let setup = try calibration.geometry(
      detectorRows: fixture.detector_shape[0], detectorColumns: fixture.detector_shape[1],
      detectorSum: detectorSum, scanRows: fixture.scan_shape[0], scanColumns: fixture.scan_shape[1])
    XCTAssertEqual(
      setup.pixels, fixture.pixels, "native BF selection differs from the exported one")
    XCTAssertEqual(Double(setup.geometry.dcValue.x), fixture.dc_value, accuracy: 0.5)
    let engine = try MetalSSBEngine(device: device, geometry: setup.geometry)
    try engine.prepare(brightfield: counts, countType: .uint8)
    return (engine, setup.geometry)
  }

  func phase(_ name: String) throws -> [Float] {
    try npy(reference.appendingPathComponent(name))
  }

  func floats(_ name: String) throws -> [Float] {
    try Data(contentsOf: root.appendingPathComponent(name)).withUnsafeBytes {
      Array($0.bindMemory(to: Float.self))
    }
  }

  func darkField() throws -> [Float] {
    try Data(contentsOf: root.appendingPathComponent("dark_field.f32"))
      .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  }

  static func mapBuffer(device: MTLDevice, url: URL) throws -> MTLBuffer {
    let handle = open(url.path, O_RDONLY)
    guard handle >= 0 else { throw XCTSkip("Cannot open \(url.lastPathComponent)") }
    defer { close(handle) }
    var info = stat()
    guard fstat(handle, &info) == 0 else { throw XCTSkip("Cannot read \(url.lastPathComponent)") }
    let length = Int(info.st_size)
    let page = Int(getpagesize())
    let mapped = (length + page - 1) / page * page
    guard let pointer = mmap(nil, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE, handle, 0),
      pointer != MAP_FAILED
    else { throw XCTSkip("Cannot map \(url.lastPathComponent)") }
    guard
      let buffer = device.makeBuffer(
        bytesNoCopy: pointer, length: mapped, options: .storageModeShared,
        deallocator: { pointer, length in munmap(pointer, length) })
    else { throw XCTSkip("Cannot wrap \(url.lastPathComponent) in a Metal buffer") }
    return buffer
  }
}

/// Little-endian float32 C-order `.npy` (version 1 or 2).
func npy(_ url: URL) throws -> [Float] {
  let data = try Data(contentsOf: url)
  guard data.count > 12, data.prefix(6) == Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]) else {
    throw XCTSkip("\(url.lastPathComponent) is not a NumPy file")
  }
  let major = data[6]
  let headerLength =
    major == 1
    ? Int(data[8]) | Int(data[9]) << 8
    : Int(data[8]) | Int(data[9]) << 8 | Int(data[10]) << 16 | Int(data[11]) << 24
  let start = (major == 1 ? 10 : 12) + headerLength
  let header = String(decoding: data[(major == 1 ? 10 : 12)..<start], as: UTF8.self)
  XCTAssertTrue(header.contains("'<f4'") && header.contains("'fortran_order': False"), header)
  return data.suffix(from: start).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

struct PhaseComparison: CustomStringConvertible {
  let correlation: Double
  let maximumDifference: Double
  let relativeL2: Double
  var description: String {
    String(
      format: "correlation %.9f · max |Δ| %.3e rad · relative L2 %.3e", correlation,
      maximumDifference, relativeL2)
  }

  init(_ actual: UnsafePointer<Float>, _ expected: [Float]) {
    var meanA = 0.0
    var meanB = 0.0
    for i in expected.indices {
      meanA += Double(actual[i])
      meanB += Double(expected[i])
    }
    meanA /= Double(expected.count)
    meanB /= Double(expected.count)
    var ab = 0.0
    var aa = 0.0
    var bb = 0.0
    var maximum = 0.0
    var error = 0.0
    var norm = 0.0
    for i in expected.indices {
      let a = Double(actual[i]) - meanA
      let b = Double(expected[i]) - meanB
      ab += a * b
      aa += a * a
      bb += b * b
      let delta = Double(actual[i]) - Double(expected[i])
      maximum = max(maximum, abs(delta))
      error += delta * delta
      norm += Double(expected[i]) * Double(expected[i])
    }
    correlation = ab / sqrt(aa * bb)
    maximumDifference = maximum
    relativeL2 = sqrt(error / norm)
  }
}

extension MeasuredSSBFixture {
  func point(_ name: String) throws -> (rotation: Double, point: SSBModelPoint) {
    let fixed = try XCTUnwrap(expected.fixed_parameters[name])
    return (
      fixed.rotation,
      SSBModelPoint(
        c10Nanometers: fixed.aberrations.C10_nm, c12Nanometers: fixed.aberrations.C12_nm,
        phi12Degrees: fixed.aberrations.phi12_deg, sampleTiltMrad: fixed.tilt,
        depthSpreadNanometers: fixed.depth)
    )
  }
}

func agreementEvaluator(_ model: MetalSSBJointModel) -> SSBAgreementSearch.Evaluate {
  { points in
    try model.agreements(
      points.map {
        SSBModelPoint(
          c10Angstroms: $0[0], c12Angstroms: $0[1], phi12Radians: $0[2],
          sampleTiltMrad: [$0[3], $0[4]],
          depthSpreadAngstroms: $0[5])
      })
  }
}

final class MetalSSBMeasuredParityTests: XCTestCase {
  /// The BF disk and the fitted detector sampling from the median-filled detector sums
  /// (`fit_detector_sampling`: semiangle / the half-plateau edge radius).
  func testBrightfieldDiskMatchesMPS() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try MeasuredSSBFixture.load(device: device)
    let shape = fixture.fixture.detector_shape
    let disk = try MetalSSBBrightfieldEstimate.measure(
      device: device, detectorSum: fixture.detectorSum, rows: shape[0], columns: shape[1])
    let c = fixture.fixture.calibration
    let sampling = try disk.detectorSampling(semiangle: c.semiangle_mrad)
    print(
      String(
        format: "MEASURED_DISK centre (%.10f, %.10f) radius %.10f edge %.10f sampling %.10f",
        disk.centerRow, disk.centerColumn, disk.selectionRadius, disk.edgeRadius, sampling))
    XCTAssertEqual(disk.centerRow, c.bright_field_center_px[0], accuracy: 1e-9)
    XCTAssertEqual(disk.centerColumn, c.bright_field_center_px[1], accuracy: 1e-9)
    XCTAssertEqual(disk.selectionRadius, c.bright_field_radius_px, accuracy: 1e-9)
    XCTAssertEqual(disk.edgeRadius, fixture.fixture.edge_radius_px, accuracy: 1e-9)
    XCTAssertEqual(sampling, c.detector_sampling_mrad[0], accuracy: 1e-12)
  }

  /// The agreement objective at the fixed parameters against the MPS resident evaluation.
  func testAgreementMatchesMPS() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try MeasuredSSBFixture.load(device: device)
    let (engine, _) = try fixture.engine(device: device)
    for name in ["tilt_corrected", "model_tilt0", "phase_variance"] {
      let (rotation, point) = try fixture.point(name)
      let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)
      XCTAssertEqual(model.bandCount, fixture.fixture.band_count)
      let started = Date()
      let agreement = try model.agreements([point])[0]
      let expected = try XCTUnwrap(fixture.fixture.fixed_agreement[name])
      let relative = abs(agreement - expected) / expected
      print(
        String(
          format: "MEASURED_AGREEMENT %@ native %.6f MPS %.6f relative %.3e · %.1f ms",
          name, agreement, expected, relative, Date().timeIntervalSince(started) * 1000))
      XCTAssertLessThan(relative, 1e-6, name)
    }
  }

  /// The default fit (tilt-corrected, rotation given), the tilt=False model fit and the standard comparison.
  func testModelFitsMatchMPS() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try MeasuredSSBFixture.load(device: device)
    let (engine, _) = try fixture.engine(device: device)
    let reference = fixture.expected.default
    let model = try engine.jointModel(scanDetectorRotationDegrees: reference.rotation_deg)
    let evaluate = agreementEvaluator(model)
    var started = Date()
    let tilt = try SSBAgreementSearch.fitTilt(evaluate)
    let tiltSeconds = Date().timeIntervalSince(started)
    let p = tilt.point
    print(
      String(
        format:
          "MEASURED_FIT tilt C10 %.4f C12 %.4f phi12 %.3f tilt (%.4f, %.4f) depth %.4f nm agreement %.6e · search %.2f s refine %.2f s total %.2f s",
        p[0] / 10, p[1] / 10, p[2] * 180 / .pi, p[3], p[4], p[5] / 10, tilt.agreement,
        tilt.searchSeconds, tilt.refinementSeconds, tiltSeconds))
    XCTAssertEqual(p[0] / 10, reference.C10_nm, accuracy: 0.01)
    XCTAssertEqual(p[1] / 10, reference.C12_nm, accuracy: 0.01)
    XCTAssertEqual(p[2] * 180 / .pi, reference.phi12_deg, accuracy: 0.2)
    XCTAssertEqual(p[3], reference.sample_tilt_mrad[0], accuracy: 0.02)
    XCTAssertEqual(p[4], reference.sample_tilt_mrad[1], accuracy: 0.02)
    XCTAssertEqual(p[5] / 10, reference.depth_spread_nm, accuracy: 0.05)
    started = Date()
    let standard = try SSBAgreementSearch.fitStandard(evaluate)
    let s = standard.point
    let control = fixture.expected.model_tilt0
    print(
      String(
        format: "MEASURED_FIT tilt=False C10 %.4f C12 %.4f phi12 %.3f nm agreement %.6e · %.2f s",
        s[0] / 10, s[1] / 10, s[2] * 180 / .pi, standard.agreement,
        Date().timeIntervalSince(started)))
    XCTAssertEqual(s[0] / 10, control.C10_nm, accuracy: 0.01)
    XCTAssertEqual(s[1] / 10, control.C12_nm, accuracy: 0.01)
    XCTAssertEqual(s[2] * 180 / .pi, control.phi12_deg, accuracy: 0.2)
    if let output = ProcessInfo.processInfo.environment["SSB_PARITY_OUTPUT"] {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(["tilt": tilt, "standard": standard]).write(
        to: URL(fileURLWithPath: output).appendingPathComponent("native_fits.json"))
    }
  }

  /// The fixed-parameter phases of all three methods, 1x and 3x, against the MPS reference.
  func testFixedParameterPhaseMatchesMPS() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try MeasuredSSBFixture.load(device: device)
    let (engine, _) = try fixture.engine(device: device)
    var report: [String: Any] = [:]
    for name in ["tilt_corrected", "model_tilt0", "phase_variance"] {
      let fixed = try XCTUnwrap(fixture.expected.fixed_parameters[name])
      let model = try engine.jointModel(scanDetectorRotationDegrees: fixed.rotation)
      let parameters = SSBJointParameters(
        aberrations: .init(
          c10Angstroms: Float(fixed.aberrations.C10_nm * 10),
          c12Angstroms: Float(fixed.aberrations.C12_nm * 10),
          phi12Radians: Float(fixed.aberrations.phi12_deg * .pi / 180)),
        depthAngstroms: Float(fixed.depth * 10), tiltRowMrad: Float(fixed.tilt[0]),
        tiltColumnMrad: Float(fixed.tilt[1]))
      for factor in [1, 3] {
        let phase = try model.reconstruct(parameters, upsamplingFactor: factor)
        let expected = try fixture.phase("fixed_\(name)_\(factor)x.npy")
        XCTAssertEqual(phase.rows * phase.columns, expected.count)
        let comparison = PhaseComparison(
          phase.phase.contents().assumingMemoryBound(to: Float.self), expected)
        print(
          "MEASURED_PHASE \(name) \(factor)x \(comparison) · \(String(format: "%.1f", phase.wallSeconds * 1000)) ms"
        )
        report["\(name)_\(factor)x"] = [
          "correlation": comparison.correlation,
          "max_abs_difference_rad": comparison.maximumDifference,
          "relative_l2": comparison.relativeL2, "wall_ms": phase.wallSeconds * 1000,
        ]
        if let output = ProcessInfo.processInfo.environment["SSB_PARITY_OUTPUT"] {
          try Data(bytes: phase.phase.contents(), count: expected.count * 4).write(
            to: URL(fileURLWithPath: output).appendingPathComponent("native_\(name)_\(factor)x.f32")
          )
        }
        XCTAssertGreaterThanOrEqual(comparison.correlation, 0.99999, "\(name) \(factor)x")
        // The separable kernels against the per-term reference kernel on the same session.
        model.separableReconstruction = false
        let reference = try model.reconstruct(parameters, upsamplingFactor: factor)
        model.separableReconstruction = true
        let a = phase.phase.contents().assumingMemoryBound(to: Float.self)
        let b = reference.phase.contents().assumingMemoryBound(to: Float.self)
        var largest: Float = 0
        for i in 0..<(phase.rows * phase.columns) { largest = max(largest, abs(a[i] - b[i])) }
        var times: [Double] = []
        for _ in 0..<5 {
          times.append(try model.reconstruct(parameters, upsamplingFactor: factor).wallSeconds)
        }
        times.sort()
        print(
          String(
            format:
              "MEASURED_SEPARABLE %@ %dx max |separable - per-term| %.3e rad · median %.1f ms (per-term %.1f ms)",
            name, factor, largest, times[2] * 1000, reference.wallSeconds * 1000))
        report["\(name)_\(factor)x_separable"] = [
          "max_vs_per_term_rad": Double(largest), "median_ms": times[2] * 1000,
        ]
      }
    }
    if let output = ProcessInfo.processInfo.environment["SSB_PARITY_OUTPUT"] {
      let data = try JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("kernel_parity.json"))
    }
  }

  /// `SSB.reconstruct()`'s default on the measured scan: the rotation from the centre-of-mass curl, both
  /// 180-degree branches fitted tilt-corrected, the dark-field column test.
  func testDefaultBranchedFitMatchesMPS() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try MeasuredSSBFixture.load(device: device)
    let (engine, _) = try fixture.engine(device: device)
    let (rows, columns) = (fixture.fixture.scan_shape[0], fixture.fixture.scan_shape[1])
    let rotation = SSBRotationBranches.curlRotationDegrees(
      row: try fixture.floats("com_row.f32"), column: try fixture.floats("com_col.f32"), rows: rows,
      columns: columns)
    print(
      String(
        format: "MEASURED_ROTATION native %.13f MPS %.13f", rotation,
        fixture.expected.rotation_estimate_deg))
    XCTAssertEqual(rotation, fixture.expected.rotation_estimate_deg, accuracy: 1e-9)
    let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)
    let started = Date()
    let result = try SSBModelFit.findAberrations(
      model: model, engine: engine, rotationDegrees: rotation, darkField: try fixture.darkField(),
      scanSampling: (
        fixture.fixture.calibration.scan_sampling_A[0],
        fixture.fixture.calibration.scan_sampling_A[1]
      ))
    let seconds = Date().timeIntervalSince(started)
    let reference = fixture.expected.default
    XCTAssertEqual(result.kept, reference.kept)
    XCTAssertEqual(result.decidedBy.rawValue, reference.decided_by)
    for (index, (fit, branch)) in zip(result.fits, reference.branches).enumerated() {
      let test = result.branches[index]
      print(
        String(
          format:
            "MEASURED_BRANCH %d rotation %.4f C10 %.4f C12 %.4f phi12 %.3f tilt (%.4f, %.4f) depth %.4f · z %.3f (MPS %.3f) effect %.4f (MPS %.4f) skew %.4f (MPS %.4f)",
          index, fit.scanDetectorRotationDegrees, fit.point.c10Nanometers, fit.point.c12Nanometers,
          fit.point.phi12Degrees, fit.point.sampleTiltMrad[0], fit.point.sampleTiltMrad[1],
          fit.point.depthSpreadNanometers, test.column?.z ?? .nan, branch.column_z ?? .nan,
          test.column?.effect ?? .nan, branch.column_effect ?? .nan, test.phaseSkewness,
          branch.phase_skewness))
      XCTAssertEqual(fit.scanDetectorRotationDegrees, branch.rotation_deg, accuracy: 1e-9)
      XCTAssertEqual(fit.point.c10Nanometers, branch.C10_nm, accuracy: 0.01)
      XCTAssertEqual(fit.point.c12Nanometers, branch.C12_nm, accuracy: 0.01)
      XCTAssertEqual(fit.point.phi12Degrees, branch.phi12_deg, accuracy: 0.2)
    }
    let kept = result.fit.point
    XCTAssertEqual(kept.sampleTiltMrad[0], reference.sample_tilt_mrad[0], accuracy: 0.02)
    XCTAssertEqual(kept.sampleTiltMrad[1], reference.sample_tilt_mrad[1], accuracy: 0.02)
    XCTAssertEqual(kept.depthSpreadNanometers, reference.depth_spread_nm, accuracy: 0.05)
    print(
      String(
        format: "MEASURED_DEFAULT %.2f s · gain %.4f · timings %@", seconds,
        result.fit.tiltFitGain ?? .nan,
        result.fits.map { $0.timings.description }.joined(separator: " | ")))
    if let output = ProcessInfo.processInfo.environment["SSB_PARITY_OUTPUT"] {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(result).write(
        to: URL(fileURLWithPath: output).appendingPathComponent("native_default.json"))
    }
  }

  /// BENCHMARK.md steps 2-7 and 9 on the exported acquisition: medians of three fits and twenty previews,
  /// written to `SSB_PARITY_OUTPUT/benchmark_native.json`. Runs only with SSB_BENCHMARK=1.
  func testBenchmarkSteps() throws {
    guard ProcessInfo.processInfo.environment["SSB_BENCHMARK"] == "1" else {
      throw XCTSkip("Set SSB_BENCHMARK=1 to time the BENCHMARK.md steps.")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try MeasuredSSBFixture.load(device: device)
    var peak = device.currentAllocatedSize
    func sample() { peak = max(peak, device.currentAllocatedSize) }
    func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
    var prepare: [Double] = []
    var session: (engine: MetalSSBEngine, geometry: MetalSSBGeometry)?
    for _ in 0..<3 {
      let started = Date()
      session = try fixture.engine(device: device)
      prepare.append(Date().timeIntervalSince(started))
      sample()
    }
    let engine = try XCTUnwrap(session).engine
    let rotation = fixture.expected.default.rotation_deg
    let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)
    let sampling = (
      fixture.fixture.calibration.scan_sampling_A[0], fixture.fixture.calibration.scan_sampling_A[1]
    )
    func time(_ body: () throws -> Void) rethrows -> Double {
      let started = Date()
      try body()
      sample()
      return Date().timeIntervalSince(started)
    }
    var results: [String: Any] = ["gpu": device.name]
    var step3: [Double] = []
    var step3b: [Double] = []
    var step4: [Double] = []
    var step5: [Double] = []
    var split: [String: Double] = [:]
    for _ in 0..<3 {
      step3.append(
        try time {
          let fit = try SSBModelFit.findAberrations(
            model: model, engine: engine, rotationDegrees: rotation, darkField: nil,
            scanSampling: sampling,
            checkRotation: false)
          split = fit.fit.timings
        })
      step3b.append(
        try time {
          _ = try SSBModelFit.findAberrations(
            model: model, engine: engine, rotationDegrees: rotation, darkField: nil,
            scanSampling: sampling,
            options: SSBFitOptions(compareStandard: false), checkRotation: false)
        })
      step4.append(
        try time {
          _ = try SSBModelFit.findAberrations(
            model: model, engine: engine, rotationDegrees: fixture.expected.rotation_estimate_deg,
            darkField: try fixture.darkField(), scanSampling: sampling)
        })
      step5.append(
        try time {
          try model.retarget(scanDetectorRotationDegrees: rotation)
          _ = try SSBModelFit.fit(model: model, engine: engine, options: SSBFitOptions(tilt: false))
        })
    }
    try model.retarget(scanDetectorRotationDegrees: rotation)
    let (_, point) = try fixture.point("tilt_corrected")
    var previews: [Int: [Double]] = [1: [], 3: []]
    for factor in [1, 3] {
      _ = try model.reconstruct(point, upsamplingFactor: factor)
      for _ in 0..<20 {
        previews[factor]!.append(
          try time { _ = try model.reconstruct(point, upsamplingFactor: factor) })
      }
    }
    results["2"] = ["s": median(prepare), "runs": prepare]
    results["3"] = ["s": median(step3), "runs": step3, "split_s": split]
    results["3b"] = ["s": median(step3b), "runs": step3b]
    results["4"] = ["s": median(step4), "runs": step4]
    results["5"] = ["s": median(step5), "runs": step5]
    results["6"] = ["ms": median(previews[1]!) * 1000]
    results["7"] = ["ms": median(previews[3]!) * 1000]
    results["9"] = ["GiB_sampled_peak": Double(peak) / Double(1 << 30)]
    let data = try JSONSerialization.data(
      withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
    print("MEASURED_BENCHMARK " + String(decoding: data, as: UTF8.self))
    if let output = ProcessInfo.processInfo.environment["SSB_PARITY_OUTPUT"] {
      try data.write(
        to: URL(fileURLWithPath: output).appendingPathComponent("benchmark_native.json"))
    }
  }
}
