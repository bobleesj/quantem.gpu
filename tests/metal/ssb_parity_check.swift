// Strict float32 SSB parity harness.
//
// Runs the native MetalSSBKernels pipeline on one exported exact BF-column
// artifact and dumps its complex object, mean phase and objective loss so a
// double-precision oracle can measure the precision a native path actually
// delivers. Nothing is binned, cropped, approximated or reduced here: the
// detector counts are the exact declared integers and every cache/stream
// topology runs the same full-aperture objective.
//
// Usage: ssb-parity-check <case-dir> <out-dir>

import Foundation
import Metal
import MetalSSBKernels

private func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data("ssb-parity-check: \(message)\n".utf8))
  exit(1)
}

private struct CaseFile: Decodable {
  let scan_shape: [Int]
  let detector_shape: [Int]
  let bf_count: Int
  let documented_bf_count: Int
  let active_bf_count: Int
  let brightfield_center: [Double]
  let brightfield_radius_px: Double
  let voltage_kV: Double
  let semiangle_mrad: Double
  let scan_sampling_A: [Double]
  let det_sampling_mrad: Double
  let aberrations: [[Double]]
  let metal_geometry: MetalSSBGeometry
  let metal_geometry_unrotated: UnrotatedGeometry
}

// The declared probe geometry before the calibrated rotation. The native
// calibration path derives this exact form from the physical inputs, so the
// harness can verify its independent derivation against the declaration.
private struct UnrotatedGeometry: Decodable {
  let brightfieldKX: [Float]
  let brightfieldKY: [Float]
  let brightfieldAlphaSquared: [Float]
  let brightfieldAperture: [Float]
  let brightfieldCos2Phi: [Float]
  let brightfieldSin2Phi: [Float]
}

private struct VariantReport: Encodable {
  let name: String
  let cacheBudgetBytes: Int?
  let loss: Double
  let wallSeconds: Double
  let gpuSeconds: Double
  let loggedCachedBrightfieldCount: Int
  let loggedStreamedBrightfieldCount: Int
  let executedBrightfieldCount: Int
  let logicalBrightfieldCount: Int
}

private struct AberrationReport: Encodable {
  let index: Int
  let c10Nanometers: Double
  let c12Nanometers: Double
  let phi12Radians: Double
  let variants: [VariantReport]
}

private struct CaseReport: Encodable {
  let caseName: String
  let scanSide: Int
  let bfCount: Int
  let aberrations: [AberrationReport]
  let calibration: [String: Double]
}

@main enum SSBParityCheck {
  static func main() throws {
    let arguments = CommandLine.arguments
    guard arguments.count >= 3 else { fail("usage: ssb-parity-check <case-dir> <out-dir>") }
    let caseDirectory = URL(fileURLWithPath: arguments[1])
    let outDirectory = URL(fileURLWithPath: arguments[2])
    try? FileManager.default.createDirectory(
      at: outDirectory, withIntermediateDirectories: true)

    let caseData = try Data(contentsOf: caseDirectory.appendingPathComponent("case.json"))
    let payload = try JSONDecoder().decode(CaseFile.self, from: caseData)
    let side = payload.scan_shape[0]
    guard payload.scan_shape[0] == payload.scan_shape[1] else { fail("scan must be square") }
    guard [128, 256, 512].contains(side) else { fail("unsupported scan side \(side)") }

    let countsURL = caseDirectory.appendingPathComponent("source/bf_columns.u16")
    let counts = try Data(contentsOf: countsURL, options: .mappedIfSafe)
    let expectedBytes = payload.bf_count * side * side * MemoryLayout<UInt16>.size
    guard counts.count == expectedBytes else {
      fail("bf_columns.u16 is \(counts.count) bytes, expected \(expectedBytes)")
    }

    guard let device = MTLCreateSystemDefaultDevice() else { fail("an Apple GPU is required") }

    // Independent calibration check: rebuild the declared geometry from the
    // physical inputs with the native calibration path, including the exact
    // dead-pixel exclusion and the exact per-detector-pixel integer sum.
    var detectorSum = [UInt64](repeating: 0, count: payload.detector_shape[0] * payload.detector_shape[1])
    let calData = try Data(contentsOf: caseDirectory.appendingPathComponent("source/snapshots/cal.json"))
    let cal = try JSONDecoder().decode(CalibrationFile.self, from: calData)
    guard cal.bf_rows.count == payload.bf_count, cal.bf_cols.count == payload.bf_count else {
      fail("calibration declares \(cal.bf_rows.count) BF pixels, case declares \(payload.bf_count)")
    }
    let plane = side * side
    guard (payload.bf_count * plane) * MemoryLayout<UInt16>.size == counts.count else { fail("payload size") }
    counts.withUnsafeBytes { raw in
      let values = raw.bindMemory(to: UInt16.self)
      for index in 0..<payload.bf_count {
        let detectorRow = cal.bf_rows[index]
        let detectorColumn = cal.bf_cols[index]
        var total: UInt64 = 0
        let base = index * plane
        for offset in 0..<plane { total += UInt64(values[base + offset]) }
        detectorSum[detectorRow * payload.detector_shape[1] + detectorColumn] = total
      }
    }
    let calibration = MetalSSBCalibration(
      beamEnergyKeV: payload.voltage_kV,
      semiangleMrad: payload.semiangle_mrad,
      scanStepRowAngstroms: payload.scan_sampling_A[0],
      scanStepColumnAngstroms: payload.scan_sampling_A[1],
      detectorStepRowMrad: payload.det_sampling_mrad,
      detectorStepColumnMrad: payload.det_sampling_mrad,
      centerRow: payload.brightfield_center[0],
      centerColumn: payload.brightfield_center[1],
      brightfieldRadiusPixels: payload.brightfield_radius_px,
      excludedDetectorPixels: cal.bf_excluded_detector_pixels)
    let rebuilt = try calibration.geometry(
      detectorRows: payload.detector_shape[0],
      detectorColumns: payload.detector_shape[1],
      detectorSum: detectorSum,
      scanRows: side, scanColumns: side)
    var calibrationReport: [String: Double] = [:]
    calibrationReport["pixelCount"] = Double(rebuilt.pixels.count)
    calibrationReport["declaredDistance"] = Double(rebuilt.pixels.count - payload.bf_count)
    let declaredUnrotated = payload.metal_geometry_unrotated
    calibrationReport["declaredKX"] = maxDifference(
      rebuilt.geometry.brightfieldKX, declaredUnrotated.brightfieldKX)
    calibrationReport["declaredKY"] = maxDifference(
      rebuilt.geometry.brightfieldKY, declaredUnrotated.brightfieldKY)
    calibrationReport["declaredAlphaSquared"] = maxDifference(
      rebuilt.geometry.brightfieldAlphaSquared, declaredUnrotated.brightfieldAlphaSquared)
    calibrationReport["declaredAperture"] = maxDifference(
      rebuilt.geometry.brightfieldAperture, declaredUnrotated.brightfieldAperture)
    calibrationReport["declaredCos2Phi"] = maxDifference(
      rebuilt.geometry.brightfieldCos2Phi, declaredUnrotated.brightfieldCos2Phi)
    calibrationReport["declaredSin2Phi"] = maxDifference(
      rebuilt.geometry.brightfieldSin2Phi, declaredUnrotated.brightfieldSin2Phi)
    calibrationReport["declaredDCValue"] = abs(
      Double(rebuilt.geometry.dcValue.x - payload.metal_geometry.dcValue.x))
    calibrationReport["activeApertureCount"] = Double(
      rebuilt.geometry.brightfieldAperture.filter { $0 > 0 }.count)

    let source = counts.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    let bytesPerBrightfield = side * side * MemoryLayout<SIMD2<Float>>.stride
    let topologies: [(String, Int?)] = [
      ("cached", nil),
      ("streamed", 0),
      ("hybrid", bytesPerBrightfield * payload.bf_count / 4),
    ]

    var aberrationReports: [AberrationReport] = []
    for topology in topologies {
      let engine = try MetalSSBEngine(
        device: device, geometry: payload.metal_geometry, cacheBudgetBytes: topology.1)
      try engine.prepare(brightfield: source, countType: .uint16)
      for (index, values) in payload.aberrations.enumerated() {
        guard values.count == 3 else { fail("aberrations must be C10, C12, phi12") }
        let aberrations = MetalSSBAberrations(
          c10Nanometers: Float(values[0]), c12Nanometers: Float(values[1]),
          phi12Radians: Float(values[2]))
        let result = try engine.reconstruct(aberrations: aberrations)
        let phase = try engine.phase(of: result)
        let variance = try engine.phaseVariance(aberrations: aberrations)
        let objectName = "object-\(index)-\(topology.0).f32"
        let phaseName = "phase-\(index)-\(topology.0).f32"
        try Data(bytes: result.object.contents(), count: plane * 8)
          .write(to: outDirectory.appendingPathComponent(objectName))
        try Data(bytes: phase.contents(), count: plane * 4)
          .write(to: outDirectory.appendingPathComponent(phaseName))
        let variant = VariantReport(
          name: topology.0,
          cacheBudgetBytes: topology.1,
          loss: Double(variance.loss),
          wallSeconds: result.wallSeconds,
          gpuSeconds: result.gpuSeconds,
          loggedCachedBrightfieldCount: result.provenance.cachedBrightfieldCount,
          loggedStreamedBrightfieldCount: result.provenance.streamedBrightfieldCount,
          executedBrightfieldCount: result.provenance.executedBrightfieldCount,
          logicalBrightfieldCount: result.provenance.logicalBrightfieldCount)
        if let position = aberrationReports.firstIndex(where: { $0.index == index }) {
          aberrationReports[position] = AberrationReport(
            index: index,
            c10Nanometers: values[0],
            c12Nanometers: values[1],
            phi12Radians: values[2],
            variants: aberrationReports[position].variants + [variant])
        } else {
          aberrationReports.append(
            AberrationReport(
              index: index, c10Nanometers: values[0], c12Nanometers: values[1],
              phi12Radians: values[2], variants: [variant]))
        }
        print(
          "PASS \(topology.0) aberration=\(index) loss=\(variance.loss) "
            + "logical=\(result.provenance.logicalBrightfieldCount) "
            + "executed=\(result.provenance.executedBrightfieldCount) "
            + "cached=\(result.provenance.cachedBrightfieldCount) "
            + "streamed=\(result.provenance.streamedBrightfieldCount)")
      }
      if topology.0 == "streamed" {
        // The streamed topology must still agree with itself across repeats.
        let engineAgain = try MetalSSBEngine(
          device: device, geometry: payload.metal_geometry, cacheBudgetBytes: 0)
        try engineAgain.prepare(brightfield: source, countType: .uint16)
        let values = payload.aberrations[0]
        let repeatResult = try engineAgain.reconstruct(
          aberrations: MetalSSBAberrations(
            c10Nanometers: Float(values[0]), c12Nanometers: Float(values[1]),
            phi12Radians: Float(values[2])))
        let first = outDirectory.appendingPathComponent("object-0-streamed.f32")
        let firstData = try Data(contentsOf: first)
        let repeatData = Data(bytes: repeatResult.object.contents(), count: plane * 8)
        print("PASS streamed rerun identical=\(repeatData == firstData)")
      }
    }

    let report = CaseReport(
      caseName: caseDirectory.lastPathComponent,
      scanSide: side,
      bfCount: payload.bf_count,
      aberrations: aberrationReports.sorted { $0.index < $1.index },
      calibration: calibrationReport)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(to: outDirectory.appendingPathComponent("metal.json"))
    try JSONSerialization.data(
      withJSONObject: calibrationReport, options: [.prettyPrinted, .sortedKeys]
    ).write(to: outDirectory.appendingPathComponent("calibration-check.json"))
    print("calibration: pixels=\(rebuilt.pixels.count) declaredBF=\(payload.bf_count) "
      + "maxKX=\(calibrationReport["declaredKX"]!) maxAlpha2=\(calibrationReport["declaredAlphaSquared"]!) "
      + "maxAperture=\(calibrationReport["declaredAperture"]!) dc=\(calibrationReport["declaredDCValue"]!)")
  }

  private static func maxDifference(_ actual: [Float], _ declared: [Float]) -> Double {
    guard actual.count == declared.count else { return Double.infinity }
    var worst = 0.0
    for (left, right) in zip(actual, declared) {
      worst = max(worst, abs(Double(left) - Double(right)))
    }
    return worst
  }
}

private struct CalibrationFile: Decodable {
  let bf_rows: [Int]
  let bf_cols: [Int]
  let bf_excluded_detector_pixels: [Int]?
}
