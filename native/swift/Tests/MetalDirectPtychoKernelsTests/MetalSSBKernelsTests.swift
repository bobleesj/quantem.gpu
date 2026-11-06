import Metal
import XCTest

@testable import MetalDirectPtychoKernels

final class MetalSSBKernelsTests: XCTestCase {
  private let size = 512

  /// A run saved by the retired non-model engine reopens: its record decodes, its stored object is returned
  /// bit for bit, and `phase(of:)` converts it on Metal.
  func testLegacySavedRunReopensWithItsPhase() throws {
    let device = try metalDevice()
    let geometry = makeGeometry(brightfieldCount: 1, dc: SIMD2<Float>(1, 0))
    let object = (0..<size * size).map { index -> SIMD2<Float> in
      let angle = Float(index % 13) * 0.4 - 2.4
      return SIMD2(cos(angle), sin(angle)) * Float(1 + index % 3)
    }
    let identity = String(repeating: "b", count: 64)
    let run = try MetalSSBSavedRun(
      result: try legacyResult(device: device, object: object), sourceIdentity: identity,
      backendRevision: "test", geometry: geometry,
      aberrations: MetalDirectPtychoAberrations(
        c10Angstroms: 150, c12Angstroms: 20, phi12Radians: 0.5),
      scanDetectorRotationDegrees: -30)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      "\(UUID().uuidString).ssb")
    defer { try? FileManager.default.removeItem(at: url) }
    try run.save(to: url)

    let reopened = try MetalSSBSavedRun.load(from: url, matchingSourceIdentity: identity)
    XCTAssertThrowsError(
      try MetalSSBSavedRun.load(
        from: url, matchingSourceIdentity: String(repeating: "c", count: 64)))
    let restored = try reopened.reconstruction(device: device)
    XCTAssertEqual(complexValues(restored.object), object)
    let engine = try MetalDirectPtychoEngine(
      device: device, geometry: geometry, cacheBudgetBytes: 0)
    let phase = try engine.phase(of: restored)
    let values = phase.contents().bindMemory(to: Float.self, capacity: size * size)
    for index in stride(from: 0, to: size * size, by: 997) {
      XCTAssertEqual(values[index], atan2(object[index].y, object[index].x), accuracy: 1e-6)
    }
  }

  /// The exported pair is the record the Python reader checks (`quantem.ssb-phase/2`): its keys,
  /// the engine's Angstrom and radians stated in nm and degrees, and the physical rotation.
  func testPhasePairWritesTheSharedRecordInPublicUnits() throws {
    let device = try metalDevice()
    let geometry = makeGeometry(brightfieldCount: 1, dc: SIMD2<Float>(1, 0))
    let aberrations = MetalDirectPtychoAberrations(
      c10Angstroms: 150, c12Angstroms: 20, phi12Radians: 0.5)
    let calibration = MetalDirectPtychoCalibration(
      voltageKV: 300, semiangleMrad: 30,
      scanSamplingRowAngstroms: 0.25, scanSamplingColumnAngstroms: 0.3,
      detectorSamplingRowMrad: 0.55, detectorSamplingColumnMrad: 0.56,
      brightfieldCenterRowPixels: 3, brightfieldCenterColumnPixels: 4, brightfieldRadiusPixels: 5)
    let identity = String(repeating: "a", count: 64)
    let run = try MetalSSBSavedRun(
      result: try legacyResult(
        device: device, object: [SIMD2<Float>](repeating: SIMD2(1, 0), count: size * size)),
      sourceIdentity: identity,
      backendRevision: "test", geometry: geometry, aberrations: aberrations,
      scanDetectorRotationDegrees: -30, calibration: calibration)
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let manifest = directory.appendingPathComponent("ssb.json")
    let phase = (0..<size * size).map { Float($0 % 7) - 3 }.withUnsafeBytes { Data($0) }
    try run.savePhasePair(to: manifest, phase: phase)

    let record = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any])
    XCTAssertEqual(
      Set(record.keys),
      [
        "schema", "source_identity", "shape", "phase_encoding", "phase_unit", "phase_sha256",
        "calibration", "aberrations", "scan_detector_rotation_deg",
        "scan_detector_rotation_flipped",
        "provenance", "run_metadata", "phase_file",
      ])
    XCTAssertEqual(record["schema"] as? String, "quantem.ssb-phase/2")
    XCTAssertEqual(record["shape"] as? [Int], [size, size])
    XCTAssertEqual(record["phase_file"] as? String, "ssb.npy")
    let stated = try XCTUnwrap(record["aberrations"] as? [String: Double])
    XCTAssertEqual(Set(stated.keys), ["C10_nm", "C12_nm", "phi12_deg"])
    XCTAssertEqual(stated["C10_nm"], 15)
    XCTAssertEqual(stated["C12_nm"], 2)
    XCTAssertEqual(try XCTUnwrap(stated["phi12_deg"]), 0.5 * 180 / .pi, accuracy: 1e-12)
    XCTAssertEqual(record["scan_detector_rotation_deg"] as? Double, 330)
    XCTAssertEqual(record["scan_detector_rotation_flipped"] as? Bool, false)
    let stored = try XCTUnwrap(record["calibration"] as? [String: Any])
    XCTAssertEqual(stored["voltage_kV"] as? Double, 300)
    XCTAssertEqual(stored["semiangle_mrad"] as? Double, 30)
    XCTAssertEqual(stored["scan_sampling_A"] as? [Double], [0.25, 0.3])
    XCTAssertEqual(stored["detector_sampling_mrad"] as? [Double], [0.55, 0.56])
    XCTAssertEqual(stored["bright_field_center_px"] as? [Double], [3, 4])
    XCTAssertEqual(stored["bright_field_radius_px"] as? Double, 5)

    let reopened = try SSBPhaseArtifact.load(from: manifest, matchingSourceIdentity: identity)
    XCTAssertEqual(reopened.phase, phase)
    XCTAssertEqual(reopened.calibration, calibration)
    XCTAssertEqual(reopened.aberrations.c10Nanometers, 15)
    // A version-1 record is refused with the reason, never read under its old keys.
    var earlier = record
    earlier["schema"] = nil
    earlier["format"] = "live.ssb"
    earlier["schemaVersion"] = 1
    try JSONSerialization.data(withJSONObject: earlier).write(to: manifest)
    XCTAssertThrowsError(try SSBPhaseArtifact.load(from: manifest)) { error in
      XCTAssertTrue(error.localizedDescription.contains("quantem.ssb-phase/2"))
    }
  }

  func testRejectsIncompleteGeometryBeforeMetalAllocation() throws {
    let device = try metalDevice()
    let geometry = MetalDirectPtychoGeometry(
      brightfieldKX: [0, 1],
      brightfieldKY: [0],
      brightfieldAlphaSquared: [0, 1],
      brightfieldAperture: [1, 1],
      brightfieldCos2Phi: [0, 1],
      brightfieldSin2Phi: [0, 1],
      qxByRow: [Float](repeating: 0, count: size),
      qyByColumn: [Float](repeating: 0, count: size),
      wavelengthAngstroms: 0.025,
      semiangleRadians: 0.02,
      angularSamplingYRadians: 0.001,
      angularSamplingXRadians: 0.001,
      dcValue: .zero,
      referenceRotationDegrees: 0
    )
    XCTAssertThrowsError(
      try MetalDirectPtychoEngine(device: device, geometry: geometry)
    )
  }

  /// A result as the retired non-model engine returned it, from a given complex object.
  private func legacyResult(device: MTLDevice, object: [SIMD2<Float>]) throws
    -> MetalDirectPtychoResult
  {
    MetalDirectPtychoResult(
      object: try makeBuffer(device: device, values: object),
      fourierSum: try makeBuffer(device: device, values: object),
      wallSeconds: 0, gpuSeconds: 0,
      provenance: MetalDirectPtychoProvenance(
        scanRows: size, scanColumns: size, sourceDType: "uint8", computeDType: "float32/complex64",
        logicalBrightfieldCount: 1, executedBrightfieldCount: 1, zeroApertureBrightfieldCount: 0,
        cachedBrightfieldCount: 1, streamedBrightfieldCount: 0, cacheBytes: 0, scanBin: 1,
        scanCrop: "none", brightfieldSelection: "exact-zero-aperture-pruning"))
  }

  private func makeGeometry(
    brightfieldCount: Int,
    dc: SIMD2<Float>
  ) -> MetalDirectPtychoGeometry {
    let kxBase: [Float] = [-0.20, 0, 0.20]
    let kyBase: [Float] = [0.10, 0, -0.10]
    let kx =
      brightfieldCount <= 3
      ? Array(kxBase.prefix(brightfieldCount))
      : (0..<brightfieldCount).map { Float($0 % 7 - 3) * 0.08 }
    let ky =
      brightfieldCount <= 3
      ? Array(kyBase.prefix(brightfieldCount))
      : (0..<brightfieldCount).map { Float($0 / 7 - 2) * 0.07 }
    let wavelength: Float = 0.025
    var alphaSquared: [Float] = []
    var cosine: [Float] = []
    var sine: [Float] = []
    for (x, y) in zip(kx, ky) {
      let radiusSquared = x * x + y * y
      alphaSquared.append(radiusSquared * wavelength * wavelength)
      if radiusSquared == 0 {
        cosine.append(0)
        sine.append(0)
      } else {
        cosine.append((x * x - y * y) / radiusSquared)
        sine.append(2 * x * y / radiusSquared)
      }
    }
    let q = (0..<size).map { index -> Float in
      Float(index < size / 2 ? index : index - size) * 0.001
    }
    return MetalDirectPtychoGeometry(
      brightfieldKX: kx,
      brightfieldKY: ky,
      brightfieldAlphaSquared: alphaSquared,
      brightfieldAperture: [Float](repeating: 1, count: brightfieldCount),
      brightfieldCos2Phi: cosine,
      brightfieldSin2Phi: sine,
      qxByRow: q,
      qyByColumn: q,
      wavelengthAngstroms: wavelength,
      semiangleRadians: 0.02,
      angularSamplingYRadians: 0.001,
      angularSamplingXRadians: 0.001,
      dcValue: dc,
      referenceRotationDegrees: 0
    )
  }

  private func metalDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal is unavailable on this runner")
    }
    return device
  }

  private func makeBuffer<T>(
    device: MTLDevice,
    values: [T]
  ) throws -> MTLBuffer {
    let length = values.count * MemoryLayout<T>.stride
    return try values.withUnsafeBytes { bytes in
      try XCTUnwrap(
        device.makeBuffer(
          bytes: bytes.baseAddress!,
          length: length,
          options: .storageModeShared
        ))
    }
  }

  private func complexValues(_ buffer: MTLBuffer) -> [SIMD2<Float>] {
    let count = size * size
    let pointer = buffer.contents().bindMemory(
      to: SIMD2<Float>.self,
      capacity: count
    )
    return Array(UnsafeBufferPointer(start: pointer, count: count))
  }
}
