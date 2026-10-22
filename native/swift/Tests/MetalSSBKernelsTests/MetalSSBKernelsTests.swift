import Metal
import XCTest

@testable import MetalSSBKernels

final class MetalSSBKernelsTests: XCTestCase {
  private let size = 512

  func testOptimizerIsDeterministicForFixedSeed() throws {
    let optimizer = SSBOptimizer(globalTrials: 24, seed: 42)
    let start = SSBOptimizationPoint(
      c10Angstroms: 30,
      c12Angstroms: 8,
      phi12Radians: 0.1
    )
    func objective(_ point: SSBOptimizationPoint) -> Double {
      pow(point.c10Angstroms - 12, 2)
        + pow(point.c12Angstroms - 3, 2)
        + 20 * pow(point.phi12Radians + 0.2, 2)
    }

    let first = try optimizer.run(start: start, evaluate: objective)
    let second = try optimizer.run(start: start, evaluate: objective)

    XCTAssertEqual(first.best, second.best)
    XCTAssertEqual(first.loss, second.loss)
    XCTAssertEqual(first.trials.count, second.trials.count)
    XCTAssertEqual(
      first.trials.map(\.loss),
      second.trials.map(\.loss)
    )
    XCTAssertLessThan(first.loss, objective(start))
  }

  func testZeroInputPreservesFullBFNormalizationAndDC() throws {
    let device = try metalDevice()
    let geometry = makeGeometry(brightfieldCount: 1, dc: SIMD2<Float>(1, 0))
    let engine = try MetalSSBEngine(device: device, geometry: geometry)
    let source = try makeBuffer(
      device: device,
      values: [UInt8](repeating: 0, count: size * size)
    )
    try engine.prepare(brightfield: source)

    let result = try engine.reconstruct(
      aberrations: MetalSSBAberrations(
        c10Angstroms: 0,
        c12Angstroms: 0,
        phi12Radians: 0
      )
    )
    let fourier = complexValues(result.fourierSum)
    XCTAssertEqual(fourier[0].x, 1, accuracy: 1e-6)
    XCTAssertEqual(fourier[0].y, 0, accuracy: 1e-6)
    for value in fourier.dropFirst() {
      XCTAssertEqual(value.x, 0, accuracy: 1e-6)
      XCTAssertEqual(value.y, 0, accuracy: 1e-6)
    }
    let expected = Float(1) / Float(size * size)
    for value in complexValues(result.object) {
      XCTAssertEqual(value.x, expected, accuracy: 2e-7)
      XCTAssertEqual(value.y, 0, accuracy: 2e-7)
    }
    XCTAssertEqual(result.provenance.scanRows, 512)
    XCTAssertEqual(result.provenance.scanColumns, 512)
    XCTAssertEqual(result.provenance.scanBin, 1)
    XCTAssertEqual(result.provenance.scanCrop, "none")
    XCTAssertEqual(result.provenance.logicalBrightfieldCount, 1)
    XCTAssertEqual(result.provenance.executedBrightfieldCount, 1)
    XCTAssertEqual(result.provenance.cachedBrightfieldCount, 1)
    XCTAssertEqual(result.provenance.streamedBrightfieldCount, 0)
    XCTAssertEqual(result.provenance.cacheBytes, size * (size / 2 + 1) * 8)

    let loss = try engine.phaseVariance(
      aberrations: MetalSSBAberrations(
        c10Angstroms: 0,
        c12Angstroms: 0,
        phi12Radians: 0
      )
    )
    XCTAssertEqual(loss.loss, 0, accuracy: 1e-8)
  }

  func testPhaseBatchTailAgreesAcrossCacheBoundaries() throws {
    let device = try metalDevice()
    let count = 33
    let geometry = makeGeometry(brightfieldCount: count, dc: SIMD2<Float>(37, 0))
    let plane = size * size
    let valueCount = count * plane
    let values: [UInt16] = (0..<valueCount).map { (index: Int) -> UInt16 in
      let ramp = index * 17
      let scanTerm = index / 113
      let detectorTerm = index / plane
      return UInt16((ramp + scanTerm + detectorTerm) % 29)
    }
    let source = try makeBuffer(device: device, values: values)
    let aberrations = MetalSSBAberrations(
      c10Angstroms: 55, c12Angstroms: 13, phi12Radians: 0.23)
    let reference = try MetalSSBEngine(device: device, geometry: geometry, cacheBudgetBytes: 0)
    try reference.prepare(brightfield: source, countType: .uint16)
    let referenceObject = complexValues(try reference.reconstruct(aberrations: aberrations).object)
    let referenceLoss = try reference.phaseVariance(aberrations: aberrations).loss

    // A complete 32-plane batch plus a tail, and a cache boundary inside it.
    let bytesPerPlane = size * (size / 2 + 1) * 8
    for cachedPlanes in [13, count] {
      let candidate = try MetalSSBEngine(
        device: device, geometry: geometry, cacheBudgetBytes: cachedPlanes * bytesPerPlane)
      try candidate.prepare(brightfield: source, countType: .uint16)
      let result = try candidate.reconstruct(aberrations: aberrations)
      XCTAssertEqual(result.provenance.executedBrightfieldCount, count)
      XCTAssertLessThan(relativeL2(complexValues(result.object), referenceObject), 1e-4)
      let loss = try candidate.phaseVariance(aberrations: aberrations).loss
      XCTAssertLessThan(abs(loss - referenceLoss) / max(abs(referenceLoss), 1e-12), 5e-5)
    }
  }

  func testCachedAndStreamedPathsAgreeForNontrivialInput() throws {
    let device = try metalDevice()
    let geometry = makeGeometry(
      brightfieldCount: 3,
      dc: SIMD2<Float>(37, -2)
    )
    let valueCount = 3 * size * size
    var values = [UInt8](repeating: 0, count: valueCount)
    for index in values.indices {
      let mixed = index * 17 + index / 113 + 3
      values[index] = UInt8(mixed % 29)
    }
    let source = try makeBuffer(device: device, values: values)
    let cached = try MetalSSBEngine(device: device, geometry: geometry)
    let streamed = try MetalSSBEngine(
      device: device,
      geometry: geometry,
      cacheBudgetBytes: 0
    )
    try cached.prepare(brightfield: source)
    try streamed.prepare(brightfield: source)
    let aberrations = MetalSSBAberrations(
      c10Angstroms: 55,
      c12Angstroms: 13,
      phi12Radians: 0.23
    )

    let cachedResult = try cached.reconstruct(aberrations: aberrations)
    let streamedResult = try streamed.reconstruct(aberrations: aberrations)
    XCTAssertEqual(cachedResult.provenance.cachedBrightfieldCount, 3)
    XCTAssertEqual(cachedResult.provenance.streamedBrightfieldCount, 0)
    XCTAssertEqual(streamedResult.provenance.cachedBrightfieldCount, 0)
    XCTAssertEqual(streamedResult.provenance.streamedBrightfieldCount, 3)
    let fourierRelativeL2 = relativeL2(
      complexValues(cachedResult.fourierSum),
      complexValues(streamedResult.fourierSum)
    )
    let cachedObject = complexValues(cachedResult.object)
    let streamedObject = complexValues(streamedResult.object)
    let objectRelativeL2 = relativeL2(cachedObject, streamedObject)
    let phaseMaximum = maximumPhaseDifference(cachedObject, streamedObject)
    XCTAssertLessThan(
      fourierRelativeL2,
      1e-4,
      "Fourier relative L2 \(fourierRelativeL2)"
    )
    XCTAssertLessThan(
      objectRelativeL2,
      1e-4,
      "object relative L2 \(objectRelativeL2)"
    )
    XCTAssertLessThan(
      phaseMaximum,
      0.02,
      "maximum wrapped phase error \(phaseMaximum) rad"
    )

    let cachedLoss = try cached.phaseVariance(aberrations: aberrations)
    let streamedLoss = try streamed.phaseVariance(aberrations: aberrations)
    let lossRelativeError =
      abs(cachedLoss.loss - streamedLoss.loss)
      / max(abs(streamedLoss.loss), 1e-12)
    XCTAssertLessThan(lossRelativeError, 5e-5)
  }

  /// The exported pair is the record the Python reader checks (`quantem.ssb-phase/2`): its keys,
  /// the engine's Angstrom and radians stated in nm and degrees, and the physical rotation.
  func testPhasePairWritesTheSharedRecordInPublicUnits() throws {
    let device = try metalDevice()
    let geometry = makeGeometry(brightfieldCount: 1, dc: SIMD2<Float>(1, 0))
    let engine = try MetalSSBEngine(device: device, geometry: geometry)
    let source = try makeBuffer(device: device, values: [UInt8](repeating: 0, count: size * size))
    try engine.prepare(brightfield: source)
    let aberrations = MetalSSBAberrations(c10Angstroms: 150, c12Angstroms: 20, phi12Radians: 0.5)
    let calibration = MetalSSBCalibration(
      voltageKV: 300, semiangleMrad: 30,
      scanSamplingRowAngstroms: 0.25, scanSamplingColumnAngstroms: 0.3,
      detectorSamplingRowMrad: 0.55, detectorSamplingColumnMrad: 0.56,
      brightfieldCenterRowPixels: 3, brightfieldCenterColumnPixels: 4, brightfieldRadiusPixels: 5)
    let identity = String(repeating: "a", count: 64)
    let run = try MetalSSBSavedRun(
      result: try engine.reconstruct(aberrations: aberrations), sourceIdentity: identity,
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
    let geometry = MetalSSBGeometry(
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
      try MetalSSBEngine(device: device, geometry: geometry)
    )
  }

  private func makeGeometry(
    brightfieldCount: Int,
    dc: SIMD2<Float>
  ) -> MetalSSBGeometry {
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
    return MetalSSBGeometry(
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

  private func relativeL2(
    _ first: [SIMD2<Float>],
    _ second: [SIMD2<Float>]
  ) -> Double {
    precondition(first.count == second.count)
    var errorSquared = 0.0
    var referenceSquared = 0.0
    for (left, right) in zip(first, second) {
      let realError = Double(left.x - right.x)
      let imaginaryError = Double(left.y - right.y)
      errorSquared += realError * realError + imaginaryError * imaginaryError
      referenceSquared += Double(left.x * left.x + left.y * left.y)
    }
    return sqrt(errorSquared / max(referenceSquared, .leastNonzeroMagnitude))
  }

  private func maximumPhaseDifference(
    _ first: [SIMD2<Float>],
    _ second: [SIMD2<Float>]
  ) -> Float {
    precondition(first.count == second.count)
    var maximum: Float = 0
    for (left, right) in zip(first, second) {
      let leftPhase = atan2(left.y, left.x)
      let rightPhase = atan2(right.y, right.x)
      let delta = leftPhase - rightPhase
      maximum = max(maximum, abs(atan2(sin(delta), cos(delta))))
    }
    return maximum
  }
}
