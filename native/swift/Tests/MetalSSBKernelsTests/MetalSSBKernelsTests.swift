import Metal
import XCTest

@testable import MetalSSBKernels

final class MetalSSBKernelsTests: XCTestCase {
  private let size = 512

  func testOptimizerIsDeterministicForFixedSeed() throws {
    let optimizer = SSBOptimizer(globalTrials: 24, seed: 42)
    let start = SSBOptimizationPoint(
      c10Nanometers: 30,
      c12Nanometers: 8,
      phi12Radians: 0.1
    )
    func objective(_ point: SSBOptimizationPoint) -> Double {
      pow(point.c10Nanometers - 12, 2)
        + pow(point.c12Nanometers - 3, 2)
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
        c10Nanometers: 0,
        c12Nanometers: 0,
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
        c10Nanometers: 0,
        c12Nanometers: 0,
        phi12Radians: 0
      )
    )
    XCTAssertEqual(loss.loss, 0, accuracy: 1e-8)
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
      c10Nanometers: 55,
      c12Nanometers: 13,
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
    let kx = Array(kxBase.prefix(brightfieldCount))
    let ky = Array(kyBase.prefix(brightfieldCount))
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
