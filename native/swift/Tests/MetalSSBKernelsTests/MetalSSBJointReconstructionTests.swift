import Foundation
import Metal
import XCTest

@testable import MetalSSBKernels

final class MetalSSBJointReconstructionTests: XCTestCase {
  func testFullScanFinePhaseAcrossCacheBoundaries() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let count = 33, plane = 512 * 512
    let values: [UInt16] = (0..<(count * plane)).map { (i: Int) -> UInt16 in
      let value = i * 17 + i / 113 + i / plane * 7
      return UInt16(value % 29)
    }
    let source = try XCTUnwrap(values.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
    })
    let points = [
      SSBJointParameters(aberrations: .init(c10Angstroms: 88, c12Angstroms: 9, phi12Radians: 1),
                         depthAngstroms: 136, tiltRowMrad: -0.7, tiltColumnMrad: -3),
      SSBJointParameters(aberrations: .init(c10Angstroms: -40, c12Angstroms: -17, phi12Radians: -0.4),
                         depthAngstroms: 0, tiltRowMrad: 0, tiltColumnMrad: 0),
    ]
    for cached in [0, 32, count] {
      let engine = try MetalSSBEngine(
        device: device, geometry: geometry(count: count),
        cacheBudgetBytes: cached * 512 * 257 * 8)
      try engine.prepare(brightfield: source, countType: .uint16)
      let model = try engine.jointModel(scanDetectorRotationDegrees: 27)
      XCTAssertTrue(model.supportsFused1536)
      for point in points {
        let expected = try model.reconstructGeneral(point, upsamplingFactor: 3)
        let actual = try model.reconstruct(point, upsamplingFactor: 3)
        XCTAssertEqual(actual.rows, 1536)
        XCTAssertEqual(actual.columns, 1536)
        XCTAssertEqual(actual.provenance.logicalBrightfieldCount, count)
        XCTAssertEqual(actual.provenance.cachedBrightfieldCount, cached)
        assertPhase(actual, equals: expected)
        let before = Data(bytes: actual.phase.contents(), count: 1536 * 1536 * 4)
        var polls = 0
        XCTAssertThrowsError(try model.reconstruct(point, upsamplingFactor: 3, isCancelled: {
          polls += 1
          return polls >= 3
        }))
        let afterCancellation = try model.reconstruct(point, upsamplingFactor: 3)
        assertPhase(afterCancellation, equals: expected)
        XCTAssertEqual(before, Data(bytes: actual.phase.contents(), count: before.count))
      }
    }
  }

  func testFinePhaseUsesGeneralTransformWhenBoundaryHasSupport() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let count = 3, plane = 512 * 512
    let values: [UInt16] = (0..<(count * plane)).map { (i: Int) -> UInt16 in
      let value = i * 13 + i / 79
      return UInt16(value % 23)
    }
    let source = try XCTUnwrap(values.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
    })
    let engine = try MetalSSBEngine(device: device, geometry: geometry(count: count, semiangle: 0.1))
    try engine.prepare(brightfield: source, countType: .uint16)
    let model = try engine.jointModel(scanDetectorRotationDegrees: 27)
    XCTAssertFalse(model.supportsFused1536)
    let point = SSBJointParameters(
      aberrations: .init(c10Angstroms: 33, c12Angstroms: 7, phi12Radians: 0.2),
      depthAngstroms: 120, tiltRowMrad: 1, tiltColumnMrad: -2)
    let expected = try model.reconstructGeneral(point, upsamplingFactor: 3)
    let actual = try model.reconstruct(point, upsamplingFactor: 3)
    XCTAssertEqual(Data(bytes: actual.phase.contents(), count: 1536 * 1536 * 4),
                   Data(bytes: expected.phase.contents(), count: 1536 * 1536 * 4))
  }

  private func assertPhase(_ actual: MetalSSBJointPhase, equals expected: MetalSSBJointPhase) {
    let a = actual.phase.contents().assumingMemoryBound(to: Float.self)
    let b = expected.phase.contents().assumingMemoryBound(to: Float.self)
    var error = 0.0, norm = 0.0, maximum = 0.0
    for i in 0..<(actual.rows * actual.columns) {
      let delta = Double(a[i]) - Double(b[i])
      error += delta * delta
      norm += Double(b[i]) * Double(b[i])
      maximum = max(maximum, abs(delta))
    }
    XCTAssertLessThan(sqrt(error / max(norm, 1e-30)), 2e-4)
    XCTAssertLessThan(maximum, 2e-4)
  }

  private func geometry(count: Int, semiangle: Float = 0.03) -> MetalSSBGeometry {
    let kx = (0..<count).map { Float($0 % 7 - 3) * 0.3 }
    let ky = (0..<count).map { Float($0 / 7 - 2) * 0.3 }
    let wavelength: Float = 0.0197
    let radius = zip(kx, ky).map { $0 * $0 + $1 * $1 }
    let cosine = zip(zip(kx, ky), radius).map { p, r in r == 0 ? 0 : (p.0 * p.0 - p.1 * p.1) / r }
    let sine = zip(zip(kx, ky), radius).map { p, r in r == 0 ? 0 : 2 * p.0 * p.1 / r }
    let q = (0..<512).map { Float($0 < 256 ? $0 : $0 - 512) / (512 * 0.373) }
    return MetalSSBGeometry(
      brightfieldKX: kx, brightfieldKY: ky,
      brightfieldAlphaSquared: radius.map { $0 * wavelength * wavelength },
      brightfieldAperture: [Float](repeating: 1, count: count),
      brightfieldCos2Phi: cosine, brightfieldSin2Phi: sine,
      qxByRow: q, qyByColumn: q, wavelengthAngstroms: wavelength,
      semiangleRadians: semiangle, angularSamplingYRadians: 0.000556,
      angularSamplingXRadians: 0.000556, dcValue: SIMD2(8_000_000, 0),
      referenceRotationDegrees: 0)
  }
}
