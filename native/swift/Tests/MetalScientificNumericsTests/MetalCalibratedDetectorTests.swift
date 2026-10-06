// Copyright (c) 2025 ophusgroup. MIT License; see the repository LICENSE.
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import MetalScientificNumerics

#if !SCIENTIFIC_NUMERICS_CHECK
  import XCTest
#endif

/// Detector reductions match restored intensities across two calibration regions.
func checkCalibratedReductions() throws {
  let ops = try MetalImageOperations()
  let source = try MetalPackedSource(
    shape: [4, 4, 64, 64], precision: MetalPrecision(device: ops.device))
  defer { source.releaseResidentStorage() }
  for region in 0..<2 {
    let values: [Float] = (0..<(8 * 4096)).map { Float($0 % 173) / 7 + Float(region * 500) }
    let image = try ops.image(values: values, rows: 8, columns: 4096)
    let precision = try MetalPrecision(device: ops.device)
    try precision.includeRange(image.buffer, count: values.count)
    try precision.calibrate(shape: [2, 4, 64, 64])
    let codes = try precision.convert(image.buffer, count: values.count)
    try precision.finish()
    try source.append(codes, frames: 8, calibration: precision)
  }
  let decoded = try source.read(0..<16)
  let values = decoded.contents().assumingMemoryBound(to: Float.self)
  var worst = 0.0
  for stride in [1, 3, 17] {
    let weights: [Float] = (0..<4096).map { $0 % stride == 0 ? 1 / 4096 : 0 }
    let observed = try ops.virtualImage(source: source, weights: weights).values()
    for frame in 0..<16 {
      let expected = (0..<4096).reduce(0.0) {
        $0 + Double(values[frame * 4096 + $1]) * Double(weights[$1])
      }
      let error = abs(Double(observed[frame]) - expected)
      worst = max(worst, error)
      guard error <= max(2e-5, abs(expected) * 3e-6) else {
        throw NSError(domain: "display-parity", code: 1)
      }
    }
  }
  do {
    _ = try ops.virtualImage(
      source: source, weights: Array(repeating: 1, count: 4096), shouldCancel: { true })
    throw NSError(domain: "missing-cancellation", code: 1)
  } catch Metal4DSTEMStreamingIOError.cancelled {}
  print("CALIBRATED_REDUCTIONS_PASS max_abs_error=\(worst) regions=2 masks=3 frames=16")
}

#if !SCIENTIFIC_NUMERICS_CHECK
  final class MetalCalibratedDetectorTests: XCTestCase {
    func testCalibratedDetectorReductions() throws {
      guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Requires Metal") }
      try checkCalibratedReductions()
    }
  }
#endif
