import Foundation
import Metal
import Metal4DSTEMStreamingIO
import MetalScientificNumerics
#if !CALIBRATED_IMAGE_CHECK
import XCTest
#endif

/// Independent full-frame reference, crossing both calibration and decode batch boundaries.
private func checkCalibratedImageProducts() throws {
  let operations = try MetalImageOperations()
  let frames = 520, rows = 7, columns = 9, pixels = rows * columns
  let source = try MetalPackedSource(shape: [1, frames, rows, columns],
    precision: MetalPrecision(device: operations.device))
  defer { source.releaseResidentStorage() }
  func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "calibrated-images", code: 1,
      userInfo: [NSLocalizedDescriptionKey: message]) }
  }
  for range in [0..<11, 11..<frames] {
    let values: [Float] = range.flatMap { frame in
      (0..<pixels).map { pixel in
        frame == 0 ? Float(0) : Float((pixel * 13 + frame * 17) % 193) / 7 + Float(range.lowerBound)
      }
    }
    let image = try operations.image(values: values, rows: range.count, columns: pixels)
    let precision = try MetalPrecision(device: operations.device)
    try precision.includeRange(image.buffer, count: values.count)
    try precision.calibrate(shape: [1, range.count, rows, columns])
    let codes = try precision.convert(image.buffer, count: values.count)
    try precision.finish()
    try source.append(codes, frames: range.count, calibration: precision)
  }
  let masks: [[Float]] = [
    Array(repeating: 1 / Float(pixels), count: pixels),
    (0..<pixels).map { $0 % 3 == 0 ? 1 : 0 },
    (0..<pixels).map { $0 % 3 == 1 ? 1 : 0 },
    (0..<pixels).map { $0 % 3 == 2 ? 1 : 0 },
    Array(repeating: 1, count: pixels),
    (0..<pixels).map { Float($0 / columns) },
    (0..<pixels).map { Float($0 % columns) },
    (0..<pixels).map { $0 % 2 == 0 ? 1 : -1 },
  ]
  let restored = try source.read(0..<frames)
  let values = restored.contents().assumingMemoryBound(to: Float.self)
  var completions: [Int] = []
  let images = try operations.virtualImages(source: source, weights: masks,
    progress: { completions.append($0); _ = $1 })
  try require(completions == [512, 520], "Each scan region must be visited once for the entire mask batch.")
  var maximumRelativeError = 0.0
  for (maskIndex, pair) in zip(masks, images).enumerated() {
    let (mask, image) = pair
    let actual = image.values()
    let single = try operations.virtualImage(source: source, weights: mask).values()
    try require(actual == single, "Batch changed scalar reduction arithmetic.")
    for frame in 0..<frames {
      let expected = (0..<pixels).reduce(0.0) {
        $0 + Double(values[frame * pixels + $1]) * Double(mask[$1])
      }
      let error = abs(Double(actual[frame]) - expected)
      maximumRelativeError = max(maximumRelativeError, error / max(1, abs(expected)))
      // Signed weights can cancel nearly to zero. For that mask use the
      // forward-error bound for one multiplication plus six float32 tree adds,
      // rather than relative error against a cancelled sum. Positive masks
      // retain the original 3e-6 relative / 2e-5 absolute display parity gate.
      let magnitude = (0..<pixels).reduce(0.0) {
        $0 + abs(Double(values[frame * pixels + $1]) * Double(mask[$1]))
      }
      let unitRoundoff = Double(Float.ulpOfOne) / 2
      let signedBound = 7 * unitRoundoff / (1 - 7 * unitRoundoff) * magnitude
      let tolerance = mask.contains(where: { $0 < 0 })
        ? max(2e-5, signedBound) : max(2e-5, abs(expected) * 3e-6)
      try require(error <= tolerance, "Reduction differs: mask=\(maskIndex) frame=\(frame) expected=\(expected) actual=\(actual[frame]) error=\(error)")
    }
  }
  let com = try operations.normalized(images[5], by: images[4]).values()
  for frame in 0..<frames {
    var total = 0.0, weighted = 0.0
    for pixel in 0..<pixels {
      let value = Double(values[frame * pixels + pixel])
      total += value; weighted += value * Double(pixel / columns)
    }
    let expected = total == 0 ? 0 : weighted / total
    try require(abs(Double(com[frame]) - expected) < 2e-5, "CoM row convention or normalization differs.")
  }
  let numerator = try operations.image(values: [-8, 3, 4, 0], rows: 2, columns: 2)
  let denominator = try operations.image(values: [2, 0, -2, 5], rows: 2, columns: 2)
  let normalized = try operations.normalized(numerator, by: denominator).values()
  try require(normalized == [-4, 0, -2, 0], "Signed and zero-total normalization differs.")
  try require(numerator.values() == [-8, 3, 4, 0] && denominator.values() == [2, 0, -2, 5],
    "Normalization modified an input image.")
  do {
    _ = try operations.virtualImages(source: source, weights: masks, shouldCancel: { true })
    try require(false, "Cancellation did not stop batch reductions.")
  } catch Metal4DSTEMStreamingIOError.cancelled { }
  print("CALIBRATED_PRODUCTS_PASS frames=520 shape=7x9 masks=8 regions=2 zero_total=true signed=true scalar_bit_equal=true max_relative_error=\(maximumRelativeError)")
}

#if CALIBRATED_IMAGE_CHECK
@main struct CalibratedImageCheck {
  static func main() throws { try checkCalibratedImageProducts() }
}
#else
final class CalibratedImageProductsTests: XCTestCase {
  func testBatchProductsMatchIndependentReference() throws {
    guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Requires Metal") }
    try checkCalibratedImageProducts()
  }
}
#endif
