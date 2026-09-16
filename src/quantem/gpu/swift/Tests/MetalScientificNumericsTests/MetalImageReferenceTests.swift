// Copyright (c) 2025 ophusgroup. MIT License; see the repository LICENSE.
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import MetalScientificNumerics

#if !SCIENTIFIC_NUMERICS_CHECK
  import XCTest
#endif

/// Run existing frozen scientific fixtures on hosts without the XCTest runtime.
func checkFrozenImageOperations(directory: URL) throws {
  func fixture(_ name: String) throws -> [String: Any] {
    try JSONSerialization.jsonObject(
      with: Data(
        contentsOf:
          directory.appendingPathComponent(name + ".json"))) as! [String: Any]
  }
  func require(_ condition: Bool, _ message: String) throws {
    if !condition {
      throw NSError(
        domain: "frozen-fixture", code: 1,
        userInfo: [NSLocalizedDescriptionKey: message])
    }
  }
  let data = try fixture("numpy")
  let ops = try MetalImageOperations()
  let raw = (data["raw"] as! [Int]).map(UInt16.init)
  let expected = (data["corrected"] as! [Int]).map(UInt16.init)
  let source = try MetalEncodedSource(
    shape: data["shape"] as! [Int],
    hotPixelIndices: data["bad"] as! [Int], device: ops.device)
  defer { source.releaseResidentStorage() }
  for range in [0..<5, 5..<17] {
    let buffer = ops.device.makeBuffer(length: range.count * 35 * 2, options: .storageModeShared)!
    Array(raw[(range.lowerBound * 35)..<(range.upperBound * 35)]).withUnsafeBytes {
      _ = memcpy(buffer.contents(), $0.baseAddress!, $0.count)
    }
    try source.append(buffer, frames: range.count, verify: true)
  }
  let decoded = try source.read(3..<15)
  let actual = Array(
    UnsafeBufferPointer(
      start: decoded.contents().assumingMemoryBound(to: UInt16.self), count: 12 * 35))
  try require(
    actual == Array(expected[(3 * 35)..<(15 * 35)]), "ANS/median counts differ from frozen NumPy")
  for (buffer, key, count) in [
    (source.meanDiffraction, "dp_mean", 35), (source.meanBrightField, "im_bf", 17),
  ] {
    let values = Array(
      UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    try require(values == (data[key] as! [Double]).map(Float.init), "Mean differs: \(key)")
  }
  let image = try ops.image(
    values: (data["image"] as! [Double]).map(Float.init), rows: 17, columns: 19)
  let blurred = try ops.gaussian(image, sigma: 1.25).values()
  for (a, b) in zip(blurred, (data["gaussian"] as! [Double]).map(Float.init)) {
    try require(abs(a - b) <= 1e-6, "Gaussian differs from frozen NumPy")
  }
  let translated = try ops.image(
    values: (data["translated"] as! [Double]).map(Float.init), rows: 17, columns: 19)
  for factor in [1, 2, 3, 100] {
    let shift = try ops.correlation(
      ops.fourier(image), ops.fourier(translated), upsample_factor: factor
    ).values()
    try require(abs(shift[0] + 3) <= 0.001 && abs(shift[1] - 4) <= 0.001, "Correlation differs")
  }
  let grid = try fixture("normalized_grid")
  let shape = grid["shape"] as! [Int]
  let shifted = try ops.shifted(
    ops.image(rows: shape[0], columns: shape[1], value: 1),
    shifts: ops.image(values: (grid["shift"] as! [Double]).map(Float.init), rows: 1, columns: 2),
    index: 0
  ).values()
  for (index, value) in zip(grid["indices"] as! [Int], grid["values"] as! [Double]) {
    try require(
      abs(shifted[index] - Float(value)) <= 2e-7, "Normalized-grid boundary differs at \(index)")
  }
  let torch = try fixture("torch_mps_parameters")
  func generated(_ rows: Int, _ columns: Int) throws -> GPUImage {
    try ops.image(
      values: (0..<(rows * columns)).map { Float(($0 * 37) % 251 - 125) * 0.03125 }, rows: rows,
      columns: columns)
  }
  func check(_ image: GPUImage, _ observation: [String: Any], operation: String) throws {
    let values = image.values()
    let indices = observation["indices"] as! [Int]
    if image.isComplex {
      for (index, expected) in zip(indices, observation["values"] as! [[Double]]) {
        try require(
          values[2 * index] == Float(expected[0]) && values[2 * index + 1] == Float(expected[1]),
          "Frozen MPS \(operation) mismatch at \(index): \(values[2 * index]), \(values[2 * index + 1]); expected \(expected)"
        )
      }
    } else {
      for (index, expected) in zip(indices, observation["values"] as! [Double]) {
        try require(
          values[index] == Float(expected),
          "Frozen MPS \(operation) mismatch at \(index): \(values[index]); expected \(Float(expected))"
        )
      }
    }
  }
  let full = try generated(512, 512)
  for (sigma, observation) in torch["gradient"] as! [String: [String: Any]] {
    try check(
      ops.gradientMagnitude(full, sigma: Double(sigma)!), observation,
      operation: "gradient sigma=\(sigma)")
  }
  try check(ops.fourier(generated(520, 520)), torch["fft"] as! [String: Any], operation: "fft")
  for (edge, observation) in torch["windows"] as! [String: [String: Any]] {
    try check(
      ops.window(ops.image(rows: 192, columns: 192, value: 1), kind: 1, edge_blend: Double(edge)!),
      observation, operation: "window edge=\(edge)")
  }
  try check(
    ops.centered(full, window: ops.image(rows: 512, columns: 512, value: 1)),
    torch["centered"] as! [String: Any], operation: "centered")
  print(
    "FROZEN_IMAGE_OPERATIONS_PASS numpy_counts_median_means=true gaussian=true correlation=true grid=true torch_mps=true"
  )
}

#if SCIENTIFIC_NUMERICS_CHECK
  @main struct ScientificNumericsCheck {
    static func main() throws {
      guard CommandLine.arguments.count == 3 else {
        fatalError("Pass the frozen image-operation and native IO fixture directories.")
      }
      try checkFrozenImageOperations(directory: URL(fileURLWithPath: CommandLine.arguments[1]))
      try checkCalibratedReductions()
      try checkEncodedSourceFiles(directory: URL(fileURLWithPath: CommandLine.arguments[2]))
    }
  }
#else
  final class MetalImageReferenceTests: XCTestCase {
    func testFrozenScientificOperations() throws {
      guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Requires Metal") }
      try checkFrozenImageOperations(
        directory: Bundle.module.resourceURL!.appendingPathComponent("Fixtures"))
    }
  }
#endif
