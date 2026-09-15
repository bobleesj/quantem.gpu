import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels

private let fineScans = 512
private let fineLeaves = 2_304
private let fineRoots = 144
private let fineFields = 2_448

private enum FinePolarError: Error { case failed(String) }

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  if !condition() { throw FinePolarError.failed(message) }
}

private func makeBuffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
  guard let result = values.withUnsafeBytes({ bytes in
    device.makeBuffer(bytes: bytes.baseAddress!, length: max(bytes.count, 1), options: .storageModeShared)
  }) else { throw FinePolarError.failed("Metal allocation failed") }
  return result
}

/// Deterministic exact uint16 detector evidence without retaining a 36 MiB raw duplicate.
private func rawCount(pixel: Int, scan: Int) -> UInt16 {
  if scan == 64 && pixel < 256 { return .max }
  switch pixel % 11 {
  case 0: return 0
  case 1: return UInt16(100 + pixel % 997)
  case 2: return scan.isMultiple(of: 37) ? .max : 0
  case 3: return UInt16((scan + pixel) & 255)
  case 4: return UInt16(truncatingIfNeeded: scan &* 257 &+ pixel &* 31)
  default: return UInt16(truncatingIfNeeded: scan &* 40_503 &+ pixel &* 7_919)
  }
}

private func reference() throws -> [UInt32] {
  var result = [UInt32](repeating: 0, count: fineFields * fineScans)
  for leaf in 0..<fineLeaves {
    for scan in 0..<fineScans {
      var sum = UInt64(0)
      for within in 0..<16 { sum += UInt64(rawCount(pixel: leaf * 16 + within, scan: scan)) }
      try require(sum <= UInt64(UInt32.max), "Fine leaf overflow")
      result[leaf * fineScans + scan] = UInt32(sum)
    }
  }
  // Compute roots independently from all 256 raw counts rather than summing the
  // already-built CPU leaf reference.
  for root in 0..<fineRoots {
    for scan in 0..<fineScans {
      var sum = UInt64(0)
      for within in 0..<256 { sum += UInt64(rawCount(pixel: root * 256 + within, scan: scan)) }
      try require(sum <= UInt64(UInt32.max), "Fine root overflow")
      result[(fineLeaves + root) * fineScans + scan] = UInt32(sum)
    }
  }
  try require(result[fineLeaves * fineScans + 64] == UInt32(16 * 16 * 65_535),
    "Fine wide-root sentinel is missing")
  return result
}

@main
struct FinePolarFieldCorrectnessProbe {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw FinePolarError.failed("Metal unavailable")
    }
    let expected = try reference()
    var input = [UInt32](repeating: 0, count: fineFields * fineScans)
    input.replaceSubrange(0..<(fineLeaves * fineScans),
      with: expected[0..<(fineLeaves * fineScans)])
    let fields = try makeBuffer(input, device: device)
    let failure = try makeBuffer([UInt32(0)], device: device)
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    guard let function = library.makeFunction(name: "paired_runtime_tans_polar_roots_in_place") else {
      throw FinePolarError.failed("Missing fine polar roots kernel")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
      throw FinePolarError.failed("Could not create fine polar command")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(fields, offset: 0, index: 0)
    encoder.setBuffer(failure, offset: 0, index: 1)
    var parameters: [UInt32] = [1, UInt32(fineLeaves), UInt32(fineFields)]
    encoder.setBytes(&parameters, length: parameters.count * 4, index: 2)
    encoder.dispatchThreads(MTLSize(width: fineRoots * fineScans, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
    try require(command.status == .completed && failure.contents().load(as: UInt32.self) == 0,
      "Fine polar roots kernel failed")
    let actual = Array(UnsafeBufferPointer(
      start: fields.contents().assumingMemoryBound(to: UInt32.self), count: fineFields * fineScans))
    try require(actual == expected,
      "Fine in-place leaves/roots differ from independent raw-count UInt64 reference")

    memset(failure.contents(), 0, failure.length)
    guard let malformed = queue.makeCommandBuffer(), let malformedEncoder = malformed.makeComputeCommandEncoder() else {
      throw FinePolarError.failed("Could not create malformed fine polar command")
    }
    malformedEncoder.setComputePipelineState(pipeline)
    malformedEncoder.setBuffer(fields, offset: 0, index: 0)
    malformedEncoder.setBuffer(failure, offset: 0, index: 1)
    var invalid: [UInt32] = [1, UInt32(fineLeaves - 1), UInt32(fineFields)]
    malformedEncoder.setBytes(&invalid, length: invalid.count * 4, index: 2)
    malformedEncoder.dispatchThreads(MTLSize(width: fineRoots * fineScans, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    malformedEncoder.endEncoding(); malformed.commit(); malformed.waitUntilCompleted()
    try require(malformed.status == .completed && failure.contents().load(as: UInt32.self) != 0,
      "Malformed fine polar extent was not rejected")
    print("PASS fine polar 2304 leaves, 144 roots, raw-count UInt64 parity, malformed extent")
  }
}
