import Foundation
import Metal

/// Standalone real-Metal check for machines with Command Line Tools only.
/// Usage: swiftc -O -parse-as-library hdf5_hot_pixels_check.swift -o check;
///        ./check path/to/qh5idx.metal
@main
struct HDF5HotPixelsCheck {
  static func main() throws {
    guard CommandLine.arguments.count == 2,
      let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue()
    else { throw failure("Supply qh5idx.metal on a Mac with Metal.") }
    let source = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
    // Compile the complete shipped decode library, not an extracted test kernel.
    // This catches duplicate entry points left by an integration conflict.
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "h5hot_pixel_median_qh5idx") else {
      throw failure("The source-marked median kernel is missing.")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    let cases: [([UInt32], [UInt32], [UInt32])] = [
      ([1, 2, 3, 4, 255, 6, 7, 8, 9], [4], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
      ([255, 255, 3, 4, 255, 6, 7, 8, 9], [0, 1, 4], [4, 4, 3, 4, 6, 6, 7, 8, 9]),
      (Array(repeating: 255, count: 9), Array(0..<9), Array(repeating: 0, count: 9)),
    ]
    var passed = 0
    for bytes in [1, 2, 4] {
      for (input, marked, expected) in cases {
        try check(input, marked: marked, expected: expected, bytes: bytes,
          device: device, queue: queue, pipeline: pipeline)
        passed += 1
      }
    }
    let high = UInt32.max - 8
    try check((0..<9).map { high + UInt32($0) }, marked: [4],
      expected: (0..<9).map { high + UInt32($0) }, bytes: 4,
      device: device, queue: queue, pipeline: pipeline)
    passed += 1
    print("PASS: \(passed) exact median cases; full HDF5 library compiled on \(device.name)")
  }

  private static func check(
    _ input: [UInt32], marked: [UInt32], expected: [UInt32], bytes: Int,
    device: MTLDevice, queue: MTLCommandQueue, pipeline: MTLComputePipelineState
  ) throws {
    // Two frames share the reusable window; verify both, including every
    // unmarked value. Corrections must not leak between detector frames.
    let frames = 2
    guard let values = device.makeBuffer(length: input.count * bytes * frames),
      let mask = device.makeBuffer(length: input.count),
      let indices = device.makeBuffer(bytes: marked, length: marked.count * 4),
      let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else { throw failure("Cannot allocate the small median test window.") }
    memset(mask.contents(), 0, mask.length)
    for pixel in marked { mask.contents().storeBytes(of: UInt8(1), toByteOffset: Int(pixel), as: UInt8.self) }
    for position in 0..<(input.count * frames) {
      let value = input[position % input.count]
      switch bytes {
      case 1: values.contents().storeBytes(of: UInt8(value), toByteOffset: position, as: UInt8.self)
      case 2: values.contents().storeBytes(of: UInt16(value), toByteOffset: position * 2, as: UInt16.self)
      default: values.contents().storeBytes(of: value, toByteOffset: position * 4, as: UInt32.self)
      }
    }
    var dimensions: [UInt32] = [UInt32(frames), 3, 3, UInt32(bytes), UInt32(marked.count)]
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(values, offset: 0, index: 0)
    encoder.setBuffer(mask, offset: 0, index: 1)
    encoder.setBuffer(indices, offset: 0, index: 2)
    encoder.setBytes(&dimensions, length: dimensions.count * 4, index: 3)
    encoder.dispatchThreads(MTLSize(width: frames * marked.count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit(); command.waitUntilCompleted()
    if let error = command.error { throw error }
    for position in 0..<(input.count * frames) {
      let actual: UInt32
      switch bytes {
      case 1: actual = UInt32(values.contents().load(fromByteOffset: position, as: UInt8.self))
      case 2: actual = UInt32(values.contents().load(fromByteOffset: position * 2, as: UInt16.self))
      default: actual = values.contents().load(fromByteOffset: position * 4, as: UInt32.self)
      }
      guard actual == expected[position % input.count] else {
        throw failure("uint\(bytes * 8), position \(position): \(actual), expected \(expected[position % input.count]).")
      }
    }
  }

  private static func failure(_ message: String) -> NSError {
    NSError(domain: "HDF5HotPixelsCheck", code: 1,
      userInfo: [NSLocalizedDescriptionKey: message])
  }
}
