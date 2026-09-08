import Foundation
import Metal

func require(_ condition: Bool, _ message: String) throws {
  if !condition { throw NSError(domain: "ExactLZ4Fixture", code: 1,
    userInfo: [NSLocalizedDescriptionKey: message]) }
}

func extensionBytes(_ value: Int) -> [UInt8] {
  Array(repeating: 255, count: value / 255) + [UInt8(value % 255)]
}

func repeated(_ prefix: [UInt8], distance: Int) -> (compressed: [UInt8], expected: [UInt8]) {
  let match = 8187 - prefix.count
  var compressed = [UInt8(min(prefix.count, 15) << 4) | 15]
  if prefix.count >= 15 { compressed += extensionBytes(prefix.count - 15) }
  compressed += prefix + [UInt8(distance), UInt8(distance >> 8)]
  compressed += extensionBytes(match - 19)
  var expected = prefix
  for _ in 0..<match { expected.append(expected[expected.count - distance]) }
  compressed += [0x50, 3, 1, 4, 1, 5]
  expected += [3, 1, 4, 1, 5]
  return (compressed, expected)
}

let device = MTLCreateSystemDefaultDevice()!
let source = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let library = try device.makeLibrary(source: source, options: nil)
let queue = device.makeCommandQueue()!
let scratch = device.makeBuffer(length: 8192, options: .storageModeShared)!
let errors = device.makeBuffer(length: 4, options: .storageModeShared)!
let literal = (0..<8192).map { UInt8(truncatingIfNeeded: $0 * 37) }
var fixtures: [(compressed: [UInt8], expected: [UInt8])] =
  [([UInt8(0xf0)] + extensionBytes(8192 - 15) + literal, literal)]
let seeds: [([UInt8], Int)] = [([0], 1), ([0, 0], 2), ([255], 1), ([0, 255], 2),
  ([1, 2, 3], 3), ([0, 0, 0, 0], 4), (Array(repeating: 0, count: 16), 16),
  ((0..<31).map { UInt8($0) }, 7)]
for (prefix, distance) in seeds { fixtures.append(repeated(prefix, distance: distance)) }
let malformed: [[UInt8]] = [[0xf0, 255], [0x10, 1, 0, 0], [0x10, 1, 2, 0],
                          [0x1f, 0, 1, 0, 255], [0x50, 1, 2]]
var exactCases = 0, rejectedCases = 0
for name in ["h5lz4dc_full_u16_aligned_fill_qh5idx", "h5lz4dc_full_u16_cooperative_qh5idx"] {
  let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
  for (ordinal, fixture) in (fixtures + malformed.map { ($0, [UInt8]()) }).enumerated() {
    let (compressed, expected) = fixture
    let input = compressed.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    var metadata = SIMD2<UInt32>(0, UInt32(compressed.count))
    let blocks = device.makeBuffer(bytes: &metadata, length: 8, options: .storageModeShared)!
    // Reuse a poisoned buffer so the experimental path must really clear it.
    memset(scratch.contents(), 0xa5, scratch.length)
    memset(errors.contents(), 0, errors.length)
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(blocks, offset: 0, index: 1)
    var zero64: UInt64 = 0, one: UInt32 = 1, pixels: UInt32 = 4096, zero: UInt32 = 0
    encoder.setBytes(&zero64, length: 8, index: 2)
    encoder.setBytes(&one, length: 4, index: 3)
    encoder.setBytes(&pixels, length: 4, index: 4)
    encoder.setBuffer(scratch, offset: 0, index: 5)
    encoder.setBytes(&zero, length: 4, index: 6)
    encoder.setBuffer(errors, offset: 0, index: 10)
    encoder.setBytes(&one, length: 4, index: 11)
    encoder.dispatchThreads(MTLSize(width: name.contains("cooperative") ? 32 : 1, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try require(command.status == .completed, "Command failed: \(name), case\(ordinal)")
    let error = errors.contents().load(as: UInt32.self)
    if expected.isEmpty {
      try require(error != 0, "Malformed input accepted: \(name), case\(ordinal)")
      rejectedCases += 1
    } else {
      let observed = Array(UnsafeBufferPointer(
        start: scratch.contents().assumingMemoryBound(to: UInt8.self), count: 8192))
      try require(error == 0 && observed == expected, "Count mismatch: \(name), case\(ordinal)")
      exactCases += 1
    }
  }
}
let result: [String: Any] = ["exact_blocks": exactCases, "malformed_streams_rejected": rejectedCases,
                            "reused_poisoned_scratch": true, "bytes_per_block": 8192]
print(String(data: try JSONSerialization.data(withJSONObject: result, options: .sortedKeys), encoding: .utf8)!)
