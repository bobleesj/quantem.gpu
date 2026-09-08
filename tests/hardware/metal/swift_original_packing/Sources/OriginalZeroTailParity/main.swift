import Foundation
import Metal
import Metal4DSTEMKernels

func require(_ condition: Bool, _ message: String) throws {
  if !condition { throw NSError(domain: "ExactLZ4Fixture", code: 1,
    userInfo: [NSLocalizedDescriptionKey: message]) }
}

func extensionBytes(_ value: Int) -> [UInt8] {
  Array(repeating: 255, count: value / 255) + [UInt8(value % 255)]
}

func repeated(_ prefix: [UInt8], distance: Int, tail: [UInt8] = [3, 1, 4, 1, 5]) -> (compressed: [UInt8], expected: [UInt8]) {
  let match = 8192 - tail.count - prefix.count
  var compressed = [UInt8(min(prefix.count, 15) << 4) | 15]
  if prefix.count >= 15 { compressed += extensionBytes(prefix.count - 15) }
  compressed += prefix + [UInt8(truncatingIfNeeded: distance), UInt8(distance >> 8)]
  compressed += extensionBytes(match - 19)
  var expected = prefix
  for _ in 0..<match { expected.append(expected[expected.count - distance]) }
  compressed += [UInt8(tail.count << 4)] + tail
  expected += tail
  return (compressed, expected)
}

let device = MTLCreateSystemDefaultDevice()!
let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
let queue = device.makeCommandQueue()!
let scratch = device.makeBuffer(length: 8192, options: .storageModeShared)!
let errors = device.makeBuffer(length: 4, options: .storageModeShared)!
let tailsBuffer = device.makeBuffer(length: 32, options: .storageModeShared)!
let literal = (0..<8192).map { UInt8(truncatingIfNeeded: $0 * 37) }
var fixtures: [(compressed: [UInt8], expected: [UInt8])] =
  [([UInt8(0xf0)] + extensionBytes(8192 - 15) + literal, literal)]
let seeds: [([UInt8], Int)] = [([0], 1), ([0, 0], 2), ([255], 1), ([0, 255], 2),
  ([1, 2, 3], 3), ([0, 0, 0, 0], 4), (Array(repeating: 0, count: 16), 16),
  ((0..<31).map { UInt8($0) }, 7)]
for (prefix, distance) in seeds { fixtures.append(repeated(prefix, distance: distance)) }
for size in [511, 512, 513, 1023, 1024, 1535] {
  var prefix = Array(repeating: UInt8(0), count: size)
  prefix[size - 1] = 255
  for distance in [1, 2, 3, 4, 31, 511] {
    fixtures.append(repeated(prefix, distance: distance))
  }
}
for plane in 0..<16 {
  var values = Array(repeating: UInt8(0), count: 8192)
  values[plane * 512 + 511] = 255
  fixtures.append(([0xf0] + extensionBytes(8192 - 15) + values, values))
}
let allZero = Array(repeating: UInt8(0), count: 8192)
fixtures.append(([0xf0] + extensionBytes(8192 - 15) + allZero, allZero))
// Terminal zeros exercise the skipped-write path; nonzero tails retain the baseline.
for fixture in fixtures {
  if fixture.compressed.suffix(6).first == 0x50 && fixture.expected.count == 8192 {
    var compressed = fixture.compressed, expected = fixture.expected
    for index in 0..<5 { compressed[compressed.count - 1 - index] = 0; expected[8191 - index] = 0 }
    fixtures.append((compressed, expected))
  }
}
for size in Array(1...33) + [511, 512, 513, 1023, 1535, 4095, 4096, 4097, 8140] {
  for count in 0..<15 {
    var prefix = Array(repeating: UInt8(0), count: size)
    if size > 3 { prefix[size - 3] = 255 }
    fixtures.append(repeated(prefix, distance: 1, tail: Array(repeating: 0, count: count)))
    if size >= 2 { fixtures.append(repeated(prefix, distance: 2, tail: Array(repeating: 0, count: count))) }
  }
}
var malformed: [[UInt8]] = [[0xf0, 255], [0x10, 1, 0, 0], [0x10, 1, 2, 0],
                          [0x1f, 0, 1, 0, 255], [0x50, 1, 2]]
for size in [1, 2, 15, 16, 17, 511, 512, 513] {
  for count in 1..<15 {
    let fixture = repeated(Array(repeating: 0, count: size), distance: 1,
                           tail: Array(repeating: 0, count: count))
    malformed.append(Array(fixture.compressed.dropLast()))
    var invalidToken = fixture.compressed
    invalidToken[invalidToken.count - count - 1] = 0xf0
    malformed.append(invalidToken)
  }
}
var exactCases = 0, rejectedCases = 0, skippedCases = 0
for name in ["h5lz4dc_full_u16_aligned_fill_qh5idx", "h5lz4dc_full_u16_zero_tail_qh5idx"] {
  let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
  for (ordinal, fixture) in (fixtures + malformed.map { ($0, [UInt8]()) }).enumerated() {
    let (compressed, expected) = fixture
    let input = compressed.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    var metadata = SIMD2<UInt32>(0, UInt32(compressed.count))
    let blocks = device.makeBuffer(bytes: &metadata, length: 8, options: .storageModeShared)!
    // Reuse a poisoned buffer so the experimental path must not consume unwritten bytes.
    memset(scratch.contents(), 0xa5, scratch.length)
    memset(errors.contents(), 0, errors.length)
    memset(tailsBuffer.contents(), 0xff, tailsBuffer.length)
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
    var tailOffset: UInt32 = 3
    if name.contains("zero_tail") {
      encoder.setBuffer(tailsBuffer, offset: 0, index: 12)
      encoder.setBytes(&tailOffset, length: 4, index: 13)
    }
    encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
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
      var observed = Array(UnsafeBufferPointer(
        start: scratch.contents().assumingMemoryBound(to: UInt8.self), count: 8192))
      if name.contains("zero_tail") {
        let tails = tailsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let tail = Int(tails[3])
        try require(tail <= 8192 && tail % 16 == 0, "Invalid aligned zero tail")
        try require((0..<8).filter { $0 != 3 }.allSatisfy { tails[$0] == UInt32.max },
                    "Tail write escaped its bounded slot")
        try require(expected[tail..<8192].allSatisfy { $0 == 0 }, "Nonzero counts omitted")
        if tail < 8192 { skippedCases += 1 }
        for byte in tail..<8192 { observed[byte] = 0 }
      }
      try require(error == 0 && observed == expected, "Count mismatch: \(name), case\(ordinal)")
      exactCases += 1
    }
  }
}
try require(skippedCases > 0, "Experimental fast path was never exercised")
let result: [String: Any] = ["exact_blocks": exactCases, "skipped_zero_tails": skippedCases, "malformed_streams_rejected": rejectedCases,
                            "reused_poisoned_scratch": true, "bytes_per_block": 8192]
print(String(data: try JSONSerialization.data(withJSONObject: result, options: .sortedKeys), encoding: .utf8)!)
print("ZERO_TAIL_EXACT_2702_REJECTED_458_PASS")
