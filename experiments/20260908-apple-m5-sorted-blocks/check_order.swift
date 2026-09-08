import Foundation
import Metal

func require(_ condition: Bool, _ message: String) throws {
  if !condition { throw NSError(domain: "SortedLZ4Fixture", code: 1,
    userInfo: [NSLocalizedDescriptionKey: message]) }
}

func extensions(_ value: Int) -> [UInt8] {
  Array(repeating: 255, count: value / 255) + [UInt8(value % 255)]
}

let device = MTLCreateSystemDefaultDevice()!
if CommandLine.arguments.count > 2 {
  _ = try device.makeLibrary(
    source: String(contentsOfFile: CommandLine.arguments[2], encoding: .utf8), options: nil)
}
let library = try device.makeLibrary(
  source: String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: nil)
let pipeline = try device.makeComputePipelineState(
  function: library.makeFunction(name: "h5lz4dc_full_u16_sorted_qh5idx")!)
let queue = device.makeCommandQueue()!
var checked = 0
for count in [9, 63, 270] {
  var compressed = [UInt8](), reference = [UInt8](), pairs = [SIMD2<UInt32>]()
  for block in 0..<count {
    let bytes = (0..<8192).map { UInt8(truncatingIfNeeded: $0 * 37 + block) }
    let encoded: [UInt8]
    let expected: [UInt8]
    if block % 3 == 0 {
      encoded = [0xf0] + extensions(8192 - 15) + bytes
      expected = bytes
    } else {
      let seed = UInt8(truncatingIfNeeded: block)
      encoded = [0x1f, seed, 1, 0] + extensions(8186 - 19) + [0x50, 3, 1, 4, 1, 5]
      expected = Array(repeating: seed, count: 8187) + [3, 1, 4, 1, 5]
    }
    pairs.append(SIMD2(UInt32(compressed.count), UInt32(encoded.count)))
    compressed += encoded
    reference += expected
  }
  for bucketed in [false, true] {
    var order = (0..<count).map { block in
      (UInt64(bucketed ? pairs[block].y >> 8 : pairs[block].y) << 32) | UInt64(block)
    }
    order.sort()
    let metadata = order.map { key -> SIMD2<UInt32> in
      let block = Int(key & 0xffffffff)
      return SIMD2(pairs[block].x, pairs[block].y | (UInt32(block) << 14))
    }
    try require(Set(order.map { $0 & 0xffffffff }).count == count, "Not a permutation")
    let input = compressed.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
    let indices = metadata.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
    let scratch = device.makeBuffer(length: count * 8192, options: .storageModeShared)!
    let errors = device.makeBuffer(length: 4, options: .storageModeShared)!
    memset(scratch.contents(), 0xa5, scratch.length)
    memset(errors.contents(), 0, 4)
    let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(indices, offset: 0, index: 1)
    var zero64: UInt64 = 0, blocks: UInt32 = 9, pixels: UInt32 = 36864
    var zero: UInt32 = 0, frames = UInt32(count / 9)
    encoder.setBytes(&zero64, length: 8, index: 2)
    encoder.setBytes(&blocks, length: 4, index: 3)
    encoder.setBytes(&pixels, length: 4, index: 4)
    encoder.setBuffer(scratch, offset: 0, index: 5)
    encoder.setBytes(&zero, length: 4, index: 6)
    encoder.setBuffer(errors, offset: 0, index: 10)
    encoder.setBytes(&frames, length: 4, index: 11)
    encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
    try require(command.status == .completed && errors.contents().load(as: UInt32.self) == 0,
      "Decode failed")
    let actual = Array(UnsafeBufferPointer(start: scratch.contents().assumingMemoryBound(to: UInt8.self), count: count * 8192))
    try require(actual == reference, "Reordered output differs")
    checked += count
  }
}
print("{\"exact_reordered_blocks\":\(checked),\"bytes_per_block\":8192,\"new_gpu_metadata_bytes\":0}")
