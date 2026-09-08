import Darwin
import Foundation
import Metal

// Standalone experiment. No source values are converted or packed on the CPU.
// Reads original bytes directly into one bounded shared Metal input buffer.
func require(_ condition: Bool, _ message: String) throws {
  if !condition { throw NSError(domain: "FloatPackingAudit", code: 1,
                                userInfo: [NSLocalizedDescriptionKey: message]) }
}

func run() throws {
  let arguments = CommandLine.arguments
  try require(arguments.count == 4, "Usage: audit ORIGINAL.npy PRODUCTION.metal AUDIT.metal")
  guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    throw NSError(domain: "MetalUnavailable", code: 1)
  }
  let shader = try String(contentsOfFile: arguments[2], encoding: .utf8)
    + "\n" + String(contentsOfFile: arguments[3], encoding: .utf8)
  let options = MTLCompileOptions()
  options.fastMathEnabled = false
  let library = try device.makeLibrary(source: shader, options: options)
  func pipeline(_ name: String) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
      throw NSError(domain: "MissingKernel", code: 1)
    }
    return try device.makeComputePipelineState(function: function)
  }
  let describe = try pipeline("empad_describe_simd")
  let pack = try pipeline("empad_pack_simd")
  let localOffsets = try pipeline("audit_local_offsets")
  let groupOffsets = try pipeline("audit_group_offsets")
  let addOffsets = try pipeline("audit_add_offsets")
  let verify = try pipeline("audit_verify")
  for state in [describe, pack, localOffsets, groupOffsets, addOffsets, verify] {
    try require(state.threadExecutionWidth == 32 && state.maxTotalThreadsPerThreadgroup >= 256,
                "Audit requires 32-lane SIMD and 256-thread-capable pipelines")
  }
  let descriptor = open(arguments[1], O_RDONLY)
  try require(descriptor >= 0, "Cannot open the original NumPy file")
  defer { close(descriptor) }
  var before = stat()
  try require(fstat(descriptor, &before) == 0, "Cannot inspect original file")
  let totalFrames = 512 * 512, pixels = 192 * 192, windowFrames = 64
  let wordCount = windowFrames * pixels, blocks = wordCount / 128, groups = blocks / 256
  try require(before.st_size == 128 + Int64(totalFrames * pixels * 4), "Incomplete original file")
  var header = [UInt8](repeating: 0, count: 128)
  try require(read(descriptor, &header, 128) == 128, "Cannot read NumPy header")
  let text = String(bytes: header.dropFirst(10), encoding: .ascii) ?? ""
  try require(Array(header.prefix(8)) == [147, 78, 85, 77, 80, 89, 1, 0]
              && header[8] == 118 && header[9] == 0
              && text.contains("'descr': '<f4'") && text.contains("'fortran_order': False")
              && text.contains("(512, 512, 192, 192)"),
              "Audit requires the verified original C-order float32 array, without conversions")
  let allocationBefore = device.currentAllocatedSize
  func buffer(_ length: Int, _ shared: Bool = false) throws -> MTLBuffer {
    guard let result = device.makeBuffer(length: length,
                                        options: shared ? .storageModeShared : .storageModePrivate) else {
      throw NSError(domain: "AllocationFailed", code: 1)
    }
    return result
  }
  let input = try buffer(wordCount * 4, true)
  let payload = try buffer(wordCount * 4)
  let descriptors = try buffer(blocks * 16)
  let totals = try buffer(groups * 4)
  let offsets = try buffer((groups + 1) * 4, true)
  let mismatches = try buffer(4, true)
  try require(device.currentAllocatedSize < 256 * 1024 * 1024, "Audit exceeded its 256 MiB budget")
  var packedBytes: UInt64 = 0, descriptorBytes: UInt64 = 0
  var readSeconds = 0.0, packingSeconds = 0.0, verifySeconds = 0.0
  var peak = device.currentAllocatedSize
  let started = CFAbsoluteTimeGetCurrent()
  func command() throws -> MTLCommandBuffer {
    guard let result = queue.makeCommandBuffer() else { throw NSError(domain: "CommandFailed", code: 1) }
    return result
  }
  func encode(_ command: MTLCommandBuffer, _ state: MTLComputePipelineState,
              _ buffers: [MTLBuffer], _ count: Int, _ threads: Int,
              _ constant: UInt32? = nil) throws {
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw NSError(domain: "EncoderFailed", code: 1)
    }
    encoder.setComputePipelineState(state)
    for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
    if var constant { encoder.setBytes(&constant, length: 4, index: buffers.count) }
    encoder.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    encoder.endEncoding()
  }
  func finish(_ command: MTLCommandBuffer) throws -> Double {
    command.commit()
    command.waitUntilCompleted()
    try require(command.status == .completed, "Metal command failed: \(String(describing: command.error))")
    return command.gpuEndTime - command.gpuStartTime
  }
  for first in stride(from: 0, to: totalFrames, by: windowFrames) {
    try autoreleasepool {
      let readStarted = CFAbsoluteTimeGetCurrent()
      var received = 0
      while received < input.length {
        let count = read(descriptor, input.contents().advanced(by: received), input.length - received)
        if count < 0 && errno == EINTR { continue }
        try require(count > 0, "Incomplete original data at frame \(first)")
        received += count
      }
      readSeconds += CFAbsoluteTimeGetCurrent() - readStarted
      let packing = try command()
      try encode(packing, describe, [input, descriptors], blocks, 32)
      try encode(packing, localOffsets, [descriptors, totals], groups, 256)
      try encode(packing, groupOffsets, [totals, offsets], 1, 256, UInt32(groups))
      try encode(packing, addOffsets, [descriptors, offsets], groups, 256)
      try encode(packing, pack, [input, descriptors, payload], blocks, 32)
      packingSeconds += try finish(packing)
      let words = offsets.contents().load(fromByteOffset: groups * 4, as: UInt32.self)
      try require(words <= wordCount, "Packed payload exceeds bounded allocation")
      packedBytes += UInt64(words) * 4
      descriptorBytes += UInt64(descriptors.length)
      mismatches.contents().storeBytes(of: UInt32(0), as: UInt32.self)
      let checking = try command()
      try encode(checking, verify, [input, payload, descriptors, mismatches], wordCount / 256, 256)
      verifySeconds += try finish(checking)
      try require(mismatches.contents().load(as: UInt32.self) == 0,
                  "Exact GPU unpacking differs from original bits at frame \(first)")
      peak = max(peak, device.currentAllocatedSize)
      try require(peak < 256 * 1024 * 1024, "Audit allocation exceeded 256 MiB")
    }
    if first % 16384 == 0 { fputs("METAL_AUDIT verified_frames=\(first + windowFrames)/\(totalFrames)\n", stderr) }
  }
  var after = stat()
  try require(fstat(descriptor, &after) == 0 && before.st_ino == after.st_ino
              && before.st_size == after.st_size && before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec
              && before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              "Original source changed during audit")
  let result: [String: Any] = [
    "scope": "Whole-source Metal bit-packing and unpacking parity, bounded windows, not full residency",
    "device": device.name, "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
    "shape": [512, 512, 192, 192], "dtype": "float32", "words_verified": totalFrames * pixels,
    "payload_bytes": packedBytes, "descriptor_bytes": descriptorBytes,
    "required_resident_bytes_before_alignment": packedBytes + descriptorBytes,
    "peak_audit_metal_allocated_bytes": peak, "metal_allocation_before_buffers": allocationBefore,
    "exact_gpu_word_parity": true, "cpu_codec": false, "cpu_prefix_scan": false,
    "read_seconds": readSeconds, "packing_gpu_seconds": packingSeconds,
    "unpack_and_verify_gpu_seconds": verifySeconds, "audit_wall_seconds": CFAbsoluteTimeGetCurrent() - started,
    "full_resident_loaded": false, "native_app_loaded": false, "cold_io_claim": false
  ]
  let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
  print(String(decoding: json, as: UTF8.self))
}

do { try run() } catch { fputs("AUDIT_FAILED \(error)\n", stderr); exit(1) }
