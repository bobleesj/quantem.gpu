import Metal

/// Internal GPU table construction; CPU loops only encode logarithmic scan
/// levels. No width, chunk, prefix, or per-chunk status array returns to the CPU.
struct CompactH5MetadataKernels: @unchecked Sendable {
  private let scan: MTLComputePipelineState
  private let addCarries: MTLComputePipelineState
  private let descriptors: MTLComputePipelineState
  private let chunks: MTLComputePipelineState
  private let reduceStatus: MTLComputePipelineState

  init(device: MTLDevice, library: MTLLibrary) throws {
    func make(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable("Missing \(name).")
      }
      return try device.makeComputePipelineState(function: function)
    }
    scan = try make("compact_h5_scan_offsets")
    addCarries = try make("compact_h5_add_offset_carries")
    descriptors = try make("compact_h5_prepare_descriptors")
    chunks = try make("compact_h5_prepare_chunks")
    reduceStatus = try make("compact_h5_reduce_decode_status")
    guard scan.threadExecutionWidth == 32, scan.maxTotalThreadsPerThreadgroup >= 256 else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Compact GPU tables require 32-lane SIMD groups and 256-thread groups.")
    }
  }

  static func buffer(_ device: MTLDevice, bytes: Int, shared: Bool = false) throws -> MTLBuffer {
    guard
      let buffer = device.makeBuffer(
        length: max(4, bytes), options: shared ? .storageModeShared : .storageModePrivate
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "compact GPU metadata", bytes: UInt64(max(4, bytes)))
    }
    return buffer
  }

  /// Returns exclusive uint32 offsets. Malformed overflow sets status bit 0.
  func encodeOffsets(
    input: MTLBuffer, count: Int, kind: UInt32, status: MTLBuffer,
    device: MTLDevice, command: MTLCommandBuffer
  ) throws -> MTLBuffer {
    var levels: [(offsets: MTLBuffer, count: Int)] = []
    var current = input
    var currentCount = count
    var currentKind = kind
    while true {
      let blockCount = (currentCount + 255) / 256
      let offsets = try Self.buffer(device, bytes: currentCount * 4)
      let sums = try Self.buffer(device, bytes: blockCount * 4)
      var parameters = SIMD2<UInt32>(UInt32(currentCount), currentKind)
      let encoder = try makeEncoder(command, pipeline: scan)
      encoder.setBuffer(current, offset: 0, index: 0)
      encoder.setBuffer(offsets, offset: 0, index: 1)
      encoder.setBuffer(sums, offset: 0, index: 2)
      encoder.setBuffer(status, offset: 0, index: 3)
      encoder.setBytes(&parameters, length: MemoryLayout.size(ofValue: parameters), index: 4)
      dispatch(encoder, count: currentCount)
      levels.append((offsets, currentCount))
      if blockCount <= 1 { break }
      current = sums
      currentCount = blockCount
      currentKind = 2
    }
    if levels.count > 1 {
      for level in stride(from: levels.count - 2, through: 0, by: -1) {
        var levelCount = UInt32(levels[level].count)
        let encoder = try makeEncoder(command, pipeline: addCarries)
        encoder.setBuffer(levels[level].offsets, offset: 0, index: 0)
        encoder.setBuffer(levels[level + 1].offsets, offset: 0, index: 1)
        encoder.setBuffer(status, offset: 0, index: 2)
        encoder.setBytes(&levelCount, length: 4, index: 3)
        dispatch(encoder, count: Int(levelCount))
      }
    }
    return levels[0].offsets
  }

  func encodeTables(
    widths: MTLBuffer, lengths: MTLBuffer, descriptorCount: Int, chunkCount: Int,
    decodedWords: UInt32, compressedBytes: UInt32, chunkBytes: UInt32,
    descriptorOutput: MTLBuffer, chunkOutput: MTLBuffer, status: MTLBuffer,
    device: MTLDevice, command: MTLCommandBuffer
  ) throws {
    let parameters = [
      UInt32(descriptorCount), UInt32(chunkCount), decodedWords, compressedBytes, chunkBytes,
    ]
    // Two fixed metadata streams, not a loop over scientific elements.
    for (input, count, kind, output, pipeline) in [
      (widths, descriptorCount, UInt32(0), descriptorOutput, descriptors),
      (lengths, chunkCount, UInt32(1), chunkOutput, chunks),
    ] {
      let offsets = try encodeOffsets(
        input: input, count: count, kind: kind, status: status, device: device, command: command)
      let encoder = try makeEncoder(command, pipeline: pipeline)
      encoder.setBuffer(input, offset: 0, index: 0)
      encoder.setBuffer(offsets, offset: 0, index: 1)
      encoder.setBuffer(output, offset: 0, index: 2)
      encoder.setBuffer(status, offset: 0, index: 3)
      parameters.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
      dispatch(encoder, count: count)
    }
  }

  func encodeDecodeStatus(
    input: MTLBuffer, count: Int, status: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    var count = UInt32(count)
    let encoder = try makeEncoder(command, pipeline: reduceStatus)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(status, offset: 0, index: 1)
    encoder.setBytes(&count, length: 4, index: 2)
    dispatch(encoder, count: Int(count))
  }

  private func makeEncoder(
    _ command: MTLCommandBuffer, pipeline: MTLComputePipelineState
  ) throws -> MTLComputeCommandEncoder {
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable("Could not encode compact GPU tables.")
    }
    encoder.setComputePipelineState(pipeline)
    return encoder
  }

  private func dispatch(_ encoder: MTLComputeCommandEncoder, count: Int) {
    encoder.dispatchThreadgroups(
      MTLSize(width: (count + 255) / 256, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
  }
}
