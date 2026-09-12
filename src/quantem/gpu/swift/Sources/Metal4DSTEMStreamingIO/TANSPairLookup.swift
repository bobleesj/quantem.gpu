import Foundation
import Metal

/// Opt-in exact codebook metadata, with no decoded scientific payload.
enum TANSPairLookup {
  static func make(
    device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary,
    decoding: MTLBuffer, bits: Int, maximumBytes: UInt64
  ) throws -> MTLBuffer {
    guard [4, 6].contains(bits), decoding.length > 0, decoding.length % 4096 == 0 else {
      throw TANSArchive.invalid("Pair lookup requires complete 1024-state models and 4 or 6 bits")
    }
    let entries = (decoding.length / 4).multipliedReportingOverflow(by: 1 << bits)
    let bytes = entries.partialValue.multipliedReportingOverflow(by: 4)
    guard !entries.overflow, !bytes.overflow, entries.partialValue <= Int(UInt32.max),
      UInt64(bytes.partialValue) <= maximumBytes, bytes.partialValue <= device.maxBufferLength
    else { throw TANSArchive.invalid("Exact pair lookup exceeds its explicit metadata budget") }
    guard let function = library.makeFunction(name: "tans_prepare_pair_lookup") else {
      throw TANSArchive.invalid("Exact pair-lookup preparation kernel is unavailable")
    }
    // Compilation may throw; finish it before opening a command encoder.
    let pipeline = try device.makeComputePipelineState(function: function)
    guard let result = device.makeBuffer(length: bytes.partialValue, options: .storageModePrivate),
      let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else { throw TANSArchive.invalid("Cannot allocate exact pair-lookup preparation") }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(decoding, offset: 0, index: 0)
    encoder.setBuffer(result, offset: 0, index: 1)
    var info = [UInt32(entries.partialValue), UInt32(bits)]
    encoder.setBytes(&info, length: 8, index: 2)
    encoder.dispatchThreads(
      MTLSize(width: entries.partialValue, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Exact pair lookup failed; no metadata published")
    }
    return result
  }
}
