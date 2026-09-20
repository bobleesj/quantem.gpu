import Foundation
import Metal
import Metal4DSTEMKernels

/// Exact IEEE words stored as two little-endian UInt16 ANS streams per float.
/// Only a bounded query window is materialized; no numeric cast is performed.
final class MetalFloatANS {
  static let codec = "float32-bit-lanes-rans-v1"
  static let schema = "quantem.gpu.float32-bit-lanes-rans/v1"
  static let pixels = 128 * 128
  static let lanes = pixels * 2
  let table: MTLBuffer
  private let decode: MTLComputePipelineState
  private let join: MTLComputePipelineState
  private let selected: MTLComputePipelineState
  private let recovery: MTLComputePipelineState
  private let changes: MTLComputePipelineState
  private let device: MTLDevice
  private var scratch: Workspace?
  private var scratchCommand: MTLCommandBuffer?

  struct Workspace {
    let lanes: MTLBuffer
    let words: MTLBuffer
    let descriptors: MTLBuffer
    let errors: MTLBuffer
  }

  init(device: MTLDevice) throws {
    self.device = device
    let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
    guard let decode = library.makeFunction(name: "streamed_counts_decode_range"),
      let join = library.makeFunction(name: "float_ans_join_words"),
      let selected = library.makeFunction(name: "float_ans_decode_selected"),
      let recovery = library.makeFunction(name: "float_ans_recovery_needed"),
      let changes = library.makeFunction(name: "float_ans_decode_changes")
    else { throw Metal4DSTEMStreamingIOError.invalidRequest("Rebuild the float ANS kernels.") }
    self.decode = try device.makeComputePipelineState(function: decode)
    self.join = try device.makeComputePipelineState(function: join)
    self.selected = try device.makeComputePipelineState(function: selected)
    self.recovery = try device.makeComputePipelineState(function: recovery)
    self.changes = try device.makeComputePipelineState(function: changes)
    let values = RuntimeANSEncoder.tables().decoding
    guard
      let table = values.withUnsafeBytes({
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    else { throw Metal4DSTEMStreamingIOError.invalidRequest("Free memory for the ANS table.") }
    self.table = table
  }

  func releaseScratch() {
    scratch = nil
    scratchCommand = nil
  }

  func recoveryFlag(
    accumulated: MTLBuffer, reset: Bool, frames: Int,
    command: MTLCommandBuffer
  ) throws -> MTLBuffer {
    guard let flag = device.makeBuffer(length: 4, options: .storageModeShared) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("Cannot allocate the ANS recovery flag.")
    }
    flag.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    if !reset {
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.invalidRequest("Cannot prepare the ANS detector update.")
      }
      encoder.setComputePipelineState(recovery)
      encoder.setBuffer(accumulated, offset: 0, index: 0)
      encoder.setBuffer(flag, offset: 0, index: 1)
      encoder.dispatchThreads(
        MTLSize(width: frames, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      encoder.endEncoding()
    }
    return flag
  }

  func encodeSelected(
    _ chunk: MetalEMPADResidentSource.Chunk, changed: MTLBuffer,
    entries: MTLBuffer, count: Int,
    mask: MTLBuffer, recovery: MTLBuffer, into workspace: Workspace,
    encoder: MTLComputeCommandEncoder
  ) throws {
    encoder.setComputePipelineState(selected)
    for (index, buffer) in [
      chunk.payload, chunk.offsets, chunk.models, table,
      workspace.errors, workspace.words, workspace.descriptors, changed, mask, recovery,
    ].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    var scans = UInt32(chunk.frameCount)
    encoder.setBytes(&scans, length: 4, index: 10)
    encoder.dispatchThreads(
      MTLSize(width: Self.pixels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    if count > 0 {
      encoder.setComputePipelineState(changes)
      encoder.setBuffer(entries, offset: 0, index: 6)
      encoder.setBytes(&scans, length: 4, index: 7)
      encoder.dispatchThreadgroups(
        MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
  }

  func workspace(frames: Int, budget: UInt64, command: MTLCommandBuffer) throws -> Workspace {
    let size = frames * Self.pixels
    // Ordered encoders in one command may reuse the same window. A different
    // in-flight command must retain its own buffers until it has completed.
    if let scratch, scratch.words.length >= size * 4,
      scratchCommand === command || scratchCommand?.status == .completed
    {
      scratchCommand = command
      return scratch
    }
    let needed = UInt64(size * 12 + frames * 128 * 16 + 65536)
    guard frames > 0, frames <= 512,
      UInt64(device.currentAllocatedSize) <= budget,
      needed <= budget - UInt64(device.currentAllocatedSize),
      let lanes = device.makeBuffer(length: size * 8, options: .storageModePrivate),
      let words = device.makeBuffer(length: size * 4, options: .storageModePrivate),
      let descriptors = device.makeBuffer(length: frames * 128 * 16, options: .storageModePrivate),
      let errors = device.makeBuffer(length: 4, options: .storageModeShared)
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Free memory for a bounded ANS query window; no dense fallback is used.")
    }
    errors.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    let result = Workspace(lanes: lanes, words: words, descriptors: descriptors, errors: errors)
    scratch = result
    scratchCommand = command
    return result
  }

  func encode(
    _ chunk: MetalEMPADResidentSource.Chunk, first: Int = 0, count: Int? = nil,
    into workspace: Workspace, command: MTLCommandBuffer
  ) throws {
    let count = count ?? chunk.frameCount
    guard first >= 0, count > 0, first + count <= chunk.frameCount,
      workspace.words.length >= count * Self.pixels * 4,
      let encoder = command.makeComputeCommandEncoder()
    else { throw Metal4DSTEMStreamingIOError.invalidRequest("Invalid float ANS decode window.") }
    encoder.setComputePipelineState(decode)
    for (index, buffer) in [
      chunk.payload, chunk.offsets, chunk.models, table,
      workspace.errors, workspace.lanes,
    ].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    var parameters: [UInt64] = [
      UInt64(chunk.frameCount), UInt64(Self.lanes), 512,
      UInt64(first), UInt64(count), 0, UInt64(Self.lanes), 2,
    ]
    encoder.setBytes(&parameters, length: parameters.count * 8, index: 6)
    encoder.dispatchThreads(
      MTLSize(width: Self.lanes, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    guard let joinEncoder = command.makeComputeCommandEncoder() else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("Float ANS word reconstruction unavailable.")
    }
    joinEncoder.setComputePipelineState(join)
    joinEncoder.setBuffer(workspace.lanes, offset: 0, index: 0)
    joinEncoder.setBuffer(workspace.words, offset: 0, index: 1)
    joinEncoder.setBuffer(workspace.descriptors, offset: 0, index: 2)
    joinEncoder.dispatchThreads(
      MTLSize(width: count * Self.pixels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    joinEncoder.endEncoding()
  }
}
