import Foundation
import Metal
import Metal4DSTEMKernels

/// Exact IEEE words stored as two little-endian UInt16 ANS streams per float.
/// Only a bounded query window is materialized; no numeric cast is performed.
final class MetalFloatANS {
  static let codec = "float32-bit-lanes-rans-v1"
  static let schema = "quantem.gpu.float32-bit-lanes-rans/v1"
  let pixels: Int
  var lanes: Int { pixels * 2 }
  let table: MTLBuffer
  private let decode: MTLComputePipelineState
  private let join: MTLComputePipelineState
  private let selected: MTLComputePipelineState
  private let recovery: MTLComputePipelineState
  private let changes: MTLComputePipelineState
  private let parallelChanges: MTLComputePipelineState
  private let parallelChangeThreshold: Int
  private let mean: MTLComputePipelineState
  private let device: MTLDevice
  private var scratch: Workspace?
  private var scratchCommand: MTLCommandBuffer?

  struct Workspace {
    let lanes: MTLBuffer
    let words: MTLBuffer
    let descriptors: MTLBuffer
    let errors: MTLBuffer
  }

  init(device: MTLDevice, pixels: Int) throws {
    self.pixels = pixels
    self.device = device
    let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
    let meanLibrary = try Metal4DSTEMKernels.makeFloatANSMeanLibrary(device: device)
    let constants = MTLFunctionConstantValues()
    var pixelCount = UInt32(pixels)
    constants.setConstantValue(&pixelCount, type: .uint, index: 20)
    guard let decode = library.makeFunction(name: "streamed_counts_decode_range"),
      let join = library.makeFunction(name: "float_ans_join_words"),
      let selected = try? library.makeFunction(
        name: "float_ans_decode_selected", constantValues: constants),
      let recovery = library.makeFunction(name: "float_ans_recovery_needed"),
      let changes = try? library.makeFunction(
        name: "float_ans_decode_changes", constantValues: constants),
      let parallelChanges = try? library.makeFunction(
        name: "float_ans_decode_changes_parallel", constantValues: constants),
      let mean = meanLibrary.makeFunction(name: "float_ans_region_mean")
    else { throw Metal4DSTEMStreamingIOError.invalidRequest("Rebuild the float ANS kernels.") }
    self.decode = try device.makeComputePipelineState(function: decode)
    self.join = try device.makeComputePipelineState(function: join)
    self.selected = try device.makeComputePipelineState(function: selected)
    self.recovery = try device.makeComputePipelineState(function: recovery)
    self.changes = try device.makeComputePipelineState(function: changes)
    self.parallelChanges = try device.makeComputePipelineState(function: parallelChanges)
    // Override only for controlled A/B tests; large edits amortize the extra
    // per-lane stream state, while small edits keep the lower-latency kernel.
    parallelChangeThreshold = Int(
      ProcessInfo.processInfo.environment["QGPU_FLOAT_ANS_PARALLEL_CHANGES"] ?? "") ?? 1600
    self.mean = try device.makeComputePipelineState(function: mean)
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

  /// Consume only selected frames, retaining the reference scan-order sum.
  /// Literal/constant lanes need no decoded window; entropy lanes are streamed
  /// through registers. Neither path materializes a dense scan volume.
  func encodeMean(
    chunks: [MetalEMPADResidentSource.Chunk], rows: Range<Int>, columns: Range<Int>,
    scanColumns: Int, shape: MetalScanRegionShape, output: MTLBuffer,
    accumulator: MTLBuffer, background: MTLBuffer?, command: MTLCommandBuffer
  ) throws {
    guard let errors = device.makeBuffer(length: 4, options: .storageModeShared),
      let encoder = command.makeComputeCommandEncoder(dispatchType: .serial)
    else { throw Metal4DSTEMStreamingIOError.invalidRequest("Cannot encode the selected mean DP.") }
    errors.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    encoder.setComputePipelineState(mean)
    encoder.setBuffer(table, offset: 0, index: 3)
    encoder.setBuffer(errors, offset: 0, index: 4)
    encoder.setBuffer(accumulator, offset: 0, index: 5)
    encoder.setBuffer(output, offset: 0, index: 6)
    encoder.setBuffer(background ?? output, offset: 0, index: 7)
    var corrected: UInt32 = background == nil ? 0 : 1
    encoder.setBytes(&corrected, length: 4, index: 8)
    let divisor = UInt32(shape.sampleCount(rowCount: rows.count, columnCount: columns.count))
    var first = true
    for chunk in chunks {
      let firstRow = max(rows.lowerBound, chunk.firstFrame / scanColumns)
      let lastRow = min(
        rows.upperBound, (chunk.firstFrame + chunk.frameCount - 1) / scanColumns + 1)
      guard firstRow < lastRow else { continue }
      var frames: [UInt32] = []
      for row in firstRow..<lastRow {
        let start = max(row * scanColumns + columns.lowerBound, chunk.firstFrame)
        let stop = min(row * scanColumns + columns.upperBound, chunk.firstFrame + chunk.frameCount)
        guard start < stop else { continue }
        for frame in start..<stop {
          if shape == .circle {
            let diameter = rows.count
            let dr = 2 * (row - rows.lowerBound) + 1 - diameter
            let dc = 2 * (frame % scanColumns - columns.lowerBound) + 1 - diameter
            if dr * dr + dc * dc > diameter * diameter { continue }
          }
          frames.append(UInt32(frame - chunk.firstFrame))
        }
      }
      guard !frames.isEmpty else { continue }
      encoder.setBuffer(chunk.payload, offset: 0, index: 0)
      encoder.setBuffer(chunk.offsets, offset: 0, index: 1)
      encoder.setBuffer(chunk.models, offset: 0, index: 2)
      var parameters = SIMD4<UInt32>(
        UInt32(chunk.frameCount), UInt32(frames.count), divisor, first ? 1 : 0)
      encoder.setBytes(&parameters, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 9)
      frames.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 10) }
      encoder.dispatchThreads(
        MTLSize(width: pixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      encoder.memoryBarrier(resources: [accumulator])
      first = false
    }
    encoder.endEncoding()
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
    encoder: MTLComputeCommandEncoder, preferParallel: Bool = true
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
      MTLSize(width: pixels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    if count > 0 {
      let parallel = preferParallel && count >= parallelChangeThreshold
      encoder.setComputePipelineState(parallel ? parallelChanges : changes)
      encoder.setBuffer(entries, offset: 0, index: 6)
      encoder.setBytes(&scans, length: 4, index: 7)
      if parallel {
        var entryCount = UInt32(count)
        encoder.setBytes(&entryCount, length: 4, index: 13)
        encoder.dispatchThreads(
          MTLSize(width: count, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      } else {
        encoder.dispatchThreadgroups(
          MTLSize(width: count, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      }
    }
  }

  func workspace(frames: Int, budget: UInt64, command: MTLCommandBuffer) throws -> Workspace {
    let size = frames * pixels
    // Ordered encoders in one command may reuse the same window. A different
    // in-flight command must retain its own buffers until it has completed.
    if let scratch, scratch.words.length >= size * 4,
      scratchCommand === command || scratchCommand?.status == .completed
    {
      scratchCommand = command
      return scratch
    }
    let needed = UInt64(size * 12 + ((size + 127) / 128) * 16 + 65536)
    guard frames > 0, frames <= 512,
      UInt64(device.currentAllocatedSize) <= budget,
      needed <= budget - UInt64(device.currentAllocatedSize),
      let lanes = device.makeBuffer(length: size * 8, options: .storageModePrivate),
      let words = device.makeBuffer(length: size * 4, options: .storageModePrivate),
      let descriptors = device.makeBuffer(
        length: ((size + 127) / 128) * 16, options: .storageModePrivate),
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
      workspace.words.length >= count * pixels * 4,
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
      UInt64(chunk.frameCount), UInt64(lanes), 512,
      UInt64(first), UInt64(count), 0, UInt64(lanes), 2,
    ]
    encoder.setBytes(&parameters, length: parameters.count * 8, index: 6)
    encoder.dispatchThreads(
      MTLSize(width: lanes, height: 1, depth: 1),
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
      MTLSize(width: count * pixels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    joinEncoder.endEncoding()
  }
}
