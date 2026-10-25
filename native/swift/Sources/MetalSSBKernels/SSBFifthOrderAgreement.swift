import Foundation
import Metal

/// Batched all-BF agreement with persistent, bounded GPU partials and GPU score reduction.
/// The model supplies its already gathered native-grid evidence and reusable probe geometry.
final class SSBFifthOrderAgreement {
  private struct Grid {
    var count, bandCount, chunks, batch: UInt32
    var wl, semiangle, angularRow, angularColumn, factor: Float
  }

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let probe, agreement, power, score: MTLComputePipelineState
  private var workspaces: [String: MTLBuffer] = [:]
  /// GPU time of the latest `evaluate` command buffer.
  private(set) var lastGPUSeconds = 0.0

  init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    self.device = device
    self.queue = queue
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalSSBError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    probe = try pipeline("fifth_probe")
    agreement = try pipeline("fifth_agreement")
    power = try pipeline("fifth_power")
    score = try pipeline("fifth_score")
  }

  private func workspace(_ name: String, bytes: Int, shared: Bool = false) throws -> MTLBuffer {
    if let existing = workspaces[name], existing.length >= bytes { return existing }
    let buffer = try SSBCoherentReconstruction.allocate(device, bytes, shared: shared)
    workspaces[name] = buffer
    return buffer
  }

  func evaluate(
    rows: [[Float]], geometry: MetalSSBGeometry, k: MTLBuffer, q: MTLBuffer,
    band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount: Int, isCancelled: () -> Bool
  ) throws -> [Double] {
    guard !rows.isEmpty else { return [] }
    let chunks = (count + 255) / 256
    let perCandidate = (chunks + 1) * bandCount * 16 + count * 4
    let batchSize = min(rows.count, max(4, ((64 << 20) / perCandidate / 4) * 4))
    let chi = try workspace("chi", bytes: count * batchSize * 4)
    let partial = try workspace("partial", bytes: chunks * bandCount * batchSize * 16)
    let powers = try workspace("power", bytes: bandCount * batchSize * 4)
    let scores = try workspace("scores", bytes: rows.count * 4, shared: true)
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    func encoder(_ pipeline: MTLComputePipelineState) throws -> MTLComputeCommandEncoder {
      guard let result = commands.makeComputeCommandEncoder() else {
        throw MetalSSBError.commandQueue
      }
      result.setComputePipelineState(pipeline)
      return result
    }
    for first in stride(from: 0, to: rows.count, by: batchSize) {
      if isCancelled() { throw CancellationError() }
      let batch = min(batchSize, rows.count - first)
      var grid = Grid(
        count: UInt32(count), bandCount: UInt32(bandCount), chunks: UInt32(chunks),
        batch: UInt32(batch), wl: geometry.wavelengthAngstroms,
        semiangle: geometry.semiangleRadians, angularRow: geometry.angularSamplingYRadians,
        angularColumn: geometry.angularSamplingXRadians,
        factor: geometry.phaseFactor ?? Float.pi / geometry.wavelengthAngstroms)
      let trials = try SSBCoherentReconstruction.buffer(
        device, rows[first..<(first + batch)].flatMap { $0 })
      let probeEncoder = try encoder(probe)
      for (index, buffer) in [k, trials, chi].enumerated() {
        probeEncoder.setBuffer(buffer, offset: 0, index: index)
      }
      probeEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 3)
      probeEncoder.dispatchThreads(
        MTLSize(width: count, height: batch, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      probeEncoder.endEncoding()
      let fitEncoder = try encoder(agreement)
      for (index, buffer) in [band, k, q, partial].enumerated() {
        fitEncoder.setBuffer(buffer, offset: 0, index: index)
      }
      fitEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 4)
      fitEncoder.setBuffer(trials, offset: 0, index: 5)
      fitEncoder.setBuffer(probeGeometry, offset: 0, index: 6)
      fitEncoder.setBuffer(chi, offset: 0, index: 7)
      fitEncoder.dispatchThreads(
        MTLSize(width: bandCount, height: chunks, depth: (batch + 3) / 4),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      fitEncoder.endEncoding()
      let powerEncoder = try encoder(power)
      for (index, buffer) in [partial, weights, powers].enumerated() {
        powerEncoder.setBuffer(buffer, offset: 0, index: index)
      }
      powerEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 3)
      powerEncoder.dispatchThreads(
        MTLSize(width: bandCount, height: batch, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      powerEncoder.endEncoding()
      let scoreEncoder = try encoder(score)
      scoreEncoder.setBuffer(powers, offset: 0, index: 0)
      scoreEncoder.setBuffer(scores, offset: first * 4, index: 1)
      scoreEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 2)
      scoreEncoder.dispatchThreadgroups(
        MTLSize(width: batch, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      scoreEncoder.endEncoding()
    }
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
    lastGPUSeconds = commands.gpuEndTime - commands.gpuStartTime
    if isCancelled() { throw CancellationError() }
    let values = scores.contents().assumingMemoryBound(to: Float.self)
    let result = rows.indices.map { Double(values[$0]) }
    guard result.allSatisfy(\.isFinite) else { throw SSBOptimizationError.nonFiniteLoss }
    return result
  }
}
