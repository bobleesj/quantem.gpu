import Foundation
import Metal

/// The low-order agreement with its exact gradient and Hessian (`ssb_low_order.metal`), for the trust-region
/// Newton of the primary fit. Coordinates are the Newton's z: C10, C12 cos 2 phi12, C12 sin 2 phi12 (Angstrom),
/// tilt row and column (mrad) and depth spread (Angstrom; the objective is even in it, so a negative z uses its
/// magnitude and the depth derivatives change sign).
///
/// One evaluation is two passes over every (band frequency, BF pixel): the sums of the objective, then, with
/// N known per band frequency, the k-sums its first and second derivatives need. Each band frequency's value,
/// gradient and Hessian are formed in float32 on the GPU from float32 BF sums; the band is summed on the host in
/// double precision, as the objective's band power.
final class SSBLowOrderDerivatives {
  struct Result {
    let value: Double
    let gradient: [Double]
    let hessian: [[Double]]
  }

  /// BF pixels per pass-2 chunk; the pass-2 partials hold 51 floats per (chunk, band frequency).
  static let derivativeChunk = 2048
  static let fields = 51
  static let outputs = 28

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let sums, reduce, derivativeSums, combine: MTLComputePipelineState
  private var workspaces: [String: MTLBuffer] = [:]

  init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    self.device = device
    self.queue = queue
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalSSBError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    sums = try pipeline("low_order_sums")
    reduce = try pipeline("low_order_reduce_sums")
    derivativeSums = try pipeline("low_order_derivative_sums")
    combine = try pipeline("low_order_derivative_combine")
  }

  private func workspace(_ name: String, bytes: Int, shared: Bool = false) throws -> MTLBuffer {
    if let existing = workspaces[name], existing.length >= bytes { return existing }
    let buffer = try SSBCoherentReconstruction.allocate(device, bytes, shared: shared)
    workspaces[name] = buffer
    return buffer
  }

  /// `grid` is the model's `JointGrid` bytes (its `chunks` counts 256-pixel chunks).
  func evaluate(
    rows: [[Double]], grid: [UInt8], bandCount: Int, count: Int, chunks: Int, band: MTLBuffer,
    k: MTLBuffer, q: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    neighbour: MTLBuffer, rowStep: SIMD2<Float>, isCancelled: () -> Bool
  ) throws -> [Result] {
    guard !rows.isEmpty else { return [] }
    guard rows.allSatisfy({ $0.count == 6 && $0.allSatisfy(\.isFinite) }) else {
      throw MetalSSBError.invalidGeometry("Supply six finite low-order coordinates per candidate.")
    }
    if isCancelled() { throw CancellationError() }
    let derivativeChunks = (count + Self.derivativeChunk - 1) / Self.derivativeChunk
    let trial = try SSBCoherentReconstruction.buffer(
      device,
      rows.flatMap { z -> [Float] in
        [
          Float(z[0]), Float(z[1]), Float(z[2]), Float(abs(z[5])), Float(z[3] / 1000),
          Float(z[4] / 1000),
        ]
      })
    let partialAB = try workspace("partial_ab", bytes: chunks * bandCount * 16)
    let partialSS = try workspace("partial_ss", bytes: chunks * bandCount * 8)
    let ab = try workspace("ab", bytes: bandCount * 16)
    let ss = try workspace("ss", bytes: bandCount * 8)
    let partial = try workspace("partial", bytes: derivativeChunks * Self.fields * bandCount * 4)
    let output = try workspace(
      "output", bytes: rows.count * Self.outputs * bandCount * 4, shared: true)
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    func encoder(_ pipeline: MTLComputePipelineState) throws -> MTLComputeCommandEncoder {
      guard let result = commands.makeComputeCommandEncoder() else {
        throw MetalSSBError.commandQueue
      }
      result.setComputePipelineState(pipeline)
      return result
    }
    var one: UInt32 = 1
    var step = rowStep
    var chunkSize = UInt32(Self.derivativeChunk)
    var chunkCount = UInt32(derivativeChunks)
    try grid.withUnsafeBytes { gridBytes in
      // Candidates share the workspaces; the tracked buffers order the passes.
      for candidate in rows.indices {
        let trialOffset = candidate * 6 * 4
        let first = try encoder(sums)
        for (index, buffer) in [band, k, q, partialAB, partialSS].enumerated() {
          first.setBuffer(buffer, offset: 0, index: index)
        }
        first.setBytes(gridBytes.baseAddress!, length: gridBytes.count, index: 5)
        first.setBuffer(trial, offset: trialOffset, index: 6)
        first.setBytes(&one, length: 4, index: 7)
        first.setBuffer(probeGeometry, offset: 0, index: 8)
        first.setBuffer(neighbour, offset: 0, index: 9)
        first.setBytes(&step, length: 8, index: 10)
        first.dispatchThreads(
          MTLSize(width: bandCount, height: chunks, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        first.endEncoding()
        let reduction = try encoder(reduce)
        for (index, buffer) in [partialAB, partialSS, ab, ss].enumerated() {
          reduction.setBuffer(buffer, offset: 0, index: index)
        }
        reduction.setBytes(gridBytes.baseAddress!, length: gridBytes.count, index: 4)
        reduction.setBytes(&one, length: 4, index: 5)
        reduction.dispatchThreads(
          MTLSize(width: bandCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        reduction.endEncoding()
        let second = try encoder(derivativeSums)
        for (index, buffer) in [band, k, q, partial].enumerated() {
          second.setBuffer(buffer, offset: 0, index: index)
        }
        second.setBytes(gridBytes.baseAddress!, length: gridBytes.count, index: 4)
        second.setBuffer(trial, offset: trialOffset, index: 5)
        second.setBytes(&one, length: 4, index: 6)
        second.setBuffer(probeGeometry, offset: 0, index: 7)
        second.setBuffer(neighbour, offset: 0, index: 8)
        second.setBytes(&step, length: 8, index: 9)
        second.setBuffer(ab, offset: 0, index: 10)
        second.setBytes(&chunkSize, length: 4, index: 11)
        second.setBytes(&chunkCount, length: 4, index: 12)
        second.dispatchThreads(
          MTLSize(width: bandCount, height: derivativeChunks, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        second.endEncoding()
        let last = try encoder(combine)
        for (index, buffer) in [partial, ab, ss, q, weights].enumerated() {
          last.setBuffer(buffer, offset: 0, index: index)
        }
        last.setBuffer(output, offset: candidate * Self.outputs * bandCount * 4, index: 5)
        last.setBytes(gridBytes.baseAddress!, length: gridBytes.count, index: 6)
        last.setBuffer(trial, offset: trialOffset, index: 7)
        last.setBytes(&one, length: 4, index: 8)
        last.setBytes(&chunkCount, length: 4, index: 9)
        last.setBytes(&step, length: 8, index: 10)
        last.dispatchThreads(
          MTLSize(width: bandCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        last.endEncoding()
      }
    }
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
    if isCancelled() { throw CancellationError() }
    let values = output.contents().assumingMemoryBound(to: Float.self)
    return try rows.indices.map { candidate in
      var totals = [Double](repeating: 0, count: Self.outputs)
      for field in 0..<Self.outputs {
        let start = (candidate * Self.outputs + field) * bandCount
        var total = 0.0
        for i in 0..<bandCount { total += Double(values[start + i]) }
        totals[field] = total
      }
      guard totals.allSatisfy(\.isFinite) else { throw SSBOptimizationError.nonFiniteLoss }
      // d|z5|/dz5 = sign(z5).
      let sign = rows[candidate][5] < 0 ? -1.0 : 1.0
      var gradient = Array(totals[1...6])
      gradient[5] *= sign
      var hessian = [[Double]](repeating: [Double](repeating: 0, count: 6), count: 6)
      var slot = 7
      for i in 0..<6 {
        for j in i..<6 {
          var value = totals[slot]
          if (i == 5) != (j == 5) { value *= sign }
          hessian[i][j] = value
          hessian[j][i] = value
          slot += 1
        }
      }
      return Result(value: totals[0], gradient: gradient, hessian: hessian)
    }
  }
}
