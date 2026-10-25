import Foundation
import Metal

/// The low-order agreement with its exact gradient and Hessian (`ssb_low_order_derivatives.metal`,
/// ssb-low-order-derivatives v0.2), for the trust-region Newton of the primary fit. Coordinates are the Newton's z:
/// C10, C12 cos 2 phi12, C12 sin 2 phi12 (Angstrom), tilt row and column (mrad) and depth spread (Angstrom; the
/// objective is even in it, so a negative z uses its magnitude and the depth derivatives change sign).
///
/// An evaluation has two stages. `values` makes one pass over every (band frequency, BF pixel) for the objective's
/// sums and keeps them; `derivatives(at:)` then makes, for any of those rows, the pass with N known per band
/// frequency that forms the k-sums of its first and second derivatives. The Newton asks for derivatives only
/// where it uses them; where it will use every row's, `values(withDerivatives: true)` makes one fused pass for
/// both (the same value sums, bitwise). Each band frequency's terms are formed in float32 on the GPU from float32
/// BF sums; the band is summed on the host in double precision, as the objective's band power.
final class SSBLowOrderDerivatives {
  struct Result {
    let value: Double
    let gradient: [Double]
    let hessian: [[Double]]
  }

  /// The model's buffers and constants; `grid` is its `JointGrid` bytes (`chunks` counts 256-pixel chunks) and
  /// `extra` the angle-sum kernels' aperture radii (`JointAgreementExtra`).
  struct Inputs {
    let grid: [UInt8]
    let extra: [UInt8]
    let bandCount: Int
    let count: Int
    let chunks: Int
    let band, k, q, weights, probeGeometry, neighbour: MTLBuffer
    let rowStep: SIMD2<Float>
  }

  /// BF pixels per pass-2 chunk; the pass-2 partials hold `fields` floats per (chunk, band frequency).
  static let derivativeChunk = 2048
  static let fields = 29
  static let fusedFields = 43
  /// The value, the six first derivatives and the 21 distinct second derivatives.
  static let outputs = 28
  static let referenceFields = 51

  /// Tests only: true evaluates with `ssb_low_order.metal` (ssb-low-order-derivatives v0.1), the parity reference
  /// of v0.2. There is no runtime fallback.
  var reference = false
  /// GPU time of the latest `values`, `derivatives(at:)` or `evaluate` (command-buffer GPU start to end, summed).
  private(set) var lastGPUSeconds = 0.0
  /// The latest `evaluate`'s GPU time of its value stage (zero for the reference, which has one command buffer).
  private(set) var lastValueGPUSeconds = 0.0

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let valueSums, valueReduce, gradientSums, gradientCombine, fusedSums, fusedCombine:
    MTLComputePipelineState
  private let referenceSums, referenceReduce, referenceDerivativeSums, referenceCombine:
    MTLComputePipelineState
  private var workspaces: [String: MTLBuffer] = [:]
  /// The rows, trial buffer and inputs of the latest `values` call, whose sums `derivatives(at:)` reads.
  private var staged: (rows: [[Double]], trial: MTLBuffer, inputs: Inputs)?
  /// Every row's derivatives when the latest `values` call was fused.
  private var fusedResults: [Result]?

  init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws {
    self.device = device
    self.queue = queue
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalSSBError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    valueSums = try pipeline("low_order_value_sums")
    valueReduce = try pipeline("low_order_value_reduce")
    gradientSums = try pipeline("low_order_gradient_sums")
    gradientCombine = try pipeline("low_order_gradient_combine")
    fusedSums = try pipeline("low_order_fused_sums")
    fusedCombine = try pipeline("low_order_fused_combine")
    referenceSums = try pipeline("low_order_sums")
    referenceReduce = try pipeline("low_order_reduce_sums")
    referenceDerivativeSums = try pipeline("low_order_derivative_sums")
    referenceCombine = try pipeline("low_order_derivative_combine")
  }

  private func workspace(_ name: String, bytes: Int, shared: Bool = false) throws -> MTLBuffer {
    if let existing = workspaces[name], existing.length >= bytes { return existing }
    let buffer = try SSBCoherentReconstruction.allocate(device, bytes, shared: shared)
    workspaces[name] = buffer
    return buffer
  }

  private func trialBuffer(_ rows: [[Double]]) throws -> MTLBuffer {
    guard rows.allSatisfy({ $0.count == 6 && $0.allSatisfy(\.isFinite) }) else {
      throw MetalSSBError.invalidGeometry("Supply six finite low-order coordinates per candidate.")
    }
    return try SSBCoherentReconstruction.buffer(
      device,
      rows.flatMap { z -> [Float] in
        [
          Float(z[0]), Float(z[1]), Float(z[2]), Float(abs(z[5])), Float(z[3] / 1000),
          Float(z[4] / 1000),
        ]
      })
  }

  private func run(_ encode: (MTLCommandBuffer) throws -> Void) throws {
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    try encode(commands)
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
    lastGPUSeconds += commands.gpuEndTime - commands.gpuStartTime
  }

  private static func encoder(
    _ commands: MTLCommandBuffer, _ pipeline: MTLComputePipelineState
  ) throws -> MTLComputeCommandEncoder {
    guard let result = commands.makeComputeCommandEncoder() else {
      throw MetalSSBError.commandQueue
    }
    result.setComputePipelineState(pipeline)
    return result
  }

  /// The agreement at each row; keeps the rows' sums for `derivatives(at:)`. `withDerivatives` forms every row's
  /// derivatives in the same pass.
  func values(
    rows: [[Double]], inputs: Inputs, withDerivatives: Bool = false, isCancelled: () -> Bool
  ) throws -> [Double] {
    if withDerivatives {
      return try fused(rows: rows, inputs: inputs, isCancelled: isCancelled).map(\.value)
    }
    lastGPUSeconds = 0
    staged = nil
    fusedResults = nil
    guard !rows.isEmpty else { return [] }
    if isCancelled() { throw CancellationError() }
    let trial = try trialBuffer(rows)
    let n = inputs.bandCount
    let partialAB = try workspace("partial_ab", bytes: rows.count * inputs.chunks * n * 16)
    let partialSS = try workspace("partial_ss", bytes: rows.count * inputs.chunks * n * 8)
    let ab = try workspace("ab", bytes: rows.count * n * 16)
    let ss = try workspace("ss", bytes: rows.count * n * 8)
    let value = try workspace("value", bytes: rows.count * n * 4, shared: true)
    var total = UInt32(rows.count)
    var step = inputs.rowStep
    try run { commands in
      try inputs.grid.withUnsafeBytes { grid in
        try inputs.extra.withUnsafeBytes { extra in
          let first = try Self.encoder(commands, valueSums)
          for (index, buffer) in [inputs.band, inputs.k, inputs.q, partialAB, partialSS].enumerated() {
            first.setBuffer(buffer, offset: 0, index: index)
          }
          first.setBytes(grid.baseAddress!, length: grid.count, index: 5)
          first.setBuffer(trial, offset: 0, index: 6)
          first.setBytes(&total, length: 4, index: 7)
          first.setBuffer(inputs.probeGeometry, offset: 0, index: 8)
          first.setBuffer(inputs.neighbour, offset: 0, index: 9)
          first.setBytes(&step, length: 8, index: 10)
          first.setBytes(extra.baseAddress!, length: extra.count, index: 11)
          first.dispatchThreads(
            MTLSize(width: n, height: inputs.chunks, depth: rows.count),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          first.endEncoding()
          let reduction = try Self.encoder(commands, valueReduce)
          for (index, buffer) in [partialAB, partialSS, ab, ss, value, inputs.q, inputs.weights]
            .enumerated()
          {
            reduction.setBuffer(buffer, offset: 0, index: index)
          }
          reduction.setBytes(grid.baseAddress!, length: grid.count, index: 7)
          reduction.setBuffer(trial, offset: 0, index: 8)
          reduction.setBytes(&total, length: 4, index: 9)
          reduction.dispatchThreads(
            MTLSize(width: n, height: rows.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          reduction.endEncoding()
        }
      }
    }
    if isCancelled() { throw CancellationError() }
    let terms = value.contents().assumingMemoryBound(to: Float.self)
    let values = try rows.indices.map { candidate -> Double in
      var total = 0.0
      for i in 0..<n { total += Double(terms[candidate * n + i]) }
      guard total.isFinite else { throw SSBOptimizationError.nonFiniteLoss }
      return total
    }
    staged = (rows, trial, inputs)
    return values
  }

  /// The gradient and Hessian at the given rows of the latest `values` call, with those values.
  func derivatives(at indices: [Int], isCancelled: () -> Bool) throws -> [Result] {
    lastGPUSeconds = 0
    guard !indices.isEmpty else { return [] }
    guard let staged, indices.allSatisfy({ staged.rows.indices.contains($0) }) else {
      throw MetalSSBError.invalidGeometry("Evaluate the low-order values before their derivatives.")
    }
    if let fusedResults { return indices.map { fusedResults[$0] } }
    if isCancelled() { throw CancellationError() }
    let inputs = staged.inputs
    let n = inputs.bandCount
    let derivativeChunks = (inputs.count + Self.derivativeChunk - 1) / Self.derivativeChunk
    let partial = try workspace(
      "partial", bytes: indices.count * derivativeChunks * Self.fields * n * 4)
    let output = try workspace(
      "output", bytes: indices.count * (Self.outputs - 1) * n * 4, shared: true)
    let selected = try SSBCoherentReconstruction.buffer(device, indices.map { UInt32($0) })
    guard let ab = workspaces["ab"], let ss = workspaces["ss"], let value = workspaces["value"] else {
      throw MetalSSBError.commandQueue
    }
    var total = UInt32(indices.count)
    var step = inputs.rowStep
    var chunkSize = UInt32(Self.derivativeChunk)
    var chunkCount = UInt32(derivativeChunks)
    try run { commands in
      try inputs.grid.withUnsafeBytes { grid in
        try inputs.extra.withUnsafeBytes { extra in
          let second = try Self.encoder(commands, gradientSums)
          for (index, buffer) in [inputs.band, inputs.k, inputs.q, partial].enumerated() {
            second.setBuffer(buffer, offset: 0, index: index)
          }
          second.setBytes(grid.baseAddress!, length: grid.count, index: 4)
          second.setBuffer(staged.trial, offset: 0, index: 5)
          second.setBytes(&total, length: 4, index: 6)
          second.setBuffer(inputs.probeGeometry, offset: 0, index: 7)
          second.setBuffer(inputs.neighbour, offset: 0, index: 8)
          second.setBytes(&step, length: 8, index: 9)
          second.setBuffer(ab, offset: 0, index: 10)
          second.setBytes(&chunkSize, length: 4, index: 11)
          second.setBytes(&chunkCount, length: 4, index: 12)
          second.setBytes(extra.baseAddress!, length: extra.count, index: 13)
          second.setBuffer(selected, offset: 0, index: 14)
          second.dispatchThreads(
            MTLSize(width: n, height: derivativeChunks, depth: indices.count),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          second.endEncoding()
          let last = try Self.encoder(commands, gradientCombine)
          for (index, buffer) in [partial, ab, ss, inputs.q, inputs.weights, output].enumerated() {
            last.setBuffer(buffer, offset: 0, index: index)
          }
          last.setBytes(grid.baseAddress!, length: grid.count, index: 6)
          last.setBuffer(staged.trial, offset: 0, index: 7)
          last.setBytes(&total, length: 4, index: 8)
          last.setBytes(&chunkCount, length: 4, index: 9)
          last.setBuffer(selected, offset: 0, index: 10)
          last.dispatchThreads(
            MTLSize(width: n, height: indices.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          last.endEncoding()
        }
      }
    }
    if isCancelled() { throw CancellationError() }
    let terms = value.contents().assumingMemoryBound(to: Float.self)
    let derivatives = output.contents().assumingMemoryBound(to: Float.self)
    return try indices.enumerated().map { slot, candidate in
      var totals = [Double](repeating: 0, count: Self.outputs)
      for i in 0..<n { totals[0] += Double(terms[candidate * n + i]) }
      for field in 1..<Self.outputs {
        totals[field] = Self.sum(derivatives + (slot * (Self.outputs - 1) + field - 1) * n, n)
      }
      return try Self.result(totals, depth: staged.rows[candidate][5])
    }
  }

  /// Value, gradient and Hessian at every row: the fused pass, or with `twoStage` the value pass and then the
  /// derivative pass (the Newton's path for rows whose derivatives it may not use).
  func evaluate(rows: [[Double]], inputs: Inputs, twoStage: Bool = false, isCancelled: () -> Bool) throws
    -> [Result]
  {
    lastValueGPUSeconds = 0
    if reference { return try evaluateReference(rows: rows, inputs: inputs, isCancelled: isCancelled) }
    guard twoStage else { return try fused(rows: rows, inputs: inputs, isCancelled: isCancelled) }
    _ = try values(rows: rows, inputs: inputs, isCancelled: isCancelled)
    let valueSeconds = lastGPUSeconds
    lastValueGPUSeconds = valueSeconds
    let results = try derivatives(at: Array(rows.indices), isCancelled: isCancelled)
    lastGPUSeconds += valueSeconds
    return results
  }

  /// One pass for the value sums and the derivative sums that do not need N, then the value reduction and the
  /// combine (which projects on conj N) in the same command buffer.
  private func fused(rows: [[Double]], inputs: Inputs, isCancelled: () -> Bool) throws -> [Result] {
    lastGPUSeconds = 0
    staged = nil
    fusedResults = nil
    guard !rows.isEmpty else { return [] }
    if isCancelled() { throw CancellationError() }
    let trial = try trialBuffer(rows)
    let n = inputs.bandCount
    let derivativeChunks = (inputs.count + Self.derivativeChunk - 1) / Self.derivativeChunk
    let partialAB = try workspace("partial_ab", bytes: rows.count * inputs.chunks * n * 16)
    let partialSS = try workspace("partial_ss", bytes: rows.count * inputs.chunks * n * 8)
    let ab = try workspace("ab", bytes: rows.count * n * 16)
    let ss = try workspace("ss", bytes: rows.count * n * 8)
    let value = try workspace("value", bytes: rows.count * n * 4, shared: true)
    let partial = try workspace(
      "fused_partial", bytes: rows.count * derivativeChunks * Self.fusedFields * n * 4)
    let output = try workspace(
      "output", bytes: rows.count * (Self.outputs - 1) * n * 4, shared: true)
    var total = UInt32(rows.count)
    var step = inputs.rowStep
    var chunkSize = UInt32(Self.derivativeChunk)
    var chunkCount = UInt32(derivativeChunks)
    try run { commands in
      try inputs.grid.withUnsafeBytes { grid in
        try inputs.extra.withUnsafeBytes { extra in
          let first = try Self.encoder(commands, fusedSums)
          for (index, buffer) in [inputs.band, inputs.k, inputs.q, partial].enumerated() {
            first.setBuffer(buffer, offset: 0, index: index)
          }
          first.setBytes(grid.baseAddress!, length: grid.count, index: 4)
          first.setBuffer(trial, offset: 0, index: 5)
          first.setBytes(&total, length: 4, index: 6)
          first.setBuffer(inputs.probeGeometry, offset: 0, index: 7)
          first.setBuffer(inputs.neighbour, offset: 0, index: 8)
          first.setBytes(&step, length: 8, index: 9)
          first.setBuffer(partialAB, offset: 0, index: 10)
          first.setBuffer(partialSS, offset: 0, index: 11)
          first.setBytes(&chunkSize, length: 4, index: 12)
          first.setBytes(&chunkCount, length: 4, index: 13)
          first.setBytes(extra.baseAddress!, length: extra.count, index: 14)
          first.dispatchThreads(
            MTLSize(width: n, height: derivativeChunks, depth: rows.count),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          first.endEncoding()
          let reduction = try Self.encoder(commands, valueReduce)
          for (index, buffer) in [partialAB, partialSS, ab, ss, value, inputs.q, inputs.weights]
            .enumerated()
          {
            reduction.setBuffer(buffer, offset: 0, index: index)
          }
          reduction.setBytes(grid.baseAddress!, length: grid.count, index: 7)
          reduction.setBuffer(trial, offset: 0, index: 8)
          reduction.setBytes(&total, length: 4, index: 9)
          reduction.dispatchThreads(
            MTLSize(width: n, height: rows.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          reduction.endEncoding()
          let last = try Self.encoder(commands, fusedCombine)
          for (index, buffer) in [partial, ab, ss, inputs.q, inputs.weights, output].enumerated() {
            last.setBuffer(buffer, offset: 0, index: index)
          }
          last.setBytes(grid.baseAddress!, length: grid.count, index: 6)
          last.setBuffer(trial, offset: 0, index: 7)
          last.setBytes(&total, length: 4, index: 8)
          last.setBytes(&chunkCount, length: 4, index: 9)
          last.dispatchThreads(
            MTLSize(width: n, height: rows.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          last.endEncoding()
        }
      }
    }
    if isCancelled() { throw CancellationError() }
    let terms = value.contents().assumingMemoryBound(to: Float.self)
    let derivatives = output.contents().assumingMemoryBound(to: Float.self)
    let results = try rows.indices.map { candidate in
      var totals = [Double](repeating: 0, count: Self.outputs)
      for i in 0..<n { totals[0] += Double(terms[candidate * n + i]) }
      for field in 1..<Self.outputs {
        totals[field] = Self.sum(derivatives + (candidate * (Self.outputs - 1) + field - 1) * n, n)
      }
      return try Self.result(totals, depth: rows[candidate][5])
    }
    staged = (rows, trial, inputs)
    fusedResults = results
    return results
  }

  /// A derivative field's band sum in double precision with four interleaved partial sums (the value keeps the
  /// sequential order of the objective's band power).
  private static func sum(_ terms: UnsafePointer<Float>, _ n: Int) -> Double {
    var a = 0.0
    var b = 0.0
    var c = 0.0
    var d = 0.0
    var i = 0
    while i + 4 <= n {
      a += Double(terms[i])
      b += Double(terms[i + 1])
      c += Double(terms[i + 2])
      d += Double(terms[i + 3])
      i += 4
    }
    while i < n {
      a += Double(terms[i])
      i += 1
    }
    return (a + b) + (c + d)
  }

  /// Field sums to the Newton's derivatives; d|z5|/dz5 = sign(z5).
  private static func result(_ totals: [Double], depth: Double) throws -> Result {
    guard totals.allSatisfy(\.isFinite) else { throw SSBOptimizationError.nonFiniteLoss }
    let sign = depth < 0 ? -1.0 : 1.0
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

  /// v0.1 (`ssb_low_order.metal`): both passes per candidate in one command buffer, sharing the workspaces.
  private func evaluateReference(rows: [[Double]], inputs: Inputs, isCancelled: () -> Bool) throws
    -> [Result]
  {
    lastGPUSeconds = 0
    staged = nil
    fusedResults = nil
    guard !rows.isEmpty else { return [] }
    if isCancelled() { throw CancellationError() }
    let n = inputs.bandCount
    let chunks = inputs.chunks
    let derivativeChunks = (inputs.count + Self.derivativeChunk - 1) / Self.derivativeChunk
    let trial = try trialBuffer(rows)
    let partialAB = try workspace("reference_partial_ab", bytes: chunks * n * 16)
    let partialSS = try workspace("reference_partial_ss", bytes: chunks * n * 8)
    let ab = try workspace("reference_ab", bytes: n * 16)
    let ss = try workspace("reference_ss", bytes: n * 8)
    let partial = try workspace(
      "reference_partial", bytes: derivativeChunks * Self.referenceFields * n * 4)
    let output = try workspace(
      "reference_output", bytes: rows.count * Self.outputs * n * 4, shared: true)
    var one: UInt32 = 1
    var step = inputs.rowStep
    var chunkSize = UInt32(Self.derivativeChunk)
    var chunkCount = UInt32(derivativeChunks)
    try run { commands in
      try inputs.grid.withUnsafeBytes { gridBytes in
        for candidate in rows.indices {
          let trialOffset = candidate * 6 * 4
          let first = try Self.encoder(commands, referenceSums)
          for (index, buffer) in [inputs.band, inputs.k, inputs.q, partialAB, partialSS].enumerated() {
            first.setBuffer(buffer, offset: 0, index: index)
          }
          first.setBytes(gridBytes.baseAddress!, length: gridBytes.count, index: 5)
          first.setBuffer(trial, offset: trialOffset, index: 6)
          first.setBytes(&one, length: 4, index: 7)
          first.setBuffer(inputs.probeGeometry, offset: 0, index: 8)
          first.setBuffer(inputs.neighbour, offset: 0, index: 9)
          first.setBytes(&step, length: 8, index: 10)
          first.dispatchThreads(
            MTLSize(width: n, height: chunks, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          first.endEncoding()
          let reduction = try Self.encoder(commands, referenceReduce)
          for (index, buffer) in [partialAB, partialSS, ab, ss].enumerated() {
            reduction.setBuffer(buffer, offset: 0, index: index)
          }
          reduction.setBytes(gridBytes.baseAddress!, length: gridBytes.count, index: 4)
          reduction.setBytes(&one, length: 4, index: 5)
          reduction.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          reduction.endEncoding()
          let second = try Self.encoder(commands, referenceDerivativeSums)
          for (index, buffer) in [inputs.band, inputs.k, inputs.q, partial].enumerated() {
            second.setBuffer(buffer, offset: 0, index: index)
          }
          second.setBytes(gridBytes.baseAddress!, length: gridBytes.count, index: 4)
          second.setBuffer(trial, offset: trialOffset, index: 5)
          second.setBytes(&one, length: 4, index: 6)
          second.setBuffer(inputs.probeGeometry, offset: 0, index: 7)
          second.setBuffer(inputs.neighbour, offset: 0, index: 8)
          second.setBytes(&step, length: 8, index: 9)
          second.setBuffer(ab, offset: 0, index: 10)
          second.setBytes(&chunkSize, length: 4, index: 11)
          second.setBytes(&chunkCount, length: 4, index: 12)
          second.dispatchThreads(
            MTLSize(width: n, height: derivativeChunks, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          second.endEncoding()
          let last = try Self.encoder(commands, referenceCombine)
          for (index, buffer) in [partial, ab, ss, inputs.q, inputs.weights].enumerated() {
            last.setBuffer(buffer, offset: 0, index: index)
          }
          last.setBuffer(output, offset: candidate * Self.outputs * n * 4, index: 5)
          last.setBytes(gridBytes.baseAddress!, length: gridBytes.count, index: 6)
          last.setBuffer(trial, offset: trialOffset, index: 7)
          last.setBytes(&one, length: 4, index: 8)
          last.setBytes(&chunkCount, length: 4, index: 9)
          last.setBytes(&step, length: 8, index: 10)
          last.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
          last.endEncoding()
        }
      }
    }
    if isCancelled() { throw CancellationError() }
    let values = output.contents().assumingMemoryBound(to: Float.self)
    return try rows.indices.map { candidate in
      var totals = [Double](repeating: 0, count: Self.outputs)
      for field in 0..<Self.outputs {
        let start = (candidate * Self.outputs + field) * n
        var total = 0.0
        for i in 0..<n { total += Double(values[start + i]) }
        totals[field] = total
      }
      return try Self.result(totals, depth: rows[candidate][5])
    }
  }
}
