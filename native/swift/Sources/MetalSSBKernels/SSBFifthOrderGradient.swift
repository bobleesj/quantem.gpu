import Foundation
import Metal

/// The fifth-order agreement and its exact gradient for a batch of candidates.
/// Pass 1 is the objective itself (`fifth_probe`, `fifth_agreement`, `fifth_power`, `fifth_score`), so the
/// agreement is bit-identical to `SSBFifthOrderAgreement`. Pass 2 (`fifth_gradient`) uses the per-frequency
/// projection a_q = N_q / D_q of pass 1. Simdgroup partials of the gradient are reduced on the host in Double.
final class SSBFifthOrderGradient {
  private struct Grid {
    var count, bandCount, chunks, batch: UInt32
    var wl, semiangle, angularRow, angularColumn, factor: Float
  }
  private struct GradientGrid {
    var count, bandCount, chunks, batch: UInt32
    var wl, semiangle, angularRow, angularColumn, factor: Float
    var span, spans, qGroups, qStride, fullBandCount: UInt32
  }

  /// One candidate: the agreement and d agreement / d row, in the row layout of the objective:
  /// [depth (A), tilt row (rad), tilt column (rad), 28 packed Cartesian coefficients (A)].
  struct Value {
    let agreement: Double
    let gradient: [Double]
  }

  /// BF pixels per gradient thread; 1024 keeps the partials small and the grid large.
  static let span = 1024

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let probe, agreement, power, score, strided, scale, gradient: MTLComputePipelineState
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
    probe = try pipeline("fifth_probe")
    agreement = try pipeline("fifth_agreement")
    power = try pipeline("fifth_power")
    score = try pipeline("fifth_score")
    strided = try pipeline("fifth_strided_agreement")
    scale = try pipeline("fifth_gradient_scale")
    gradient = try pipeline("fifth_gradient")
  }

  private func workspace(_ name: String, bytes: Int, shared: Bool = false) throws -> MTLBuffer {
    if let existing = workspaces[name], existing.length >= bytes { return existing }
    let buffer = try SSBCoherentReconstruction.allocate(device, bytes, shared: shared)
    workspaces[name] = buffer
    return buffer
  }

  private struct Retained {
    let rows: [[Float]]
    let qStride: Int
    let trials: MTLBuffer
    let values: [Double]
  }
  /// The last value-only evaluation, whose projections a_q and probe phases stay in the workspaces.
  private var retained: Retained?

  /// - Parameters:
  ///   - qStride: 1 evaluates the objective and its gradient. A larger stride evaluates only the gradient of the
  ///     agreement restricted to every qStride-th band frequency (for a steering Hessian); its `agreement` is NaN.
  ///   - gradients: false evaluates the agreements only (empty gradients) and retains their projections, so
  ///     `retainedGradients` can add the gradient pass for the same rows without repeating the objective.
  func evaluate(
    rows: [[Float]], geometry: MetalSSBGeometry, k: MTLBuffer, q: MTLBuffer,
    band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount fullBandCount: Int, qStride: Int = 1, gradients withGradients: Bool = true,
    isCancelled: () -> Bool
  ) throws -> [Value] {
    guard !rows.isEmpty else { return [] }
    precondition(qStride >= 1)
    retained = nil
    let layout = Layout(count: count, fullBandCount: fullBandCount, qStride: qStride, rows: rows.count)
    let chi = try workspace("chi", bytes: count * layout.batchSize * 4)
    let partial = try workspace("partial", bytes: layout.chunks * layout.bandCount * layout.batchSize * 16)
    let powers = try workspace("power", bytes: layout.bandCount * layout.batchSize * 4)
    let scales = try workspace("scale", bytes: layout.bandCount * layout.batchSize * 16)
    let scores = try workspace("scores", bytes: rows.count * 4, shared: true)
    let gradients = try workspace(
      "gradients", bytes: rows.count * layout.perGradient * 4, shared: true)
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    var lastTrials: MTLBuffer?
    for first in stride(from: 0, to: rows.count, by: layout.batchSize) {
      if isCancelled() { throw CancellationError() }
      let batch = min(layout.batchSize, rows.count - first)
      var grid = layout.grid(geometry, batch: batch)
      var gradientGrid = layout.gradientGrid(geometry, batch: batch)
      let trials = try SSBCoherentReconstruction.buffer(
        device, rows[first..<(first + batch)].flatMap { $0 })
      lastTrials = trials
      let probeEncoder = try encoder(commands, probe)
      for (index, buffer) in [k, trials, chi].enumerated() {
        probeEncoder.setBuffer(buffer, offset: 0, index: index)
      }
      probeEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 3)
      probeEncoder.dispatchThreads(
        MTLSize(width: count, height: batch, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      probeEncoder.endEncoding()
      let fitEncoder = try encoder(commands, qStride == 1 ? agreement : strided)
      for (index, buffer) in [band, k, q, partial].enumerated() {
        fitEncoder.setBuffer(buffer, offset: 0, index: index)
      }
      if qStride == 1 {
        fitEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 4)
      } else {
        fitEncoder.setBytes(&gradientGrid, length: MemoryLayout<GradientGrid>.stride, index: 4)
      }
      fitEncoder.setBuffer(trials, offset: 0, index: 5)
      fitEncoder.setBuffer(probeGeometry, offset: 0, index: 6)
      fitEncoder.setBuffer(chi, offset: 0, index: 7)
      fitEncoder.dispatchThreads(
        MTLSize(width: layout.bandCount, height: layout.chunks, depth: (batch + 3) / 4),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      fitEncoder.endEncoding()
      if qStride == 1 {
        let powerEncoder = try encoder(commands, power)
        for (index, buffer) in [partial, weights, powers].enumerated() {
          powerEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        powerEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 3)
        powerEncoder.dispatchThreads(
          MTLSize(width: layout.bandCount, height: batch, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        powerEncoder.endEncoding()
        let scoreEncoder = try encoder(commands, score)
        scoreEncoder.setBuffer(powers, offset: 0, index: 0)
        scoreEncoder.setBuffer(scores, offset: first * 4, index: 1)
        scoreEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 2)
        scoreEncoder.dispatchThreadgroups(
          MTLSize(width: batch, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        scoreEncoder.endEncoding()
      }
      let scaleEncoder = try encoder(commands, scale)
      for (index, buffer) in [partial, weights, scales].enumerated() {
        scaleEncoder.setBuffer(buffer, offset: 0, index: index)
      }
      scaleEncoder.setBytes(&gradientGrid, length: MemoryLayout<GradientGrid>.stride, index: 3)
      scaleEncoder.dispatchThreads(
        MTLSize(width: layout.bandCount, height: batch, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      scaleEncoder.endEncoding()
      if withGradients {
        try encodeGradient(
          commands, layout: layout, grid: &gradientGrid, batch: batch, first: first,
          buffers: (band, k, q, gradients, trials, probeGeometry, chi, scales))
      }
    }
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
    if isCancelled() { throw CancellationError() }
    let pointer = scores.contents().assumingMemoryBound(to: Float.self)
    let values = rows.indices.map { qStride == 1 ? Double(pointer[$0]) : .nan }
    guard qStride > 1 || values.allSatisfy(\.isFinite) else {
      throw SSBOptimizationError.nonFiniteLoss
    }
    if !withGradients {
      // One batch keeps every row's projections in the workspaces.
      if rows.count <= layout.batchSize, let lastTrials {
        retained = Retained(rows: rows, qStride: qStride, trials: lastTrials, values: values)
      }
      return values.map { Value(agreement: $0, gradient: []) }
    }
    return try reduce(values: values, gradients: gradients, layout: layout, geometry: geometry)
  }

  /// The gradients of the rows of the last value-only `evaluate`, by the gradient pass alone; other rows are
  /// evaluated in full.
  func retainedGradients(
    rows: [[Float]], geometry: MetalSSBGeometry, k: MTLBuffer, q: MTLBuffer,
    band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount fullBandCount: Int, isCancelled: () -> Bool
  ) throws -> [Value] {
    guard let kept = retained, kept.rows == rows, kept.qStride == 1,
      let chi = workspaces["chi"], let scales = workspaces["scale"]
    else {
      return try evaluate(
        rows: rows, geometry: geometry, k: k, q: q, band: band, weights: weights,
        probeGeometry: probeGeometry, count: count, bandCount: fullBandCount,
        isCancelled: isCancelled)
    }
    retained = nil
    let layout = Layout(count: count, fullBandCount: fullBandCount, qStride: 1, rows: rows.count)
    let gradients = try workspace(
      "gradients", bytes: rows.count * layout.perGradient * 4, shared: true)
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    var gradientGrid = layout.gradientGrid(geometry, batch: rows.count)
    try encodeGradient(
      commands, layout: layout, grid: &gradientGrid, batch: rows.count, first: 0,
      buffers: (band, k, q, gradients, kept.trials, probeGeometry, chi, scales))
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
    if isCancelled() { throw CancellationError() }
    return try reduce(values: kept.values, gradients: gradients, layout: layout, geometry: geometry)
  }

  private struct Layout {
    let count, fullBandCount, qStride, bandCount, chunks, spans, qGroups, batchSize, perGradient: Int
    init(count: Int, fullBandCount: Int, qStride: Int, rows: Int) {
      self.count = count
      self.fullBandCount = fullBandCount
      self.qStride = qStride
      bandCount = (fullBandCount + qStride - 1) / qStride
      chunks = (count + 255) / 256
      spans = (count + SSBFifthOrderGradient.span - 1) / SSBFifthOrderGradient.span
      qGroups = (bandCount + 63) / 64 * 2
      let perCandidate = (chunks + 1) * bandCount * 16 + count * 4 + bandCount * 16
      batchSize = min(rows, max(4, ((64 << 20) / perCandidate / 4) * 4))
      perGradient = spans * qGroups * 32
    }
    func grid(_ geometry: MetalSSBGeometry, batch: Int) -> Grid {
      Grid(
        count: UInt32(count), bandCount: UInt32(bandCount), chunks: UInt32(chunks),
        batch: UInt32(batch), wl: geometry.wavelengthAngstroms,
        semiangle: geometry.semiangleRadians, angularRow: geometry.angularSamplingYRadians,
        angularColumn: geometry.angularSamplingXRadians,
        factor: geometry.phaseFactor ?? Float.pi / geometry.wavelengthAngstroms)
    }
    func gradientGrid(_ geometry: MetalSSBGeometry, batch: Int) -> GradientGrid {
      let g = grid(geometry, batch: batch)
      return GradientGrid(
        count: g.count, bandCount: g.bandCount, chunks: g.chunks, batch: g.batch,
        wl: g.wl, semiangle: g.semiangle, angularRow: g.angularRow,
        angularColumn: g.angularColumn, factor: g.factor,
        span: UInt32(SSBFifthOrderGradient.span), spans: UInt32(spans), qGroups: UInt32(qGroups),
        qStride: UInt32(qStride), fullBandCount: UInt32(fullBandCount))
    }
  }

  private func encoder(_ commands: MTLCommandBuffer, _ pipeline: MTLComputePipelineState) throws
    -> MTLComputeCommandEncoder
  {
    guard let result = commands.makeComputeCommandEncoder() else { throw MetalSSBError.commandQueue }
    result.setComputePipelineState(pipeline)
    return result
  }

  private func encodeGradient(
    _ commands: MTLCommandBuffer, layout: Layout, grid: inout GradientGrid, batch: Int, first: Int,
    buffers: (
      band: MTLBuffer, k: MTLBuffer, q: MTLBuffer, gradients: MTLBuffer, trials: MTLBuffer,
      probeGeometry: MTLBuffer, chi: MTLBuffer, scales: MTLBuffer
    )
  ) throws {
    let gradientEncoder = try encoder(commands, gradient)
    for (index, buffer) in [buffers.band, buffers.k, buffers.q].enumerated() {
      gradientEncoder.setBuffer(buffer, offset: 0, index: index)
    }
    gradientEncoder.setBuffer(buffers.gradients, offset: first * layout.perGradient * 4, index: 3)
    gradientEncoder.setBytes(&grid, length: MemoryLayout<GradientGrid>.stride, index: 4)
    gradientEncoder.setBuffer(buffers.trials, offset: 0, index: 5)
    gradientEncoder.setBuffer(buffers.probeGeometry, offset: 0, index: 6)
    gradientEncoder.setBuffer(buffers.chi, offset: 0, index: 7)
    gradientEncoder.setBuffer(buffers.scales, offset: 0, index: 8)
    gradientEncoder.dispatchThreadgroups(
      MTLSize(width: layout.qGroups / 2, height: layout.spans, depth: batch),
      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    gradientEncoder.endEncoding()
  }

  /// Simdgroup partials to Double sums; the coefficient entries take the basis factor 2 pi / wl.
  private func reduce(values: [Double], gradients: MTLBuffer, layout: Layout, geometry: MetalSSBGeometry)
    throws -> [Value]
  {
    let partials = gradients.contents().assumingMemoryBound(to: Float.self)
    let basis = 2 * Double.pi / Double(geometry.wavelengthAngstroms)
    var result: [Value] = []
    for row in values.indices {
      var sums = [Double](repeating: 0, count: 31)
      let base = partials + row * layout.perGradient
      for group in 0..<(layout.spans * layout.qGroups) {
        let entry = base + group * 32
        for j in 0..<31 { sums[j] += Double(entry[j]) }
      }
      for j in 3..<31 { sums[j] *= basis }
      result.append(Value(agreement: values[row], gradient: sums))
    }
    guard result.allSatisfy({ $0.gradient.allSatisfy(\.isFinite) }) else {
      throw SSBOptimizationError.nonFiniteLoss
    }
    return result
  }
}

extension SSBFifthOrderCoordinates {
  /// d / dz from d / d row, through the linear `pack` and the row's depth (A) and tilt (rad) entries.
  func gradient(row: [Double]) -> [Double] {
    var z = Array(repeating: 0.0, count: 28)
    z[0] = 50 * row[3]
    z[1] = 50 * row[5]
    z[2] = 50 * row[6]
    z[3] = 5e-3 * row[1]
    z[4] = 5e-3 * row[2]
    z[5] = 100 * row[0]
    var j = 6
    for (i, (n, m)) in Self.orders.enumerated() where n >= 2 {
      let scale = wavelengthAngstroms / (2 * Double.pi * pow(semiangleRadians, Double(n + 1)))
      z[j] = scale * row[3 + 2 * i]
      j += 1
      if m != 0 {
        z[j] = scale * row[3 + 2 * i + 1]
        j += 1
      }
    }
    return z
  }
}
