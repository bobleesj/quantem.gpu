import Foundation
import Metal

/// The fifth-order agreement and its exact gradient for a batch of candidates, on all BF pixels or every n-th.
/// Pass 1 is the objective itself (`SSBFifthOrderAgreement.encodeBatch`: `fifth_probe`, `fifth_agreement_rows`,
/// `fifth_power`, `fifth_score`), so the agreement on all BF pixels is bit-identical to `fifthOrderAgreements`.
/// Pass 2 (`fifth_gradient_rows`, ssb-fifth-gradient v0.2) uses the per-frequency projection a_q = N_q / D_q of
/// pass 1 (`fifth_gradient_scale`). Threadgroup partials of the gradient are reduced on the host in Double.
final class SSBFifthOrderGradient {
  private struct Grid {
    var count, bandCount, chunks, batch: UInt32
    var wl, semiangle, angularRow, angularColumn, factor: Float
  }
  private struct GradientGrid {
    var count, bandCount, batch, kStride, span, spans, tiles: UInt32
    var wl, semiangle, angularRow, angularColumn, factor, inner2, outer2: Float
  }

  /// One candidate: the agreement and d agreement / d row, in the row layout of the objective:
  /// [depth (A), tilt row (rad), tilt column (rad), 28 packed Cartesian coefficients (A)].
  struct Value {
    /// The float32 score of `fifth_score` (the objective).
    let agreement: Double
    /// The same float32 per-frequency powers summed in Double on the host (finer than the float32 total).
    let agreement64: Double
    let gradient: [Double]
  }

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let objective: SSBFifthOrderAgreement
  private let scale, gradient: MTLComputePipelineState
  private var workspaces: [String: MTLBuffer] = [:]
  /// d chi / d packed coefficient as row^a col^b polynomials (`SSBRedrawPolynomial.full` of each unit coefficient).
  private var basis: (wavelength: Float, polynomials: [[Double]])?
  /// GPU time of the latest command buffer (`evaluate` or `retainedGradients`).
  private(set) var lastGPUSeconds = 0.0

  /// `objective` evaluates the value pass and supplies its band tiles.
  init(
    device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, objective: SSBFifthOrderAgreement
  ) throws {
    self.device = device
    self.queue = queue
    self.objective = objective
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalSSBError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    scale = try pipeline("fifth_gradient_scale")
    gradient = try pipeline("fifth_gradient_rows")
  }

  private func workspace(_ name: String, bytes: Int, shared: Bool = false) throws -> MTLBuffer {
    if let existing = workspaces[name], existing.length >= bytes { return existing }
    let buffer = try SSBCoherentReconstruction.allocate(device, bytes, shared: shared)
    workspaces[name] = buffer
    return buffer
  }

  private struct Retained {
    let rows: [[Float]]
    let kStride: Int
    let batch: SSBFifthOrderAgreement.Batch
    let values: [Double]
    let values64: [Double]
  }
  /// The last value-only evaluation, whose projections a_q and probe phases stay in the workspaces.
  private var retained: Retained?

  /// - Parameters:
  ///   - kStride: 1 evaluates the objective and its gradient. A larger stride evaluates the agreement restricted to
  ///     every kStride-th BF pixel and its gradient: a steering model, never the objective.
  ///   - gradients: false evaluates the agreements only (empty gradients) and retains their projections, so
  ///     `retainedGradients` can add the gradient pass for the same rows without repeating the objective.
  func evaluate(
    rows: [[Float]], geometry: MetalSSBGeometry, k: MTLBuffer, q: MTLBuffer,
    band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount: Int, kStride: Int = 1, gradients withGradients: Bool = true,
    isCancelled: () -> Bool
  ) throws -> [Value] {
    guard !rows.isEmpty else { return [] }
    precondition(kStride >= 1)
    retained = nil
    let layout = Layout(
      count: count, bandCount: bandCount, kStride: kStride, tiles: objective.tileCount,
      rows: rows.count)
    let chi = try workspace("chi", bytes: count * layout.batchSize * 4)
    let partial = try workspace("partial", bytes: layout.chunks * bandCount * layout.batchSize * 16)
    let powers = try workspace("power", bytes: bandCount * rows.count * 4, shared: true)
    let scales = try workspace("scale", bytes: bandCount * layout.batchSize * 16)
    let scores = try workspace("scores", bytes: rows.count * 4, shared: true)
    let gradients = try workspace(
      "gradients", bytes: rows.count * layout.perGradient * 4, shared: true)
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    var last: SSBFifthOrderAgreement.Batch?
    for first in stride(from: 0, to: rows.count, by: layout.batchSize) {
      if isCancelled() { throw CancellationError() }
      let batch = try objective.batch(
        rows[first..<min(first + layout.batchSize, rows.count)], geometry: geometry)
      last = batch
      try objective.encodeBatch(
        commands, batch, kStride: kStride, geometry: geometry, k: k, q: q, band: band,
        weights: weights, probeGeometry: probeGeometry, count: count, bandCount: bandCount,
        chi: chi, partial: partial, powers: powers, powersOffset: first * bandCount * 4,
        scores: scores, first: first)
      var grid = layout.grid(geometry, batch: batch.count)
      try encode(commands, scale) { encoder in
        for (index, buffer) in [partial, weights, scales].enumerated() {
          encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 3)
        encoder.dispatchThreads(
          MTLSize(width: bandCount, height: batch.count, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      }
      if withGradients {
        try encodeGradient(
          commands, layout: layout, geometry: geometry, batch: batch, first: first,
          buffers: (band, k, q, gradients, probeGeometry, chi, scales))
      }
    }
    try finish(commands)
    if isCancelled() { throw CancellationError() }
    let pointer = scores.contents().assumingMemoryBound(to: Float.self)
    let values = rows.indices.map { Double(pointer[$0]) }
    let power = powers.contents().assumingMemoryBound(to: Float.self)
    let values64 = rows.indices.map { row in
      (0..<bandCount).reduce(0.0) { $0 + Double(power[row * bandCount + $1]) }
    }
    guard values.allSatisfy(\.isFinite), values64.allSatisfy(\.isFinite) else {
      throw SSBOptimizationError.nonFiniteLoss
    }
    if !withGradients {
      // One batch keeps every row's projections in the workspaces.
      if rows.count <= layout.batchSize, let last {
        retained = Retained(
          rows: rows, kStride: kStride, batch: last, values: values, values64: values64)
      }
      return zip(values, values64).map { Value(agreement: $0, agreement64: $1, gradient: []) }
    }
    return try reduce(
      values: values, values64: values64, gradients: gradients, layout: layout, geometry: geometry)
  }

  /// The gradients of the rows of the last value-only `evaluate`, by the gradient pass alone; other rows are
  /// evaluated in full.
  func retainedGradients(
    rows: [[Float]], geometry: MetalSSBGeometry, k: MTLBuffer, q: MTLBuffer,
    band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount: Int, kStride: Int, isCancelled: () -> Bool
  ) throws -> [Value] {
    guard let kept = retained, kept.rows == rows, kept.kStride == kStride,
      let chi = workspaces["chi"], let scales = workspaces["scale"]
    else {
      return try evaluate(
        rows: rows, geometry: geometry, k: k, q: q, band: band, weights: weights,
        probeGeometry: probeGeometry, count: count, bandCount: bandCount, kStride: kStride,
        isCancelled: isCancelled)
    }
    retained = nil
    let layout = Layout(
      count: count, bandCount: bandCount, kStride: kStride, tiles: objective.tileCount,
      rows: rows.count)
    let gradients = try workspace(
      "gradients", bytes: rows.count * layout.perGradient * 4, shared: true)
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    try encodeGradient(
      commands, layout: layout, geometry: geometry, batch: kept.batch, first: 0,
      buffers: (band, k, q, gradients, probeGeometry, chi, scales))
    try finish(commands)
    if isCancelled() { throw CancellationError() }
    return try reduce(
      values: kept.values, values64: kept.values64, gradients: gradients, layout: layout,
      geometry: geometry)
  }

  private struct Layout {
    let count, bandCount, kStride, tiles, chunks, span, spans, batchSize, perGradient: Int
    init(count: Int, bandCount: Int, kStride: Int, tiles: Int, rows: Int) {
      self.count = count
      self.bandCount = bandCount
      self.kStride = kStride
      self.tiles = tiles
      let used = (count + kStride - 1) / kStride
      chunks = SSBFifthOrderAgreement.chunks(count: count, kStride: kStride)
      // At most 1024 used pixels per gradient threadgroup, and down to 256 so that a single candidate on a BF
      // subsample still dispatches about 4096 threadgroups.
      let wanted = max(1, 4096 / max(1, tiles))
      var spans = max((used + 1023) / 1024, min((used + 255) / 256, wanted))
      span = ((used + spans - 1) / spans + 15) / 16 * 16
      spans = (used + span - 1) / span
      self.spans = spans
      let perCandidate = (chunks + 1) * bandCount * 16 + count * 4 + bandCount * 16
      batchSize = min(rows, max(4, ((64 << 20) / perCandidate / 4) * 4))
      perGradient = spans * tiles * 28
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
      let radii = SSBFifthOrderAgreement.apertureRadii(geometry)
      return GradientGrid(
        count: g.count, bandCount: g.bandCount, batch: g.batch, kStride: UInt32(kStride),
        span: UInt32(span), spans: UInt32(spans), tiles: UInt32(tiles), wl: g.wl,
        semiangle: g.semiangle, angularRow: g.angularRow, angularColumn: g.angularColumn,
        factor: g.factor, inner2: radii.inner2, outer2: radii.outer2)
    }
  }

  private func encode(
    _ commands: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
    _ body: (MTLComputeCommandEncoder) -> Void
  ) throws {
    guard let encoder = commands.makeComputeCommandEncoder() else { throw MetalSSBError.commandQueue }
    encoder.setComputePipelineState(pipeline)
    body(encoder)
    encoder.endEncoding()
  }

  private func finish(_ commands: MTLCommandBuffer) throws {
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
    lastGPUSeconds = commands.gpuEndTime - commands.gpuStartTime
  }

  private func encodeGradient(
    _ commands: MTLCommandBuffer, layout: Layout, geometry: MetalSSBGeometry,
    batch: SSBFifthOrderAgreement.Batch, first: Int,
    buffers: (
      band: MTLBuffer, k: MTLBuffer, q: MTLBuffer, gradients: MTLBuffer, probeGeometry: MTLBuffer,
      chi: MTLBuffer, scales: MTLBuffer
    )
  ) throws {
    var grid = layout.gradientGrid(geometry, batch: batch.count)
    try encode(commands, gradient) { encoder in
      for (index, buffer) in [buffers.band, buffers.k, buffers.q].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBuffer(buffers.gradients, offset: first * layout.perGradient * 4, index: 3)
      encoder.setBytes(&grid, length: MemoryLayout<GradientGrid>.stride, index: 4)
      for (index, buffer) in [
        batch.trials, buffers.probeGeometry, batch.polynomials, objective.tiles, buffers.chi,
        buffers.scales,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: 5 + index)
      }
      encoder.setThreadgroupMemoryLength(2 * SSBFifthOrderAgreement.pixelBatch * 2 * 32, index: 0)
      encoder.setThreadgroupMemoryLength(SSBFifthOrderAgreement.pixelBatch * 6 * 16, index: 1)
      encoder.setThreadgroupMemoryLength((52 + 4 * 28) * 4, index: 2)
      encoder.dispatchThreadgroups(
        MTLSize(width: layout.tiles, height: layout.spans, depth: batch.count),
        threadsPerThreadgroup: MTLSize(width: SSBFifthOrderAgreement.tileWidth, height: 1, depth: 1))
    }
  }

  /// d chi / d c_j for the 28 packed coefficients as 7 x 7 row^a col^b polynomials (1/A units, with 2 pi / wl).
  private func basisPolynomials(_ wavelength: Float) -> [[Double]] {
    if let basis, basis.wavelength == wavelength { return basis.polynomials }
    let polynomials = (0..<28).map { j in
      SSBRedrawPolynomial.fullDouble(
        coefficients: (0..<28).map { $0 == j ? 1.0 : 0.0 }, wavelength: Double(wavelength))
    }
    basis = (wavelength, polynomials)
    return polynomials
  }

  /// Threadgroup partials to Double sums: depth and tilt directly; the monomial sums T_a through the basis
  /// polynomials to the 28 packed coefficients.
  private func reduce(
    values: [Double], values64: [Double], gradients: MTLBuffer, layout: Layout,
    geometry: MetalSSBGeometry
  ) throws -> [Value] {
    let partials = gradients.contents().assumingMemoryBound(to: Float.self)
    let basis = basisPolynomials(geometry.wavelengthAngstroms)
    // (index in the degree-ordered partial, a, b) of every monomial row^a col^b, 2 <= a + b <= 6.
    let monomials = (2...6).flatMap { d in (0...d).map { b in (d * (d + 1) / 2 + b, d - b, b) } }
    var result: [Value] = []
    for row in values.indices {
      var sums = [Double](repeating: 0, count: 28)
      let base = partials + row * layout.perGradient
      for group in 0..<(layout.spans * layout.tiles) {
        let entry = base + group * 28
        for j in 0..<28 { sums[j] += Double(entry[j]) }
      }
      var gradient = Array(sums[0..<3])
      for polynomial in basis {
        gradient.append(monomials.reduce(0.0) { $0 + polynomial[$1.1 * 7 + $1.2] * sums[$1.0] })
      }
      result.append(Value(agreement: values[row], agreement64: values64[row], gradient: gradient))
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
