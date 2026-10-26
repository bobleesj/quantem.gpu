import Foundation
import Metal

/// Batched all-BF fifth-order agreement with persistent, bounded GPU partials and GPU score reduction.
/// The model supplies its already gathered native-grid evidence and reusable probe geometry.
///
/// Evaluation is `fifth_agreement_rows` (ssb-fifth-agreement, `ssb_fifth_agreement.metal`), also the value pass of
/// `DirectPtychoFifthOrderGradient` on all or every n-th BF pixel.
final class DirectPtychoFifthOrderAgreement {
  private struct Grid {
    var count, bandCount, chunks, batch: UInt32
    var wl, semiangle, angularRow, angularColumn, factor: Float
  }
  private struct RowsGrid {
    var count, bandCount, chunks, batch, first, kStride: UInt32
    var wl, semiangle, angularRow, angularColumn, factor, inner2, outer2: Float
  }
  /// Band frequencies per threadgroup and BF pixels per shared batch (`FIFTH_S` in the kernel).
  static let tileWidth = 128
  static let pixelBatch = 16

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let library: MTLLibrary
  private let probe, power, score: MTLComputePipelineState
  private var rowsPipelines: [Int: MTLComputePipelineState] = [:]
  /// The band tiles of `fifth_agreement_rows`, shared with the gradient pass.
  let tiles: MTLBuffer
  let tileCount: Int
  private var workspaces: [String: MTLBuffer] = [:]
  /// GPU time of the latest `evaluate` command buffer.
  private(set) var lastGPUSeconds = 0.0

  /// `q` holds the band frequencies (row, column) in band order, each band row contiguous, as
  /// `MetalDirectPtychoJointModel` builds them.
  init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, q: [SIMD2<Float>]) throws {
    self.device = device
    self.queue = queue
    self.library = library
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalDirectPtychoError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    probe = try pipeline("fifth_probe")
    power = try pipeline("fifth_power")
    score = try pipeline("fifth_score")
    let table = Self.tiles(q, width: Self.tileWidth)
    tiles = try SSBCoherentReconstruction.buffer(device, table)
    tileCount = table.count
  }

  /// Threadgroup tiles: (first band index, count, lanes in the first band row, rows) of up to `width` consecutive
  /// band frequencies spanning at most two band rows (equal row coordinate).
  static func tiles(_ q: [SIMD2<Float>], width: Int) -> [SIMD4<UInt32>] {
    var rowStart: [Int] = []
    for i in q.indices where i == 0 || q[i].x != q[i - 1].x { rowStart.append(i) }
    rowStart.append(q.count)
    var rowOf = [Int](repeating: 0, count: q.count)
    for r in 0..<(rowStart.count - 1) {
      for i in rowStart[r]..<rowStart[r + 1] { rowOf[i] = r }
    }
    var result: [SIMD4<UInt32>] = []
    var first = 0
    while first < q.count {
      var end = min(first + width, q.count)
      if rowOf[end - 1] - rowOf[first] > 1 { end = rowStart[rowOf[first] + 2] }
      let single = rowOf[end - 1] == rowOf[first]
      let split = single ? end - first : rowStart[rowOf[first] + 1] - first
      result.append(SIMD4(UInt32(first), UInt32(end - first), UInt32(split), single ? 1 : 2))
      first = end
    }
    return result
  }

  private func rowsPipeline(_ perThread: Int) throws -> MTLComputePipelineState {
    if let existing = rowsPipelines[perThread] { return existing }
    let constants = MTLFunctionConstantValues()
    var value = UInt32(perThread)
    constants.setConstantValue(&value, type: .uint, index: 40)
    let function = try library.makeFunction(name: "fifth_agreement_rows", constantValues: constants)
    let pipeline = try device.makeComputePipelineState(function: function)
    rowsPipelines[perThread] = pipeline
    return pipeline
  }

  private func workspace(_ name: String, bytes: Int, shared: Bool = false) throws -> MTLBuffer {
    if let existing = workspaces[name], existing.length >= bytes { return existing }
    let buffer = try SSBCoherentReconstruction.allocate(device, bytes, shared: shared)
    workspaces[name] = buffer
    return buffer
  }

  private static func batchSize(count: Int, bandCount: Int, candidates: Int) -> Int {
    let chunks = (count + 255) / 256
    let perCandidate = (chunks + 1) * bandCount * 16 + count * 4
    return min(candidates, max(4, ((64 << 20) / perCandidate / 4) * 4))
  }

  /// The 256-pixel chunks of the partial sums over every `kStride`-th of `count` BF pixels.
  static func chunks(count: Int, kStride: Int) -> Int {
    ((count + kStride - 1) / kStride + 255) / 256
  }

  /// Squared radii inside which the soft aperture is exactly 1 and outside which it is exactly 0 (worst-case
  /// anisotropic sampling); they move inward/outward by 1e-4, far beyond the float32 rounding of the edge expression.
  static func apertureRadii(_ geometry: MetalDirectPtychoGeometry) -> (inner2: Float, outer2: Float)
  {
    let wl = Double(geometry.wavelengthAngstroms)
    let semi = Double(geometry.semiangleRadians)
    let angular = Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians))
    let inner = (semi - 0.5 * angular) / wl
    let outer = (semi + 0.5 * angular) / wl
    return (inner > 0 ? Float(inner * inner * (1 - 1e-4)) : -1, Float(outer * outer * (1 + 1e-4)))
  }

  private func grid(
    geometry: MetalDirectPtychoGeometry, count: Int, bandCount: Int, batch: Int
  ) -> Grid {
    Grid(
      count: UInt32(count), bandCount: UInt32(bandCount), chunks: UInt32((count + 255) / 256),
      batch: UInt32(batch), wl: geometry.wavelengthAngstroms,
      semiangle: geometry.semiangleRadians, angularRow: geometry.angularSamplingYRadians,
      angularColumn: geometry.angularSamplingXRadians,
      factor: geometry.phaseFactor ?? Float.pi / geometry.wavelengthAngstroms)
  }

  private func encode(
    _ commands: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
    _ body: (MTLComputeCommandEncoder) -> Void
  ) throws {
    guard let encoder = commands.makeComputeCommandEncoder() else {
      throw MetalDirectPtychoError.commandQueue
    }
    encoder.setComputePipelineState(pipeline)
    body(encoder)
    encoder.endEncoding()
  }

  /// chi(k) of each candidate (the reference `fifth_probe`).
  private func encodeProbe(
    _ commands: MTLCommandBuffer, grid: inout Grid, k: MTLBuffer, trials: MTLBuffer, chi: MTLBuffer
  ) throws {
    try encode(commands, probe) { encoder in
      for (index, buffer) in [k, trials, chi].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 3)
      encoder.dispatchThreads(
        MTLSize(width: Int(grid.count), height: Int(grid.batch), depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
  }

  /// Chunk and weighted-band reduction to one float32 score per candidate (`fifth_power`, `fifth_score`).
  private func encodeReduction(
    _ commands: MTLCommandBuffer, grid: inout Grid, partial: MTLBuffer, weights: MTLBuffer,
    powers: MTLBuffer, powersOffset: Int = 0, scores: MTLBuffer, first: Int
  ) throws {
    try encode(commands, power) { encoder in
      encoder.setBuffer(partial, offset: 0, index: 0)
      encoder.setBuffer(weights, offset: 0, index: 1)
      encoder.setBuffer(powers, offset: powersOffset, index: 2)
      encoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 3)
      encoder.dispatchThreads(
        MTLSize(width: Int(grid.bandCount), height: Int(grid.batch), depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    try encode(commands, score) { encoder in
      encoder.setBuffer(powers, offset: powersOffset, index: 0)
      encoder.setBuffer(scores, offset: first * 4, index: 1)
      encoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 2)
      encoder.dispatchThreadgroups(
        MTLSize(width: Int(grid.batch), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
  }

  private func finish(_ commands: MTLCommandBuffer) throws {
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalDirectPtychoError.commandExecution(error.localizedDescription)
    }
    lastGPUSeconds = commands.gpuEndTime - commands.gpuStartTime
  }

  /// The GPU inputs of one batch of rows: the rows themselves and their collapsed chi polynomials.
  struct Batch {
    let trials: MTLBuffer
    let polynomials: MTLBuffer
    let count: Int
  }

  func batch(_ rows: ArraySlice<[Float]>, geometry: MetalDirectPtychoGeometry) throws -> Batch {
    let polynomials = rows.flatMap {
      SSBRedrawPolynomial.full(
        coefficients: Array($0[3..<31]), wavelength: geometry.wavelengthAngstroms)
    }
    return Batch(
      trials: try SSBCoherentReconstruction.buffer(device, rows.flatMap { $0 }),
      polynomials: try SSBCoherentReconstruction.buffer(device, polynomials), count: rows.count)
  }

  /// Encodes the agreement of one batch on every `kStride`-th BF pixel: chi(k) of each row (`fifth_probe`, all
  /// pixels), the row-collapsed chunk partials (`fifth_agreement_rows`, four candidates per thread, then one dispatch
  /// for the remaining one to three), the float32 per-frequency powers at `powersOffset` and the scores at `first`.
  /// The chunk partials and chi stay in `partial` and `chi` for a gradient pass.
  func encodeBatch(
    _ commands: MTLCommandBuffer, _ batch: Batch, kStride: Int, geometry: MetalDirectPtychoGeometry,
    k: MTLBuffer, q: MTLBuffer, band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount: Int, chi: MTLBuffer, partial: MTLBuffer, powers: MTLBuffer,
    powersOffset: Int, scores: MTLBuffer, first: Int
  ) throws {
    let chunks = Self.chunks(count: count, kStride: kStride)
    var grid = grid(geometry: geometry, count: count, bandCount: bandCount, batch: batch.count)
    try encodeProbe(commands, grid: &grid, k: k, trials: batch.trials, chi: chi)
    let radii = Self.apertureRadii(geometry)
    var rowsGrid = RowsGrid(
      count: UInt32(count), bandCount: UInt32(bandCount), chunks: UInt32(chunks),
      batch: UInt32(batch.count), first: 0, kStride: UInt32(kStride), wl: grid.wl,
      semiangle: grid.semiangle, angularRow: grid.angularRow, angularColumn: grid.angularColumn,
      factor: grid.factor, inner2: radii.inner2, outer2: radii.outer2)
    let full = batch.count / 4 * 4
    for (perThread, offset, groups) in [(4, 0, full / 4), (batch.count - full, full, 1)]
    where perThread > 0 && groups > 0 {
      rowsGrid.first = UInt32(offset)
      try encode(commands, try rowsPipeline(perThread)) { encoder in
        for (index, buffer) in [band, k, q, partial].enumerated() {
          encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.setBytes(&rowsGrid, length: MemoryLayout<RowsGrid>.stride, index: 4)
        for (index, buffer) in [batch.trials, probeGeometry, batch.polynomials, tiles, chi]
          .enumerated()
        {
          encoder.setBuffer(buffer, offset: 0, index: 5 + index)
        }
        encoder.setThreadgroupMemoryLength(2 * Self.pixelBatch * perThread * 64, index: 0)
        encoder.setThreadgroupMemoryLength(perThread * 49 * 4, index: 1)
        encoder.dispatchThreadgroups(
          MTLSize(width: tileCount, height: chunks, depth: groups),
          threadsPerThreadgroup: MTLSize(width: Self.tileWidth, height: 1, depth: 1))
      }
    }
    grid.chunks = UInt32(chunks)
    try encodeReduction(
      commands, grid: &grid, partial: partial, weights: weights, powers: powers,
      powersOffset: powersOffset, scores: scores, first: first)
  }

  /// One score per row (depth, tilt row, tilt column, then the 28 packed coefficients), in the order given.
  func evaluate(
    rows input: [[Float]], geometry: MetalDirectPtychoGeometry, k: MTLBuffer, q: MTLBuffer,
    band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount: Int, isCancelled: () -> Bool
  ) throws -> [Double] {
    guard !input.isEmpty else { return [] }
    // Candidates with equal (depth, tilt) become neighbours and share their depth-sinc weights in the kernel.
    let order = input.indices.sorted {
      let a = input[$0]
      let b = input[$1]
      return (a[0], a[1], a[2], $0) < (b[0], b[1], b[2], $1)
    }
    let sorted = order.map { input[$0] }
    let chunks = Self.chunks(count: count, kStride: 1)
    let batchSize = Self.batchSize(count: count, bandCount: bandCount, candidates: sorted.count)
    let chi = try workspace("chi", bytes: count * batchSize * 4)
    let partial = try workspace("partial", bytes: chunks * bandCount * batchSize * 16)
    let powers = try workspace("power", bytes: bandCount * batchSize * 4)
    let scores = try workspace("scores", bytes: sorted.count * 4, shared: true)
    guard let commands = queue.makeCommandBuffer() else {
      throw MetalDirectPtychoError.commandQueue
    }
    for first in stride(from: 0, to: sorted.count, by: batchSize) {
      if isCancelled() { throw CancellationError() }
      let rows = sorted[first..<min(first + batchSize, sorted.count)]
      try encodeBatch(
        commands, try batch(rows, geometry: geometry), kStride: 1, geometry: geometry, k: k, q: q,
        band: band, weights: weights, probeGeometry: probeGeometry, count: count,
        bandCount: bandCount, chi: chi, partial: partial, powers: powers, powersOffset: 0,
        scores: scores, first: first)
    }
    try finish(commands)
    if isCancelled() { throw CancellationError() }
    let values = scores.contents().assumingMemoryBound(to: Float.self)
    var result = [Double](repeating: 0, count: input.count)
    for (position, index) in order.enumerated() { result[index] = Double(values[position]) }
    guard result.allSatisfy(\.isFinite) else { throw SSBOptimizationError.nonFiniteLoss }
    return result
  }
}
