import Foundation
import Metal

/// Batched all-BF fifth-order agreement with persistent, bounded GPU partials and GPU score reduction.
/// The model supplies its already gathered native-grid evidence and reusable probe geometry.
///
/// Production evaluation is `fifth_agreement_rows` (ssb-fifth-agreement v0.2, `ssb_fifth_agreement.metal`). The
/// earlier `fifth_agreement` (`ssb_fifth_order.metal`) is kept only as the parity reference of the tests
/// (`evaluateReference`); there is no runtime fallback.
final class SSBFifthOrderAgreement {
  private struct Grid {
    var count, bandCount, chunks, batch: UInt32
    var wl, semiangle, angularRow, angularColumn, factor: Float
  }
  private struct RowsGrid {
    var count, bandCount, chunks, batch, first: UInt32
    var wl, semiangle, angularRow, angularColumn, factor, inner2, outer2: Float
  }
  /// Band frequencies per threadgroup and BF pixels per shared batch (`FIFTH_S` in the kernel).
  static let tileWidth = 128
  static let pixelBatch = 16

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let library: MTLLibrary
  private let probe, agreement, power, score: MTLComputePipelineState
  private var rowsPipelines: [Int: MTLComputePipelineState] = [:]
  private let tiles: MTLBuffer
  private let tileCount: Int
  private var workspaces: [String: MTLBuffer] = [:]
  /// GPU time of the latest `evaluate` command buffer.
  private(set) var lastGPUSeconds = 0.0

  /// `q` holds the band frequencies (row, column) in band order, each band row contiguous, as
  /// `MetalSSBJointModel` builds them.
  init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, q: [SIMD2<Float>]) throws {
    self.device = device
    self.queue = queue
    self.library = library
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

  private func grid(
    geometry: MetalSSBGeometry, count: Int, bandCount: Int, batch: Int
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
      throw MetalSSBError.commandQueue
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
    powers: MTLBuffer, scores: MTLBuffer, first: Int
  ) throws {
    try encode(commands, power) { encoder in
      for (index, buffer) in [partial, weights, powers].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 3)
      encoder.dispatchThreads(
        MTLSize(width: Int(grid.bandCount), height: Int(grid.batch), depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    try encode(commands, score) { encoder in
      encoder.setBuffer(powers, offset: 0, index: 0)
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
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
    lastGPUSeconds = commands.gpuEndTime - commands.gpuStartTime
  }

  /// One score per row (depth, tilt row, tilt column, then the 28 packed coefficients), in the order given.
  func evaluate(
    rows input: [[Float]], geometry: MetalSSBGeometry, k: MTLBuffer, q: MTLBuffer,
    band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount: Int, isCancelled: () -> Bool
  ) throws -> [Double] {
    guard !input.isEmpty else { return [] }
    // Candidates with equal (depth, tilt) become neighbours and share their depth-sinc weights in the kernel.
    let order = input.indices.sorted {
      let a = input[$0], b = input[$1]
      return (a[0], a[1], a[2], $0) < (b[0], b[1], b[2], $1)
    }
    let sorted = order.map { input[$0] }
    let polynomials = sorted.map {
      SSBRedrawPolynomial.full(
        coefficients: Array($0[3..<31]), wavelength: geometry.wavelengthAngstroms)
    }
    let chunks = (count + 255) / 256
    let batchSize = Self.batchSize(count: count, bandCount: bandCount, candidates: sorted.count)
    let chi = try workspace("chi", bytes: count * batchSize * 4)
    let partial = try workspace("partial", bytes: chunks * bandCount * batchSize * 16)
    let powers = try workspace("power", bytes: bandCount * batchSize * 4)
    let scores = try workspace("scores", bytes: sorted.count * 4, shared: true)
    // The soft aperture is exactly 1 inside `inner` and 0 outside `outer` (worst-case anisotropic sampling);
    // the squared radii move inward/outward by 1e-4, far beyond the float32 rounding of the edge expression.
    let wl = Double(geometry.wavelengthAngstroms), semi = Double(geometry.semiangleRadians)
    let angular = Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians))
    let inner = (semi - 0.5 * angular) / wl, outer = (semi + 0.5 * angular) / wl
    let base = grid(geometry: geometry, count: count, bandCount: bandCount, batch: 0)
    var rowsGrid = RowsGrid(
      count: UInt32(count), bandCount: UInt32(bandCount), chunks: UInt32(chunks), batch: 0, first: 0,
      wl: base.wl, semiangle: base.semiangle, angularRow: base.angularRow,
      angularColumn: base.angularColumn, factor: base.factor,
      inner2: inner > 0 ? Float(inner * inner * (1 - 1e-4)) : -1,
      outer2: Float(outer * outer * (1 + 1e-4)))
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    for first in stride(from: 0, to: sorted.count, by: batchSize) {
      if isCancelled() { throw CancellationError() }
      let batch = min(batchSize, sorted.count - first)
      var grid = grid(geometry: geometry, count: count, bandCount: bandCount, batch: batch)
      rowsGrid.batch = UInt32(batch)
      let trials = try SSBCoherentReconstruction.buffer(
        device, sorted[first..<(first + batch)].flatMap { $0 })
      let poly = try SSBCoherentReconstruction.buffer(
        device, polynomials[first..<(first + batch)].flatMap { $0 })
      try encodeProbe(commands, grid: &grid, k: k, trials: trials, chi: chi)
      // Four candidates per thread, then one dispatch for the remaining one to three.
      let full = batch / 4 * 4
      for (perThread, offset, groups) in [(4, 0, full / 4), (batch - full, full, 1)]
      where perThread > 0 && groups > 0 {
        rowsGrid.first = UInt32(offset)
        try encode(commands, try rowsPipeline(perThread)) { encoder in
          for (index, buffer) in [band, k, q, partial].enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
          }
          encoder.setBytes(&rowsGrid, length: MemoryLayout<RowsGrid>.stride, index: 4)
          for (index, buffer) in [trials, probeGeometry, poly, tiles, chi].enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: 5 + index)
          }
          encoder.setThreadgroupMemoryLength(2 * Self.pixelBatch * perThread * 64, index: 0)
          encoder.setThreadgroupMemoryLength(perThread * 49 * 4, index: 1)
          encoder.dispatchThreadgroups(
            MTLSize(width: tileCount, height: chunks, depth: groups),
            threadsPerThreadgroup: MTLSize(width: Self.tileWidth, height: 1, depth: 1))
        }
      }
      try encodeReduction(
        commands, grid: &grid, partial: partial, weights: weights, powers: powers, scores: scores,
        first: first)
    }
    try finish(commands)
    if isCancelled() { throw CancellationError() }
    let values = scores.contents().assumingMemoryBound(to: Float.self)
    var result = [Double](repeating: 0, count: input.count)
    for (position, index) in order.enumerated() { result[index] = Double(values[position]) }
    guard result.allSatisfy(\.isFinite) else { throw SSBOptimizationError.nonFiniteLoss }
    return result
  }

  /// Tests only: the retired per-term kernel `fifth_agreement` (ssb-per-term v0.1), the parity reference of v0.2.
  func evaluateReference(
    rows: [[Float]], geometry: MetalSSBGeometry, k: MTLBuffer, q: MTLBuffer,
    band: MTLBuffer, weights: MTLBuffer, probeGeometry: MTLBuffer,
    count: Int, bandCount: Int
  ) throws -> [Double] {
    guard !rows.isEmpty else { return [] }
    let chunks = (count + 255) / 256
    let batchSize = Self.batchSize(count: count, bandCount: bandCount, candidates: rows.count)
    let chi = try workspace("chi", bytes: count * batchSize * 4)
    let partial = try workspace("partial", bytes: chunks * bandCount * batchSize * 16)
    let powers = try workspace("power", bytes: bandCount * batchSize * 4)
    let scores = try workspace("scores", bytes: rows.count * 4, shared: true)
    guard let commands = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    for first in stride(from: 0, to: rows.count, by: batchSize) {
      let batch = min(batchSize, rows.count - first)
      var grid = grid(geometry: geometry, count: count, bandCount: bandCount, batch: batch)
      let trials = try SSBCoherentReconstruction.buffer(
        device, rows[first..<(first + batch)].flatMap { $0 })
      try encodeProbe(commands, grid: &grid, k: k, trials: trials, chi: chi)
      try encode(commands, agreement) { encoder in
        for (index, buffer) in [band, k, q, partial].enumerated() {
          encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 4)
        encoder.setBuffer(trials, offset: 0, index: 5)
        encoder.setBuffer(probeGeometry, offset: 0, index: 6)
        encoder.setBuffer(chi, offset: 0, index: 7)
        encoder.dispatchThreads(
          MTLSize(width: bandCount, height: chunks, depth: (batch + 3) / 4),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      }
      try encodeReduction(
        commands, grid: &grid, partial: partial, weights: weights, powers: powers, scores: scores,
        first: first)
    }
    try finish(commands)
    let values = scores.contents().assumingMemoryBound(to: Float.self)
    return rows.indices.map { Double(values[$0]) }
  }
}
