import Foundation
import Metal
import MetalPerformanceShadersGraph

/// The regularized direct reconstructions (`phase`, `complex`) of Python's CUDA `ssb.cuda.direct.PreparedDirect` on
/// Metal (`ssb_direct.metal`).
///
/// Per scan frequency Q of the source half plane (DC and, on even grids, the Nyquist row and column excluded) the
/// supported output aliases q = Q + (i, j) * (scan frequency period), |q| < (2 semiangle + max angular sampling) / wl,
/// are fitted jointly: the Gram T^H W T, the right side T^H W g and the signal power of the weak-object transfer T
/// over the BF pixels (detector weights W, g = G / dose) form the prepared system. Regularization acts on the
/// solution: (T^H W T + diag(damping * scale * prior(q))) x = T^H W g with prior (1 + |q|^2 / q0^2)^2 (1 for
/// q0 = 0) and scale = sum_k w_k A(k)^4. The solution's aliases fill the output spectrum (times factor^2, and their
/// Hermitian mirrors), whose inverse FFT gives eta (log-amplitude) and phi; the result is exp(eta + i phi).
///
/// The layout (frequencies, aliases, priors, detector weights, dose) depends on the data and sampling only and is
/// kept while those agree; the prepared normal equations depend on the optics too and are kept per method while
/// the optics agree, so a damping change only solves again. A complex system holds the phase system as its
/// lower-right block and serves both methods. Normal equations are float32 per BF pixel, accumulated in float-float
/// across BF batches on Metal; the regularized systems are solved in float64 (Cholesky) on the CPU, as CUDA solves
/// them in float64; the inverse FFT is float32.
/// Stage times of one direct reconstruction (benchmarks).
@_spi(SSBBenchmark) public struct SSBDirectTiming: Sendable {
  /// Frequencies, aliases, priors, weights and dose (zero when kept from the previous call).
  public var layoutSeconds = 0.0
  /// Normal equations at these optics, wall and GPU (zero when kept from the previous call).
  public var prepareSeconds = 0.0
  public var prepareGPUSeconds = 0.0
  /// Float64 regularized solves and the spectrum scatter on the CPU.
  public var solveSeconds = 0.0
  /// Inverse FFT and the wave's phase and amplitude.
  public var waveSeconds = 0.0
  public var preparedFromCache = false
}

final class SSBDirectSolver {
  struct LayoutKey: Hashable {
    var factor: Int
    var smoothness: Double
    var weighting: SSBDetectorWeighting

    init(factor: Int, regularization: SSBDirectRegularization) {
      self.factor = factor
      smoothness = regularization.smoothnessInverseAngstroms
      weighting = regularization.detectorWeighting
    }
  }
  struct PreparedKey: Equatable {
    var layout: LayoutKey
    var point: SSBModelPoint
    var rotation: Double
  }

  /// The frequencies with one count of supported aliases.
  final class Group {
    let count: Int
    let size: Int
    /// (source row, source column) per frequency.
    let frequency: MTLBuffer
    /// (q row, q column) per frequency and alias, 1/A.
    let alias: MTLBuffer
    /// Output index R * N + C per frequency and alias.
    let positions: [Int32]
    /// The prior (1 + |q|^2 / q0^2)^2 per frequency and alias.
    let prior: [Double]
    init(count: Int, size: Int, frequency: MTLBuffer, alias: MTLBuffer, positions: [Int32], prior: [Double]) {
      self.count = count
      self.size = size
      self.frequency = frequency
      self.alias = alias
      self.positions = positions
      self.prior = prior
    }
  }

  final class Layout {
    let key: LayoutKey
    let outputN: Int
    let groups: [Group]
    /// Detector weight of every active BF pixel.
    let weights: MTLBuffer
    let scale: Double
    let dose: Double
    init(key: LayoutKey, outputN: Int, groups: [Group], weights: MTLBuffer, scale: Double, dose: Double) {
      self.key = key
      self.outputN = outputN
      self.groups = groups
      self.weights = weights
      self.scale = scale
      self.dose = dose
    }
  }

  final class Prepared {
    let key: PreparedKey
    let complex: Bool
    /// Per group: the augmented upper triangle per frequency, float-float (hi re, hi im, lo re, lo im).
    let sums: [MTLBuffer]
    init(key: PreparedKey, complex: Bool, sums: [MTLBuffer]) {
      self.key = key
      self.complex = complex
      self.sums = sums
    }
  }


  struct Grid {
    var frequencies, count, bfOffset, planeStride, gCols: UInt32
    var wl, semiangle, angularRow, angularColumn, factor, depth, tiltRow, tiltColumn, outer2: Float
  }

  /// Augmented entries per dispatch: the float32 running sums stay in registers below this.
  static let maximumShare = 192
  static let maximumAliases = 16
  static let batch = 512

  let device: MTLDevice
  let queue: MTLCommandQueue
  let library: MTLLibrary
  let geometry: MetalSSBGeometry
  private let activeIndices: [Int]
  let n: Int
  private let pixelPipeline, dcPipeline, wavePipeline: MTLComputePipelineState
  private var pipelines: [Kernel: MTLComputePipelineState] = [:]
  var layout: Layout?
  /// The prepared systems of the current optics, phase (false) and complex (true).
  private var prepared: [Bool: Prepared] = [:]
  /// Drag windows (tiles) per control, most recently used first (`SSBDirectDragWindow.swift`).
  var windows: [SSBDragControl: [DragWindow]] = [:]
  var useCounter = 0
  private var windowPipelines: [String: MTLComputePipelineState] = [:]
  private var fft: [Int: (MPSGraph, MPSGraphTensor, MPSGraphTensor)] = [:]
  var lastTiming = SSBDirectTiming()
  let sourceSHA256: String

  init(
    device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, geometry: MetalSSBGeometry,
    activeIndices: [Int]
  ) throws {
    self.device = device
    self.queue = queue
    self.library = library
    self.geometry = geometry
    self.activeIndices = activeIndices
    n = geometry.qxByRow.count
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else { throw MetalSSBError.missingFunction(name) }
      return try device.makeComputePipelineState(function: function)
    }
    pixelPipeline = try pipeline("direct_pixel_terms")
    dcPipeline = try pipeline("direct_dc")
    wavePipeline = try pipeline("direct_wave")
    sourceSHA256 = try SSBKernelRegistry.sourceSHA256(SSBKernelRegistry.direct)
  }

  /// Augmented entries of a system with `count` aliases, and its split into dispatches.
  static func parts(count: Int, complex: Bool) -> (outputs: Int, ranges: [Range<Int>]) {
    let width = (complex ? 2 * count : count) + 1
    let outputs = width * (width + 1) / 2
    let parts = (outputs + maximumShare - 1) / maximumShare
    let share = (outputs + parts - 1) / parts
    return (outputs, stride(from: 0, to: outputs, by: share).map { $0..<min($0 + share, outputs) })
  }

  /// One specialized normal-equations dispatch: `direct_normal_tiled` (one thread per frequency and alias) for complex
  /// systems of two or more aliases, `direct_normal_equations` (one thread per frequency, entries split into parts)
  /// otherwise; each is the faster one there on the M5 Max.
  struct Kernel: Hashable {
    var count: Int
    var complex: Bool
    /// Entries of this part (`direct_normal_equations`); every entry for the tiled kernel.
    var range: Range<Int>
    var tiled: Bool
    var name: String { tiled ? "direct_normal_tiled" : "direct_normal_equations" }
  }

  static func kernels(count: Int, complex: Bool) -> [Kernel] {
    if complex && count >= 2 {
      return [Kernel(count: count, complex: true, range: 0..<parts(count: count, complex: true).outputs, tiled: true)]
    }
    return parts(count: count, complex: complex).ranges.map {
      Kernel(count: count, complex: complex, range: $0, tiled: false)
    }
  }

  /// Blocks of 32 frequencies per threadgroup of the tiled kernel: at least 128 threads.
  static func blocks(count: Int) -> Int { max(1, 4 / count) }
  /// BF pixels per threadgroup tile of the tiled kernel: the tile's columns take about 16 KB.
  static func tile(count: Int, complex: Bool) -> Int {
    max(1, min(32, 16384 / (256 * count * (complex ? 2 : 1) * blocks(count: count))))
  }

  private func makePipeline(_ kernel: Kernel) throws -> MTLComputePipelineState {
    let constants = MTLFunctionConstantValues()
    var c = UInt32(kernel.count), x = kernel.complex
    var first = UInt32(kernel.range.lowerBound), last = UInt32(kernel.range.upperBound)
    var tile = UInt32(Self.tile(count: kernel.count, complex: kernel.complex))
    var blocks = UInt32(Self.blocks(count: kernel.count))
    constants.setConstantValue(&c, type: .uint, index: 60)
    constants.setConstantValue(&x, type: .bool, index: 61)
    constants.setConstantValue(&first, type: .uint, index: 62)
    constants.setConstantValue(&last, type: .uint, index: 63)
    constants.setConstantValue(&tile, type: .uint, index: 64)
    constants.setConstantValue(&blocks, type: .uint, index: 65)
    let function = try library.makeFunction(name: kernel.name, constantValues: constants)
    return try device.makeComputePipelineState(function: function)
  }

  /// A drag-window kernel (`direct_window_accumulate`, `_finalize`, `_evaluate`).
  func windowPipeline(_ name: String) throws -> MTLComputePipelineState {
    if let pipeline = windowPipelines[name] { return pipeline }
    guard let function = library.makeFunction(name: name) else { throw MetalSSBError.missingFunction(name) }
    let pipeline = try device.makeComputePipelineState(function: function)
    windowPipelines[name] = pipeline
    return pipeline
  }

  /// Specialize every pipeline a layout needs, in parallel (the cold cost of a new alias count).
  private func warmPipelines(_ layout: Layout, complex: Bool) throws {
    let needed = layout.groups.flatMap { Self.kernels(count: $0.count, complex: complex) }
      .filter { pipelines[$0] == nil }
    if needed.isEmpty { return }
    let lock = NSLock()
    nonisolated(unsafe) var failure: Error?
    nonisolated(unsafe) var made: [Kernel: MTLComputePipelineState] = [:]
    DispatchQueue.concurrentPerform(iterations: needed.count) { index in
      do {
        let pipeline = try makePipeline(needed[index])
        lock.lock()
        made[needed[index]] = pipeline
        lock.unlock()
      } catch {
        lock.lock()
        failure = error
        lock.unlock()
      }
    }
    if let failure { throw failure }
    pipelines.merge(made) { first, _ in first }
  }

  /// Encode one alias group's normal equations for one batch of BF pixels (buffers and grid already bound).
  private func encodeNormalEquations(_ compute: MTLComputeCommandEncoder, group: Group, complex: Bool) throws {
    for kernel in Self.kernels(count: group.count, complex: complex) {
      guard let pipeline = pipelines[kernel] else { throw MetalSSBError.missingFunction(kernel.name) }
      compute.setComputePipelineState(pipeline)
      if kernel.tiled {
        let blocks = Self.blocks(count: group.count), tile = Self.tile(count: group.count, complex: complex)
        let columns = blocks * tile * group.count * 32 * (complex ? 2 : 1) * 8
        compute.setThreadgroupMemoryLength((columns + 15) / 16 * 16, index: 0)
        compute.setThreadgroupMemoryLength((blocks * tile * 32 * 8 + 15) / 16 * 16, index: 1)
        compute.dispatchThreadgroups(
          MTLSize(width: (group.size + 32 * blocks - 1) / (32 * blocks), height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32 * group.count * blocks, height: 1, depth: 1))
      } else {
        compute.dispatchThreads(
          MTLSize(width: group.size, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      }
    }
  }

  // MARK: - Layout

  private func makeLayout(
    _ key: LayoutKey, visit: MetalSSBJointModel.VisitSpectra, isCancelled: () -> Bool
  ) throws -> Layout {
    guard n >= 3 else { throw SSBDirectError.tooFewScanPositions }
    let factor = key.factor
    let outputN = factor * n
    let wl = geometry.modelWavelengthAngstroms ?? Double(geometry.wavelengthAngstroms)
    let semiangle = geometry.modelSemiangleRadians ?? Double(geometry.semiangleRadians)
    let angular = Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians))
    let cutoff = (2 * semiangle + angular) / wl
    let cutoff2 = Float(cutoff * cutoff)
    let dqRow = geometry.qxByRow[1], dqColumn = geometry.qyByColumn[1]
    let half = n / 2 + 1
    let smoothness2 = Float(key.smoothness * key.smoothness)
    // Signed output frequency of every output row and column, as the CUDA layout: int -> float32, times the step.
    func signed(_ index: Int) -> Int { index <= (outputN - 1) / 2 ? index : index - outputN }
    let qRow = (0..<outputN).map { Float(signed($0)) * dqRow }
    let qColumn = (0..<outputN).map { Float(signed($0)) * dqColumn }
    // Kept source frequencies (row-major) and the supported aliases of each, as a bit pattern over factor^2.
    var kept: [(row: Int, column: Int, pattern: Int)] = []
    kept.reserveCapacity(n * half)
    for r in 0..<n {
      for c in 0..<half {
        var keep = c > 0 || (r > 0 && r <= (n - 1) / 2)
        if n % 2 == 0 { keep = keep && r != n / 2 && c != n / 2 }
        guard keep else { continue }
        var pattern = 0
        for ar in 0..<factor {
          let qr = qRow[r + ar * n]
          for ac in 0..<factor {
            let qc = qColumn[c + ac * n]
            if qr * qr + qc * qc < cutoff2 { pattern |= 1 << (ar * factor + ac) }
          }
        }
        // Frequencies without a supported alias add only to the signal power, which no output uses.
        if pattern != 0 { kept.append((r, c, pattern)) }
      }
    }
    var byCount: [Int: [Int]] = [:]
    for (index, entry) in kept.enumerated() { byCount[entry.pattern.nonzeroBitCount, default: []].append(index) }
    guard let largest = byCount.keys.max(), largest <= Self.maximumAliases else {
      throw MetalSSBError.invalidGeometry(
        "Direct inversion supports up to \(Self.maximumAliases) aliases per scan frequency; choose a lower sampling.")
    }
    var groups: [Group] = []
    for count in byCount.keys.sorted() {
      // Frequencies with the same supported aliases, in scan order: a SIMD group's neighbouring threads then hold
      // neighbouring q of every alias and take the same aperture branches.
      let members = byCount[count]!.sorted { (kept[$0].pattern, $0) < (kept[$1].pattern, $1) }
      var frequency: [SIMD2<UInt32>] = []
      var q: [SIMD2<Float>] = []
      var positions: [Int32] = []
      var prior: [Double] = []
      frequency.reserveCapacity(members.count)
      q.reserveCapacity(members.count * count)
      positions.reserveCapacity(members.count * count)
      prior.reserveCapacity(members.count * count)
      for index in members {
        let (r, c, pattern) = kept[index]
        frequency.append(SIMD2(UInt32(r), UInt32(c)))
        for alias in 0..<(factor * factor) where pattern & (1 << alias) != 0 {
          let outputRow = r + (alias / factor) * n, outputColumn = c + (alias % factor) * n
          let value = SIMD2(qRow[outputRow], qColumn[outputColumn])
          q.append(value)
          positions.append(Int32(outputRow * outputN + outputColumn))
          if key.smoothness == 0 {
            prior.append(1)
          } else {
            // float32, as the CUDA layout's (1 + q^2 / s^2)^2
            let t = 1 + (value.x * value.x + value.y * value.y) / smoothness2
            prior.append(Double(t * t))
          }
        }
      }
      groups.append(
        Group(
          count: count, size: members.count, frequency: try Self.buffer(device, frequency),
          alias: try Self.buffer(device, q), positions: positions, prior: prior))
    }
    // Dose: sum_k G_k(0) / (scan positions * sum_k A(k)^2) over every logical BF pixel; the common scan mean.
    let logical = geometry.logicalBrightfieldCount
    let total = Double(geometry.dcValue.x) * Double(logical)
    let aperture2 = geometry.brightfieldAperture.map { $0 * $0 }
    let dose = total / (Double(n * n) * aperture2.reduce(0.0) { $0 + Double($1) })
    guard dose.isFinite, dose > 0 else { throw SSBDirectError.noIncidentIntensity }
    var logicalWeights = [Float](repeating: 1, count: logical)
    if key.weighting == .inverseMean {
      logicalWeights = try inverseMeanWeights(visit: visit, isCancelled: isCancelled)
    }
    let scale = zip(logicalWeights, aperture2).reduce(0.0) { $0 + Double($1.0 * ($1.1 * $1.1)) }
    let weights = try Self.buffer(device, activeIndices.map { logicalWeights[$0] })
    return Layout(key: key, outputN: outputN, groups: groups, weights: weights, scale: scale, dose: dose)
  }

  /// 1 / max(mean count, 5 % of the median mean), normalized to mean 1 over the BF pixels (Python `inverse_mean`).
  private func inverseMeanWeights(visit: MetalSSBJointModel.VisitSpectra, isCancelled: () -> Bool) throws -> [Float] {
    let logical = geometry.logicalBrightfieldCount
    guard activeIndices.count == logical else {
      throw MetalSSBError.invalidGeometry(
        "Inverse-mean weighting needs every BF pixel inside the calibrated aperture; reduce the BF radius or use uniform weighting."
      )
    }
    let dc = try Self.allocate(device, logical * 4, shared: true)
    var pending: MTLCommandBuffer? = try command()
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit(Self.batch) { [self] source, offset, half, bfOffset, count in
        if cancelled() { throw CancellationError() }
        let commands = try pending ?? command()
        var p = SIMD3<UInt32>(UInt32(count), UInt32(half ? n * (n / 2 + 1) : n * n), UInt32(bfOffset))
        let encoder = try self.encoder(commands, dcPipeline)
        encoder.setBuffer(source, offset: offset, index: 0)
        encoder.setBuffer(dc, offset: 0, index: 1)
        encoder.setBytes(&p, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
        dispatch(encoder, width: count)
        if half {
          pending = commands
        } else {
          try finish(commands)
          pending = nil
        }
      }
    }
    if let pending { try finish(pending) }
    let values = UnsafeBufferPointer(start: dc.contents().assumingMemoryBound(to: Float.self), count: logical)
    let positions = Float(n * n)
    let mean = values.map { max($0 / positions, 0) }
    let sorted = mean.sorted()
    let middle = sorted.count / 2
    let median = sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
    let floor = median * 0.05
    guard floor > 0 else { throw SSBDirectError.noPositiveCounts }
    var weights = mean.map { 1 / max($0, floor) }
    let average = Float(weights.reduce(0.0) { $0 + Double($1) } / Double(weights.count))
    weights = weights.map { $0 / average }
    return weights
  }

  // MARK: - Prepared normal equations

  func prepare(
    _ key: PreparedKey, complex: Bool, layout: Layout, k: MTLBuffer, visit: MetalSSBJointModel.VisitSpectra,
    isCancelled: () -> Bool
  ) throws -> (Prepared, Double) {
    try warmPipelines(layout, complex: complex)
    let point = key.point
    let coefficients = try Self.buffer(device, point.reconstructionCoefficients)
    let wl = geometry.wavelengthAngstroms
    let radius =
      (Double(geometry.semiangleRadians)
        + 0.5 * Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians)))
      / Double(wl)
    var grid = Grid(
      frequencies: 0, count: UInt32(activeIndices.count), bfOffset: 0, planeStride: 0, gCols: 0,
      wl: wl, semiangle: geometry.semiangleRadians, angularRow: geometry.angularSamplingYRadians,
      angularColumn: geometry.angularSamplingXRadians,
      factor: geometry.phaseFactor ?? Float(Double.pi / Double(wl)),
      depth: Float(point.depthSpreadAngstroms), tiltRow: Float(point.sampleTiltMrad[0] * 1e-3),
      tiltColumn: Float(point.sampleTiltMrad[1] * 1e-3), outer2: Float(radius * radius * (1 + 1e-4)))
    let pixel = try Self.allocate(device, activeIndices.count * 32)
    var sums: [MTLBuffer] = []
    let first = try command()
    let pixelEncoder = try encoder(first, pixelPipeline)
    pixelEncoder.setBuffer(k, offset: 0, index: 0)
    pixelEncoder.setBuffer(layout.weights, offset: 0, index: 1)
    pixelEncoder.setBuffer(coefficients, offset: 0, index: 2)
    pixelEncoder.setBuffer(pixel, offset: 0, index: 3)
    pixelEncoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 4)
    dispatch(pixelEncoder, width: activeIndices.count)
    guard let blit = first.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
    for group in layout.groups {
      let length = group.size * Self.parts(count: group.count, complex: complex).outputs * 16
      let buffer = try Self.allocate(device, length, shared: true)
      blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
      sums.append(buffer)
    }
    blit.endEncoding()
    var gpuSeconds = 0.0
    var pending: MTLCommandBuffer? = first
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit(Self.batch) { [self] source, offset, half, bfOffset, count in
        if cancelled() { throw CancellationError() }
        let commands = try pending ?? command()
        var p = grid
        p.count = UInt32(count)
        p.bfOffset = UInt32(bfOffset)
        p.planeStride = UInt32(half ? n * (n / 2 + 1) : n * n)
        p.gCols = UInt32(half ? n / 2 + 1 : n)
        guard let compute = commands.makeComputeCommandEncoder() else { throw MetalSSBError.commandQueue }
        compute.setBuffer(source, offset: offset, index: 0)
        compute.setBuffer(pixel, offset: 0, index: 3)
        compute.setBuffer(coefficients, offset: 0, index: 4)
        for (index, group) in layout.groups.enumerated() {
          p.frequencies = UInt32(group.size)
          compute.setBuffer(group.frequency, offset: 0, index: 1)
          compute.setBuffer(group.alias, offset: 0, index: 2)
          compute.setBuffer(sums[index], offset: 0, index: 5)
          compute.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 6)
          try encodeNormalEquations(compute, group: group, complex: complex)
        }
        compute.endEncoding()
        if half {
          pending = commands
        } else {
          // A streamed batch reuses the engine's spectrum workspace: finish before the next batch.
          try finish(commands)
          gpuSeconds += commands.gpuEndTime - commands.gpuStartTime
          pending = nil
        }
      }
    }
    if let pending {
      try finish(pending)
      gpuSeconds += pending.gpuEndTime - pending.gpuStartTime
    }
    return (Prepared(key: key, complex: complex, sums: sums), gpuSeconds)
  }

  // MARK: - Reconstruction

  func reconstruct(
    _ point: SSBModelPoint, factor: Int, complex: Bool, regularization: SSBDirectRegularization,
    rotation: Double, k: MTLBuffer, visit: MetalSSBJointModel.VisitSpectra, isCancelled: () -> Bool
  ) throws -> (phase: MTLBuffer, amplitude: MTLBuffer?, n: Int) {
    var timing = SSBDirectTiming()
    var started = Date()
    let layoutKey = LayoutKey(factor: factor, regularization: regularization)
    let layout = try currentLayout(layoutKey, visit: visit, isCancelled: isCancelled)
    timing.layoutSeconds = Date().timeIntervalSince(started)
    started = Date()
    let key = PreparedKey(layout: layoutKey, point: point, rotation: rotation)
    if prepared.values.contains(where: { $0.key != key }) { prepared = [:] }
    // A complex system serves the phase method as its lower-right block.
    let system: Prepared
    if let existing = prepared[complex] ?? (complex ? nil : prepared[true]) {
      system = existing
      timing.preparedFromCache = true
    } else {
      let (made, gpuSeconds) = try prepare(
        key, complex: complex, layout: layout, k: k, visit: visit, isCancelled: isCancelled)
      prepared[complex] = made
      system = made
      timing.prepareGPUSeconds = gpuSeconds
    }
    timing.prepareSeconds = Date().timeIntervalSince(started)
    lastTiming = timing
    if isCancelled() { throw CancellationError() }
    let result = try solveAndWave(
      system, layout: layout, complex: complex, regularization: regularization, timing: &timing)
    lastTiming = timing
    return result
  }

  /// The layout of these settings, made again (and every prepared system and drag window dropped) when they change.
  func currentLayout(_ key: LayoutKey, visit: MetalSSBJointModel.VisitSpectra, isCancelled: () -> Bool) throws
    -> Layout
  {
    if let layout, layout.key == key { return layout }
    layout = nil
    prepared = [:]
    windows = [:]
    let made = try makeLayout(key, visit: visit, isCancelled: isCancelled)
    layout = made
    return made
  }

  /// The regularized solves of a prepared (or window-evaluated) system, the scatter and the wave.
  func solveAndWave(
    _ system: Prepared, layout: Layout, complex: Bool, regularization: SSBDirectRegularization,
    timing: inout SSBDirectTiming
  ) throws -> (phase: MTLBuffer, amplitude: MTLBuffer?, n: Int) {
    var started = Date()
    let outputN = layout.outputN
    let spectrum = try Self.allocate(device, outputN * outputN * 8, shared: true)
    memset(spectrum.contents(), 0, spectrum.length)
    try solve(
      system, layout: layout, complex: complex, regularization: regularization,
      spectrum: spectrum.contents().assumingMemoryBound(to: SIMD2<Float>.self))
    timing.solveSeconds = Date().timeIntervalSince(started)
    started = Date()
    let result = try wave(spectrum: spectrum, n: outputN, amplitude: complex)
    timing.waveSeconds = Date().timeIntervalSince(started)
    return (result.phase, result.amplitude, outputN)
  }

  /// Solve every frequency's regularized system in float64 and place its aliases (times factor^2, and the
  /// Hermitian mirrors) in the packed spectrum eta + i phi.
  func solve(
    _ system: Prepared, layout: Layout, complex: Bool, regularization: SSBDirectRegularization,
    spectrum: UnsafeMutablePointer<SIMD2<Float>>
  ) throws {
    let outputN = layout.outputN
    let gain = Float(layout.key.factor * layout.key.factor)
    let phaseStrength = regularization.phaseDamping * layout.scale
    let amplitudeStrength = regularization.amplitudeDamping * layout.scale
    let inverseDose = 1 / layout.dose
    let invalid = UnsafeMutablePointer<Int32>.allocate(capacity: 1)
    invalid.initialize(to: 0)
    defer { invalid.deallocate() }
    for (index, group) in layout.groups.enumerated() {
      let count = group.count
      let stored = Self.parts(count: count, complex: system.complex).outputs
      let width = (system.complex ? 2 * count : count) + 1
      // The system's rows of the augmented Gram: all for complex, the phase block of a complex system for phase.
      let dim = complex ? 2 * count : count
      let offset = system.complex && !complex ? count : 0
      let sums = system.sums[index].contents().assumingMemoryBound(to: SIMD4<Float>.self)
      let chunk = 256
      let chunks = (group.size + chunk - 1) / chunk
      DispatchQueue.concurrentPerform(iterations: chunks) { part in
        withUnsafeTemporaryAllocation(of: Double.self, capacity: 4 * dim * dim + 4 * dim) { scratch in
          let lr = scratch.baseAddress!, li = lr + dim * dim
          let yr = li + dim * dim, yi = yr + dim
          for f in (part * chunk)..<min((part + 1) * chunk, group.size) {
            let base = f * stored
            // Lower triangle of A = Gram + diag(penalty): A[i][j] = conj(M[j][i]) for i > j.
            for j in 0..<dim {
              for i in j..<dim {
                let (re, im) = Self.entry(sums, base, width, offset + j, offset + i)
                lr[i * dim + j] = re
                li[i * dim + j] = i == j ? 0 : -im
              }
              let alias = j % count
              let strength = complex && j < count ? amplitudeStrength : phaseStrength
              lr[j * dim + j] += strength * group.prior[f * count + alias]
              let (br, bi) = Self.entry(sums, base, width, offset + j, width - 1)
              yr[j] = br * inverseDose
              yi[j] = bi * inverseDose
            }
            // Cholesky A = L L^H in place.
            var ok = true
            for j in 0..<dim {
              var d = lr[j * dim + j]
              for k in 0..<j { d -= lr[j * dim + k] * lr[j * dim + k] + li[j * dim + k] * li[j * dim + k] }
              guard d > 0, d.isFinite else {
                ok = false
                break
              }
              d = d.squareRoot()
              lr[j * dim + j] = d
              for i in (j + 1)..<dim {
                var sr = lr[i * dim + j], si = li[i * dim + j]
                for k in 0..<j {
                  // L[i][k] conj(L[j][k])
                  let ar = lr[i * dim + k], ai = li[i * dim + k], br = lr[j * dim + k], bi = -li[j * dim + k]
                  sr -= ar * br - ai * bi
                  si -= ar * bi + ai * br
                }
                lr[i * dim + j] = sr / d
                li[i * dim + j] = si / d
              }
            }
            if !ok {
              invalid.pointee = 1
              continue
            }
            for i in 0..<dim {
              var sr = yr[i], si = yi[i]
              for k in 0..<i {
                let ar = lr[i * dim + k], ai = li[i * dim + k]
                sr -= ar * yr[k] - ai * yi[k]
                si -= ar * yi[k] + ai * yr[k]
              }
              yr[i] = sr / lr[i * dim + i]
              yi[i] = si / lr[i * dim + i]
            }
            for i in stride(from: dim - 1, through: 0, by: -1) {
              var sr = yr[i], si = yi[i]
              for k in (i + 1)..<dim {
                // conj(L[k][i]) x[k]
                let ar = lr[k * dim + i], ai = -li[k * dim + i]
                sr -= ar * yr[k] - ai * yi[k]
                si -= ar * yi[k] + ai * yr[k]
              }
              yr[i] = sr / lr[i * dim + i]
              yi[i] = si / lr[i * dim + i]
            }
            // Coefficients round to complex64, then scale by factor^2, as the CUDA scatter.
            for a in 0..<count {
              let p = Int(group.positions[f * count + a])
              let row = p / outputN, column = p % outputN
              let mirror = ((outputN - row) % outputN) * outputN + (outputN - column) % outputN
              let phi = SIMD2<Float>(Float(yr[dim - count + a]), Float(yi[dim - count + a])) * gain
              let eta = complex ? SIMD2<Float>(Float(yr[a]), Float(yi[a])) * gain : .zero
              guard phi.x.isFinite, phi.y.isFinite, eta.x.isFinite, eta.y.isFinite else {
                invalid.pointee = 1
                continue
              }
              // eta + i phi at q, conj(eta) + i conj(phi) at -q.
              spectrum[p] = SIMD2(eta.x - phi.y, eta.y + phi.x)
              spectrum[mirror] = SIMD2(eta.x + phi.y, -eta.y + phi.x)
            }
          }
        }
      }
    }
    if invalid.pointee != 0 { throw SSBDirectError.diverged }
  }

  /// Row a, column b (a <= b) of a stored augmented upper triangle (row-major), float-float to Double.
  @inline(__always)
  private static func entry(
    _ sums: UnsafeMutablePointer<SIMD4<Float>>, _ base: Int, _ width: Int, _ a: Int, _ b: Int
  ) -> (Double, Double) {
    let s = sums[base + a * width - a * (a - 1) / 2 + (b - a)]
    return (Double(s.x) + Double(s.z), Double(s.y) + Double(s.w))
  }

  /// Inverse FFT of the packed spectrum, then the phase and (complex) amplitude of exp(eta + i phi).
  func wave(spectrum: MTLBuffer, n: Int, amplitude: Bool) throws -> (phase: MTLBuffer, amplitude: MTLBuffer?) {
    let field = try Self.allocate(device, n * n * 8)
    let shape: [NSNumber] = [NSNumber(value: n), NSNumber(value: n)]
    if fft[n] == nil {
      let graph = MPSGraph()
      graph.options = .none
      let input = graph.placeholder(shape: shape, dataType: .complexFloat32, name: "spectrum")
      let descriptor = MPSGraphFFTDescriptor()
      descriptor.inverse = true
      descriptor.scalingMode = .size
      let output = graph.fastFourierTransform(input, axes: [0, 1], descriptor: descriptor, name: "ifft2")
      fft[n] = (graph, input, output)
    }
    let (graph, input, output) = fft[n]!
    let commands = try command()
    let scheduled = MPSCommandBuffer(commandBuffer: commands)
    let execution = MPSGraphExecutionDescriptor()
    execution.waitUntilCompleted = false
    graph.encode(
      to: scheduled,
      feeds: [input: MPSGraphTensorData(spectrum, shape: shape, dataType: .complexFloat32)],
      targetOperations: nil,
      resultsDictionary: [output: MPSGraphTensorData(field, shape: shape, dataType: .complexFloat32)],
      executionDescriptor: execution)
    let phase = try Self.allocate(device, n * n * 4, shared: true)
    let magnitude = amplitude ? try Self.allocate(device, n * n * 4, shared: true) : nil
    let invalid = try Self.allocate(device, 4, shared: true)
    invalid.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    var p = SIMD2<UInt32>(UInt32(n * n), amplitude ? 1 : 0)
    let encoder = try self.encoder(scheduled, wavePipeline)
    encoder.setBuffer(field, offset: 0, index: 0)
    encoder.setBuffer(phase, offset: 0, index: 1)
    encoder.setBuffer(magnitude ?? phase, offset: 0, index: 2)
    encoder.setBuffer(invalid, offset: 0, index: 3)
    encoder.setBytes(&p, length: 8, index: 4)
    dispatch(encoder, width: n * n)
    try finish(scheduled)
    if invalid.contents().load(as: UInt32.self) != 0 { throw SSBDirectError.diverged }
    return (phase, magnitude)
  }

  // MARK: - Metal helpers

  func command() throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    return command
  }
  func encoder(_ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState) throws
    -> MTLComputeCommandEncoder
  {
    guard let encoder = command.makeComputeCommandEncoder() else { throw MetalSSBError.commandQueue }
    encoder.setComputePipelineState(pipeline)
    return encoder
  }
  func dispatch(_ encoder: MTLComputeCommandEncoder, width: Int) {
    encoder.dispatchThreads(
      MTLSize(width: width, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
  }
  func finish(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw MetalSSBError.commandExecution(error.localizedDescription) }
  }
  static func allocate(_ device: MTLDevice, _ length: Int, shared: Bool = false) throws -> MTLBuffer {
    guard
      let buffer = device.makeBuffer(
        length: max(4, length), options: shared ? .storageModeShared : .storageModePrivate)
    else {
      throw MetalSSBError.allocation(
        "direct reconstruction workspace (\(length) bytes). Close inactive datasets and retry")
    }
    return buffer
  }
  static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
    if values.isEmpty { return try allocate(device, 4, shared: true) }
    guard
      let result = values.withUnsafeBytes({ bytes in
        device.makeBuffer(bytes: bytes.baseAddress!, length: max(4, bytes.count), options: .storageModeShared)
      })
    else { throw MetalSSBError.allocation("direct reconstruction layout") }
    return result
  }
}
