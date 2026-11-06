import Foundation
import Metal

/// Host side of the separable coherent reconstruction (`ssb_separable.metal`), Python MPS
/// `_separable_geometry` and `_separable_spectrum`: support, phasor parts reduced modulo 2 pi in double precision,
/// the depth parts, the alias tuples, and the launches over the cached G(q, k) chunks.
final class SSBSeparableReconstruction {
  struct Grid {
    var rows, cols, colsAll, extra, nyquistRI, sourceN, gCols, count: UInt32
    var chunk, groups, groupOffset, tuples, bfOffset, n, pad0, pad1: UInt32
    var depthScale, pad2, pad3, pad4: Float
  }
  /// BF pixels per culling group (`_SEPARABLE_CHUNK`); it divides every cache-chunk offset.
  static let chunk = 16
  struct NotCached: Error {}

  private struct Geometry {
    let n: Int
    let rows, cols, upper: [Int]
    let nyquistRI: Int
    let rowF, colF: [Double]  // all columns: support then the Nyquist row's upper half
    let rowIndices, colIndices, rowFBuffer, colFBuffer, rowPosition, colPosition: MTLBuffer
    let tuples: [Int: MTLBuffer]
    let tupleCounts: [Int: Int]
  }

  private let device: MTLDevice
  private let geometry: MetalDirectPtychoGeometry
  private let k: [SIMD2<Float>]
  private let tables, depthTables, single, paired, scatter: MTLComputePipelineState
  private let depth: [Int: MTLComputePipelineState]
  private var kData: MTLBuffer?
  private var bounds: MTLBuffer?
  private var cache: [Int: Geometry] = [:]
  /// Reused workspaces by purpose; every call waits for its GPU work before returning.
  private var workspaces: [String: MTLBuffer] = [:]
  private func workspace(_ name: String, _ length: Int) throws -> MTLBuffer {
    if let buffer = workspaces[name], buffer.length >= length { return buffer }
    let buffer = try Self.allocate(device, length)
    workspaces[name] = buffer
    return buffer
  }

  init(
    device: MTLDevice, library: MTLLibrary, geometry: MetalDirectPtychoGeometry, k: [SIMD2<Float>]
  ) throws {
    self.device = device
    self.geometry = geometry
    self.k = k
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalDirectPtychoError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    tables = try pipeline("sep_tables")
    depthTables = try pipeline("sep_depth_tables")
    single = try pipeline("sep_coherent")
    paired = try pipeline("sep_coherent_paired")
    scatter = try pipeline("sep_scatter")
    depth = [
      1: try pipeline("sep_coherent_depth1"), 2: try pipeline("sep_coherent_depth2"),
      3: try pipeline("sep_coherent_depth3"),
    ]
  }

  // MARK: Double-precision parts (NumPy semantics)

  static func remainder(_ x: Double, _ y: Double) -> Double {
    var value = fmod(x, y)
    if value != 0 && (y < 0) != (value < 0) { value += y }
    return value
  }

  /// `_split_mod`: a value reduced modulo 2 pi as three float32 parts, the first two with 12 significant bits.
  static func splitMod(_ value: Double) -> [Float] {
    var rest = remainder(value + .pi, 2 * .pi) - .pi
    var parts: [Double] = []
    for _ in 0..<2 {
      let exponent = rest == 0 ? 0 : frexp(rest).1
      let scale = ldexp(1.0, 12 - exponent)
      let part = (rest * scale).rounded(.toNearestOrEven) / scale
      parts.append(part)
      rest -= part
    }
    parts.append(rest)
    return parts.map { Float($0) }
  }

  static func phasor(_ angle: Double) -> SIMD2<Float> {
    let reduced = remainder(angle, 2 * .pi)
    return SIMD2(Float(cos(reduced)), Float(sin(reduced)))
  }

  static func frequency(_ index: Int, _ n: Int) -> Double {
    let integer = index < (n - 1) / 2 + 1 ? index : index - n
    return Double(integer) * (1.0 / Double(n)) * Double(n)
  }

  /// `_alias_row_pairs`: row-table indices in groups of `size` sharing a source row, 0xffffffff for no partner.
  static func aliasTuples(_ rows: [Int], sourceN: Int, size: Int) -> [UInt32] {
    var order: [Int] = []
    var members: [Int: [Int]] = [:]
    for (index, row) in rows.enumerated() {
      let key = row % sourceN
      if members[key] == nil { order.append(key) }
      members[key, default: []].append(index)
    }
    var tuples: [UInt32] = []
    for key in order {
      let list = members[key]!
      for first in stride(from: 0, to: list.count, by: size) {
        let group = list[first..<min(first + size, list.count)].map(UInt32.init)
        tuples += group + [UInt32](repeating: 0xFFFF_FFFF, count: size - group.count)
      }
    }
    return tuples
  }

  private func geometryFor(factor: Int) throws -> Geometry {
    if let cached = cache[factor] { return cached }
    let source = geometry.qxByRow.count
    let n = source * factor
    let columns = n / 2 + 1
    var radius =
      (Double(geometry.semiangleRadians)
        + 0.5 * Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians)))
      / Double(geometry.wavelengthAngstroms)
    radius *= 1 + 8 * Double(Float.ulpOfOne)
    func support(_ value: Double, _ step: Float, _ count: Int) -> [Int] {
      let limit = (value / Double(abs(step))).rounded(.up) + 1
      return (0..<count).filter { abs(Self.frequency($0, n)) <= limit }
    }
    let rows = support(Double(k.map { abs($0.x) }.max() ?? 0) + radius, geometry.qxByRow[1], n)
    let cols = support(
      Double(k.map { abs($0.y) }.max() ?? 0) + radius, geometry.qyByColumn[1], columns)
    let nyquist = n / 2
    var upper: [Int] = []
    if rows.contains(nyquist) {
      let limit = cols.map { abs(Self.frequency($0, n)) }.max() ?? 0
      upper = (columns..<n).filter { abs(Self.frequency($0, n)) <= limit }
      guard
        upper.isEmpty
          || (upper.last == n - 1 && zip(upper.dropFirst(), upper).allSatisfy { $0 - $1 == 1 }),
        upper.count <= cols.count
      else {
        throw MetalDirectPtychoError.invalidGeometry(
          "Unexpected Nyquist-row support for the separable SSB kernel.")
      }
    }
    let allCols = cols + upper
    let rowF = rows.map { Self.frequency($0, n) }
    let colF = allCols.map { Self.frequency($0, n) }
    var rowPosition = [Int32](repeating: -1, count: n)
    for (i, row) in rows.enumerated() { rowPosition[row] = Int32(i) }
    var colPosition = [Int32](repeating: -1, count: n)
    for (i, col) in cols.enumerated() { colPosition[col] = Int32(i) }
    var tuples: [Int: MTLBuffer] = [:]
    var counts: [Int: Int] = [:]
    for size in Set([2, max(1, min(3, factor))]) {
      let values = Self.aliasTuples(rows, sourceN: source, size: size)
      tuples[size] = try Self.buffer(device, values)
      counts[size] = values.count / size
    }
    let result = Geometry(
      n: n, rows: rows, cols: cols, upper: upper,
      nyquistRI: upper.isEmpty ? 0 : rows.firstIndex(of: nyquist)!, rowF: rowF, colF: colF,
      rowIndices: try Self.buffer(device, rows.map(UInt32.init)),
      colIndices: try Self.buffer(device, allCols.map(UInt32.init)),
      rowFBuffer: try Self.buffer(device, rowF.map { Float($0) }),
      colFBuffer: try Self.buffer(device, colF.map { Float($0) }),
      rowPosition: try Self.buffer(device, rowPosition),
      colPosition: try Self.buffer(device, colPosition),
      tuples: tuples, tupleCounts: counts)
    cache[factor] = result
    return result
  }

  /// k_data (k row, k column, aperture at k, 0) and the per-group bounds (row and column extent, smallest
  /// aperture), from the probe's float32 aperture.
  func prepare(probe: MTLBuffer) throws {
    try ensureKData(probe: probe)
  }

  private func ensureKData(probe: MTLBuffer) throws {
    if kData != nil { return }
    let values = probe.contents().assumingMemoryBound(to: Float.self)
    let data = k.indices.map { SIMD4<Float>(k[$0].x, k[$0].y, values[$0 * 5 + 3], 0) }
    let groups = (k.count + Self.chunk - 1) / Self.chunk
    var boundValues = [Float](repeating: 0, count: groups * 8)
    for group in 0..<groups {
      let members = (0..<Self.chunk).map { data[min(group * Self.chunk + $0, k.count - 1)] }
      boundValues[group * 8] = members.map(\.x).min()!
      boundValues[group * 8 + 1] = members.map(\.x).max()!
      boundValues[group * 8 + 2] = members.map(\.y).min()!
      boundValues[group * 8 + 3] = members.map(\.y).max()!
      boundValues[group * 8 + 4] = members.map(\.z).min()!
    }
    kData = try Self.buffer(device, data)
    bounds = try Self.buffer(device, boundValues)
  }

  /// Encode the corrected spectrum into `full` (n x n) on `commands` and the cached batches `visit` yields; the
  /// probe must hold this geometry's apertures. Throws `NotCached` when a batch is a streamed full spectrum.
  func encode(
    params: [Float], factor: Int, probe: MTLBuffer, full: MTLBuffer, commands: MTLCommandBuffer,
    visit: (Int, @escaping MetalDirectPtychoJointModel.SpectrumVisitor) throws -> Void,
    isCancelled: () -> Bool
  ) throws {
    try ensureKData(probe: probe)
    let geo = try geometryFor(factor: factor)
    let source = geometry.qxByRow.count
    let stored = k.count
    let wl = Double(params[0])
    let c10 = Double(params[4])
    let c12 = Double(params[5])
    let c2 = Double(params[6])
    let s2 = Double(params[7])
    let factorChi = Double(params[8])
    let depthValue = Double(params[9])
    let scale = factorChi * wl * wl
    let m00 = scale * (c10 + c12 * c2)
    let m01 = scale * c12 * s2
    let m11 = scale * (c10 - c12 * c2)
    let stepR = Double(geometry.qxByRow[1])
    let stepC = Double(geometry.qyByColumn[1])
    let pRow = k.flatMap { Self.splitMod(2 * stepR * (m00 * Double($0.x) + m01 * Double($0.y))) }
    let pCol = k.flatMap { Self.splitMod(2 * stepC * (m01 * Double($0.x) + m11 * Double($0.y))) }
    let psiA = m00 * stepR * stepR
    let psiB = 2 * m01 * stepR * stepC
    let psiC = m11 * stepC * stepC
    let psiRow = geo.rowF.map { Self.phasor(psiA * $0 * $0) }
    let psiCol = geo.colF.map { Self.phasor(psiC * $0 * $0) }
    let psiCross = geo.rowF.flatMap { Self.splitMod(psiB * $0) }
    let rows = geo.rows.count
    let cols = geo.cols.count
    let extra = geo.upper.count
    let colsAll = cols + extra
    var grid = Grid(
      rows: UInt32(rows), cols: UInt32(cols), colsAll: UInt32(colsAll), extra: UInt32(extra),
      nyquistRI: UInt32(geo.nyquistRI), sourceN: UInt32(source), gCols: UInt32(source / 2 + 1),
      count: UInt32(stored), chunk: UInt32(Self.chunk), groups: 0, groupOffset: 0, tuples: 0,
      bfOffset: 0,
      n: UInt32(geo.n), pad0: 0, pad1: 0, depthScale: 0, pad2: 0, pad3: 0, pad4: 0)
    let hasDepth = depthValue != 0
    let pRowBuffer = try Self.buffer(device, pRow)
    let pColBuffer = try Self.buffer(device, pCol)
    let tRow = try workspace("rows", stored * rows * (hasDepth ? 16 : 8))
    let tCol = try workspace("columns", stored * colsAll * (hasDepth ? 16 : 8))
    var depthBuffers: [MTLBuffer] = []
    let encoderTables = try encoder(commands, hasDepth ? depthTables : tables)
    encoderTables.setBuffer(pRowBuffer, offset: 0, index: 0)
    encoderTables.setBuffer(pColBuffer, offset: 0, index: 1)
    if hasDepth {
      let depthF = depthValue * scale
      let thetaR = Double(params[10])
      let thetaC = Double(params[11])
      let dRow = k.flatMap { Self.splitMod(-depthF * stepR * Double($0.x)) }
      let dCol = k.flatMap { Self.splitMod(-depthF * stepC * Double($0.y)) }
      encoderTables.setBuffer(try Self.buffer(device, dRow), offset: 0, index: 2)
      encoderTables.setBuffer(try Self.buffer(device, dCol), offset: 0, index: 3)
      encoderTables.setBuffer(geo.rowFBuffer, offset: 0, index: 4)
      encoderTables.setBuffer(geo.colFBuffer, offset: 0, index: 5)
      encoderTables.setBuffer(tRow, offset: 0, index: 6)
      encoderTables.setBuffer(tCol, offset: 0, index: 7)
      encoderTables.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 8)
      for (frequencies, step, theta) in [(geo.rowF, stepR, thetaR), (geo.colF, stepC, thetaC)] {
        var minus: [SIMD2<Float>] = []
        var plus: [SIMD2<Float>] = []
        var parts: [Float] = []
        for frequency in frequencies {
          let q = frequency * step
          let p = 0.5 * depthF * q * q
          let c = -Double.pi * depthValue * theta * q
          minus.append(Self.phasor(c - p))
          plus.append(Self.phasor(c + p))
          parts += [Float(p), Float(c)]
        }
        depthBuffers += [
          try Self.buffer(device, minus), try Self.buffer(device, plus),
          try Self.buffer(device, parts),
        ]
      }
      grid.depthScale = Float(depthF)
    } else {
      encoderTables.setBuffer(geo.rowFBuffer, offset: 0, index: 2)
      encoderTables.setBuffer(geo.colFBuffer, offset: 0, index: 3)
      encoderTables.setBuffer(tRow, offset: 0, index: 4)
      encoderTables.setBuffer(tCol, offset: 0, index: 5)
      encoderTables.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 6)
    }
    encoderTables.dispatchThreads(
      MTLSize(width: max(rows, colsAll), height: stored, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoderTables.endEncoding()
    let compact = try workspace("compact", (rows + 1) * cols * 8)
    guard let blit = commands.makeBlitCommandEncoder() else {
      throw MetalDirectPtychoError.commandQueue
    }
    blit.fill(buffer: compact, range: 0..<((rows + 1) * cols * 8), value: 0)
    blit.endEncoding()
    let psiRowBuffer = try Self.buffer(device, psiRow)
    let psiColBuffer = try Self.buffer(device, psiCol)
    let psiCrossBuffer = try Self.buffer(device, psiCross)
    var paramsCopy = params
    let aliases = max(1, min(3, factor))
    let usePaired = !hasDepth && factor == 2
    let pipeline = hasDepth ? depth[aliases]! : (usePaired ? paired : single)
    let tupleSize = hasDepth ? aliases : 2
    let tupleCount = (hasDepth || usePaired) ? geo.tupleCounts[tupleSize]! : 0
    let tableStride = hasDepth ? 16 : 8
    guard let kData, let bounds else { throw MetalDirectPtychoError.commandQueue }
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit(1 << 20) { spectra, offset, half, bfOffset, count in
        if cancelled() { throw CancellationError() }
        guard half, bfOffset % Self.chunk == 0 else { throw NotCached() }
        var launch = grid
        launch.count = UInt32(count)
        launch.groups = UInt32((count + Self.chunk - 1) / Self.chunk)
        launch.tuples = UInt32(tupleCount)
        let compute = try self.encoder(commands, pipeline)
        compute.setBuffer(spectra, offset: offset, index: 0)
        compute.setBuffer(self.qRow(geo), offset: 0, index: 1)
        compute.setBuffer(self.qCol(geo), offset: 0, index: 2)
        compute.setBuffer(kData, offset: bfOffset * 16, index: 3)
        compute.setBytes(&paramsCopy, length: paramsCopy.count * 4, index: 4)
        compute.setBuffer(bounds, offset: bfOffset / Self.chunk * 32, index: 5)
        compute.setBuffer(geo.rowIndices, offset: 0, index: 6)
        compute.setBuffer(geo.colIndices, offset: 0, index: 7)
        compute.setBuffer(tRow, offset: bfOffset * rows * tableStride, index: 8)
        compute.setBuffer(tCol, offset: bfOffset * colsAll * tableStride, index: 9)
        compute.setBuffer(psiRowBuffer, offset: 0, index: 10)
        compute.setBuffer(psiColBuffer, offset: 0, index: 11)
        compute.setBuffer(psiCrossBuffer, offset: 0, index: 12)
        compute.setBuffer(geo.colFBuffer, offset: 0, index: 13)
        compute.setBuffer(compact, offset: 0, index: 14)
        compute.setBytes(&launch, length: MemoryLayout<Grid>.stride, index: 15)
        if hasDepth || usePaired { compute.setBuffer(geo.tuples[tupleSize]!, offset: 0, index: 16) }
        if hasDepth {
          for (index, buffer) in [
            depthBuffers[0], depthBuffers[3], depthBuffers[1], depthBuffers[4],
            depthBuffers[2], depthBuffers[5],
          ].enumerated() {
            compute.setBuffer(buffer, offset: 0, index: 17 + index)
          }
        }
        let height = (hasDepth || usePaired ? tupleCount : rows) + (extra > 0 ? 1 : 0)
        compute.dispatchThreads(
          MTLSize(width: cols, height: height, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        compute.endEncoding()
      }
    }
    var first = Int32(geo.upper.first ?? -1)
    let mirror = try encoder(commands, scatter)
    mirror.setBuffer(compact, offset: 0, index: 0)
    mirror.setBuffer(full, offset: 0, index: 1)
    mirror.setBuffer(geo.rowPosition, offset: 0, index: 2)
    mirror.setBuffer(geo.colPosition, offset: 0, index: 3)
    mirror.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 4)
    mirror.setBytes(&first, length: 4, index: 5)
    mirror.dispatchThreads(
      MTLSize(width: geo.n, height: geo.n, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    mirror.endEncoding()
  }

  private var qBuffers: [Int: (MTLBuffer, MTLBuffer)] = [:]
  private func frequencies(_ geo: Geometry) -> (MTLBuffer, MTLBuffer) {
    if let buffers = qBuffers[geo.n] { return buffers }
    let n = geo.n
    func values(_ step: Float) -> [Float] {
      (0..<n).map { Float(Self.frequency($0, n) * Double(step)) }
    }
    let buffers = (
      try! Self.buffer(device, values(geometry.qxByRow[1])),
      try! Self.buffer(device, values(geometry.qyByColumn[1]))
    )
    qBuffers[n] = buffers
    return buffers
  }
  private func qRow(_ geo: Geometry) -> MTLBuffer { frequencies(geo).0 }
  private func qCol(_ geo: Geometry) -> MTLBuffer { frequencies(geo).1 }

  private func encoder(_ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState) throws
    -> MTLComputeCommandEncoder
  {
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw MetalDirectPtychoError.commandQueue
    }
    encoder.setComputePipelineState(pipeline)
    return encoder
  }
  static func allocate(_ device: MTLDevice, _ length: Int) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: max(16, length), options: .storageModePrivate)
    else {
      throw MetalDirectPtychoError.allocation(
        "separable SSB workspace (\(length) bytes). Close inactive datasets and retry")
    }
    return buffer
  }
  static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
    guard
      let result = values.withUnsafeBytes({ bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!, length: max(16, bytes.count), options: .storageModeShared)
      })
    else { throw MetalDirectPtychoError.allocation("separable SSB tables") }
    return result
  }
}
