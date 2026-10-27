import Foundation
import Metal

/// One parameter of the model point that a slider drags: what a drag window of the regularized reconstructions
/// follows. Values are in the point's own units: Angstrom for C10, C12, a higher-order magnitude and the depth
/// spread, radians for phi12 and a higher-order azimuth, mrad for the sample tilt.
public enum SSBDragControl: Hashable, Codable, Sendable {
  case c10, c12, phi12
  /// The magnitude of the higher-order term C(order)(symmetry).
  case magnitude(order: Int, symmetry: Int)
  /// The azimuth of the higher-order term C(order)(symmetry).
  case angle(order: Int, symmetry: Int)
  case tiltRow, tiltColumn, depth

  /// This control's value in `point`; an absent higher-order term is 0.
  public func value(of point: SSBModelPoint) -> Double {
    switch self {
    case .c10: point.c10Angstroms
    case .c12: point.c12Angstroms
    case .phi12: point.phi12Radians
    case .magnitude(let order, let symmetry):
      point.higherOrder.first { $0.order == order && $0.symmetry == symmetry }?.magnitudeAngstroms
        ?? 0
    case .angle(let order, let symmetry):
      point.higherOrder.first { $0.order == order && $0.symmetry == symmetry }?.angleRadians ?? 0
    case .tiltRow: point.sampleTiltMrad[0]
    case .tiltColumn: point.sampleTiltMrad[1]
    case .depth: point.depthSpreadAngstroms
    }
  }

  /// `point` with this control set to `value`.
  public func setting(_ value: Double, in point: SSBModelPoint) -> SSBModelPoint {
    var point = point
    switch self {
    case .c10: point.c10Angstroms = value
    case .c12: point.c12Angstroms = value
    case .phi12: point.phi12Radians = value
    case .magnitude(let order, let symmetry), .angle(let order, let symmetry):
      var index = point.higherOrder.firstIndex { $0.order == order && $0.symmetry == symmetry }
      if index == nil {
        point.higherOrder.append(SSBModelHigherOrder(order: order, symmetry: symmetry))
        index = point.higherOrder.count - 1
      }
      if case .magnitude = self {
        point.higherOrder[index!].magnitudeAngstroms = value
      } else {
        point.higherOrder[index!].angleRadians = value
      }
    case .tiltRow: point.sampleTiltMrad[0] = value
    case .tiltColumn: point.sampleTiltMrad[1] = value
    case .depth: point.depthSpreadAngstroms = value
    }
    return point
  }
}

/// Marks a `.phase` or `.complex` result as a drag preview: evaluated from a drag window along `control`, not
/// solved from the measured spectra at its own optics. Nil on every exact result. A preview is for display while a
/// slider moves; committed, saved and exported results are always exact (`reconstruct`).
public struct SSBDragPreview: Codable, Equatable, Sendable {
  public let control: SSBDragControl
  /// The window's centre and half width in the control's units; it covers center +- halfWidth (all values when
  /// the control does not change the model).
  public let center: Double
  public let halfWidth: Double
  /// The Chebyshev order: the window holds order + 1 exact preparations.
  public let order: Int

  public init(control: SSBDragControl, center: Double, halfWidth: Double, order: Int) {
    self.control = control
    self.center = center
    self.halfWidth = halfWidth
    self.order = order
  }
}

/// Progress of the drag window along one control at one point.
public struct SSBDragWindowState: Equatable, Sendable {
  public let control: SSBDragControl
  /// Exact preparations done and needed.
  public let built: Int
  public let total: Int
  /// The values the window covers, in the control's units.
  public let range: ClosedRange<Double>
  public var ready: Bool { built == total }
}

/// A Chebyshev series, along one control, of every prepared entry (Python's CUDA `DragWindow`, `chebyshev.Series`).
///
/// With everything else held, each Gram entry and right side is a sum over BF pixels of f_k exp(i (c - c0) u_k),
/// an entire function of the control c, so on [c0 - h, c0 + h] its Chebyshev series converges like
/// 2 (z / 2)^(N + 1) / (N + 1)!, z = h max|u|. The window adds one exact preparation per node
/// c0 + h cos(pi (j + 1/2) / (N + 1)) and evaluates any value inside as sum_n c_n T_n((c - c0) / h), followed by the
/// same float64 solves as an exact reconstruction. A complex window holds the phase system as its lower-right block.
final class DragWindow {
  /// Everything the prepared entries depend on but the control: the layout, the point with the control at zero,
  /// the rotation.
  struct Key: Equatable {
    var layout: SSBDirectSolver.LayoutKey
    var held: SSBModelPoint
    var rotation: Double
  }

  let control: SSBDragControl
  let key: Key
  let complex: Bool
  let center: Double
  let halfWidth: Double
  let order: Int
  private(set) var added = 0
  /// Per alias group while nodes are added: float-float coefficients, [n][frequency][entry].
  var accumulators: [MTLBuffer]
  /// Per alias group once complete: the float32 coefficients, [n][frequency][entry].
  var coefficients: [MTLBuffer]?
  /// Per alias group: a preview's evaluated entries, in the prepared sums' layout (reused across frames).
  var evaluated: [MTLBuffer] = []
  var lastUse = 0

  init(
    control: SSBDragControl, key: Key, complex: Bool, center: Double, halfWidth: Double, order: Int,
    accumulators: [MTLBuffer]
  ) {
    self.control = control
    self.key = key
    self.complex = complex
    self.center = center
    self.halfWidth = halfWidth
    self.order = order
    self.accumulators = accumulators
  }

  var ready: Bool { coefficients != nil }
  var bytes: Int { (accumulators + (coefficients ?? []) + evaluated).reduce(0) { $0 + $1.length } }
  var range: ClosedRange<Double> {
    halfWidth.isFinite
      ? (center - halfWidth)...(center + halfWidth) : -Double.infinity...Double.infinity
  }
  func covers(_ value: Double) -> Bool { !halfWidth.isFinite || abs(value - center) <= halfWidth }
  /// The control value of node j.
  func node(_ j: Int) -> Double {
    halfWidth.isFinite
      ? center + halfWidth * cos(Double.pi * (Double(j) + 0.5) / Double(order + 1)) : center
  }
  /// Weight of node j in coefficient n: (2 / (N + 1)) cos(n theta_j), halved for n = 0.
  func weights(node j: Int) -> [Float] {
    let theta = Double.pi * (Double(j) + 0.5) / Double(order + 1)
    return (0...order).map { n in
      Float((n == 0 ? 1.0 : 2.0) / Double(order + 1) * cos(Double(n) * theta))
    }
  }
  /// T_n((value - center) / h) as float-float (hi, lo).
  func basis(at value: Double) -> [SIMD2<Float>] {
    let x = halfWidth.isFinite ? min(1, max(-1, (value - center) / halfWidth)) : 0
    let angle = acos(x)
    return (0...order).map { n in
      let t = cos(Double(n) * angle)
      let hi = Float(t)
      return SIMD2(hi, Float(t - Double(hi)))
    }
  }
  func nodeAdded() { added += 1 }
  var state: SSBDragWindowState {
    SSBDragWindowState(control: control, built: added, total: order + 1, range: range)
  }
  var provenance: SSBDragPreview {
    SSBDragPreview(control: control, center: center, halfWidth: halfWidth, order: order)
  }
}

extension SSBDirectSolver {
  /// Tiles kept per control: a drag back and forth across one tile edge needs two (Python `TILES`).
  static let tilesPerControl = 3

  /// The half width and Chebyshev order of a window along `control` at `point` (Python `chebyshev.series`,
  /// `half_width`): h = reach / max|u|, max|u| the spread of the transfer phase (or depth-sinc argument) per unit
  /// of the control over the aperture, reach 10 for a magnitude, tilt or depth and 20 for an azimuth, and the
  /// smallest order from 12 whose bound 2 (reach / 2)^(N + 1) / (N + 1)! is at most 1e-10 (1e-14 for an azimuth).
  func windowShape(_ control: SSBDragControl, at point: SSBModelPoint) -> (
    halfWidth: Double, order: Int
  ) {
    let wl = Double(geometry.wavelengthAngstroms)
    let alpha =
      Double(geometry.semiangleRadians)
      + 0.5 * Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians))
    let cutoff =
      (2 * Double(geometry.semiangleRadians)
        + Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians))) / wl
    func perAngstrom(_ order: Int) -> Double {
      2 * .pi / wl * pow(alpha, Double(order + 1)) / Double(order + 1)
    }
    let azimuth: Bool
    let spread: Double
    switch control {
    case .c10:
      (azimuth, spread) = (false, perAngstrom(1))
    case .c12:
      (azimuth, spread) = (false, 2 * perAngstrom(1))
    case .phi12:
      (azimuth, spread) = (true, 4 * perAngstrom(1) * abs(point.c12Angstroms))
    case .magnitude(let order, let symmetry):
      (azimuth, spread) = (false, (symmetry > 0 ? 2 : 1) * perAngstrom(order))
    case .angle(let order, let symmetry):
      let magnitude = SSBDragControl.magnitude(order: order, symmetry: symmetry).value(of: point)
      (azimuth, spread) = (true, 2 * Double(symmetry) * perAngstrom(order) * abs(magnitude))
    case .tiltRow, .tiltColumn:
      // the depth sinc's argument moves by pi t |q| per rad of tilt; a Gram entry multiplies two sincs
      (azimuth, spread) = (false, 2 * .pi * point.depthSpreadAngstroms * cutoff * 1e-3)
    case .depth:
      let tilt = point.sampleTiltMrad.map { abs($0) }.max() ?? 0
      (azimuth, spread) = (
        false, Double.pi * wl * (alpha / wl) * (alpha / wl) + 2 * .pi * cutoff * tilt * 1e-3
      )
    }
    guard spread > 0, spread.isFinite else { return (.infinity, 0) }
    let reach = azimuth ? 20.0 : 10.0
    let truncation = azimuth ? 1e-14 : 1e-10
    var order = 12
    while log(2) + Double(order + 1) * log(reach / 2) - lgamma(Double(order + 2)) > log(truncation)
    { order += 1 }
    return (reach / spread, order)
  }

  /// Drop every window whose held inputs differ from `point` (a complex window holds about 1.5 GB at 3x).
  func dropStaleWindows(point: SSBModelPoint, rotation: Double, layout: LayoutKey) {
    for (control, tiles) in windows {
      let key = DragWindow.Key(
        layout: layout, held: control.setting(0, in: point), rotation: rotation)
      let kept = tiles.filter { $0.key == key }
      windows[control] = kept.isEmpty ? nil : kept
    }
  }

  /// Add up to `nodes` exact preparations to the window along `control` that covers `point` (a new tile when
  /// none does), and return its state.
  func buildWindow(
    along control: SSBDragControl, at point: SSBModelPoint, factor: Int, complex: Bool,
    regularization: SSBDirectRegularization, rotation: Double, k: MTLBuffer, nodes: Int,
    visit: MetalSSBJointModel.VisitSpectra, isCancelled: () -> Bool
  ) throws -> SSBDragWindowState {
    let layoutKey = LayoutKey(factor: factor, regularization: regularization)
    let layout = try currentLayout(layoutKey, visit: visit, isCancelled: isCancelled)
    dropStaleWindows(point: point, rotation: rotation, layout: layoutKey)
    let key = DragWindow.Key(
      layout: layoutKey, held: control.setting(0, in: point), rotation: rotation)
    let value = control.value(of: point)
    var tiles = windows[control] ?? []
    // A complex window serves both methods; a phase window is replaced by a complex one when complex is needed.
    if complex { tiles.removeAll { !$0.complex } }
    let window: DragWindow
    if let index = tiles.firstIndex(where: { $0.covers(value) && ($0.complex || !complex) }) {
      window = tiles.remove(at: index)
    } else {
      let shape = windowShape(control, at: point)
      var center = value
      if let anchor = tiles.last, anchor.halfWidth.isFinite {
        // tiles on a grid c0 + 2 h k, so crossing an edge and coming back reuses both
        let width = 2 * anchor.halfWidth
        center = anchor.center + width * ((value - anchor.center) / width).rounded()
      }
      let accumulators = try layout.groups.map { group in
        let length =
          (shape.order + 1) * group.size * Self.parts(count: group.count, complex: complex).outputs
          * 16
        return try Self.allocate(device, length)
      }
      try clear(accumulators)
      window = DragWindow(
        control: control, key: key, complex: complex, center: center, halfWidth: shape.halfWidth,
        order: shape.order, accumulators: accumulators)
    }
    windows[control] = [window] + tiles.prefix(Self.tilesPerControl - 1)
    useCounter += 1
    window.lastUse = useCounter
    trimWindows(keeping: window)
    var remaining = nodes
    while remaining > 0, !window.ready {
      if isCancelled() { throw CancellationError() }
      let nodePoint = control.setting(window.node(window.added), in: point)
      let (node, _) = try prepare(
        PreparedKey(layout: layoutKey, point: nodePoint, rotation: rotation),
        complex: window.complex,
        layout: layout, k: k, visit: visit, isCancelled: isCancelled)
      try accumulate(node, into: window, layout: layout)
      remaining -= 1
    }
    return window.state
  }

  /// The ready window along `control` covering `point` with these held inputs, if any.
  func readyWindow(
    along control: SSBDragControl, at point: SSBModelPoint, layout: LayoutKey, complex: Bool,
    rotation: Double
  ) -> DragWindow? {
    dropStaleWindows(point: point, rotation: rotation, layout: layout)
    let value = control.value(of: point)
    return windows[control]?.first { $0.ready && $0.covers(value) && ($0.complex || !complex) }
  }

  /// The state of the window along `control` covering `point`, if one exists.
  func windowState(
    along control: SSBDragControl, at point: SSBModelPoint, factor: Int, complex: Bool,
    regularization: SSBDirectRegularization, rotation: Double
  ) -> SSBDragWindowState? {
    let layoutKey = LayoutKey(factor: factor, regularization: regularization)
    guard layout?.key == layoutKey else { return nil }
    dropStaleWindows(point: point, rotation: rotation, layout: layoutKey)
    let value = control.value(of: point)
    return windows[control]?.first { $0.covers(value) && ($0.complex || !complex) }?.state
  }

  /// The preview at `point` from a ready window, or nil when none covers it. Never prepares or builds anything.
  func preview(
    _ point: SSBModelPoint, along control: SSBDragControl, factor: Int, complex: Bool,
    regularization: SSBDirectRegularization, rotation: Double
  ) throws -> (phase: MTLBuffer, amplitude: MTLBuffer?, n: Int, window: SSBDragPreview)? {
    let layoutKey = LayoutKey(factor: factor, regularization: regularization)
    guard let layout, layout.key == layoutKey,
      let window = readyWindow(
        along: control, at: point, layout: layoutKey, complex: complex, rotation: rotation),
      let coefficients = window.coefficients
    else { return nil }
    useCounter += 1
    window.lastUse = useCounter
    var timing = SSBDirectTiming()
    let started = Date()
    if window.evaluated.isEmpty {
      window.evaluated = try layout.groups.map { group in
        try Self.allocate(
          device, group.size * Self.parts(count: group.count, complex: window.complex).outputs * 16,
          shared: true)
      }
    }
    var basis = window.basis(at: control.value(of: point))
    let commands = try command()
    for (index, group) in layout.groups.enumerated() {
      var p = SIMD2<UInt32>(
        UInt32(group.size * Self.parts(count: group.count, complex: window.complex).outputs),
        UInt32(window.order + 1))
      let encoder = try self.encoder(commands, try windowPipeline("direct_window_evaluate"))
      encoder.setBuffer(coefficients[index], offset: 0, index: 0)
      encoder.setBuffer(window.evaluated[index], offset: 0, index: 1)
      encoder.setBytes(&basis, length: basis.count * 8, index: 2)
      encoder.setBytes(&p, length: 8, index: 3)
      dispatch(encoder, width: Int(p.x))
    }
    try finish(commands)
    timing.prepareGPUSeconds = commands.gpuEndTime - commands.gpuStartTime
    timing.prepareSeconds = Date().timeIntervalSince(started)
    timing.preparedFromCache = true
    let system = Prepared(
      key: PreparedKey(layout: layoutKey, point: point, rotation: rotation),
      complex: window.complex,
      sums: window.evaluated)
    let result = try solveAndWave(
      system, layout: layout, complex: complex, regularization: regularization, timing: &timing)
    lastTiming = timing
    return (result.phase, result.amplitude, result.n, window.provenance)
  }

  /// Drop every drag window.
  func releaseWindows() { windows = [:] }

  // MARK: - Window arithmetic

  /// Add one node's exact entries to the window's coefficients; round them to float32 after the last node.
  private func accumulate(_ node: Prepared, into window: DragWindow, layout: Layout) throws {
    var weights = window.weights(node: window.added)
    let commands = try command()
    for (index, group) in layout.groups.enumerated() {
      var p = SIMD2<UInt32>(
        UInt32(group.size * Self.parts(count: group.count, complex: window.complex).outputs),
        UInt32(window.order + 1))
      let encoder = try self.encoder(commands, try windowPipeline("direct_window_accumulate"))
      encoder.setBuffer(node.sums[index], offset: 0, index: 0)
      encoder.setBuffer(window.accumulators[index], offset: 0, index: 1)
      encoder.setBytes(&weights, length: weights.count * 4, index: 2)
      encoder.setBytes(&p, length: 8, index: 3)
      dispatch(encoder, width: Int(p.x))
    }
    try finish(commands)
    window.nodeAdded()
    guard window.added == window.order + 1 else { return }
    let finalize = try command()
    let rounded = try window.accumulators.map { try Self.allocate(device, $0.length / 2) }
    for (accumulator, output) in zip(window.accumulators, rounded) {
      var count = UInt32(accumulator.length / 16)
      let encoder = try self.encoder(finalize, try windowPipeline("direct_window_finalize"))
      encoder.setBuffer(accumulator, offset: 0, index: 0)
      encoder.setBuffer(output, offset: 0, index: 1)
      encoder.setBytes(&count, length: 4, index: 2)
      dispatch(encoder, width: Int(count))
    }
    try finish(finalize)
    window.coefficients = rounded
    window.accumulators = []
  }

  private func clear(_ buffers: [MTLBuffer]) throws {
    let commands = try command()
    guard let blit = commands.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
    for buffer in buffers { blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0) }
    blit.endEncoding()
    try finish(commands)
  }

  /// Keep the windows within a quarter of the device's recommended working set, least recently used dropped first.
  private func trimWindows(keeping current: DragWindow) {
    let budget = Int(device.recommendedMaxWorkingSetSize / 4)
    var all = windows.values.flatMap { $0 }.sorted { $0.lastUse > $1.lastUse }
    var total = all.reduce(0) { $0 + $1.bytes }
    while total > budget, let last = all.last, last !== current {
      all.removeLast()
      total -= last.bytes
      for (control, tiles) in windows {
        let kept = tiles.filter { $0 !== last }
        windows[control] = kept.isEmpty ? nil : kept
      }
    }
  }
}
