import Foundation
import Metal
import XCTest

@testable @_spi(DirectPtychoBenchmark) import MetalDirectPtychoKernels

/// Drag windows of the regularized reconstructions on the synthetic scene of `SSBModelReconstructionTests`: inside a
/// window a preview agrees with the exact `reconstruct` at the same point; a preview needs a complete window whose
/// held inputs match, and is marked as one.
///
/// Stated tolerance: phase rms within 3e-6 of the exact image's standard deviation and maximum within 1e-7 rad,
/// amplitude rms within 2e-7 of its mean (float32 series coefficients, Chebyshev truncation at 1e-10 of its bound).
/// Observed: rms 1e-7 to 1.5e-6 of the std, maximum 3.1e-8 rad, amplitude 6.1e-8, over C10, C12, phi12, a
/// higher-order magnitude and azimuth, tilt and depth.
final class SSBDirectDragWindowTests: XCTestCase {
  typealias Scene = SSBModelReconstructionTests.Scene

  private func model(_ s: Scene, device: MTLDevice) throws -> MetalDirectPtychoJointModel {
    let input = try XCTUnwrap(
      s.counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let engine = try MetalDirectPtychoEngine(device: device, geometry: s.geometry)
    try engine.prepare(brightfield: input, countType: .uint16)
    return try engine.jointModel(scanDetectorRotationDegrees: 17.25)
  }

  private func values(_ buffer: MTLBuffer, _ count: Int) -> [Double] {
    let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
    return (0..<count).map { Double(pointer[$0]) }
  }

  /// (phase rms / exact std, phase max |difference|, amplitude rms / mean or 0).
  private func agreement(
    _ preview: MetalDirectPtychoJointPhase, _ exact: MetalDirectPtychoJointPhase
  ) -> (
    Double, Double, Double
  ) {
    let count = exact.rows * exact.columns
    let a = values(preview.phase, count)
    let b = values(exact.phase, count)
    let mean = b.reduce(0, +) / Double(count)
    var squared = 0.0
    var maximum = 0.0
    var power = 0.0
    for i in 0..<count {
      var d = a[i] - b[i]
      d -= 2 * .pi * (d / (2 * .pi)).rounded()
      squared += d * d
      maximum = max(maximum, abs(d))
      power += (b[i] - mean) * (b[i] - mean)
    }
    var amplitude = 0.0
    if let x = preview.amplitude, let y = exact.amplitude {
      let p = values(x, count)
      let q = values(y, count)
      let average = q.reduce(0, +) / Double(count)
      // A loop: Swift 6.1 cannot type-check the zip-map-reduce form in time.
      var squaredError = 0.0
      for (u, v) in zip(p, q) { squaredError += (u - v) * (u - v) }
      amplitude = (squaredError / Double(count)).squareRoot() / average
    }
    return ((squared / power).squareRoot(), maximum, amplitude)
  }

  private static let thick = DirectPtychoModelPoint(
    c10Angstroms: -38, c12Angstroms: 9, phi12Radians: -0.4, sampleTiltMrad: [2.5, -4],
    depthSpreadAngstroms: 140,
    higherOrder: [
      .init(order: 2, symmetry: 1, magnitudeAngstroms: 900, angleRadians: 0.3),
      .init(order: 3, symmetry: 0, magnitudeAngstroms: -4e4),
    ])

  func testPreviewsAgreeWithExactSolvesInsideTheWindow() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try SSBModelReconstructionTests.scene()
    let model = try model(s, device: device)
    let point = Self.thick
    let controls: [(DirectPtychoDragControl, DirectPtychoReconstructionMethod, Int)] = [
      (.c10, .complex, 2), (.c12, .phase, 2), (.phi12, .complex, 1),
      (.magnitude(order: 2, symmetry: 1), .phase, 2),
      (.angle(order: 2, symmetry: 1), .complex, 2), (.tiltRow, .complex, 2), (.depth, .phase, 2),
    ]
    for (control, method, factor) in controls {
      let state = try model.buildDragWindow(
        along: control, at: point, upsamplingFactor: factor, method: method, nodes: 1000)
      XCTAssertTrue(state.ready, "\(control)")
      XCTAssertEqual(
        state.total, [.phi12, .angle(order: 2, symmetry: 1)].contains(control) ? 50 : 29,
        "\(control)")
      XCTAssertTrue(state.range.contains(control.value(of: point)))
      let half = (state.range.upperBound - state.range.lowerBound) / 2
      XCTAssertTrue(half.isFinite && half > 0)
      for fraction in [-0.83, 0.0, 0.31, 0.97] {
        let moved = control.setting(control.value(of: point) + fraction * half, in: point)
        let preview = try XCTUnwrap(
          try model.dragPreview(moved, along: control, upsamplingFactor: factor, method: method),
          "\(control)")
        let exact = try model.reconstruct(moved, upsamplingFactor: factor, method: method)
        XCTAssertNil(exact.preview)
        XCTAssertEqual(preview.preview?.control, control)
        XCTAssertEqual(preview.method, method)
        XCTAssertEqual(preview.point, moved)
        XCTAssertEqual(preview.rows, exact.rows)
        XCTAssertEqual(preview.amplitude == nil, method == .phase)
        XCTAssertTrue(preview.kernel?.variant.contains("drag-window") == true)
        let (rms, maximum, amplitude) = agreement(preview, exact)
        print(
          "DRAG \(control) \(method.rawValue) \(factor)x at \(fraction): rms/std \(rms) max \(maximum) amp \(amplitude)"
        )
        XCTAssertLessThan(rms, 3e-6, "\(control) at \(fraction)")
        XCTAssertLessThan(maximum, 1e-7, "\(control) at \(fraction)")
        XCTAssertLessThan(amplitude, 2e-7, "\(control) at \(fraction)")
      }
      model.releaseDragWindows()
    }
  }

  func testPreviewNeedsACompleteWindowWithTheSameHeldInputs() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try SSBModelReconstructionTests.scene()
    let model = try model(s, device: device)
    let point = Self.thick
    func preview(
      _ p: DirectPtychoModelPoint, _ method: DirectPtychoReconstructionMethod = .complex,
      _ regularization: DirectPtychoRegularization = .init()
    ) throws -> MetalDirectPtychoJointPhase? {
      try model.dragPreview(
        p, along: .c10, upsamplingFactor: 2, method: method, regularization: regularization)
    }
    XCTAssertNil(try preview(point))
    XCTAssertNil(
      model.dragWindowState(along: .c10, at: point, upsamplingFactor: 2, method: .complex))
    // One node at a time; no preview before the last.
    var state = try model.buildDragWindow(
      along: .c10, at: point, upsamplingFactor: 2, method: .complex, nodes: 3)
    XCTAssertEqual(state.built, 3)
    XCTAssertFalse(state.ready)
    XCTAssertNil(try preview(point))
    state = try model.buildDragWindow(
      along: .c10, at: point, upsamplingFactor: 2, method: .complex, nodes: 1000)
    XCTAssertTrue(state.ready)
    XCTAssertEqual(
      model.dragWindowState(along: .c10, at: point, upsamplingFactor: 2, method: .complex), state)
    XCTAssertNotNil(try preview(point))
    // A complex window serves phase, with any damping, as the exact solve would.
    let damping = DirectPtychoRegularization(phaseDamping: 0.2, amplitudeDamping: 0.05)
    let phase = try XCTUnwrap(try preview(point, .phase, damping))
    let (rms, _, _) = agreement(
      phase,
      try model.reconstruct(point, upsamplingFactor: 2, method: .phase, regularization: damping))
    XCTAssertLessThan(rms, 3e-6)
    // Outside the window: none.
    var outside = point
    outside.c10Angstroms = state.range.upperBound + 1
    XCTAssertNil(try preview(outside))
    // Another control moved: the window is dropped.
    var held = point
    held.c12Angstroms += 1
    XCTAssertNil(try preview(held))
    XCTAssertNil(
      model.dragWindowState(along: .c10, at: point, upsamplingFactor: 2, method: .complex))
    // A phase window does not serve complex; a new prior or rotation drops it.
    _ = try model.buildDragWindow(
      along: .c10, at: point, upsamplingFactor: 2, method: .phase, nodes: 1000)
    XCTAssertNotNil(try preview(point, .phase))
    XCTAssertNil(try preview(point, .complex))
    XCTAssertNil(try preview(point, .phase, .init(smoothnessInverseAngstroms: 0.5)))
    _ = try model.reconstruct(
      point, upsamplingFactor: 2, method: .phase,
      regularization: .init(smoothnessInverseAngstroms: 0.5))
    XCTAssertNil(try preview(point, .phase))
    _ = try model.buildDragWindow(
      along: .c10, at: point, upsamplingFactor: 2, method: .phase, nodes: 1000)
    try model.retarget(scanDetectorRotationDegrees: 18)
    XCTAssertNil(try preview(point, .phase))
    // SSB has its own redraw: no window.
    XCTAssertNil(try model.dragPreview(point, along: .c10, upsamplingFactor: 2, method: .ssb))
    XCTAssertThrowsError(
      try model.buildDragWindow(along: .c10, at: point, upsamplingFactor: 2, method: .ssb))
    XCTAssertThrowsError(
      try model.buildDragWindow(
        along: .c10, at: point, upsamplingFactor: 2, method: .phase, isCancelled: { true })
    ) { XCTAssertTrue($0 is CancellationError) }
  }

  func testAControlWithoutEffectNeedsOneNode() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let s = try SSBModelReconstructionTests.scene()
    let model = try model(s, device: device)
    // phi12 at C12 = 0 and the tilt without a depth spread do not change the model.
    let flat = DirectPtychoModelPoint(c10Angstroms: 61, c12Angstroms: 0, phi12Radians: 0.7)
    for control in [DirectPtychoDragControl.phi12, .tiltColumn] {
      let state = try model.buildDragWindow(
        along: control, at: flat, upsamplingFactor: 2, method: .phase, nodes: 5)
      XCTAssertEqual(state.total, 1)
      XCTAssertTrue(state.ready)
      let moved = control.setting(control.value(of: flat) + 0.4, in: flat)
      let preview = try XCTUnwrap(
        try model.dragPreview(moved, along: control, upsamplingFactor: 2, method: .phase))
      XCTAssertEqual(preview.preview?.halfWidth, .infinity)
      let (rms, _, _) = agreement(
        preview, try model.reconstruct(moved, upsamplingFactor: 2, method: .phase))
      XCTAssertLessThan(rms, 3e-6)
    }
  }

  func testControlsReadAndSetTheirParameter() throws {
    let point = Self.thick
    let cases: [(DirectPtychoDragControl, Double)] = [
      (.c10, -38), (.c12, 9), (.phi12, -0.4), (.magnitude(order: 2, symmetry: 1), 900),
      (.angle(order: 2, symmetry: 1), 0.3), (.magnitude(order: 5, symmetry: 6), 0), (.tiltRow, 2.5),
      (.tiltColumn, -4), (.depth, 140),
    ]
    for (control, value) in cases {
      XCTAssertEqual(control.value(of: point), value, "\(control)")
      let moved = control.setting(value + 0.5, in: point)
      XCTAssertEqual(control.value(of: moved), value + 0.5, "\(control)")
      // every other parameter is unchanged
      let back = control.setting(value, in: moved)
      XCTAssertEqual(
        back.reconstructionCoefficients, point.reconstructionCoefficients, "\(control)")
      XCTAssertEqual(back.sampleTiltMrad, point.sampleTiltMrad, "\(control)")
      XCTAssertEqual(back.depthSpreadAngstroms, point.depthSpreadAngstroms, "\(control)")
    }
    let preview = DirectPtychoDragPreview(
      control: .angle(order: 3, symmetry: 2), center: 1, halfWidth: 2, order: 49)
    XCTAssertEqual(
      try JSONDecoder().decode(DirectPtychoDragPreview.self, from: JSONEncoder().encode(preview)),
      preview)
  }
}
