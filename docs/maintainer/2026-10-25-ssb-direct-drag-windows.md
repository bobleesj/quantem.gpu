# Drag previews of the regularized direct reconstructions (native Metal)

Status: implemented 2026-10-25 (`ssb-direct` v0.2). Port of Python's CUDA `DragWindow` (`direct_ptycho/cuda/direct.py`,
`direct_ptycho/cuda/chebyshev.py`, `direct_ptycho/direct_workflow.py` `drag`/`prepare`/`advance`).

## Problem

`.phase` and `.complex` (`MetalSSBJointModel.reconstruct(_:upsamplingFactor:method:regularization:)`) solve, per scan
frequency, the damped normal equations of every supported output alias. The normal equations depend on the optics:
every aberration, tilt or depth change prepares them again from all BF pixels. On an Apple M5 Max at 3x on a
512 x 512 scan that takes about 0.17 s (phase) and 0.17-0.19 s (complex) per change, so a slider drag redraws at about
5 frames per second, against 30 for SSB (`ssb-redraw`). The preparation is bound by the per (frequency, BF pixel)
transfer and products and by streaming the 9 GB of cached spectra; it cannot reach 33 ms.

## Design

While one control moves and every other input is held, each prepared entry (Gram, right side) is a sum over BF pixels
of f_k exp(i (c - c0) u_k): an entire function of the control c. On a window [c0 - h, c0 + h] its Chebyshev series
converges like 2 (z / 2)^(N + 1) / (N + 1)!, z = h max|u|, so N + 1 exact preparations at the Chebyshev nodes
determine every entry inside the window to the truncation bound; a drag frame then evaluates the series
(sum_n c_n T_n) and runs the same float64 solves, scatter and inverse FFT as an exact reconstruction.

| Choice | Value | Reason |
| --- | --- | --- |
| Reach z, truncation | 10 and 1e-10 for a magnitude, tilt or depth (N = 28); 20 and 1e-14 for an azimuth (N = 49) | Python's measured choice: the image reaches the float32 floor; a looser bound leaves visible error after the solve (cancellation in the Gram) |
| Half width h | reach / max|u|; max|u| the spread over the aperture of d(chi)/d(control) (doubled for m > 0 and for an azimuth), of the depth-sinc argument for tilt and depth | Python's `half_width`; C10 +- 6.8 nm and C12 +- 3.4 nm at 300 kV, 30 mrad |
| Nodes | exact preparations at the node optics (the same kernels as `reconstruct`) | one code path, so every node is the exact system; any control works |
| Coefficients | float-float while nodes are added, rounded to float32 once complete; evaluated in float-float | float32 storage halves the memory (about 1.4 GB per complex window at 3x) at the float32 level of the preparation itself |
| Tiles | up to three per control on a grid c0 + 2 h k | a drag back and forth across an edge reuses both (Python `TILES`) |
| Validity | layout (factor, prior, weighting), every other parameter of the point, rotation | any change drops the window; a damping change does not (solve-time) |
| Methods | a complex window also serves `.phase` (its lower-right block) | as the prepared systems |
| Memory | at most a quarter of the device's recommended working set, least recently used dropped first | a complex window at 3x is about 1.4 GB (2.9 GB while building) |

Previews are never exact solves: `MetalSSBJointPhase.preview` (`SSBDragPreview`: control, centre, half width, order)
is set on every preview and nil on every exact result. Callers show it as a preview and settle on `reconstruct` when
the drag ends; they must not save or export a preview.

## API (`MetalSSBKernels`)

```swift
public enum SSBDragControl: Hashable, Codable, Sendable {
  case c10, c12, phi12, magnitude(order: Int, symmetry: Int), angle(order: Int, symmetry: Int), tiltRow, tiltColumn, depth
  public func value(of point: SSBModelPoint) -> Double         // the point's units: A, rad, mrad
  public func setting(_ value: Double, in point: SSBModelPoint) -> SSBModelPoint
}
public struct SSBDragWindowState { control, built, total, range, ready }
public struct SSBDragPreview: Codable { control, center, halfWidth, order }
extension MetalSSBJointModel {
  func buildDragWindow(along:at:upsamplingFactor:method:regularization:nodes:isCancelled:) throws -> SSBDragWindowState
  func dragWindowState(along:at:upsamplingFactor:method:regularization:) -> SSBDragWindowState?
  func dragPreview(_:along:upsamplingFactor:method:regularization:) throws -> MetalSSBJointPhase?  // nil: no window
  func releaseDragWindows()
}
// MetalSSBJointPhase.preview: SSBDragPreview?  (nil for exact results)
```

`buildDragWindow` adds `nodes` exact preparations (one by default) so a viewer can interleave them with frames and
yield to user requests; `dragPreview` never prepares or builds anything. `.ssb` has its own interactive redraw and no
window. The viewer contract mirrors Python's `drag` / `prepare` / `advance`: while dragging, show the preview when a
window covers the value, otherwise the exact solve while the window builds; after an exact image, build windows for
the controls dragged next (Live4DSTEM: the last dragged, then C10, C12 and phi12) between interactions.

## Alternatives not taken

- **Reduced BF subset while dragging** (the MPS SSB preview): about 15x fewer pixels would be needed for 33 ms at 3x,
  so a visibly noisier image, and still a different solution of the regularized model. A window is exact to the
  float32 level.
- **Per-alias (decoupled) solve as a preview**: the cost of an SSB redraw, but the off-diagonal alias blocks carry
  about a third of the Gram's products at 3x; the image is not the method's.
- **SSB redraw while dragging Phase or Complex**: a different method's image, no amplitude.
- **Faster exact preparation only**: the one-thread-per-alias kernel (`direct_normal_tiled`) brings complex 3x from
  about 0.28 to 0.17-0.19 s; the floor of the per-pixel work and the spectrum stream stays far above 33 ms.

## Measured (Apple M5 Max 40-core GPU, 512 x 512 scans, 300 kV, 30 mrad, 2026-10-25)

Benchmark (`metal-ssb-fit-benchmark --direct-drag CASE`, a measured 0.373 A acquisition, Auto 3x, window along C10):

| | build (29 nodes) | preview frame median (max) | evaluate | preview vs `reconstruct` at the same point |
| --- | --- | --- | --- | --- |
| complex | 5.4 s | 8.9 ms (22) | 3.5 ms | phase rms 9e-7 of the image std, max 6e-7 rad; amplitude rms 1.3e-7 of its mean |
| phase | 5.2 s | 4.8 ms (10) | 1.4 ms | phase rms 3.5e-7 of the image std, max 4e-7 rad |

The window covers C10 +- 6.8 nm (C12 +- 3.4 nm). In the Live4DSTEM app (headed, built-in display, 4 s drags of C10
and C12 after the fifth-order fit; median per-render wall time and median interval between published renders):
complex 9.8-10.4 ms per render, a render every 18-21 ms; phase 5.4-5.9 ms, every 16-18 ms, on two 0.373 A
acquisitions at 3x; on a 0.495 A acquisition at 4x complex 26-27 ms (every 33-47 ms) and phase 11.7 ms (every
18-20 ms), with windows of 12-15 s to build. Every render of every drag was a preview; the exact image follows the
release in 0.24-0.29 s at 3x and 0.46-0.54 s at 4x.

Open: a window takes about 29 exact preparations (5 s at 3x, 12-15 s at 4x), so a drag beyond the prepared tile
shows exact frames (about 5 per second at 3x) until the next tile is built. Building several nodes per pass of the
spectra (the transfer once, B rotated node columns) would cut that by about 2x; the products per node bound it.
