# Drift across a time series

A time series of the same specimen drifts: each frame sees it a little further along. `quantem.gpu.drift` measures how
far between the frames' phase images and moves the frames back into register. It corrects translation only: no
scan distortion, no change of the scan positions.

```python
from quantem.gpu.drift import drift

found = drift(phases, sampling_A=0.124)    # phases: (frames, rows, cols)
found                                      # table: each frame's displacement and σ, pairs rejected, closure
found.show(phases)                         # frame 0, the mean before and after correction, spectra, the path
aligned, mean = found.corrected(phases)    # every frame on frame 0, over the region all frames share
```

`phases` can be a GPU tensor or CuPy array, a NumPy array (moved to the GPU), a list of 2D images, a list of Direct Ptycho
results, or the Direct Ptycho stream itself. A stream or result supplies its own sampling and a stream its frame names, so
`drift(stream)` needs no other argument. On a stream, `stream.align()` runs the same `drift` and also re-places the
stream's own combined image with it; call `drift(stream)` for the Drift alone.

## What the numbers mean

`found.displacement_A[f]` is `(row, col)` in Å: where frame `f` sees the specimen against frame 0. Frame `f`'s image is
frame 0's moved by it, `I_f(r) = I_0(r - d_f)`; `corrected` reads `I_f(r + d_f)` by an exact Fourier shift and keeps the
rows and columns inside every frame. `found.sigma_A[f]` is its one-sigma uncertainty from the least-squares residuals.

| attribute | meaning |
|---|---|
| `displacement_A` | (frames, 2) Å, frame 0 at zero |
| `sigma_A` | (frames, 2) Å, one sigma |
| `pairs` | every registered pair: shift, confidence, whether it was kept |
| `closure_A` | `d_ij + d_jk - d_ik` for each consecutive triple: near zero when the pairs agree |
| `frames` | the frame labels (a stream's names, else 0, 1, 2, ...) |

## Which registration

| call | when | how |
|---|---|---|
| `drift(phases)` | each frame images the specimen well (a bilayer at full dose) | every pair up to 3 frames apart by its own lattice-safe search, a confident chain outvoting a pair that took a wrong lattice image, solved by least squares |
| `running(phases)` | frames too sparse to place one against another | each frame against the sum of all the others in register; no pairs, so no σ or closure |

## Which reconstruction to register on

It does not matter. On a 40-frame experimental MoS2 bilayer series the drift measured on SSB, phase-only and amplitude + phase agree to
0.05 Å over 116 Å of drift, the size of the drift's own σ. Measure it on whichever you have (the stream gives SSB) and
correct any reconstruction of the same frames with it: `found.corrected(phase_only_stack)`. The frames must share frame
0's origin; a stack at another pixel size moves by its own pixels (`corrected(stack, sampling_A=...)`, read from a
stream or results when they carry it).
Evidence: [method comparison](../performance/ssb/reports/2026-10-24-drift-method-comparison.md).

## Advanced

`drift(..., frames=None, neighbours=3, band_A=(50.0, 0.67), reach_A=30.0, outlier_A=0.5)`: the frame labels (a
stream's names, else 0, 1, 2, ...), how many frames apart pairs are registered, the band of periods compared (Å), how
far a pair's search reaches (Å), and how far a pair may disagree with the confident chain before it is rejected (Å).
`running(..., passes=2)`: how many rounds each frame is re-placed against all the others.
`register(reference, moving, sampling_A=...)` registers one pair. `Drift.plot(others)` draws the path beside other
measures of it (an ADF drift, an earlier run).
