# SSB joint-fit benchmark, 2026-10-21

> **Historical:** describes the code as of 2026-10-21; its fits refined with the finite-difference Newton, since replaced by exact derivatives, so its fit timings are not current. See the [current fit methods](../../../api/ssb.md).

The reusable command is `quantem-gpu benchmark-ssb`. Its
[measurement protocol](../joint-fit.md) and repository
[agent skill](../../../../.agents/skills/gpu-benchmark/SKILL.md) keep subsequent
measurements comparable. This record qualifies the measurements below; it is
not a release announcement or a new optimization result.

Private experiment ID: `ssb-benchmark-release-20261021`. Raw JSON, frozen
reference phases, logs and specimen images remain in the private archive.

## Workload and provenance

- Complete 512 × 512 scan, 192 × 192 detector, resident `uint8` counts;
  8,807 bright-field pixels, no binning or subsampling.
- One seed (42), 200 trials followed by Newton. Estimated full-field rotation
  evaluates both branches: 400 completed trials. Regional fits hold that
  rotation and jointly fit aberrations, tilt and depth with 200 trials.
- Auto reconstruction is 3×, producing the full 1536 × 1536 phase. A regional
  image is cropped only after a full-scan reconstruction with its local fit.
- Six measurements in one process: first observation plus five warm samples.
  Storage caches were not controlled. Loading occurred once, outside fit timing.
- MPS: Apple M5 Max, Python 3.13.13, Torch 2.14.1, NumPy 2.5.4, MLX 0.32.4.
  GPU fallback and fast math were disabled; CPU thread limits were four.
- Fresh installed candidate wheels; no editable or `PYTHONPATH` source imports.
  GPU package source fingerprint:
  `59777fd00705552fe3c3bd6d0d2e61785d7261539b3d627c8d49cf3ef84ef5c7`.
  Local version labels were GPU `0.0.1rc17` and widget `0.0.1rc43`; labels
  alone do not identify these unpublished artifacts.

Region bounds use `(row_start, row_stop, col_start, col_stop)`, with exclusive
stops: 256² `(128, 384, 128, 384)`; 128² `(192, 320, 192, 320)`;
moved 128² `(224, 352, 224, 352)`; resized `(224, 351, 224, 352)`.
All calibration and fitted parameter values are retained in the private JSON.

## Installed MPS measurements

| Operation | First (s) | Warm median (s) | Warm p90 (s) |
|---|---:|---:|---:|
| Full-field fit, both rotation branches | 16.493 | 17.858 | 18.751 |
| Full-field Auto reconstruction | 0.122 | 0.132 | 0.136 |
| 256 × 256 region | 3.964 | 3.731 | 3.885 |
| 128 × 128 region | 1.597 | 1.729 | 1.839 |
| Moved 128 × 128 region | 1.358 | 1.464 | 1.885 |
| Resized 127 × 128 region | 1.450 | 1.496 | 1.606 |

Regional rows include preparation, joint fitting, full-scan reconstruction and
cropping. The full-field fit includes its native reconstruction/loss diagnostic;
Auto output is timed separately. Browser transfer and paint are excluded.
Load was 4.659 s, missing detector-calibration fitting 0.229 s, and session
construction had a warm median of 0.613 s. These separate boundaries must not
be added to a claimed warm slider rate.

Every measurement passed the declared `atol=rtol=1e-6` phase checks. Against the
frozen earlier MPS Auto phase, the maximum absolute difference was
`1.7881393432617188e-7` rad and RMS difference `1.709563258527498e-8` rad in
all six measurements. This is tolerance agreement, **not bitwise equality**.
The prior runtime used Torch 2.13.0, NumPy 2.4.6 and MLX 0.32.0; this comparison
does not isolate the effect of a runtime change.

## Installed widget acceptance on MPS

The browser test drew two regions, moved and resized the second, and verified
that the first region and full-field fit remained unchanged. Each regional
fit used one seed and all 200 trials plus Newton. It saved calibration and
restored the full-field fit in a second fresh kernel with new fitting disabled.
Both rendered screenshots were inspected. Hardware browser rendering and MPS
numerical fitting were checked separately.

| Browser interaction | Completion time (s) |
|---|---:|
| Draw first region | 2.111 |
| Draw second region | 1.602 |
| Move second region | 2.109 |
| Resize second region | 2.118 |

The retained regional fit times were 1.239 s and 1.333 s; the separate browser
run's full-field fit was 21.253 s. Interaction measurements include notebook
communication and waiting for the completed result. They are single
observations, not an FPS benchmark or a timing distribution.

## Limits and release state

The earlier one-off table (MPS full field 9.80 s, regions 1.99/0.88/0.97 s)
was not reproduced as a repeated distribution here. Regional coordinates and
runtime versions differ; this is not a matched before/after speedup claim.
The full-field timing difference remains unexplained by this measurement.

CUDA qualification is pending a free cooperative compute slot. A contended
allocation failure and two timed-out lock waits are excluded from results.
MPS evidence does not qualify CUDA. Dedicated hardware CI runners also need
registration before the new release gates can run. No RC was published.

The focused MPS regression set passed 132 tests on its first run and one on a
corrected retry; a CUDA-only test was skipped. The retry changed only its GPU
integer-sum reference to avoid an unavailable fused Torch MPS operation.
Widget source tests passed 692 cases on MPS; five Git-dependent checks passed
on the source host, and the notebook-export test passed after registering the
test environment's own kernel. Browser tests were run separately as described
above. Frontend tests passed 378 cases, with two skipped. Skips do not count as
backend acceptance.
