# SSB API

Primary imports:

```python
from quantem.gpu.ssb import (
    SSB,
    SSBEngine,
    defocus_sweep,
    ssb,
    ssb_fit_mps,
    ssb_preview_mps,
    ssb_series,
)
```

Use `ssb()` or `SSBEngine` for CUDA/reference workflows. Use `ssb_preview_mps`
and `ssb_fit_mps` for the Apple GPU path.

For Apple GPU live review, `ssb_preview_mps(...)` uses an exact BF-averaged
object-wave path by default when `compute_loss=False`:

```python
preview = ssb_preview_mps(
    loaded.data,
    voltage_kV=300,
    semiangle_mrad=21.4,
    scan_sampling_A=1.0,
    bf_intensity_threshold=0.0,
    bf_radius=53,
)
```

This is fast because the BF average is taken in Fourier space before one final
inverse FFT. It is not the same scientific quantity as
`phase_mode="mean"`, which computes the mean of per-BF phase images and
supports phase-variance loss. Requesting `compute_loss=True` uses the exact
`phase_mode="mean"` path automatically.

For full-BF Apple GPU review, the setup chunk and redraw chunk are tuned
separately. The default high-memory Mac path prepares Hermitian `G_qk` in
larger BF chunks for a short first-use wait, then redraws object-wave changes
with smaller BF chunks for lower interaction latency. Advanced users can
override these with `QUANTEM_MPS_SSB_OBJECT_CHUNK_BF` and
`QUANTEM_MPS_SSB_OBJECT_REDRAW_CHUNK_BF`.

The SSB API is intentionally pure compute. It should return arrays, metrics, and
metadata that widget/live callers can display or save.

Current performance checkpoint, 2026-07-18:

- CUDA real `512x512` field, full active BF (`8827` BF), Hermitian
  `G_qk=(8827,512,257)`: phase-only `31.10 ms` mean (`32.2 FPS`) and
  phase+loss `31.27 ms` mean (`32.0 FPS`) on GPU1.
- CUDA full calibration path on the same field:
  latest full-BF rerun loaded in `1.06 s`, constructed Hermitian `G_qk` in
  `0.265 s`, ran `optimize(n_trials=200)` in `7.39 s`, ran `refine()` in
  `1.14 s` / `36` evaluations, and produced final `result()` in `0.046 s`.
- CUDA synthetic `1024x1024`, full active BF-style (`8809` BF), Hermitian
  `G_qk=(8809,1024,513)`: split-512 exact phase/loss now measures
  `190-198 ms` mean (`~5 FPS`) on GPU1. This is faster than the old
  `382.24 ms` path, but it is not yet a real-time 10/30 FPS solution.
- MPS prepared full active BF-style matrix (`8809` BF), Hermitian `G_qk`:
  exact phase+loss is about `8.3 ms` at `128x128`, about `34-35 ms` at
  `256x256`, warm `~170 ms` at `512x512`, and warm `~0.8-1.0 s` at
  `1024x1024`. The 256 phase-only path reaches about `32.75 ms`; the 1024
  fused Metal path is about `2-2.6x` faster than the old generic MLX route but
  is still not live-interactive.
- MPS object-wave steering is a separate fast review quantity: it is useful
  for object-wave inspection, but it is not a parity claim for exact
  mean-phase or phase-variance loss.

Signoff expectations:

- Use real data, not only synthetic controls.
- Compare CUDA and MPS on the same BF-pixel selection.
- Include images and difference maps, not only scalar tables.
- Do not use object mode or any other fast review mode for exact phase/loss
  parity claims.
- Keep temporal/joint SSB experiments separate until the improvement metric is
  clear.
