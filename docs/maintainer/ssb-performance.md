# SSB performance notes

This page records the current native CUDA SSB live-redraw contract so future
work starts from measured behavior, not from memory.

## User-facing target

Microscopists need to steer aberration controls while viewing the same
full-BF reconstruction used for the final result. For native full-resolution
live display, the current targets are:

- `512x512`: at least 30 FPS (`<=33.3 ms/redraw`).
- `1024x1024`: at least 10 FPS (`<=100 ms/redraw`).

Do not claim these numbers from detector binning, scan cropping, fewer BF
pixels, or saved complex64 caches. Those are separate preview/export choices.

## Current CUDA/MPS checkpoint, 2026-07-17

This checkpoint excludes WebGPU by request. It uses full active BF evidence:
no scan crop, detector binning, BF subsampling, preview/settle split, or saved
derived float32/complex64 cache.

Microscopist workflow, real 512 field:

```text
source: private HDF5 master file
shape: (512, 512, 192, 192), uint16, 19.33 GB
BF policy: threshold=0.0, bf_radius=53, full active BF
active BF: 8827
resident G_qk: Hermitian complex64, (8827, 512, 257), 9.29 GB
```

CUDA GPU1 live-control timing, sustained real-data run:

| Quantity | Mean | p50 | p95 | FPS | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Phase-only | `31.10 ms` | `31.13 ms` | `31.20 ms` | `32.2` | Passes 30 FPS |
| Phase+loss | `31.27 ms` | `31.27 ms` | `31.37 ms` | `32.0` | Passes 30 FPS |

CUDA GPU1 calibration path, same data and BF policy:

| Stage | Wall time | Notes |
| --- | ---: | --- |
| HDF5 load | `1.19 s` | CUDA load of full `(512,512,192,192)` uint16 block |
| SSB construction / Gqk | `0.27 s` | Hermitian `G_qk`, `9.29 GB` |
| `optimize(n_trials=200)` | `7.39 s` | Full BF, no `bf_subsample`, latest rerun used `8793` active BF |
| `refine()` | `1.14 s` | `gpu-full-ifft-nmead`, `36` evaluations |
| Final `result()` | `0.046 s` | Final loss `0.0464598909` |
| Total through final result | `9.91 s` | End-to-end CUDA/Python compute path |

CUDA synthetic matrix on GPU1, Hermitian `G_qk`, `8809` BF:

| Scan | Object mean / FPS | Phase mean / FPS | Phase+loss mean / FPS |
| --- | ---: | ---: | ---: |
| `128x128` | `4.85 ms / 206.1` | `8.30 ms / 120.5` | `8.35 ms / 119.7` |
| `256x256` | `2.20 ms / 454.0` | `20.94 ms / 47.8` | `20.99 ms / 47.7` |
| `512x512` | `8.59 ms / 116.3` | `27.24 ms / 36.7` | `27.33 ms / 36.6` |
| `1024x1024` | `41.79 ms / 23.9` | `195.54 ms / 5.11` | `197.71 ms / 5.06` |

The latest CUDA 1024 exact phase/loss path uses a split-512 row and column IFFT
topology over transposed scratch. It keeps the same full active BF evidence,
Hermitian complex64 `G_qk`, float32 phase/loss arithmetic, scan size, and
objective definition. On the synthetic full active BF-style benchmark, this
moved phase+loss from `382.24 ms` (`2.62 FPS`) to `197.71 ms` (`5.06 FPS`)
while keeping the CuPy memory-pool footprint about `45.9 GB`. This is a real
exact-kernel speedup, but it still fails the 10 FPS and 30 FPS targets.

Nsight Compute on the current CUDA 1024 exact phase/loss path shows the row and
column FFT kernels are scheduler/shared-memory limited, not disk or DRAM
bandwidth limited:

| Kernel | Eligible warps/scheduler | No eligible cycles | Achieved occupancy | Memory throughput | Main stall |
| --- | ---: | ---: | ---: | ---: | --- |
| `ifft1024_rows_fused_pk_split512_t64_packed` | `0.36` | `74.9%` | `31.6%` | `520 GB/s` | `123` regs/thread, MIO/short scoreboard, shared-memory and memory-pipe pressure |
| `ifft1024_cols_accumulate_split512_t64` | `0.70` | `58.8%` | `32.8%` | `888 GB/s` | `121` regs/thread, MIO/short scoreboard, high memory-pipe use |

The accepted split-512 topology is the first larger CUDA 1024 phase/loss
breakthrough in this sequence. The next breakthrough must reduce the
register-heavy row/gamma stage or the amount of exact per-BF phase work; chunk
size and BF-group retuning were measured and stayed flat.

MPS real 512 timing, same full active BF selection:

| Quantity | Mean | p50 | p95 | FPS | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Object redraw | `22.25 ms` | `21.65 ms` | `23.44 ms` | `44.9` | Usable for object-wave steering |
| Phase-only | `168.84 ms` | `168.60 ms` | `171.66 ms` | `5.9` | Fails exact phase target |
| Phase+loss | `165.20 ms` | `165.57 ms` | `166.71 ms` | `6.1` | Fails exact loss target |

Parity gates for this checkpoint:

- CUDA: `tests/test_ssb_cuda_128.py` + `tests/test_ssb_batch_optuna.py`,
  `29 passed` on GPU1.
- MPS: `tests/test_ssb_mps_cuda_reference.py`, `14 passed, 2 skipped` on a
  Mac MPS machine.
  This includes the fused 128/256/512/1024 column phase/loss helpers and the
  fused 128/256/1024 dynamic row-IFFT helpers.

MPS scalar-loss reduction is scientifically valid, but it did not produce a
large wall-time win: avoiding a full phase-squared image write leaves the
row/column IFFT work dominant. The default exact phase/loss chunk is now
`3072` BF on 96 GB-class Macs, `1024` BF on 64 GB-class Macs, and `512` BF on
smaller Macs. The 96 GB Mac setting is a small warmed steady-state win, but it
is still not a real-time exact phase/loss breakthrough. The next MPS
breakthrough needs a different exact row/column FFT topology, not another
scalar-loss or chunk-size tweak.

Latest MPS exact-loss chunk repeat on the same real `512x512` field kept the
`3072` BF default: `3072` BF measured `165.47 ms` mean / `166.96 ms` p95,
while `1024` BF measured `169.03 ms` mean / `171.66 ms` p95.

## Exact object redraw path

`SSB.result()` displays the complex object wave and then exposes its phase.
For that object path, the inverse FFT can move outside the BF average:

```text
mean_bf(ifft2(corrected_bf)) == ifft2(mean_bf(corrected_bf))
```

The CUDA object redraw path now uses this identity for large native scans:

1. Compute the same per-BF `pk` correction as the full custom SSB path.
2. Sum corrected Fourier-domain terms in BF groups on the GPU.
3. Reduce those group sums to one corrected Fourier image.
4. Run one `ifft2` for the final object.

This changes the memory/computation topology only. It does not change the BF
selection, scan size, precision type, or object definition.

The path is intentionally separate from the phase-variance optimizer path.
`reconstruct()` and `reconstruct_with_loss()` average per-BF phase and still
need their own optimized kernels because `angle(mean(object))` is not the same
operation as `mean(angle(object))`.

## CUDA Hermitian G_qk storage

The CUDA object redraw path now uses a lower-memory resident `G_qk` layout by
default:

```python
ssb = SSB(...)                       # default: gqk_storage="herm"
```

`gqk_storage="herm"` stores only scan-frequency columns `0..N/2`. The CUDA
object Fourier-sum kernels mirror-conjugate the missing half-plane directly
when they fetch `G(q, k)`. The CUDA phase/loss row kernels also fetch the
Hermitian half-plane directly through `ld_gqk_maybe_herm`; the engine no
longer materializes transient full-plane `G_qk` chunks for those paths. This
relies on the exact Hermitian symmetry of the FFT of real virtual-BF images.
The object-wave definition, phase/loss definition, and BF selection are
unchanged; only the resident storage layout and fetch path change.

Persistent `gqk_storage="full"` has been removed from the public SSB runtime
path. Full-plane `G_qk` appears only in low-level parity tests as a canonical
expansion from the Hermitian half-plane; it is not a user-facing mode.

Resident `G_qk` memory becomes:

```text
full: num_bf * N * N         * sizeof(complex64)
herm: num_bf * N * (N/2 + 1) * sizeof(complex64)
```

For a microscopist this is useful when the goal is to fit aberrations and steer
the final object view without spending the persistent VRAM budget on redundant
Fourier columns. Phase mean, phase variance/loss, `optimize()`, `refine()`,
`grid_search()`, defocus sweeps, and higher-order aberration paths now keep the
same resident Hermitian storage and fetch missing columns on demand.

Implementation status from the 2026-07-17 pass:

- CUDA object kernels `128/256/512/1024` use Hermitian half-plane `G_qk` in the
  public SSB path; full-plane references are constructed only inside tests.
- CUDA phase/loss paths `128/256/512/1024` preserve Hermitian resident storage
  and fetch the missing half-plane directly inside the row kernels; no
  transient full-plane `G_qk` chunk is built for the current phase/loss paths.
- CUDA `512x512` phase-only redraw has a sum-only column accumulator so
  `reconstruct()` does not compute phase variance when `reconstruct_with_loss()`
  is not requested. The measured speed did not improve materially, which shows
  the remaining floor is FFT topology rather than the removed `sumsq` writes.
- `_extract_gqk(...)` builds the half-plane directly after the BF-stack FFT,
  avoiding a persistent full `G_qk` allocation.
- Parity tests compare default Hermitian end-to-end `SSB(...).result()` against
  explicit canonical full storage. A raw `cp.fft.fft2` redundant half-plane can
  differ from exact conjugate symmetry at the expected fp32 arithmetic-noise
  floor, so full storage is canonicalized from the half-plane rather than using
  that redundant noise as a separate reference.
- MPS fixed-preview and sparse-fit prepared paths now store the same Hermitian
  half-plane. The cached sparse objective Metal kernel fetches missing columns
  by mirror-conjugate symmetry; non-cached MLX paths expand only per chunk.

Synthetic storage benchmark on GPU1, `8809` BF pixels, object-redraw mode:

| Scan | Storage | Resident `G_qk` | Mean | p95 | FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `512x512` | full | `18.47 GB` | `15.65 ms` | `16.70 ms` | `63.9` |
| `512x512` | herm | `9.27 GB` | `14.96 ms` | `15.07 ms` | `66.8` |
| `1024x1024` | full | `73.90 GB` | `70.85 ms` | `74.41 ms` | `14.1` |
| `1024x1024` | herm | `37.02 GB` | `66.08 ms` | `67.99 ms` | `15.1` |

Interpretation: this pass is a memory-topology win with no observed object-
redraw penalty. It is not a 30 FPS breakthrough for `1024x1024`; reaching that
target on one GPU still needs a deeper FFT/reduction topology change.

Public constructor-to-result smoke profile on GPU1, synthetic
`(256, 256, 20, 20)` uint16 data with `47` BF pixels:

| Storage | Resident `G_qk` | Warm `result()` mean | Object parity vs full |
| --- | ---: | ---: | ---: |
| default `herm` | `12.42 MB` | `0.20 ms` | `p99.9 abs = 0.0` |
| explicit `full` | `24.64 MB` | `0.50 ms` | reference |

This is an end-to-end API check (`SSB(...) -> result()`), not only a raw kernel
probe. The exact-zero parity here comes from canonicalizing both storage modes
from the same Hermitian half-plane.

## Historical measured baseline

This subsection is retained as the pre-Hermitian-only CUDA benchmark history.
The later "Hermitian-only and MPS matrix follow-up" section is the current
source of truth for public runtime storage, MPS status, and post-removal test
results.

Hardware: RTX PRO 6000 Blackwell-class CUDA workstation.

Input: synthetic complex64 `G_qk`, fitted 192-pixel detector BF disk radius
`53 px`, `8809` BF pixels, native scan size, no crop, no binning.

Benchmark: `SSBEngine.reconstruct_object(C10, C12, phi12)` after cache warmup.

| Scan | Mean | p50 | p95 | FPS | VRAM pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128x128` | `0.80 ms` | `0.80 ms` | `0.81 ms` | `1247.9` | `2.3 GB` |
| `256x256` | `3.97 ms` | `3.37 ms` | `5.55 ms` | `251.6` | `9.4 GB` |
| `512x512` | `12.29 ms` | `12.30 ms` | `12.53 ms` | `81.4` | `19.1 GB` |
| `1024x1024` | `56.22 ms` | `55.54 ms` | `62.20 ms` | `17.8` | `76.2 GB` |

This meets the current live-object target for both sizes. Treat it as a kernel
microbenchmark, not as complete scientist-workflow signoff. Real-data HDF5
load, hot-pixel filtering, BF-mask setup, browser/widget interaction, and
display readback still need end-to-end checks for every public workflow.

The same run measured the existing phase-mean and phase+loss paths. Those
paths are different scientific quantities and remain the next optimization
target:

| Mode | `128x128` | `256x256` | `512x512` | `1024x1024` |
| --- | ---: | ---: | ---: | ---: |
| Object redraw | `0.80 ms / 1247.9 FPS` | `3.97 ms / 251.6 FPS` | `12.29 ms / 81.4 FPS` | `56.22 ms / 17.8 FPS` |
| Phase redraw | `10.05 ms / 99.5 FPS` | `22.49 ms / 44.5 FPS` | `72.39 ms / 13.8 FPS` | `342.71 ms / 2.9 FPS` |
| Phase+loss | `9.17 ms / 109.1 FPS` | `19.55 ms / 51.2 FPS` | `69.48 ms / 14.4 FPS` | `326.71 ms / 3.1 FPS` |

### Direct Hermitian phase/loss follow-up

The 2026-07-17 follow-up removed the transient full-plane `G_qk` chunk from
the CUDA phase/loss row kernels and routed `512x512` phase-only redraw through
the existing radix-8 column topology used by the variance kernel. Direct
Hermitian fetch alone was not a 2x breakthrough because `G_qk` fetch is not the
dominant cost; the radix-8 column path is the first real phase redraw speedup
from this pass.

Focused CUDA parity after the direct-fetch change:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q tests/test_ssb_cuda_128.py

24 passed
```

Synthetic `512x512`, `8809` BF timing on GPU1:

| Mode | Storage | Mean | p50 | FPS |
| --- | --- | ---: | ---: | ---: |
| Object redraw | herm | `16.49 ms` | `16.82 ms` | `60.6` |
| Phase redraw | full | `86.35 ms` | `87.34 ms` | `11.6` |
| Phase redraw | herm direct fetch only | `86.66 ms` | `87.20 ms` | `11.5` |
| Phase redraw | herm + radix-8 column | `73.24 ms` | `73.25 ms` | `13.7` |
| Phase+loss | full | `89.98 ms` | `90.60 ms` | `11.1` |
| Phase+loss | herm + radix-8 column | `73.57 ms` | `73.59 ms` | `13.6` |

GPU event profile for the `512x512`, `8809` BF Hermitian phase redraw:

| Component | Total |
| --- | ---: |
| `pk` update | `0.35 ms` |
| Row gamma + row IFFT | `36.91 ms` |
| Column IFFT + phase accumulation | `37.75 ms` |
| Partial-sum reduction | `0.80 ms` |
| Profiled GPU total | `75.81 ms` |

After the radix-8 column route, the same component probe measured:

| Component | Total |
| --- | ---: |
| `pk` update | `0.33 ms` |
| Row gamma + row IFFT | `37.59 ms` |
| Radix-8 column IFFT + phase accumulation | `24.90 ms` |
| Partial-sum reduction | `0.81 ms` |
| Profiled GPU total | `63.63 ms` |

Interpretation: direct Hermitian fetch is the right storage architecture, but
live exact phase redraw is still limited by doing all per-BF row/column IFFTs.
The radix-8 column path reduces the column phase accumulation cost enough for a
single-GPU `1.22x` phase-redraw speedup (`89 ms -> 73 ms`) and brings
phase+loss to the same range, but it is still not the `30 FPS` target.
cuFFT was checked as a topology baseline for a `1024`-BF `512x512` chunk:
`ifft2` alone took `8.91 ms` and `angle().sum(axis=0)` another `2.93 ms`,
where the custom row/column/phase path takes about `8.7 ms` for the same
chunk. A naive cuFFT replacement is therefore not the next breakthrough.

### 512 exact phase/loss GPU1 push

The 2026-07-17 GPU1 optimization pass targeted the exact full-BF
`512x512`, `8809` BF phase/loss path directly. The target was `30 FPS`, or
`33.3 ms` per exact redraw. No detector binning, scan cropping, BF reduction,
preview path, persistent derived float/complex cache, or multi-GPU work was
counted as a win.

Accepted kernel changes:

- Added a `64`-thread radix-8 row/gamma kernel for the `C10/C12/phi12`
  phase/loss hot path. This replaced the older `128`-thread radix-4 row kernel
  for that path.
- Changed the row staging layout to `[bf, col, row]`, so the column
  phase/loss kernel reads coalesced memory. This intentionally trades more
  expensive row writes for a much cheaper column pass.
- Updated the batch variance row staging layout to match the transposed column
  reader, preserving parity for batched optimizer candidates.
- Tested larger `512x512` column phase/loss BF groups as an intermediate
  partial-plane optimization, but the durable direct-accumulate path keeps the
  fixed 32-BF variance grouping. The row-variance kernel is specialized for
  32 BF pixels per group; changing only the wrapper group count under-counts
  BF evidence and is not valid.
- Relaxed the two 512 radix-8 hot kernels from `__launch_bounds__(64, 10)` to
  `__launch_bounds__(64, 8)`, which gave a small scheduling win without parity
  changes.
- Added a 512 direct-accumulate path where the column phase/loss kernel
  atomically accumulates into the final phase planes. This removes the
  per-chunk partial-plane reduction launches; atomic cost is lower than the
  removed launch/reduction overhead at this size.
- Added an exact safe-inside aperture branch to `compute_geometry()`. For the
  Samsung-like `512`, `8809` BF benchmark, both shifted apertures are exactly
  `1.0` for `99.9995%` of points, so the row kernel now avoids the soft-edge
  aperture `sqrt/div` path except at the edge while preserving parity.
- Enabled CUDA fast math for these RawModules, switched global-load cache
  policy from `dlcm=cg` to `dlcm=ca`, changed the 512 column phase/loss loop
  from unroll `8` to unroll `2`, and retuned the 512 row/gamma launch bound to
  `__launch_bounds__(64, 10)` with the column kernel staying at
  `__launch_bounds__(64, 8)`.
- Replaced the hot 512 column `atan2f` calls with a degree-6 polynomial
  `atan2` helper. Full CUDA parity tests still pass; this is a small
  column-side win after `--use_fast_math`, not the main breakthrough.

Steady-state synthetic `512x512`, `8809` BF timing on GPU1:

| Mode | Before this pass | After radix-8 row | After transposed staging | After 64-BF groups | FPS after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase redraw | `70.57 ms` | `58.10 ms` | `53.46 ms` | `52.27 ms` | `19.1` |
| Phase+loss | `69.26 ms` | `58.09 ms` | `52.98 ms` | `52.36 ms` | `19.1` |

Component timing for phase redraw after the accepted changes:

| Component | Total |
| --- | ---: |
| `pk` update | `0.02 ms` |
| Row gamma + row IFFT + transposed write | `33.79 ms` |
| Column IFFT + phase accumulation | `14.40 ms` |
| Partial-sum reduction | `0.75 ms` |
| Profiled GPU total | `48.99 ms` |

Longer worker measurements are slightly slower than the isolated component
probe because they include the full `reconstruct()` / `reconstruct_with_loss()`
loop overhead and steady GPU clocks:

```text
phase: mean 53.46 ms, p50 53.74 ms, p95 53.91 ms, 18.7 FPS
loss:  mean 52.98 ms, p50 53.29 ms, p95 53.43 ms, 18.9 FPS
with 64-BF column groups:
phase: mean 52.63 ms, p50 53.12 ms, p95 53.31 ms, 19.0 FPS
loss:  mean 52.83 ms, p50 53.42 ms, p95 53.61 ms, 18.9 FPS
with 64-BF groups and relaxed launch bounds:
phase: mean 52.45 ms, p50 52.97 ms, p95 53.09 ms, 19.1 FPS
loss:  mean 52.67 ms, p50 53.31 ms, p95 53.44 ms, 19.0 FPS
with direct accumulation:
phase: mean 52.27 ms, p50 52.78 ms, p95 52.90 ms, 19.1 FPS
loss:  mean 52.36 ms, p50 52.97 ms, p95 53.11 ms, 19.1 FPS
with aperture shortcut, fast math/cache retune, column unroll 2, row launch
bound 10, and polynomial atan:
phase: mean 45.18 ms, p50 45.33 ms, p95 45.47 ms, 22.1 FPS
loss:  mean 45.11 ms, p50 45.32 ms, p95 45.42 ms, 22.2 FPS
```

Final component timing for the accepted 2026-07-17 incremental pass:

| Component | p50 total |
| --- | ---: |
| `pk` update | `0.015 ms` |
| Row gamma + row IFFT + transposed write | `29.99 ms` |
| Column IFFT + phase/loss accumulation | `13.71 ms` |
| Partial/direct final accumulation overhead | `0.59 ms` |
| Profiled GPU total | `44.39 ms` |

### 512 paired-BF phase/loss follow-up

The next GPU1 pass paired bright-field pixels at `+k` and `-k` for the
`512x512` C10/C12 exact phase/loss path. For the even C10/C12 probe,
`P(-k) == P(+k)` under the paired BF map, so the row kernel can share the
shifted probe geometry, `sincos`, and gamma normalization for the pair while
still applying each BF pixel's own `G_qk` evidence. This preserves the same
BF disk, scan size, and phase/loss definition; no preview path, binning,
cropping, saved `g_bf` cache, or multi-GPU work is counted.

Focused parity during this pass:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q \
  tests/test_ssb_cuda_128.py -k 'engine_matches_explicit or phase_loss'

6 passed, 18 deselected
```

Sustained synthetic `512x512`, `8809` BF timing on GPU1 after the paired path:

| Mode | Mean | p50 | p95 | FPS |
| --- | ---: | ---: | ---: | ---: |
| Phase redraw | `43.30 ms` | `43.76 ms` | `43.87 ms` | `23.1` |
| Phase+loss | `43.30 ms` | `43.83 ms` | `43.93 ms` | `23.1` |

The first few samples in a run can report `37-38 ms`, but sustained samples
settle around `43-44 ms`. During the same run, `nvidia-smi dmon` showed GPU1
at `100%` SM, `83-85%` memory activity, about `280 W`, and about `1550 MHz`
graphics clock under the `300 W` board power limit. An unprivileged attempt
to raise GPU1 to its reported `325 W` max was rejected by the driver. GPU0 was
also checked as a single-GPU comparison but was slower under current machine
load (`p50 ~47.8 ms`), so GPU1 remains the cleaner benchmark device.

Current component split for the paired full-storage path:

| Component | p50 total |
| --- | ---: |
| `pk` update | `0.015 ms` |
| Row gamma + row IFFT + transposed write | `28.55 ms` |
| Column IFFT + phase/loss accumulation | `13.69 ms` |
| Singleton/final overhead | `0.18 ms` |
| Profiled GPU total | `42.45 ms` |

Follow-up on GPU1 kept two exact-path micro-improvements for the `512x512`,
`8809` BF C10/C12 phase/loss path:

- The paired row kernel now processes four rows per block, reducing row-kernel
  block scheduling pressure while preserving the same `+k/-k` arithmetic.
- The C10/C12 chunked core can use a memory-aware transient staging buffer.
  On the 96 GB GPU1 test run it staged the full `8809 x 512 x 512` complex64
  intermediate in VRAM (`~37.5 GB` CuPy pool including source `G_qk`) and
  reduced row/column launch count. This is not a saved cache and does not
  change BF selection, scan size, precision, or phase/loss definition.
- The paired C10/C12 row helper now evaluates the quadratic phase directly
  from `r^2`, `dx^2 - dy^2`, and `2dxdy` instead of forming
  `cos(2phi)`/`sin(2phi)` through a division. This is restricted to the
  `512x512` paired C10/C12 hot path; broader polar/aberration paths still use
  the shared geometry helper.
- The paired row kernel now stages each 4-row `+k/-k` output tile in shared
  memory and writes row-contiguous groups to the transposed intermediate. This
  keeps the column reader's fast layout while reducing excessive global store
  sectors from the row stage.

Sustained timing after those changes:

| Mode | Mean | p50 | p95 | FPS |
| --- | ---: | ---: | ---: | ---: |
| Phase redraw | `39.26 ms` | `39.40 ms` | `39.46 ms` | `25.5` |
| Phase+loss | `39.28 ms` | `39.44 ms` | `39.54 ms` | `25.5` |

This is a small checkpoint, not the `30 FPS` target. The exact path still
misses the `33.3 ms` frame budget by about `6.1 ms` p50.

Component split for the full-staging four-row path:

| Component | p50 total |
| --- | ---: |
| `pk` update | `0.012 ms` |
| Row gamma + row IFFT + coalesced transposed write | `25.43 ms` |
| Column IFFT + phase/loss accumulation | `13.17 ms` |
| Singleton/final overhead | `0.11 ms` |
| Profiled GPU total | `~38.7 ms` |

### 512 real subpixel-BF dual path

The real Samsung `512x512` central field has a subpixel fitted BF center. Under
the exact integer detector-pixel mirror test, that means there are no usable
`+k/-k` pairs:

```text
source: /home/owner/ssd/data/samsung/logic_pmos_1p3Mx_30pA_1mrad_5um_17mradtilt/maped/logic_pmos_1p3Mx_30pA_1mrad_5um_17mradtilt_0.0x_0.0y_20260129_15-41-52_master.h5
shape: (512, 512, 192, 192), uint16, 19.33 GB
BF policy: bf_radius=53, threshold=0.0
active BF: 8822
exact symmetry pairs: 0
```

For that scientist workflow, the accepted exact optimization is an arbitrary
dual-BF row kernel. It pairs the remaining singleton BF pixels two at a time
for launch/staging efficiency, but computes each BF pixel's own `kx/ky`, probe
correction, `G_qk` fetch, inverse FFT, phase, and loss contribution. It is not
a symmetry approximation and does not reduce the BF disk, scan size, detector
sampling, or precision.

Focused parity now includes this condition directly:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q tests/test_ssb_cuda_128.py

25 passed
```

The new regression test constructs a `512x512` subpixel-BF center with zero
exact symmetry pairs, asserts that the dual path is used, and compares
`reconstruct_with_loss()` against an explicit chunked CuPy reference.

Real Samsung GPU1 timing after the dual-BF path, plus the Hermitian-specialized
dual row fetch:

| Mode | Storage | Active BF | Mean | p50 | p95 | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Phase+loss | herm | `8822` | `35.41 ms` | `35.40 ms` | `35.64 ms` | `28.2` |

Follow-up scalar-loss checkpoint: the exact C10/C12 loss path now keeps the
mean phase image as before but accumulates the phase-squared term into one
scalar for the no-pair dual path. This matches the optimizer objective,
because the loss only needs the global mean of `phase^2`; it avoids writing and
clearing a full per-pixel variance plane for the hot real Samsung condition.
Focused CUDA parity still passes:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/owner/repos/quantem.gpu/src \
  python -m pytest -q tests/test_ssb_cuda_128.py

25 passed
```

Sustained real Samsung GPU1 timing after the scalar-loss path:

| Mode | Storage | Active BF | Mean | p50 | p95 | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Phase+loss | herm | `8822` | `34.97 ms` | `34.95 ms` | `35.47 ms` | `28.6` |

This is a valid incremental win, but not a signoff. The exact full-BF 512
phase/loss path still misses the `30 FPS` frame budget (`33.3 ms`) by about
`1.6-1.7 ms` on GPU1.

Component timing for the same no-pair condition:

| Component | p50 total |
| --- | ---: |
| `pk` update | `0.012 ms` |
| Dual row gamma + row IFFT + transposed write | `~21.3 ms` |
| Column IFFT + phase/loss accumulation | `~13.5 ms` |
| Final mean/loss bookkeeping | `<0.2 ms` |

This is real progress for the microscopist: the exact full-BF 512 phase/loss
view is now around `28 FPS` on the real central Samsung field. It is still a
fail against the declared `30 FPS` target because the frame budget is
`33.3 ms`, leaving a sustained `~2.1 ms` gap.

Hardware note from the same 2026-07-17 push: sustained GPU1 runs sit at the
`300 W` power limit with `100%` SM utilization, memory utilization around
`80-83%`, and SM clocks around `1.41-1.45 GHz`. A one-off comparison on GPU0
(`600 W` limit, but shared with display and another process) crossed the target
at p50 (`32.98 ms`) while mean/p95 were noisy (`35.67 ms` mean,
`47.51 ms` p95). Treat GPU0 as evidence that the current kernel is near the
target on a higher-power card, not as a clean signoff. The GPU1 goal still
needs a code-side `~2-3 ms` sustained reduction or a deliberate power-limit
change by an operator.

Rejected follow-ups from this subpixel-BF pass:

| Candidate | Result | Decision |
| --- | --- | --- |
| Full-plane resident `G_qk` instead of Hermitian fetch | Real Samsung p50 regressed to about `37.8 ms` and doubled resident `G_qk` from `9.29 GB` to `18.50 GB`. | Rejected. |
| Dual row launch bound relaxed from `__launch_bounds__(256, 4)` to `256,2` | Parity passed, but real Samsung timing regressed to about `35.5 ms` p50 in short runs. | Reverted. |
| Dual row blocks reduced from 4 rows/block to 2 rows/block | Parity passed, but real Samsung timing regressed to about `36.3 ms` p50. | Reverted. |
| Precomputed row/q/k term helper for the dual row kernel | Parity passed, but register pressure made real timing worse (`~36.6 ms` p50). | Reverted. |
| Wrapper-only `_colvar_group` change from 32 to 64/128 | Initially looked faster, but it was invalid because `ifft512_rows_var_radix8_t64` hard-codes 32 BF pixels per group. A dynamic-k attempt hit illegal memory access at full BF. | Reverted; do not repeat without a separate parity-tested fixed-size kernel. |
| Column launch-bound tightening (`64,10`, `64,9`, `64,12`) | Full parity passed for the tested bounds. `64,10` produced no sustained real-data win (`36.17 ms` p50 before the Hermitian branch), and `64,12` regressed column p50 to about `14.4 ms`. | Rejected: register pressure/spills outweighed the occupancy attempt. |
| Hermitian row offset precompute | Full parity passed, but sustained real-data timing stayed about `35.43 ms` p50 versus `35.40 ms` for the simpler Hermitian branch. | Rejected: extra live offsets did not beat the compiler's simpler inline path. |
| Reusing loaded `qy` values across A/B dual gamma calls | Full parity passed, but component p50 did not improve and added register-pressure risk. | Rejected: the compiler/read-only cache already handles this cheaply enough. |
| Runtime shared-memory carveout `100` on the no-pair row/column kernels | Scratch component timing left row p50 about `21.3 ms` and combined dual p50 about `34.7 ms`, not a sustained breakthrough. | Rejected: larger carveout did not turn the shared-memory occupancy warning into real FPS. |
| One-BF/four-row index row kernel | Existing parity still passed, but forcing all BF pixels through the experimental topology gave row p50 about `24.2 ms` and row+column p50 about `37.6 ms`. | Rejected: doubling the independent BF blocks did not compensate for lost dual-BF staging efficiency. |
| Closed-form radix-8 source indices instead of `octal_reverse_512(tid*8+s)` | Full focused CUDA parity passed. A 240-step real Samsung run briefly measured `34.90 ms` p50, but a 600-step sustained run settled at `35.53 ms` p50, matching or slightly regressing the accepted `35.50 ms` baseline. | Rejected: not a robust wall-time win for a microscopist dragging controls. |
| Exact inside-aperture phase-identity branch for normalized gamma | Full focused CUDA parity passed, but real Samsung loss regressed to `37.43 ms` p50 (`26.7 FPS`). | Rejected: the added branch, `chi_k` load, and changed special-function mix cost more than the removed normalization. |
| Degree-5 column `atan2` polynomial | Full focused CUDA parity passed and a 240-step run measured `35.22 ms` p50, but the 600-step sustained run regressed to `35.79 ms` p50. | Rejected: removing one FMA was not a durable column-side speedup. |
| Column loop unroll reduced from `2` to `1` | Full focused CUDA parity passed. A 240-step run stayed near baseline (`35.53 ms` p50), and the 600-step sustained run regressed to `36.27 ms` p50. | Rejected: lower unroll did not overcome the register/L1TEX floor. |
| Column launch bound relaxed from `__launch_bounds__(64, 8)` to `64,4` | Full focused CUDA parity passed, but the 240-step real Samsung run regressed to `35.79 ms` p50. | Rejected: giving the compiler more register freedom did not beat the occupancy loss. |
| Module `--maxrregcount` reduced from `96` to `80` | Full focused CUDA parity passed and a 240-step run improved to `35.12 ms` p50, but the 600-step sustained run settled at `35.72 ms` p50. | Rejected: lower register cap was not a durable sustained win. |
| Column phase/loss group reduced from `32` BF to `16` BF | Full focused CUDA parity passed. A 240-step run stayed near baseline (`35.52 ms` p50), and the 600-step sustained run regressed to `36.25 ms` p50. | Rejected: extra scheduler parallelism did not offset the doubled group/atomic overhead. |
| Fixed `64`-BF column phase/loss kernel after scalar-loss path | Full focused CUDA parity passed. A 240-step real Samsung run improved to `34.62 ms` mean/p50, but the 600-step sustained run regressed to `35.25 ms` mean and `35.28 ms` p50. | Rejected: halving phase-sum atomics was not durable under sustained GPU1 power/occupancy behavior. |
| In-place `phase_sum` normalization after scalar-loss path | Full focused CUDA parity passed. A 240-step run had `34.82 ms` p50 but worse mean/p95, and the 600-step sustained run regressed to `35.61 ms` mean and `35.62 ms` p50. | Rejected: allocation avoidance in finalization did not beat the existing CuPy expression path under sustained timing. |
| Reusing `_sum_buffer` for phase accumulation after scalar-loss path | Full focused CUDA parity passed. A 240-step run had similar p50 but worse mean/p95, and the 600-step sustained run regressed to `35.63 ms` mean and `35.64 ms` p50. | Rejected: CuPy's fresh zeroed plane path is more stable than reusing and filling the internal accumulation buffer here. |
| Dedicated scalar-loss column kernel without per-pixel `sumsq0..7` accumulators | Full focused CUDA parity passed, but the 240-step real Samsung run regressed to `35.23 ms` p50. NCU showed registers dropped only from `115` to `108`, theoretical occupancy stayed `33.3%`, and L1TEX/eligible-warp stalls were unchanged. | Rejected: removing per-pixel sumsq registers was not enough to change the column kernel's occupancy class or latency floor. |
| Dual row `float4` row-IFFT packing for arbitrary no-pair BF pixels | Full focused CUDA parity passed after isolating the probe to the real Samsung no-pair dual row kernel. The 240-step real Samsung run regressed to `36.18 ms` mean and `36.17 ms` p50 (`27.6 FPS`) versus the accepted scalar-loss baseline of about `34.97 ms` mean / `34.95 ms` p50. | Rejected: packing the A/B row FFTs into `float4` reduced no useful sustained wall time; the extra register/instruction pressure outweighed fewer shared-memory slots and helper calls. |
| Sequential-index mode for the arbitrary no-pair dual row kernel | The real Samsung full-BF cache has `8822` sequential singleton BF pixels and dual pairs `[0,1], [2,3], ...`, so a guarded mode replaced `pair_a/pair_b` loads with affine `idx_a = base + 2*pair`. Full focused CUDA parity passed, but the 240-step real Samsung run regressed to `35.31 ms` mean and `35.31 ms` p50. | Rejected: pair-index gathers are not a meaningful part of the remaining row/column floor. |
| Row-aware singleton pairing for the arbitrary no-pair dual row kernel | Reordered singleton BF pairs to maximize same-detector-row pairs (`4403/4411` versus `4351/4411`) while keeping every BF pixel and no tail. Full focused CUDA parity passed, but the 240-step real Samsung run regressed to `35.24 ms` mean and `35.26 ms` p50. | Rejected: pairing locality alone does not remove enough row-kernel work without a different same-`ky` kernel. |
| Same-`ky` dual gamma helper inside the arbitrary no-pair row kernel | Added an exact branch for pairs with the same detector row so A/B share `dy_m` and `dy_p` while still computing each BF's own `kx`, `pk`, `G_qk`, IFFT, phase, and loss. Full focused CUDA parity passed, but the 240-step real Samsung run regressed to `35.40 ms` mean and `35.40 ms` p50. | Rejected: the extra branch/code pressure outweighed the small duplicated-geometry savings. |
| Warp-shuffle scalar-loss block reduction in the column kernel | Replaced the scalar `phase^2` shared-memory tree reduction with warp shuffles plus one shared handoff. Full focused CUDA parity passed, but the 240-step real Samsung run regressed to `35.46 ms` mean and `35.47 ms` p50. | Rejected: the block reduction barriers are not the dominant column cost; the register/L1TEX FFT path remains the floor. |
| Dual row transposed copy-out changed from `float2` stores to adjacent-row `float4` stores | Full focused CUDA parity passed and a 240-step run measured `35.34 ms` p50, but the 600-step sustained run regressed to `36.06 ms` p50. | Rejected: fewer store instructions did not improve the sustained power-capped row path. |
| Dual row Hermitian path forced at compile time by replacing the `gqk_cols == 257` branch with `true` | Exploratory Hermitian-only timing regressed to `35.79 ms` p50 in a 240-step real Samsung run. | Rejected: a separate Hermitian-only row kernel is not justified by this branch-cost probe. |

Nsight Compute on the accepted dual row kernel:

- Grid `(1, 128, 4404)`, block `(64, 4, 1)`.
- `64` registers/thread and `32.77 KB` static shared memory/block.
- `50.0%` theoretical occupancy, `49.2%` achieved occupancy.
- About `1.06` eligible warps/scheduler, with `~52%` cycles having no
  eligible warp.
- About `972 GB/s` memory throughput, `57.5%` memory busy, `64.6%` L2 hit
  rate, and `37.4%` L1/TEX hit rate.

Interpretation: the no-pair real-data path is no longer dominated by HDF5
load, BF selection, or Python. The remaining floor is the row/column FFT
topology: row is shared-memory/scheduler limited, and column still spends
about `13-14 ms` doing exact per-BF phase/loss accumulation. The next
breakthrough should target one of these structural costs, not another storage
flag.

Nsight Compute on the four-row paired row kernel with
`__launch_bounds__(256, 3)`:

- `77` registers/thread, no local/shared spills.
- `50.0%` theoretical occupancy, `48.7%` achieved occupancy.
- `0.80` eligible warps/scheduler and `61.7%` cycles with no eligible warp.
- `993 GB/s` memory throughput, `57.9%` memory busy, `65.1%` L2 hit rate.
- Main stall: still MIO/shared-memory pressure, but the coalesced tile write
  roughly halves warp cycles per issued instruction (`27.9 -> 14.5`) versus
  the prior direct transposed stores.

Nsight Compute on the column phase/loss kernel:

- `115` registers/thread, no local/shared spills.
- `33.3%` theoretical occupancy, `33.1%` achieved occupancy.
- `0.57` eligible warps/scheduler and `68.4%` cycles with no eligible warp.
- `846 GB/s` memory throughput but low L2 hit rate, with L1TEX scoreboard
  stalls. Launch-bound forcing to reduce registers regressed timing.

Follow-up full-plane profiling on GPU1, matching the synthetic FPS benchmark
storage mode, confirmed the same limit after the coalesced row-store commit:

- Paired row/gamma kernel: `77` registers/thread, no spills, `50.0%`
  theoretical occupancy, `49.2%` achieved occupancy, `0.57` eligible
  warps/scheduler, about `1.09 TB/s` memory throughput, and source counters
  reported about `50%` excessive shared-memory wavefronts.
- Column phase/loss kernel: `115` registers/thread, no spills, `33.3%`
  theoretical occupancy, `33.1%` achieved occupancy, `0.57` eligible
  warps/scheduler, about `847 GB/s` memory throughput, near-zero L1 hit rate,
  and source counters reported L1TEX scoreboard stalls plus about `33%`
  excessive shared-memory wavefronts.

Hardware note from the same continuation: GPU1 was power-capped at `300 W`.
During a sustained 512 loss benchmark it held `100%` GPU utilization with
throttle reason `0x4` and SM clocks around `1.59-1.61 GHz`. The driver reports
`325 W` as the max power limit, but raising GPU1 to `325 W` failed with
insufficient permissions. This is real clock headroom, but the available
`300 -> 325 W` increase is too small to explain the full `39.5 -> 33.3 ms`
target gap by itself.

Nsight Compute on the paired row kernel:

- `94` registers/thread, no local/shared spills.
- `41.7%` theoretical occupancy, `38.9%` achieved occupancy.
- `0.37` eligible warps/scheduler and `79%` cycles with no eligible warp.
- `577 GB/s` memory throughput, `91%` memory busy, `85.8%` L2 hit rate.
- The primary warning is still uncoalesced global traffic: about `63%`
  excessive global sectors, plus about `33%` excessive shared wavefronts.

Interpretation: the paired BF symmetry is a real exact-path improvement, but
it does not reach the `33.3 ms` / `30 FPS` target. The remaining bottleneck is
the same topology issue: the row stage writes a full transposed complex
intermediate so the column stage can read coalesced data. Reaching 30 FPS
requires a deeper row/column topology change that reduces this intermediate
traffic or coalesces both sides without changing the per-BF phase/loss math.

Final Nsight samples on a `1024`-BF chunk still show the structural limit:

- Row/gamma kernel: `93` registers/thread, no spills, `38.2%` achieved
  occupancy, `0.53` eligible warps/scheduler, high memory-pipe/MIO pressure,
  and about `550 GB/s` memory throughput. The row kernel is still the largest
  single cost.
- Column phase/loss kernel: `115` registers/thread, no spills, `32.1%`
  achieved occupancy, `0.59` eligible warps/scheduler, L1TEX scoreboard
  stalls, and about `845 GB/s` memory throughput.
- A scratch gamma-bypass lower-bound experiment, which is not scientifically
  valid and was reverted, still landed around `36 ms` component time. That
  means gamma-only shortcuts cannot reach the `33.3 ms` budget by themselves;
  the row/column staging and column phase accumulation topology also has to
  change.

Rejected candidates from the same pass:

| Candidate | Result | Decision |
| --- | --- | --- |
| Phase-only radix-8 column variant | Parity passed, but timing stayed around `69.5 ms` before the row/transposed breakthrough. | Reverted. |
| Resident aperture-pair cache | Parity passed, but full-BF frame time regressed to about `178 ms` because the row kernel streamed two huge aperture arrays every redraw. | Reverted. |
| Shared-memory padding in row/column radix-8 kernels | Parity passed, but short-run timing was neutral (`~50.4 ms`) and component timing did not improve. | Reverted. |
| Legacy radix-4 column accumulator | Column pass measured about `43 ms`, compared with about `14 ms` for the transposed radix-8 column path. | Rejected. |
| Batch throughput for optimizer candidates | Batch sizes `4/8/16` stayed near `19.2` exact eval/s (`~52 ms/eval`). | Not a 30 FPS breakthrough. |
| Replacing `sqrtf`/division geometry with `rsqrtf` geometry in the exact C10/C12 helper | Focused parity failed the 1024 explicit-reference gate (`p99.9` phase error `4.35e-4` versus `3e-4`). | Reverted. |
| Lowering global CUDA `--maxrregcount` from `96` to `64`/`48` | Component timings were only noise-level better, and sequential full-loop timing stayed around `53 ms`. | Reverted. |
| Row-major row output followed by an out-of-place tiled GPU transpose | Parity passed, but `512` phase redraw regressed to `71.4 ms` (`14 FPS`) because the added full-stack transpose outweighed the row-store savings. | Reverted. |
| 512 phase-only skip of `sumsq` accumulation | Parity passed, but phase-only timing barely moved and loss did not improve. | Reverted as extra complexity without a target-path win. |
| 512 column `__launch_bounds__(64, 10)` under the final compiler settings | Parity passed, but steady p50/p95 regressed relative to column launch bound `8`. | Reverted. |
| 512 row `__launch_bounds__(64, 12)` under the final compiler settings | Parity passed, but steady p50/p95 regressed relative to row launch bound `10`. | Reverted. |
| Computing 8 rows per block and writing transposed tiles directly | Parity passed, but `512` phase redraw regressed to `60.1 ms` (`16.6 FPS`) from lower occupancy/shared-memory cost. | Reverted. |
| Computing 4 rows per block and writing transposed tiles directly | Parity passed, but `512` phase redraw still regressed to `55.7 ms` (`18.0 FPS`). | Reverted. |
| Computing 2 rows per block and writing transposed tiles directly | Parity passed, but `512` phase redraw regressed to `54.5 ms` (`18.4 FPS`). | Reverted. |
| Skipping `sincos` when shifted apertures are exactly zero | Parity passed, but `512` phase redraw stayed around `53.1 ms`; branch/control-flow cost offset the skipped work. | Reverted. |
| Raising the exact phase/loss chunk cap from `2 GB` to `4 GB` | Parity passed, but phase/loss timing stayed around `52.7-52.9 ms` while using more transient memory. | Reverted. |
| Raising the column phase/loss BF group from `64` to `128` | Parity passed, but phase/loss timing regressed slightly to `52.8-53.0 ms`. | Reverted. |
| Relaxing 512 radix-8 launch bounds further from `8` to `6` blocks | Parity passed, but phase timing regressed to `52.6 ms` from the `52.45 ms` launch-bounds-8 result. | Reverted. |
| Raising the direct-accumulate BF group from `64` to `128` | Parity passed, but phase timing regressed to `52.47 ms` from the `52.27 ms` direct 64-BF result. | Reverted. |
| Paired row FFT helper applying `+k` and `-k` simultaneously | Parity passed, but sustained p50 regressed to `43.1 ms`; higher register pressure erased the saved barrier/twiddle work. | Reverted. |
| Exact phase/loss staging chunk raised from `2 GB` to `4 GB` with paired rows | Parity passed, but timing stayed around `42.9 ms` while using more transient VRAM. | Reverted. |
| Global CUDA `--maxrregcount=80` with paired rows | Parity passed, but p50 stayed around `42.7 ms`; this broad compile knob was not worth the risk. | Reverted. |
| Paired row launch bound `__launch_bounds__(64, 12)` | Parity passed, but p50 regressed to about `42.9 ms`; `64,10` remains better. | Reverted. |
| Column BF group `128` after the paired row change | Parity passed, but sustained p50 stayed around `43.9 ms` and mean worsened slightly. | Reverted. |
| Column BF group `16` after the paired row change | Parity passed, but sustained p50 regressed to about `43.8 ms`; it added more atomic/group overhead. | Reverted. |
| Column BF group `48` after the paired row change | Parity passed, but sustained p50 regressed to about `43.9 ms`; `32` was the best measured group in this pass. | Reverted. |
| Column BF group `64` after the coalesced paired-row write | Parity passed, but exact loss p50 regressed to `39.56 ms`; halving group/atomic count reduced wavefront parallelism too much. | Reverted. |
| Partial-plane reduction instead of direct atomics for paired chunks | Scratch component timing was slower (`pair_col` p50 about `13.3 ms` plus reduction) than direct accumulation. | Rejected. |
| Two-lane column phase/loss block `(64,2,1)` | Parity passed, but sustained p50 stayed about `43.8 ms`; extra shared-memory reduction and lower occupancy offset the halved BF-loop iterations. | Reverted. |
| Direct full-plane `G_qk` loads in the paired row kernel | Full-storage timing regressed to p50 `44.0 ms`; the Hermitian-capable helper branch is not the row bottleneck. | Reverted. |
| `16x16` tiled intermediate layout balancing row writes and column reads | Parity passed, but p50 regressed to `50.5 ms`; improved row-store locality was outweighed by worse column-load locality. | Reverted. |
| Eight rows per paired row block using dynamic 64 KB shared memory | Parity passed, but p50 regressed to `46.0 ms`; lower occupancy/shared-memory pressure outweighed lower block count. | Reverted. |
| Packed `float4` helper transforming the `+k/-k` row FFTs together | Parity passed, but p50 regressed to `43.7 ms`; extra register and shuffle pressure outweighed saved barriers. | Reverted. |
| Column launch bound `__launch_bounds__(64, 12)` | Parity passed, but p50 regressed to `44.25 ms`; forcing more blocks over-constrained the compiler. | Reverted. |
| Column launch bound `__launch_bounds__(64, 10)` | Parity passed, but p50 regressed to `43.21 ms`; the original `64,8` launch bound remains best for the column kernel. | Reverted. |
| Paired row launch bound relaxed from `__launch_bounds__(256, 3)` to `256,2` | Parity passed, but p50 regressed to `44.08 ms`; lower occupancy outweighed any extra compiler freedom. | Reverted. |
| Paired row blocks reduced from 4 rows/block to 2 rows/block | Parity passed, but p50 regressed to `43.11 ms`; lower shared memory per block did not hide the row-stage stalls. | Reverted. |
| Paired row blocks reduced to 1 row/block | Parity passed, but p50 regressed to `43.35 ms`; more independent blocks added overhead without enough latency hiding. | Reverted. |
| Bit-reversed transient row layout plus contiguous column loads | Parity passed after fixing the direct/partial address mode, but the split-kernel version stayed around p50 `42.8 ms` and the branch version raised column registers from `115` to `127` for only a noise-level win. | Reverted. |
| Contiguous `G_qk` row-load microscope for a hypothetical column-permuted storage layout | Synthetic constant-`G_qk` throughput probe regressed to p50 `47.8 ms`; row `G_qk` column order is not the current breakthrough. | Reverted. |
| CUDA cache policy `-Xptxas=-dlcm=cg` instead of `ca` | Exact loss p50 regressed to `42.83 ms`; cache-all remains better for the current row/column mix. | Reverted. |
| Column no-prefetch register-reduction variant | Parity passed and registers dropped from `115` to `101`, but exact loss p50 stayed about `39.42 ms` and L1TEX scoreboard stalls worsened. | Reverted. |
| Degree-5 `atan2` polynomial in the phase accumulator | Full CUDA parity passed, but A/B timing was noise-level (`~0.04-0.09 ms` p50) and not worth a precision-sensitive change. | Reverted. |
| Negative `use_partial` phase-only branch to skip dummy `sumsq` writes | Small CUDA parity passed, but the full `8809` BF phase benchmark hit `CUDA_ERROR_ILLEGAL_ADDRESS`, reproducing the earlier unsafe branch failure mode. | Reverted. |
| Column read-only `ld_float2` loads for the transposed intermediate | Full CUDA parity passed, but exact loss p50 regressed to `39.48 ms`; the plain global-load path remains better for the streaming intermediate. | Reverted. |
| Global CUDA `--maxrregcount=80` after the coalesced paired-row write | Full CUDA parity passed, but exact loss p50 stayed around `39.29 ms`; occupancy pressure is not solved by this broad cap. | Reverted. |
| Global CUDA `--maxrregcount=128` after the coalesced paired-row write | Full CUDA parity passed, but exact loss p50 stayed around `39.46 ms`; extra compiler freedom did not reduce the dependency floor. | Reverted. |
| Runtime preferred shared-memory carveout on paired row/column kernels | Scratch timings regressed to about `41-47 ms` p50 for carveout values `0-50`; the default driver carveout remained best. | Rejected. |
| Row-level aperture-fast microscope for the synthetic geometry | Only rows `250..262` can touch the soft aperture edge, but hard-coding aperture=1 elsewhere still measured about `39.47 ms` p50; the skipped branch is not the row bottleneck. | Reverted. |

GPU1 was saturated during the long run (`100%` SM at the `300 W` power cap,
about `66%` memory controller). The remaining exact-path bottleneck is not
data loading or React/browser rendering. It is the row/column IFFT topology:
the column pass is now much cheaper, but the row pass pays for exact gamma
math, row IFFT, and strided transposed stores. The next single-GPU breakthrough
needs a topology that gives both coalesced row writes and coalesced column
reads, or fuses/tile-transposes the two dimensions without changing the exact
per-BF phase/loss definition.

Do not assume the half-plane source storage permits a half-complex inverse FFT
for exact phase/loss. A direct symmetry probe on GPU1 showed the source
`fft2(real)` plane was Hermitian to `3e-7` relative error, but after the SSB
`q-k`/`q+k` phase/aperture correction the corrected Fourier plane had relative
Hermitian error about `2.0`. The corrected per-BF image is complex, so a
real-output half-complex IFFT would be mathematically wrong for exact phase
mean/loss.

Rejected broad exact-path experiment: replacing the standard C10/C12 polar
phase calculation with an algebraically equivalent Cartesian polynomial across
the general fixed-size kernels reduced some row-stage math on paper, but
changed float32 rounding enough to fail the current 1024 explicit-reference
gate (`p99.9` phase error `3.89e-4` versus `3e-4`). Do not generalize that
shortcut without an explicit reference/tolerance decision. The narrower
512-only C10/C12 hot-path helper used by the accepted paired/dual kernels is
covered by the focused CUDA parity suite above.

Synthetic `1024x1024`, `1382` BF Hermitian timing on GPU1:

| Mode | Mean | p50 | FPS |
| --- | ---: | ---: | ---: |
| Object redraw | `9.42 ms` | `9.42 ms` | `106.1` |
| Phase redraw | `63.63 ms` | `63.35 ms` | `15.7` |
| Phase+loss | `71.25 ms` | `65.95 ms` | `14.0` |

The practical next breakthrough is not another `G_qk` layout flag. Keep the
roadmap single-GPU: redesign the row/column FFT topology with less
shared-memory/barrier pressure, reduce repeated row-stage math, or use a
clearly labeled preview/settle workflow when the user is willing to inspect
object phase during drag and exact mean phase on release.

## 12-cell backend tracking matrix

Track native SSB live-redraw work as a 12-cell backend matrix: three GPU
backends by four native scan sizes. Each cell must record implementation
status, parity status, and the best measured performance before it is treated
as a supported scientist workflow.

| Backend / size | `128x128` | `256x256` | `512x512` | `1024x1024` |
| --- | --- | --- | --- | --- |
| CUDA object redraw | Implemented. High-BF exact fallback mean `4.81 ms`, p95 `5.08 ms`, `208.1 FPS`. | Implemented. Mean `1.74 ms`, p95 `1.78 ms`, `575.2 FPS`. | Implemented. Mean `6.97 ms`, p95 `7.30 ms`, `143.5 FPS`. | Implemented. Older full-BF mean `56.22 ms`, p95 `62.20 ms`, `17.8 FPS`; current pass measured `3000` BF mean `12.41 ms`, p95 `12.60 ms`, `80.6 FPS`. |
| MPS Hermitian preview/free-fit | Implemented on a Mac MPS machine. Sparse `3.60 ms` / exact `4.02 ms` at `128` BF. | Implemented on a Mac MPS machine. Sparse `10.25 ms` / exact `10.68 ms` at `96` BF. | Implemented on a Mac MPS machine. Sparse `28.27 ms` / exact `33.26 ms` at `64` BF. | Implemented on a Mac MPS machine. Sparse `44.66 ms` / exact `50.39 ms` at `24` BF. |
| WebGPU phase/loss path | Implemented in `quantem.widget`; migration pending. Synthetic browser parity passed. | Implemented in `quantem.widget`; migration pending. Synthetic browser parity passed. | Implemented in `quantem.widget`; migration pending. Real 512 full-BF drive measured mean `31.4 ms` GPU and `41.8 ms` UI for C10 changes at `9070/9070` BF. | Implemented in `quantem.widget`; migration pending. Real 1024 BF-column load passes on Mac Chrome Metal. Full active-BF controls work but remain about `168-170 ms` UI/GPU, about `5.9 FPS`, below the 30 FPS target. |

Interpretation:

- CUDA is the only backend with all four object-redraw cells implemented and
  parity-tested against the previous per-BF IFFT object path. The current
  `1024x1024` full-BF synthetic rerun needs a quiet GPU because the Hermitian
  resident `G_qk` allocation is about `37 GB` before scratch.
- MPS now stores prepared `G_qk` as the same Hermitian half-plane and supports
  `128/256/512/1024` MLX/Metal preview/free-fit runs. Treat the MPS table as
  prepared-data MPS evidence, not as CUDA object Fourier-sum parity or full-BF
  real-data signoff.
- WebGPU currently lives in `quantem.widget` because it is bundled for browser
  export. The maintenance target is to move reusable kernel source, shape
  guards, and parity fixtures into `quantem.gpu`, then let `quantem.widget`
  import/build from that source for display.

### WebGPU 1024 status from 2026-07-16

The browser kernel was extended to support `1024x1024` by using a 256-thread
WGSL topology with looped row/column load-store and looped butterflies. This
avoids relying on a 1024-thread workgroup, which common WebGPU limits reject.

Browser performance signoff must record the WebGPU adapter. SwiftShader,
llvmpipe, or any other software adapter is a CPU fallback. It can prove that an
HTML page opens, fetches data, and avoids crashes, but it is not valid evidence
for FPS, GPU latency, or end-to-end interactive performance. Re-run those tests
on Mac Chrome Metal, a working NVIDIA Chrome/WebGPU session, or the native
CUDA/MPS path before claiming responsiveness.

Headed Chrome/CDP evidence on NVIDIA Blackwell:

| Case | Data | Result |
| --- | --- | --- |
| Synthetic shape matrix | `128/256/512/1024`, 8 BF pixels | Browser parity passed for phase and FFT log-magnitude at every size. |
| Synthetic stress | `1024x1024`, 64 BF pixels | WGSL compute mean `8.2 ms`; page wall mean about `503 ms` because the standalone parity/demo page repaints and compares too much on the CPU. |
| Real Samsung full BF | `512x512`, `9070/9070` BF | C10 keyboard drive mean `31.4 ms` GPU, mean `41.8 ms` UI, about `23.9 FPS`; screenshot/report under `/tmp/showptycho-webgpu-size-matrix/real_samsung_fullbf_c10_keys/`. |

Local real `1024x1024` data target used for browser signoff:

```text
/path/to/local/1024_scan.h5
dataset: /entry/data/data
native shape: (1024, 1024, 192, 192) via flattened (1048576, 192, 192)
dtype: uint16 on disk; exact max count 12, so uint8 is lossless for browsing/load
wrapper: /path/to/local/1024_scan_master_wrapper.h5
```

### Evidence-selective WebGPU BF loading

The preferred 1024 browser source is no longer a persistent float32/complex64
`g_bf` cache. It is an exact detector-major BF-column companion:

```text
/path/to/local/1024_bf_columns.u4
layout: [bf, scan]
shape: [1805, 1048576]
encoding: uint4
size: 946.3 MB / 902.5 MiB
```

This companion stores raw detector counts only for the BF candidates, packed
losslessly because the real dataset's maximum count is `10`. It is not a
derived `g_bf` cache, does not reduce the scan, and does not bin the detector.
The browser range-reads only the active BF columns required by the current BF
policy:

| BF policy | Selected BF | Active aperture BF | Bytes fetched |
| --- | ---: | ---: | ---: |
| `BF=0.3` | `542/1805` | `379` | `198.7 MB` (`189.5 MiB`) |
| `BF=1.0` | `1805/1805` | `1382` | `724.6 MB` (`691.0 MiB`) |

Interpretation for the microscopist: `BF=1.0` is the full active-BF review
path, but it still should not fetch non-BF detector pixels or decode the whole
scan-major HDF5 file. `BF<1.0` is a selected-BF or preview policy and must be
reported as such when comparing speed.

Headed Phil Chrome Metal result after switching the real 1024 target to the
BF-column companion:

| Step | Result |
| --- | --- |
| Adapter gate | `apple metal-3`; `software=false`. This is valid browser WebGPU evidence. |
| BF-column dispatch fix | Changed the BF-column unpack dispatch from the old `16 x 16` shape to `32 x 8`, so `1024 x 1024` uses `32768` workgroups in X instead of exceeding Chrome's `65535` per-dimension WebGPU limit. |
| Default BF setup | `BF=0.3`, `542/1805` selected, `379` active, `0.199 GB` fetched, `558 ms` fetch, `157 ms` unpack, `40 ms` FFT, `757 ms` total. |
| Full active BF setup | `BF=0.99-1.0`, `1783-1805/1805` selected, `1382` active, `0.725 GB` fetched, `1668 ms` fetch, `640 ms` unpack, `216 ms` FFT, `2525 ms` total. |
| Full active BF controls | C10, C12, phi12, and scan rotation all updated the phase image; UI mean `169.5 ms`, GPU mean `167.8 ms`, about `5.9 FPS`. |
| Screenshots | `/tmp/phil-showptycho-bfcols-after-load.png`, `/tmp/phil-showptycho-bfcols-bf1-exact.png`, `/tmp/phil-showptycho-bfcols-controls-visible.png`. |

This is a real improvement in first-use loading and source layout, not a solved
redraw target. For full active BF at 1024, the browser still recomputes the
phase/loss path over `1382` active BF columns on each control change. The next
breakthrough must change that WebGPU math topology, for example by porting an
exact object Fourier-sum formulation or another parity-tested reduction that
keeps the same BF policy and precision. Do not compare the CUDA object-redraw
`17.8 FPS` figure directly with this browser phase/loss path without naming the
different scientific quantity being timed.

### Browser WebGPU SSB checklist

Before a browser result is treated as a supported scientist workflow, record:

- [ ] Hardware adapter, with `software=false`; reject SwiftShader/llvmpipe FPS.
- [ ] Native phase size: `128`, `256`, `512`, or `1024`.
- [ ] Source mode: BF-column companion or compressed HDF5 fallback.
- [ ] Selected BF and active aperture BF; label preview BF separately from
  full active BF.
- [ ] First-use profile: bytes fetched, fetch, unpack/decode, FFT/reducer
  setup, and total time.
- [ ] Interaction profile for C10, C12, phi12, scan rotation, histogram,
  colormap, FFT, and flip where those controls are visible.
- [ ] Mean, p50, p95, and FPS-equivalent UI/GPU timing from repeated drives.
- [ ] Screenshot/report paths and console/WebGPU errors.
- [ ] Pass/fail against the declared target frame budget.

Headed Chrome result on mjgoat after adding the range-index HDF5 source path:

| Step | Result |
| --- | --- |
| Initial compressed-source load | Reducer ready in `15.6 s` wall; profile `parse 103 ms`, `decode 5013 ms`, `gather 2286 ms`, `fft 65 ms`, total setup `13.05 s`. |
| Network shape | One small master fetch, one `16 MB` chunk-index fetch, then `206` byte-range reads for the `2.7 GB` compressed HDF5 data file. The previous single `200` full-file fetch failed in Chrome before WebGPU work started. |
| 0.3 BF interaction | `542/1805` selected BF, `379` active aperture BF. C10/C12/phi12/scan-rotation coordinate drives updated live, with UI readouts `148-200 ms` and GPU readouts `134-189 ms`. |
| Near-full BF setup | `1767/1805` selected BF, `1382` active aperture BF. HDF5 setup completed in `14.1 s` wall; profile total `13.11 s`. |
| Near-full BF interaction | C10 drive updated live at about `141 ms` GPU and `152 ms` UI. |

Interpretation for the microscopist: the full native 1024 field can now be
opened from the compressed HDF5 source without saving `g_bf.c64`, and the
controls do update the scientific image and FFT at 1024. It is not yet a
30 FPS steering experience. The next WebGPU work is reducing redraw latency
for the 1024 phase/loss path, not further reducing or binning the dataset.

This is the correct real-data target for WebGPU 1024 workflow testing. Do not
substitute a synthetic 1024 page for final signoff.

## Parity evidence

Focused CUDA parity tests live in `tests/test_ssb_cuda_128.py`.

The object Fourier-sum path is compared against the previous per-BF chunked
IFFT object path for `128x128`, `256x256`, `512x512`, and `1024x1024`:

- `p99.9(abs_err) < 5e-9`
- `p99.9(rel_err) < 1e-4`

The Hermitian `G_qk` object and phase/loss paths are compared against a
canonical full-plane reference for the same four scan sizes, and `_extract_gqk`
is tested to keep only the nonredundant columns. Default constructor-to-
`result()` parity is tested against a test-only full-plane expansion of the
same half-plane, including the diagnostic loss. Focused CUDA check from
2026-07-17 on GPU1:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q \
  tests/test_ssb_cuda_128.py -k 'hermitian or phase_loss'

14 passed, 14 deselected
```

Full CUDA SSB test file from the same pass:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q tests/test_ssb_cuda_128.py

28 passed
```

MPS Hermitian half-plane checks from a Mac MPS machine:

```text
cd /path/to/quantem.gpu
PYTHONPATH=src python -m pytest -q \
  tests/test_ssb_mps_cuda_reference.py

2 passed, 2 skipped
```

The skipped MPS cases are optional real-data CUDA-reference comparisons that
require local `QUANTEM_GPU_SSB_MASTER` / `QUANTEM_GPU_SSB_REFERENCE_NPZ`
fixtures. The executed checks cover supported sparse row masks through
`1024x1024` and exact Hermitian half-plane expansion against `fft2`.

The `1024x1024` reconstruct-with-loss path is compared to an explicit CuPy
reference with a tolerance that allows rare `atan2` branch-cut pixels while
requiring the scalar objective and 99.9% of phase pixels to match.

Run before changing SSB kernels:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/owner/repos/quantem.gpu/src \
pytest -q /home/owner/repos/quantem.gpu/tests/test_ssb_cuda_128.py
```

## Repeat the native benchmark

Use the local Codex skill benchmark for synthetic kernel timing:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/owner/repos/quantem.gpu/src \
python /home/owner/.codex/skills/quantem-ssb-kernel-optimization/scripts/ssb_native_bench.py \
  --repo /home/owner/repos/quantem.gpu \
  --sizes 128,256,512,1024 \
  --num-bf 8809 \
  --iters 4 \
  --mode object
```

Repeat with `--mode phase` and `--mode loss` for the full 12-run matrix. Do not
compare object-mode FPS to phase-mode optimizer FPS without saying which
scientific quantity is being drawn.

## Rejected 512 phase/loss probes

Follow-up GPU1 probes after the accepted `~45-47 ms` exact `512x512`, `8809`
BF path did not produce another accepted kernel change. Keep these as negative
evidence so the next pass starts at the remaining bottleneck instead of
repeating local minima.

| Probe | Result | Decision |
| --- | ---: | --- |
| Quadratic C10/C12 gamma fast branch inside the aperture | Parity passed, but row p50 stayed about `30.6 ms`; full loss p50 was about `46.8 ms`. | Rejected: branch/scheduling pressure erased the special-math savings. |
| Precomputed `sign(sin(Q(q)))` plus BF row/column phase tables | Parity passed, but full loss p50 worsened to about `48.3 ms`. | Rejected: extra table loads were worse than recomputing gamma. |
| Column BF group `64 -> 128` | Parity passed, full loss p50 about `46.8 ms`. | Rejected: atomics/group count is not the dominant cost. |
| Adaptive/full 8809-BF staging chunk | Component total improved slightly, but engine p50 improved only about `0.1 ms` while staging memory rose to about `37 GB`. | Rejected as a default: too much memory for negligible wall-time gain. |
| Coalesced radix-8 row output plus legacy column reader | Row p50 dropped to about `23.8 ms`, but column p50 rose to about `22.3 ms`; total about `46.8 ms`. | Rejected: legacy column path gives back the row-store win. |
| Coalesced radix-8 row output plus radix-8 normal-layout column reader | Column p50 rose to about `28.3 ms`; total about `52.8 ms`. | Rejected: strided column reads dominate. |
| Coalesced radix-8 row output plus tiled explicit transpose | Parity passed, but full loss p50 worsened to about `66 ms`. | Rejected: explicit full transpose costs far more than the row-store savings. |
| Degree-2 column `atan2` polynomial | Focused parity passed, full loss p50 improved only about `0.1 ms`. | Rejected: too little speedup for a rougher scientific approximation. |
| Fixed 64-BF variant of `ifft512_rows_var_radix8_t64` | Full CUDA parity passed, but real Samsung no-pair loss p50 regressed to `36.09 ms` from the `~35.97 ms` short-run baseline. | Rejected: halving the group count/atomics reduced useful parallelism enough to erase the savings. |
| Dual row launch bound relaxed from `__launch_bounds__(256, 4)` to `256,3` | Full CUDA parity passed, but real Samsung no-pair loss p50 regressed to `36.79 ms`. | Rejected: the compiler freedom did not overcome the row-stage scheduling/shared-memory floor. |
| No-pair dual partial-plane reduction instead of direct atomics | Scratch profile p50 moved from about `13.4 ms` direct column accumulation to about `14.0 ms` with partial reduction plus summing. | Rejected: extra global writes/reduction cost more than atomics for this path. |
| Same-row dual `kx` reuse branch | Full CUDA parity passed and `4351/4411` real Samsung dual pairs shared detector row, but real loss p50 stayed about `36.6 ms` and row p50 stayed about `21.8 ms`. | Rejected: saved scalar `dx` work was too small and added branch/register pressure. |
| One-shared-buffer arbitrary dual row IFFT | Focused CUDA parity passed, but row p50 regressed from about `21.6 ms` to `24.2 ms`; sustained real Samsung loss regressed to `38.78 ms` p50 (`25.8 FPS`). | Rejected: halving shared memory serialized the A/B row IFFTs and added barriers/stores, so occupancy pressure was not the dominant floor. |
| Phase partial planes plus scalar loss (`use_partial=3`) | Full CUDA parity passed, but real Samsung loss regressed to `35.46 ms` mean / `35.44 ms` p50 in a 240-step run. | Rejected: replacing phase atomics with partial writes plus a separate reduction costs more than the current in-kernel atomics. |
| Analytic `qy` index formula in the dual row kernel | Full CUDA parity passed and a 240-step run was neutral (`35.03 ms` mean), but the 600-step sustained run regressed to `35.66 ms` mean / `35.66 ms` p50. | Rejected: `qy_1d` cached loads are not a durable row-stage bottleneck. |
| Double-zero aperture early return before `sincos` | Full CUDA parity passed, but real Samsung loss regressed to `35.28 ms` mean / `35.28 ms` p50. | Rejected: the exact branch and code pressure cost more than the skipped outside-aperture work for this dataset. |
| Column launch bounds relaxed from `__launch_bounds__(64, 8)` to `64,6` | Full CUDA parity passed, but real Samsung loss regressed to `35.60 ms` mean / `35.61 ms` p50. | Rejected: the column kernel still wants the current occupancy constraint. |
| Column staged-data loads through `ld_float2`/read-only path | Full CUDA parity passed, but real Samsung loss regressed to `35.69 ms` mean / `35.66 ms` p50. | Rejected: the staged row output does not benefit from the read-only cache path. |
| Column no-prefetch schedule | Full CUDA parity passed, but real Samsung loss regressed to `35.49 ms` mean / `35.48 ms` p50. | Rejected: lower register lifetime did not beat the lost global-load overlap. |
| PTX cache policy `-dlcm=ca -> -dlcm=cg` | Full CUDA parity passed, but real Samsung loss regressed to `35.76 ms` mean / `35.72 ms` p50. | Rejected: L1 caching is still the better default for this mixed row/column path. |
| Column BF group `32 -> 16` | Full CUDA parity passed, but real Samsung loss regressed to `35.92 ms` mean / `35.91 ms` p50. | Rejected: extra phase-atomic groups outweighed lower loop pressure. |
| Hermitian-only duplicate arbitrary-dual row kernel | Full CUDA parity passed, but real Samsung loss regressed to `35.80 ms` mean / `35.80 ms` p50. | Rejected: removing the runtime `gqk_cols` branch increased code footprint/register pressure enough to lose. |
| Two-stream row/column chunk overlap prototype | Exact BF chunks ran with two staging buffers, but best chunked p50 was still about `35.15 ms` for row+column work only. | Rejected as a breakthrough path: the kernels do not overlap enough on one GPU to close the budget. |
| 8-row arbitrary-dual row block with 64 KB dynamic shared memory | Small focused parity passed, but the real full-BF Samsung launch hit `CUDA_ERROR_ILLEGAL_ADDRESS` inside the row kernel under `CUDA_LAUNCH_BLOCKING=1`. | Rejected: the larger row-store topology is unsafe on the target grid and must not be carried without a fresh memory-correct redesign. |
| 6-row arbitrary-dual row block with static shared memory | Focused CUDA parity passed, but real Samsung loss regressed to `37.21 ms` mean / `37.20 ms` p50 (`26.9 FPS`) in a 240-step run. | Rejected: the larger block lowered scheduling efficiency and did not recover enough row-stage throughput. |
| Global module register cap tightened from `96` to `80`/`72` | Focused CUDA parity passed. Real Samsung loss measured `35.32 ms` mean at `80` and `35.44 ms` mean at `72`. | Rejected: forcing lower register allocation did not overcome the column occupancy limiter and likely traded occupancy for spills/scheduling pressure. |

Accepted 2026-07-17 breakthrough: native `512x512` exact phase/loss now uses
64-BF staging chunks by default. This keeps the row-IFFT producer/consumer
working set small instead of writing and rereading one `18+ GB` intermediate.
The scientific contract is unchanged: same full active BF disk (`8822` BF on
the central Samsung field), same Hermitian `G_qk`, same per-BF phase/loss
arithmetic, no binning, no crop, and no preview/settle split. Focused CUDA
parity passed (`25/25`).

Sustained real Samsung GPU1 result after the default change:

| Mode | Steps | Mean | p50 | p95 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase+loss | 600 | `32.49 ms` | `32.15 ms` | `33.31 ms` | `30.78 FPS` |
| Phase-only | 240 | `31.94 ms` | `31.93 ms` | `32.08 ms` | `31.31 FPS` |

Earlier full-buffer component timing was approximately `22.6 ms` row/gamma,
`12.3 ms` column/phase, and `<0.2 ms` finalization. The new chunking win is not
from changing gamma algebra or scalar loss; it is from making the producer/
consumer memory working set smaller. GPU1 remains a 300 W card and sustained
runs can still show `pviol=100%`, so the p95 margin around the 30 FPS frame
budget should be watched on other machines.

Current conclusion: the `512x512` exact full-BF phase/loss target is met on the
real central field. The next structural performance target is
`1024x1024`, where the same exact phase/loss path still needs a larger topology
change, likely fused/tiled row-column work or another exact formulation.

### 512 full-BF calibration timing follow-up

The same real `512x512` field was used to test the full calibration workflow,
not only live redraw. Source: private HDF5 master file loaded as full
scan/full detector `uint16`, `9070` active BF after the fitted
aperture, Hermitian `G_qk=(9070,512,257)` / `9.55 GB`.

The important workflow split is:

- Live interaction uses the exact full-IFFT `reconstruct_with_loss()` path.
- Default historical calibration uses sparse `variance_loss_batch()`, a
  different scalar objective.
- Forced exact calibration uses the exact full-IFFT path for optimizer and
  refiner objective evaluations.

Accepted calibration improvement:

| Stage | Before | After | Notes |
| --- | ---: | ---: | --- |
| Exact 200-trial optimize | `~6.7 s` | `~6.7 s` | Still bounded by `~32-33 ms` per exact full-BF candidate. |
| Exact Nelder-Mead refine | `~7.1 s / 200 evals` in the earlier forced path | `1.12 s / 34 evals` | Exact full-BF objective; default exact-fallback tolerances now stop before the invisible flat tail. |
| Load -> Gqk -> optimize -> refine -> widget | `17.05 s` | `11.32 s` | Full BF, no binning, no crop, no trial-count reduction. |

The new exact-refine default for this fallback path was chosen from a sweep:
`xatol=0.25`, `fatol=2e-6`, `max_iter=160`. Compared with the longer exact
baseline, the selected policy gave loss delta about `4-6e-7` and phase deltas
around `5.7e-5 rad` mean / `3.9e-4 rad` p99.9 in the real-data probe. The
too-loose four-evaluation policies were rejected because they produced
`~0.0014 rad` mean and `~0.01 rad` max phase deltas.

Rejected calibration hypotheses:

| Hypothesis | Measurement | Decision |
| --- | --- | --- |
| Host-sync-free GPU scalar losses for exact fallback | Parity passed (`2.2e-8` max scalar-loss delta), but batch-4 exact stayed about `130.6 ms` (`32.6 ms/candidate`). | Rejected and removed from production code: no measurable throughput win for the added complexity. |
| Concurrent exact candidates on separate CUDA streams | Four candidates were slower concurrently (`140.9 ms`) than sequentially (`129.7 ms`). | Rejected: kernels contend for the same shared-memory/scheduler resources. |
| Larger exact BF chunks for calibration | Real-data sweep showed `32/64` BF chunks at `~32.0-32.1 ms`; `96+` chunks regressed to `35-36 ms`. | Keep `64` as the default; larger chunks are not a calibration win. |
| Fewer exact Optuna trials | `150` trials + exact refine reached `6.72 s` optimize+refine with `0.00054 rad` p99.9 phase delta versus the 200-trial baseline. | Useful evidence for an opt-in fast-calibration mode, but do not present it as the full 200-trial default. |

Nsight on the exact 512 row kernels confirms the next kernel-level ceiling:
`ifft512_rows_fused_pk_dual_radix8_t64_packed` runs at about 50% theoretical
occupancy, limited by shared memory, with about `49%` no-eligible scheduler
cycles and MIO/short-scoreboard stalls. The column phase/loss accumulator
`ifft512_rows_var_radix8_t64` is register-limited, reaches about 20% achieved
occupancy in the small chunk launch, and also shows scheduler starvation. The
next kernel breakthrough therefore needs a different row/column topology or a
parity-tested exact multi-candidate formulation, not bigger chunks or streams.

### Hermitian-only and MPS matrix follow-up

The 2026-07-17 follow-up made Hermitian `G_qk` the only public runtime storage
mode. `SSB(..., gqk_storage="full")` now raises a corrective `ValueError`.
Full-plane `G_qk` remains available only as a test-only canonical expansion of
the Hermitian half-plane, so parity references are stable without carrying
redundant FFT roundoff as a separate public mode.

CUDA validation after the removal:

```text
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q \
  tests/test_ssb_cuda_128.py tests/test_ssb_batch_optuna.py

29 passed in 3.91s
```

The `128x128` high-BF object path currently uses the exact fused-IFFT fallback
when `num_bf > 1024`. The small-BF Fourier-sum kernel remains parity-tested,
but a high-BF synthetic stress probe left the CUDA context in an illegal-
address state. The fallback keeps the user path exact and fast at `128x128`
while that microkernel is investigated.

CUDA synthetic Hermitian timing from this pass:

| Scan | Mode | BF | Resident `G_qk` | Mean | p50 | p95 | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `128x128` | object, fused-IFFT fallback | `8809` | `586 MB` | `4.81 ms` | `4.70 ms` | `5.08 ms` | `208.1` |
| `256x256` | object | `8809` | `2.33 GB` | `1.74 ms` | `1.73 ms` | `1.78 ms` | `575.2` |
| `512x512` | object | `8809` | `9.27 GB` | `6.97 ms` | `6.82 ms` | `7.30 ms` | `143.5` |
| `1024x1024` | object | `8809` | `37.02 GB` | `41.79 ms` | `42.32 ms` | `43.93 ms` | `23.9` |
| `1024x1024` | phase-only, split-512 row/column | `8809` | `37.02 GB` | `195.54 ms` | `194.99 ms` | `200.84 ms` | `5.11` |
| `1024x1024` | phase+loss, split-512 row/column | `8809` | `37.02 GB` | `197.71 ms` | `199.00 ms` | `199.24 ms` | `5.06` |
| `128x128` | phase+loss | `8809` | `586 MB` | `7.86 ms` | `7.80 ms` | `7.99 ms` | `127.2` |
| `256x256` | phase+loss | `8809` | `2.33 GB` | `18.31 ms` | `18.20 ms` | `18.51 ms` | `54.6` |
| `512x512` | phase+loss | `8809` | `9.27 GB` | `26.98 ms` | `27.11 ms` | `27.12 ms` | `37.1` |

The `1024x1024` exact phase/loss path now uses a split-512 row and column IFFT:
each 1024 IFFT is decomposed into exact even/odd 512-point radix-8 transforms
plus a final radix-2 combine. The row kernel writes transposed scratch, and the
column kernel consumes that layout directly. This preserves the default math
and focused CUDA parity while cutting the synthetic full-BF phase+loss time
from `382.24 ms` to `197.71 ms`.

The current `1024x1024` exact phase/loss default still uses a `1024` BF staging
chunk for high-BF runs. Chunk sweeps from `512` to `4096` BF measured about
`197-200 ms`; BF-group sweeps kept `32` BF as the most stable column
accumulation group. Those knobs reduce memory footprint or stabilize the run,
but the next FPS win must come from the register-heavy row/gamma path or a new
exact formulation.

MPS validation and timing were run on a Mac MPS machine with MLX `0.32.0`.
Prepared MPS `G_qk` now stores the same Hermitian half-plane. The cached sparse
objective Metal kernel fetches missing columns by mirror-conjugate symmetry,
and the non-cached MLX preview path expands only the active chunk.

Mac MPS test:

```text
PYTHONPATH=src python -m pytest -q \
  tests/test_ssb_mps_cuda_reference.py

2 passed, 2 skipped in 1.29s
```

Mac MPS synthetic prepared-data timing:

| Scan | BF | Resident `G_qk` | Prep | Sparse objective mean / p95 / FPS | Exact preview mean / p95 / FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128x128` | `128` | `8.52 MB` | `0.163 s` | `3.60 ms` / `4.25 ms` / `278` | `4.02 ms` / `4.40 ms` / `248` |
| `256x256` | `96` | `25.36 MB` | `0.036 s` | `10.25 ms` / `11.08 ms` / `97.5` | `10.68 ms` / `11.36 ms` / `93.6` |
| `512x512` | `64` | `67.37 MB` | `0.067 s` | `28.27 ms` / `30.67 ms` / `35.4` | `33.26 ms` / `33.61 ms` / `30.1` |
| `1024x1024` | `24` | `100.86 MB` | `0.120 s` | `44.66 ms` / `48.79 ms` / `22.4` | `50.39 ms` / `55.98 ms` / `19.8` |
| `512x512` pushed | `96` | `101.06 MB` | `0.166 s` | `44.18 ms` / `45.90 ms` / `22.6` | `49.71 ms` / `50.20 ms` / `20.1` |
| `1024x1024` pushed | `30` | `126.07 MB` | `0.092 s` | `54.27 ms` / `55.75 ms` / `18.4` | `61.05 ms` / `61.58 ms` / `16.4` |

Interpretation for Mac users: MPS now supports the native size matrix for the
prepared SSB path and is interactive for moderate BF counts, especially at
`128/256/512`. This is not yet a full-BF `1024x1024` Mac signoff. Report the
BF count with every MPS FPS number because Mac unified memory and MLX FFT cost
scale directly with the number of active BF columns.

### MPS full-BF 512 follow-up

The next MPS pass targeted Mac no-crop/no-bin review on the real `512x512`
logic dataset:

```text
private HDF5 master file
native shape: (262144, 192, 192), uint16
BF policy: threshold=0.0, bf_radius=53
selected BF: 8827
Hermitian G_qk: (8827, 512, 257), 9.29 GB
```

Accepted MPS changes:

- Added batched `ChunkedFrames.columns(rows, cols)` so SSB preparation reads BF
  columns in groups instead of calling `column()` once per detector pixel.
- Changed the non-prefix MPS column gather to fill scan-major chunks with
  `np.take` from the flattened detector grid, then enabled threaded extraction
  over independent Metal chunks for large BF selections. This keeps the same
  raw detector evidence and only changes the host extraction topology.
- Added a fused dynamic Metal correction kernel for large-BF prepared paths.
  It computes the same C10/C12/phi12 correction and fetches missing Hermitian
  columns directly, avoiding expanded `G_qk` and large MLX geometry
  temporaries.
- Added an explicit `phase_mode="object"` path for `ssb_preview_mps`. This is
  an exact BF-averaged object wave, using Fourier-domain BF summation followed
  by one inverse FFT. It is now the default no-loss preview mode; requesting
  `compute_loss=True` or `phase_mode="mean"` uses the exact
  mean-of-per-BF phase/loss quantity.
- Precomputed the BF-only probe term `p(k)` once per live aberration setting
  and passed it into the object Fourier-sum Metal kernel. The kernel still
  computes the `q-k` and `q+k` terms per pixel, but it no longer recomputes the
  same BF-only probe phase for every output pixel.
- Split object-mode chunking into separate setup and redraw defaults:
  `QUANTEM_MPS_SSB_OBJECT_CHUNK_BF` controls first-use BF FFT setup, while
  `QUANTEM_MPS_SSB_OBJECT_REDRAW_CHUNK_BF` controls repeated live redraw. On
  the Mac MPS machine, setup is fastest at `1024` BF chunks, but synchronized live redraw is
  fastest with the object redraw default `128`; tying those together regressed
  interaction timing.
- Added `QUANTEM_MPS_SSB_OBJECT_THREADGROUP` for repeated object redraw and
  set the high-memory Mac default to a `16`-thread Metal threadgroup after launch-shape
  sweeps.
- Replaced separate object-kernel `sin`/`cos` calls with
  `metal::fast::sincos` for the two shifted aperture phases. Mac MPS parity
  stayed green, and this was the decisive redraw speedup.
- Added a 512-only fused Metal column-IFFT + phase/loss accumulator for the
  prepared MPS exact mean-phase path. The route still uses the same full BF
  disk and the same float32 per-BF phase/loss definition, but it no longer
  materializes the full per-BF object chunk before summing phases.
- Added a 512-only fused dynamic correction + row-IFFT Metal kernel for the
  prepared MPS exact mean-phase path. This removes the MLX corrected-plane
  materialization and MLX row-IFFT from the large-BF loop while preserving the
  same Hermitian `G_qk`, BF disk, and float32 phase/loss definition.

Real Mac MPS before/after:

| Path | Before | After | FPS after | Status |
| --- | ---: | ---: | ---: | --- |
| Full no-bin MPS load | not retimed in same harness | `0.94 s` final run, earlier `1.05-1.85 s` | n/a | Pass. |
| BF select | not retimed in same harness | `0.28-0.33 s` | n/a | Pass. |
| Prepare Hermitian `G_qk` | `22.67 s` | `3.21 s` final run, earlier `2.91-3.04 s` | n/a | `~7x` faster than the old per-column setup. |
| Public object-mode preview after load | `24.84 s` | about `3.3-3.5 s` | n/a | `~7x` faster first review, no BF reduction. |
| Repeated object-wave redraw | `~211 ms` scratch loop; first optimized pass `50.6 ms` mean, p95 `65.0 ms` | mean `24.01 ms`, p50 `23.85 ms`, p95 `26.34 ms` | `41.65 FPS` | Exact object-wave path passes the 30 FPS target after warm-up. |
| Repeated exact phase+loss redraw | `710 ms`; pre-column-fusion best `~550-640 ms`; fused-column-only best `335.18 ms` mean | best `169.62 ms` mean, `169.68 ms` p50, `170.24 ms` p95 | `5.90 FPS` | Fused correction+row plus fused column is exact and about `2x` faster than the fused-column-only path, but still fails the 30 FPS live target. |

Latest setup split with `chunk_bf=1024`:

```text
load_s=1.05, bf_select_s=0.33
prepare_total_s=2.91
  gather_s=1.49
  cpu_to_mlx_stage_s=0.48
  rfft2_eval_s=0.54
  concat_s=0.30
selected_bf=8827
```

Final default-path object redraw timing:

```text
load_s=1.196, prepare_s=3.230
selected_bf=8827
Hermitian G_qk=(8827, 512, 257)
object redraw defaults: chunk_bf=128, threadgroup=16
best measured chunk_bf=256:
  mean_ms=21.82, p50_ms=21.88, p95_ms=23.16, fps=45.82
default chunk_bf=128:
  mean_ms=22.21, p50_ms=22.19, p95_ms=22.98, fps=45.03
public ssb_preview_mps default after load:
  wall_s=3.436, loss=None
public ssb_preview_mps explicit object after load:
  wall_s=3.257, loss=None
```

Final exact mean-phase/phase-loss redraw timing after fused correction+row and
fused column accumulation:

```text
load_s=1.196, prepare_s=3.230
selected_bf=8827
Hermitian G_qk=(8827, 512, 257)
phase-only best chunk_bf=1024:
  mean_ms=168.77, p50_ms=168.52, p95_ms=169.86, fps=5.93
phase+loss best warmed chunk_bf=3072:
  mean_ms=165.20, p50_ms=165.57, p95_ms=166.71, fps=6.05
component split at chunk_bf=1024:
  correction+row-IFFT mean=74.30 ms
  column-IFFT+phase/loss mean=93.76 ms
  accumulation mean=2.87 ms
public ssb_preview_mps explicit mean after load:
  wall_s=4.281, loss=None
public ssb_preview_mps compute_loss=True after load:
  latest default chunk_bf=3072 wall_s=4.751, loss=0.050665274262428284
```

Follow-up on 2026-07-17 changed `_default_phase_loss_chunk_bf()` to return
`3072` on 96 GB-class Macs, `1024` on 64 GB-class Macs, and `512` otherwise.
The high-memory Mac path reports `default_phase_chunk=3072`; the public real
`ssb_preview(..., compute_loss=True, chunk_bf=16)` path completed with
`8827` BF and `512x512` phase/amplitude in `4.75 s` after a `0.97 s` full
MPS load. The prepared steady-state exact phase+loss redraw is still only
about `6 FPS`, so do not present this as a solved 30 FPS MPS path.

Real full-BF parity against the previous MLX row-IFFT + fused-column path:

```text
loss_old=0.021938618272542953
loss_new=0.021938620135188103
phase_max_abs=1.4901161193847656e-08
phase_mean_abs=1.894959078541092e-09
```

### MPS native full-BF matrix follow-up, 2026-07-18

This pass extended the exact fused Metal phase/loss topology beyond the earlier
512-only path. It keeps the same full active BF-style evidence, native scan
size, Hermitian complex64 `G_qk`, and float32 mean-phase/loss definition. No
scan crop, detector binning, BF subsampling, preview/settle split, or derived
float/complex cache is used.

Accepted MPS changes:

- Added fused 128/256 Metal column-IFFT + phase/loss accumulation.
- Added fused 128/256 Metal dynamic correction + row-IFFT.
- Added fused 1024 Metal dynamic correction + row-IFFT and column-IFFT +
  phase/loss accumulation.
- Fixed the top-level fused MPS loss accumulator so all fused exact sizes
  (`128/256/512/1024`) use scalar `phase^2` accumulation instead of
  accidentally broadcasting the scalar loss tile through an image-shaped
  accumulator.
- Changed the 128/256 MPS exact phase/loss default chunk to a large
  full-BF-capable chunk on 96 GB-class Macs. This removes avoidable loop and
  partial-reduction overhead without changing BF evidence.
- Changed the 1024 MPS exact phase/loss default chunk to `512` BF after the
  scalar-loss fix. Isolated sweeps show `512/768` are close; the default keeps
  the smaller, safer working set.
- Changed 128/256 phase-only column grouping to 8 columns per Metal
  threadgroup. Phase+loss keeps 4 columns because 8 columns regressed the loss
  path in repeated timing.
- Changed repeated MPS object redraw to use a `64`-threadgroup default for
  `512x512` and larger scans after a small launch-shape sweep.
- Added `QUANTEM_MPS_SSB_PHASE_COL_K_BF` as an internal tuning knob. Sweeps
  from `8` to `128` BF did not produce a durable 512 breakthrough, so the
  default remains `32`.

Mac MPS parity gate:

```text
PYTHONPATH=src python -m pytest -q tests/test_ssb_mps_cuda_reference.py

15 passed, 2 skipped
```

MPS synthetic prepared full-BF-style matrix, Hermitian `G_qk`, `8809` BF:

| Scan | Object mean / FPS | Phase mean / FPS | Phase+loss mean / FPS | Notes |
| --- | ---: | ---: | ---: | --- |
| `128x128` | `2.45 ms / 408.4` | `~8.0 ms / 122-126` | `~8.3 ms / 119-121` | Exact fused row/column path passes 30 FPS. |
| `256x256` | `8.62 ms / 116.1` | `32.75 ms / 30.5` | `~34-35 ms / 28.6-29.4` | Phase-only reaches 30 FPS; phase+loss remains just above the strict `33.3 ms` budget. |
| `512x512` | `37.65 ms / 26.6` | warm `166-182 ms / 5-6` | warm `170-173 ms / 5-6` | No new durable 512 exact phase/loss breakthrough; long stress loops throttle upward. |
| `1024x1024` | clean `~156 ms / 6.4`, hot `~222 ms / 4.5` | warm `~0.78-1.0 s / 1.0-1.3` | warm `~0.79-1.0 s / 1.0-1.3` | Fused 1024 is about 2-2.6x faster than the generic MLX path, but still far from 10/30 FPS. |

Before/after for the exact MPS phase/loss paths in the same synthetic prepared
full-BF-style harness:

| Scan | Quantity | Before | After | Speedup | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `128x128` | phase | `38.70 ms` | `~8.0 ms` | `~4.8x` | Passes 30 FPS. |
| `128x128` | phase+loss | `37.33 ms` | `~8.3 ms` | `~4.5x` | Passes 30 FPS. |
| `256x256` | phase | `144.98 ms` | `32.75 ms` | `4.4x` | Passes 30 FPS by mean; p95 remains close to budget. |
| `256x256` | phase+loss | `146.51 ms` | `~34-35 ms` | `~4.2x` | Near miss for 30 FPS. |
| `1024x1024` | phase | `1994 ms` | warm `~0.78-1.0 s` | `~2.0-2.6x` | Still fails 10/30 FPS. |
| `1024x1024` | phase+loss | `1984 ms` | warm `~0.79-1.0 s` | `~2.0-2.5x` | Still fails 10/30 FPS. |

Rejected or non-breakthrough MPS probes from this pass:

| Probe | Result | Decision |
| --- | --- | --- |
| 512 exact phase/loss chunk sweep down to `64` BF | Small CUDA-like chunks regressed (`~226 ms` at `64` BF). Larger chunks stayed best (`~166-170 ms`). | Keep the high-memory 512 default at `3072` BF. MPS benefits from fewer loop launches here. |
| 512 column BF grouping `8/16/32/64/128` | Best cases moved only `1-2 ms`; larger groups regressed. | Not a topology breakthrough; keep default `32`. |
| 1024 fused chunk sweep `128/256/512/768/1024` after scalar-loss fix | `512/768` were close and better than the earlier `256` cap in isolated runs, but order and thermal state moved the result by hundreds of ms. | Set 1024 default to `512` for a smaller safe working set; this remains far from interactive. |
| Object threadgroup sweep `8/16/32/64/128` | `64/128` modestly improved large-object redraw. | Keep `64` for `512+`; it is a small object-mode win, not a phase/loss breakthrough. |
| 128/256 in-kernel scalar-loss reduction | Parity passed, but 256 phase+loss regressed to about `34.5 ms`. | Reverted. Smaller loss outputs did not beat the extra threadgroup barriers. |
| 128/256 8-column Metal grouping | Phase-only improved enough for 256 to reach about `32.75 ms`, but phase+loss regressed. | Use 8 columns only for phase-only; keep phase+loss at 4 columns. |

Interpretation for the microscopist: on an Apple GPU, exact full-BF
mean-phase/loss is now real-time at `128x128`, very close at `256x256`, still
review-only at `512x512`, and not live-interactive at `1024x1024`. The 1024
fused Metal path proves the generic MLX FFT route was a major bottleneck, but
the remaining wall time is still per-BF exact phase work. The next MPS
breakthrough needs a different exact 512/1024 row-column topology or a
scientifically equivalent reformulation, not BF reduction.

Rejected MPS probes from the same pass:

| Probe | Result | Decision |
| --- | --- | --- |
| Direct one-threadgroup-per-Fourier-pixel reduce | Parity passed, but real full-BF timing was `~103-168 ms` depending on thread count because it destroyed useful parallelism. | Removed from production code. |
| Lazy aperture branch that moved astigmatism/trig work after support checks | Parity passed, but real timing regressed because the extra branching hurt Metal occupancy. | Reverted. |
| Old redraw launch defaults `chunk_bf=48/64`, threadgroup `64/256` | Worked, but sustained real-data redraw stayed roughly `18-24 FPS`. | Replaced by `chunk_bf=128`, threadgroup `16`. |
| Phase-only sum kernel that skipped `sumsq` | Parity passed, but phase-only still measured about `334-348 ms` because the inverse FFT and correction dominate. | Kept only where it is simple; do not expect it to be the MPS breakthrough. |
| Column BF grouping `k_bf=8/16/32/64/128/256` after fused row | Real full-BF column+accumulation stayed best around `k_bf=16/32` at `~92 ms`; larger groups reduced partial outputs but lost useful BF parallelism. | Keep `k_bf=32`; the next exact breakthrough needs a different row-column topology, not a grouping constant. |
| CUDA 1024 direct atomic phase/loss accumulation | Focused parity passed, but `1024x1024`, `1382` BF loss regressed to `210.8 ms` mean before reverting; the known partial-sum path measured `62.0 ms` in the same condition. | Removed. Direct atomics are not the 1024 breakthrough. |
| CUDA 1024 column grouping `k_bf=8` | The isolated column kernel moved slightly, but full `8809` BF loss regressed from `406.5 ms` to `427.3 ms`. | Keep `k_bf=32`. |
| CUDA 1024 Cartesian row correction helper | Focused parity failed the 1024 CuPy reference gate (`99.9%` phase error `3.26e-4` vs `3e-4`). | Reverted. Keep exact reference parity. |
| CUDA 1024 launch bounds on row/column kernels | Focused parity passed, but full `8809` BF loss regressed to `859 ms`. | Reverted. Occupancy pressure is not solved by forcing launch bounds. |
| CUDA 1024 polynomial `atan2` in column phase/loss | Focused parity passed, but warmed full `8809` BF loss was noise-level (`389.0 ms` vs `389.5 ms` same-session baseline). | Rejected as a non-breakthrough. |
| CUDA 1024 specialized Hermitian `G_qk` fetch helper | Focused parity passed, but default full `8809` BF timing did not improve (`380.84 ms` phase, `382.67 ms` loss). | Reverted; the generic fetch is not the bottleneck. |
| CUDA 1024 split-512 launch bound tightened from `64,8` to `64,12` | Focused parity passed, but component timing regressed from about `15.2 ms` row / `6.9 ms` column to about `17.0 ms` row / `11.5 ms` column per 1024-BF chunk. | Reverted. The higher-register compiler choice is faster despite lower occupancy. |
| CUDA 1024 coalesced-load split row | Focused parity passed, but warmed full `8809` BF loss stayed about `200-201 ms`, slower than the direct split row at about `197-199 ms`. | Removed. The extra shared gather and synchronization cost more than the improved source-load order. |

The object-mode parity guard compares the fused object Fourier-sum kernel
against the looped corrected-object reference on a Mac MPS machine:

```text
PYTHONPATH=src python -m pytest -q \
  tests/test_ssb_mps_cuda_reference.py

5 passed, 2 skipped
```

Interpretation for the microscopist: on an Apple GPU, a full-BF `512x512`
field can now load no-bin data, build the BF evidence, and show an
exact object-wave phase review a few seconds after load, then steer the object
view above `40 FPS` after warm-up on a Mac MPS machine. The stricter exact
mean-phase/phase-variance loss view improved from about `550-640 ms` to about
`335 ms`, so it is better but still not usable for live full-BF steering on
MPS. The next real MPS breakthrough has to fuse the correction + row FFT side
of the exact phase/loss path or port the CUDA row/column topology more fully;
another BF-column gather or UI flag will not close the remaining budget.

## Next performance work

Problem: the live object redraw target is met for the real Mac MPS `512x512`
full-BF object-wave workflow, and the fused column accumulator improved exact
phase/loss to about `3 FPS`, but exact phase/loss is still far below real-time
on MPS.

Action: port or prototype the fused correction + row-FFT half of the CUDA
topology on MPS, then rerun the same before/after table with full BF.

Problem: the `512x512` exact phase/loss path now meets the `33.3 ms` / `30 FPS`
target on the real central Samsung field, but the p95 margin is small on the
300 W GPU1 power envelope.

Action: keep the 64-BF default chunking for 512, and re-run a 600-step
real-data signoff whenever row/column kernels, power settings, BF policy, or
driver/toolkit versions change.

Problem: the phase-variance optimizer path cannot inherit the object Fourier-
sum result because `mean(angle(object_bf))` is a different scientific quantity
from `angle(mean(object_bf))`.

Action: keep optimizing `reconstruct()` and `reconstruct_with_loss()` with
dedicated native variance kernels or an equivalent exact reformulation; do not
reuse the object-path claim for the optimizer objective.

Problem: `1024x1024` batched optimizer variance is still disabled.

Action: implement and parity-test a dedicated 1024 batch variance kernel before
enabling batch trials at that size.

Problem: MPS and WebGPU are implemented for native-size SSB review, but they
are not yet equivalent to CUDA full-BF real-data signoff.

Action: keep extending the 12-cell matrix with real-data MPS and WebGPU runs.
For MPS, measure the same BF policies used by scientists on a Mac MPS machine and compare
against CUDA-reference fixtures. For WebGPU, move reusable WGSL kernels and
parity fixtures from `quantem.widget` into `quantem.gpu` without using
SwiftShader performance numbers.
