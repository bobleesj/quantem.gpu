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
ssb_full = SSB(..., gqk_storage="full")
```

`gqk_storage="herm"` stores only scan-frequency columns `0..N/2`. The CUDA
object Fourier-sum kernels mirror-conjugate the missing half-plane directly
when they fetch `G(q, k)`. The CUDA phase/loss row kernels also fetch the
Hermitian half-plane directly through `ld_gqk_maybe_herm`; the engine no
longer materializes transient full-plane `G_qk` chunks for those paths. This
relies on the exact Hermitian symmetry of the FFT of real virtual-BF images.
The object-wave definition, phase/loss definition, and BF selection are
unchanged; only the resident storage layout and fetch path change.

`gqk_storage="full"` is now a canonical full-plane expansion from the same
half-plane, not an independent source of redundant lower-half FFT roundoff.
That keeps full-vs-Hermitian object comparisons at the kernel arithmetic floor
instead of preserving backend-specific noise in mathematically redundant
columns.

Resident `G_qk` memory becomes:

```text
full: num_bf * N * N         * sizeof(complex64)
herm: num_bf * N * (N/2 + 1) * sizeof(complex64)
```

For a microscopist this is useful when the goal is to fit aberrations and steer
the final object view without spending the persistent VRAM budget on redundant
Fourier columns. Phase mean, phase variance/loss, `optimize()`, `refine()`,
`grid_search()`, defocus sweeps, and higher-order aberration paths now keep the
same resident Hermitian storage and fetch missing columns on demand where those
kernels have been ported. Use `gqk_storage="full"` when explicitly
benchmarking full storage or comparing against legacy full-residency behavior.

Implementation status from the 2026-07-17 pass:

- CUDA object kernels `128/256/512/1024` accept either full-plane or Hermitian
  half-plane `G_qk`.
- CUDA phase/loss paths `128/256/512/1024` preserve Hermitian resident storage
  and fetch the missing half-plane directly inside the row kernels; no
  transient full-plane `G_qk` chunk is built for the current phase/loss paths.
- CUDA `512x512` phase-only redraw has a sum-only column accumulator so
  `reconstruct()` does not compute phase variance when `reconstruct_with_loss()`
  is not requested. The measured speed did not improve materially, which shows
  the remaining floor is FFT topology rather than the removed `sumsq` writes.
- `_extract_gqk(..., gqk_storage="herm")` builds the half-plane directly after
  the BF-stack FFT, avoiding a persistent full `G_qk` allocation.
- Parity tests compare default Hermitian end-to-end `SSB(...).result()` against
  explicit canonical full storage. A raw `cp.fft.fft2` redundant half-plane can
  differ from exact conjugate symmetry at the expected fp32 arithmetic-noise
  floor, so full storage is canonicalized from the half-plane rather than using
  that redundant noise as a separate reference.
- MPS has not received this half-plane storage path yet.

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

## Current measured baseline

Hardware: RTX PRO 6000 Blackwell-class GPU on `mjgoat`.

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
- Raised the `512x512` column phase/loss BF group from `32` to `64`, reducing
  partial-plane overhead without changing the per-BF phase/loss arithmetic.

Steady-state synthetic `512x512`, `8809` BF timing on GPU1:

| Mode | Before this pass | After radix-8 row | After transposed staging | After 64-BF groups | FPS after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase redraw | `70.57 ms` | `58.10 ms` | `53.46 ms` | `52.63 ms` | `19.0` |
| Phase+loss | `69.26 ms` | `58.09 ms` | `52.98 ms` | `52.83 ms` | `18.9` |

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
```

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
| Computing 8 rows per block and writing transposed tiles directly | Parity passed, but `512` phase redraw regressed to `60.1 ms` (`16.6 FPS`) from lower occupancy/shared-memory cost. | Reverted. |
| Computing 4 rows per block and writing transposed tiles directly | Parity passed, but `512` phase redraw still regressed to `55.7 ms` (`18.0 FPS`). | Reverted. |
| Computing 2 rows per block and writing transposed tiles directly | Parity passed, but `512` phase redraw regressed to `54.5 ms` (`18.4 FPS`). | Reverted. |
| Skipping `sincos` when shifted apertures are exactly zero | Parity passed, but `512` phase redraw stayed around `53.1 ms`; branch/control-flow cost offset the skipped work. | Reverted. |
| Raising the exact phase/loss chunk cap from `2 GB` to `4 GB` | Parity passed, but phase/loss timing stayed around `52.7-52.9 ms` while using more transient memory. | Reverted. |

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

Rejected exact-path experiment: replacing the standard C10/C12 polar phase
calculation with an algebraically equivalent Cartesian polynomial reduced some
row-stage math on paper, but changed float32 rounding enough to fail the
current 1024 explicit-reference gate (`p99.9` phase error `3.89e-4` versus
`3e-4`). Keep it out of the exact path unless the reference formulation and
tolerance policy are intentionally updated.

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
| CUDA object redraw | Implemented. Mean `0.80 ms`, p95 `0.81 ms`, `1247.9 FPS`. | Implemented. Mean `3.97 ms`, p95 `5.55 ms`, `251.6 FPS`. | Implemented. Mean `12.29 ms`, p95 `12.53 ms`, `81.4 FPS`. | Implemented. Mean `56.22 ms`, p95 `62.20 ms`, `17.8 FPS`. |
| MPS object redraw | Pending object Fourier-sum port. | Pending object Fourier-sum port. | Pending object Fourier-sum port. | Pending object Fourier-sum port. |
| WebGPU phase/loss path | Implemented in `quantem.widget`; migration pending. Synthetic browser parity passed. | Implemented in `quantem.widget`; migration pending. Synthetic browser parity passed. | Implemented in `quantem.widget`; migration pending. Real Samsung 512 full-BF drive measured mean `31.4 ms` GPU and `41.8 ms` UI for C10 changes at `9070/9070` BF. | Implemented in `quantem.widget`; migration pending. Real Berk 1024 BF-column load passes on Phil Chrome Metal. Full active-BF controls work but remain about `168-170 ms` UI/GPU, about `5.9 FPS`, below the 30 FPS target. |

Interpretation:

- CUDA is the only backend with all four object-redraw cells implemented and
  parity-tested against the previous per-BF IFFT object path.
- MPS has useful SSB preview/free-fit infrastructure, but it has not received
  the exact object Fourier-sum topology. Do not claim MPS parity from image
  agreement or reduced optimizer settings.
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
on Phil/Mac Chrome Metal, a working NVIDIA Chrome/WebGPU session, or the native
CUDA/MPS path before claiming responsiveness.

Headed Chrome/CDP evidence on NVIDIA Blackwell:

| Case | Data | Result |
| --- | --- | --- |
| Synthetic shape matrix | `128/256/512/1024`, 8 BF pixels | Browser parity passed for phase and FFT log-magnitude at every size. |
| Synthetic stress | `1024x1024`, 64 BF pixels | WGSL compute mean `8.2 ms`; page wall mean about `503 ms` because the standalone parity/demo page repaints and compares too much on the CPU. |
| Real Samsung full BF | `512x512`, `9070/9070` BF | C10 keyboard drive mean `31.4 ms` GPU, mean `41.8 ms` UI, about `23.9 FPS`; screenshot/report under `/tmp/showptycho-webgpu-size-matrix/real_samsung_fullbf_c10_keys/`. |

Real `1024x1024` data target used for browser signoff:

```text
/home/owner/ssd/data/berk_tomo_20260716_one_tilt/pos_38_tilt0.h5
dataset: /entry/data/data
native shape: (1024, 1024, 192, 192) via flattened (1048576, 192, 192)
dtype: uint16 on disk; exact max count 12, so uint8 is lossless for browsing/load
wrapper: /home/owner/data/reports/berk_tomo_20260716_one_tilt_ssb/pos_38_tilt0_master_wrapper.h5
```

### Evidence-selective WebGPU BF loading

The preferred 1024 browser source is no longer a persistent float32/complex64
`g_bf` cache. It is an exact detector-major BF-column companion:

```text
/home/owner/ssd/agent-show/berk-showptycho-webgpu-1024-bfcols-20260716/source/bf_columns.u4
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

Interpretation for the microscopist: the full native Berk field can now be
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
`result()` parity is also tested against explicit full storage, including the
diagnostic loss. Focused CUDA check from 2026-07-17 on GPU1:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q \
  tests/test_ssb_cuda_128.py -k 'hermitian or phase_loss'

12 passed, 12 deselected
```

Full CUDA SSB test file from the same pass:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q tests/test_ssb_cuda_128.py

24 passed
```

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

## Next performance work

Problem: the live object redraw target is met in the synthetic native-kernel
benchmark, but real-data workflow signoff is still incomplete.

Action: run the Samsung/BTO HDF5 path end to end, including load/decode,
hot-pixel filtering, BF-mask formation, Nelder-Mead/SSB setup, live controls,
FFT display, and browser/widget reporting.

Problem: the `512x512` exact phase/loss path is faster after the radix-8 row
and transposed-staging work, but `~53 ms` is still only `~19 FPS`, not the
`33.3 ms` / `30 FPS` target.

Action: attack the row-stage topology next. The current accepted kernel made
column reads coalesced by making row writes strided; a real breakthrough needs
coalesced row output and coalesced column input, or a fused/tiled row-column
design that preserves the exact per-BF phase/loss arithmetic.

Problem: the phase-variance optimizer path cannot inherit the object Fourier-
sum result because `mean(angle(object_bf))` is a different scientific quantity
from `angle(mean(object_bf))`.

Action: keep optimizing `reconstruct()` and `reconstruct_with_loss()` with
dedicated native variance kernels or an equivalent exact reformulation; do not
reuse the object-path claim for the optimizer objective.

Problem: `1024x1024` batched optimizer variance is still disabled.

Action: implement and parity-test a dedicated 1024 batch variance kernel before
enabling batch trials at that size.

Problem: MPS and WebGPU are not yet at parity with the CUDA native-size matrix.

Action: run and update the 12-cell backend matrix above. MPS has
`quantem.gpu.ssb.mps` and existing CUDA-reference tests, but the object
Fourier-sum topology has not been ported there. WebGPU SSB currently lives in
`quantem.widget`; 1024 support exists there, but reusable WGSL kernels and
parity fixtures still need to move into `quantem.gpu`.
