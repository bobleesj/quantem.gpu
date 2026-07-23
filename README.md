# quantem.gpu

`quantem.gpu` is the multi-backend accelerated STEM package for QuantEM.
It is built primarily for NVIDIA CUDA workstations and Apple Silicon MPS Macs,
with CPU reference paths for availability and reference agreement checks.

## Quick Start

Install the current release candidate from TestPyPI:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc5"
```

For CUDA machines, install the CUDA extra in an environment with a matching
CUDA runtime:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda]==0.0.1rc5"
```

For Apple Silicon MPS testing:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps]==0.0.1rc5"
```

For GIF/MP4 movie rendering, include the `movie` extra. Combine extras when
you also need a device-specific backend:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[movie]==0.0.1rc5"

python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps,movie]==0.0.1rc5"
```

Check which backend will be used:

```python
import quantem.gpu as qgpu

report = qgpu.device_report()
print(report.selected)
print(report.cuda_available, report.mps_available, report.cpu_available)
```

Load a scan crop from an HDF5 master file. On CUDA this returns a CuPy array
without loading the full scan first.

```python
from quantem.gpu import load

result = load(
    "scan_master.h5",
    scan_region=(0, 32, 0, 32),  # row_start, row_stop, col_start, col_stop
)

data = result.data
print(data.shape, data.dtype, type(data))
```

Load stochastic scan positions for ptychography-style minibatches. Use
`random_positions=` when QuantEM should sample global scan positions for you;
use `scan_indices=` when your sampler already chose the positions. In both
cases, the returned array follows stochastic order, while the loader internally
sorts and de-duplicates HDF5 frame indices before GPU bitshuffle/LZ4
decompression.

```python
import numpy as np
from quantem.gpu import load

single = load(
    "scan_master.h5",
    random_positions=1000,
    scan_shape=(512, 512),
    seed=42,
)
print(single.data.shape)  # (1000, 192, 192)

rng = np.random.default_rng(42)
per_frame = np.vstack([  # explicit user-provided positions
    rng.choice(512 * 512, size=1000, replace=False)
    for _ in range(40)
])
series = load(
    master_paths[:40],
    scan_indices=per_frame,
    scan_shape=(512, 512),
    prep_workers=4,
)
print(series.data.shape)  # (40, 1000, 192, 192)

random_series = load(
    master_paths[:40],
    random_positions=1000,
    scan_shape=(512, 512),
    seed=42,
)
```

For multi-master stochastic batches, `prep_workers=` controls how many HDF5
masters are prepared in parallel before the GPU bitshuffle/LZ4 decoder runs.
Benchmark this on the storage path you will use. On one real-data 40-master
`512x512x192x192` random-position benchmark, a true cold scattered read took
about `8.90 s` with the default single worker; `2`, `4`, and `8` workers were
not faster (`8.98 s`, `9.47 s`, `9.97 s`). With the OS page cache warm, the
same 40 x 1000-position native-`uint16` batch loads in about `1.0-1.6 s`, and
`8` workers still regresses. Use more workers only when local measurement shows
that the payload files live on storage that scales with concurrent reads.

This sparse path is designed for no-bin ptychography on modest VRAM. A full
`1024x1024x192x192 uint16` acquisition is about `77 GB` and cannot be resident
on a 24 GB GPU, but a stochastic `1000x192x192 uint16` batch is only about
`74 MB` before ptychography working buffers. Build BF/DF/CoM/rotation products
once as a small cache, then decode random HDF5 batches into VRAM for the
optimizer step and release them.

For screen-style launch, do not recompute BF/DF/DPC from the raw HDF5 every
time. Use the cached product API:

```python
from quantem.gpu import load_calibration_products

products = load_calibration_products(
    "scan_master.h5",
    backend="auto",
)

print(products.loaded_from_cache, products.elapsed_s)
print(products.bf.shape, products.df.shape, products.rotation_deg)
```

On a cache miss this streams the raw HDF5 once with GPU bitshuffle/LZ4 decode
and backend-native BF/DF/CoM kernels. CUDA uses the optimized RawKernel path;
MPS uses chunk-backed Metal reductions and the same crop-first row streaming
policy. The default cache build estimates the BF disk from the first decoded row
chunk so it does not pay a second HDF5 pass before the streaming reduction; pass
`sample_positions>0` only when an explicit random probe sample is needed. On a
cache hit it reads only the small derived arrays, so UI launch can be well below
the `0.5 s` target. Cache hits are backend-neutral.

By default, `load_calibration_products()` inspects current free CUDA VRAM on
CUDA machines and otherwise uses a conservative streaming plan. Pass
`memory_budget_gb=` only to force a smaller or larger working set. For a real
`1024x1024x192x192 uint16` compressed master, CUDA cache-miss timing is about
`7.1 s` with a `24 GB` budget, `7.3 s` with `48 GB`, and `3.8 s` with `96 GB`
because the full `77 GB` raw scan fits and can use the optimized full-master
loader. On a real `512x512x192x192 uint16` master, MPS cache generation with
64-row chunks measured `3.96 s` and matched CUDA products exactly for mean DP,
BF, and DF, with CoM max error `7.63e-6`. Once cached, loading
BF/DF/CoM/rotation products is about `0.01-0.2 s`, and repeating the rotation
search on the cached CoM maps is about `0.027 s` median. Ptychography sweeps
should reuse this calibration cache rather than recomputing BF/DF/rotation for
every trial.

Compute common BF, DF, ADF, and DPC images directly through `quantem.gpu`:

```python
from quantem.gpu import adf, bf, df, dpc, virtual

bright = bf(data)
annular = adf(data, inner=40, outer=90, unit="px")
dark = df(data)
dpc_result = dpc(data)
custom = virtual(data, mode="BF")
```

Check whether a large planned virtual-image workflow has a custom GPU kernel
before allocating the array:

```python
from quantem.gpu import virtual_image_kernel_support

support = virtual_image_kernel_support(
    backend="cuda",
    shape=(1024, 1024, 192, 192),
    dtype="uint8",
    bf_radius=30,
)
print(support.resident_gib, support.mask_paths)
```

## Documentation

The docs site lives in `docs/` and mirrors the `quantem.widget` documentation
shape at a smaller compute-package scale:

- install and backend checks
- HDF5 loading and scan-region tutorials
- BF/DF/ADF, DPC, ptychographic SSB, and movie tutorials
- display-with-widget notes

`quantem.gpu.webgpu` ships canonical browser-compute TypeScript/WGSL sources
for widget/export bundling:

```python
from quantem.gpu import webgpu

print(webgpu.source_names())
compute_ts = webgpu.source_text("compute.ts")
```

The shipped Show4DSTEM WebGPU source covers GPU-resident BF/DF/ADF masked
reductions, DPC row/col reducers, and fixed-rotation iDPC; `quantem.widget`
bundles these sources for browser/offline HTML use while keeping the widget
package focused on UI.

Build it locally with:

```bash
python -m pip install -e ".[docs]"
jupyter-book build docs
```

Use the widget display migration branch with this release candidate:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[movie]>=0.0.1rc5"
```

Then existing widget calls continue to work while the heavy load and compute
paths route through `quantem.gpu`:

```python
from quantem.widget import Show4DSTEM, load

result = load("scan_master.h5", scan_region=(0, 32, 0, 32))
viewer = Show4DSTEM(result.data)
viewer
```

## Charter

`quantem.gpu` owns:

- GPU IO and decompression: bitshuffle/LZ4 HDF5 chunk decode, chunk assembly,
  load-to-device, pinned/zero-copy host transfer paths, and detector masking
  during decode.
- Heavy compute: virtual images, BF/DF/DPC, reductions, and later SSB engines.
- Device policy: explicit `cuda`, `mps`, and `cpu` selection with honest errors.

`quantem.widget` owns frontend behavior: anywidget UI, export, interaction, and
display. It should call `quantem.gpu` for accelerated load and compute.

`quantem.live` apps, CLI, and dashboard should move to `quantem.gpu` over time
instead of keeping second permanent copies of load/decompress/math paths.

Dependency arrow:

```text
file -> quantem.gpu (load + decompress + to_device) -> arrays
     -> quantem.gpu (BF/DF/DPC / SSB / movies) -> quantem.widget (display)
```

## Backends

- `cuda`: CuPy RawKernel bitshuffle/LZ4 decompression, GPU arrays, and
  CUDA/NVENC MP4 rendering. This is the phase-1 migrated hot path.
- `mps`: Apple Silicon Metal/MLX paths for MacBook-scale 4D-STEM. The raw
  Metal loaders keep data chunk-backed and avoid materializing one giant
  Torch-MPS tensor, which matters because Torch-MPS can hit 32-bit indexing /
  `>2^31` element limits and unified-memory pressure on full 4D-STEM stacks.
  BF/DF/DPC, Metal bitshuffle/LZ4 IO, MPS `uint8` browse loads, SSB
  preview/free-fit, and movie rendering run through Apple GPU paths where
  implemented.
- `cpu`: h5py/hdf5plugin reference decode for availability and reference agreement.
- `webgpu`: canonical browser-compute sources shipped in `quantem.gpu.webgpu`;
  widget bundles them into anywidget JavaScript and exported HTML.

### Feature matrix

Status terms: `Done` means implemented with real-data parity and performance
evidence; `Partial` means source exists but the full signoff matrix is not
complete; `Gap` means the backend does not implement that capability yet.

| Capability | CUDA | MPS | WebGPU | Notes |
|---|---|---|---|---|
| Device report and explicit selection | Done | Done | NA | WebGPU adapter selection happens in the browser; software adapters are rejected for timing claims. |
| HDF5 master metadata and discovery | Done | Done | Done | One shared API should serve widget and live callers. |
| Full HDF5 bitshuffle/LZ4 load/decompress | Done | Done | Done | CUDA uses CuPy/CUDA kernels; MPS uses Metal chunk-backed unified memory; WebGPU uses browser local-file HDF5 plus WGSL decode. WebGPU strict full-stack no-bin `1024x1024x192x192` browse is intentionally rejected as a memory-policy path; use product-first, crop, or explicit bin. |
| `load(..., scan_region=...)` crop-first IO | Done | Done | Done | CUDA/MPS crop during load; WebGPU slices frame windows before upload/decode. |
| Detector bin during load, min-memory | Done | Done | Done | WebGPU has an explicit count-preserving `detBin` load option in the local-H5 source; full `512x512x192x192` `detBin=2/4/8` headed parity is exact on a real NVIDIA WebGPU adapter, including native non-low8 `uint16` `detBin=2`. |
| BF/DF/ADF resident kernels | Done | Done | Done | CUDA RawKernel, MPS Metal, and WebGPU WGSL selected reducers are implemented. |
| Dense DF/ADF strategy | Done | Done | Done | Uses cached full-detector total minus complement when that is cheaper than scanning dense masks. |
| CoM/DPC resident kernels | Done | Done | Done | CUDA and MPS have fused moment kernels; WebGPU row/col DPC has full no-bin headed signoff on real hardware. |
| iDPC | Done | Done | Done | WebGPU has a fixed-rotation browser iDPC solver using paired DPC buffers and a dual-real FFT. It matches the Python reference within float32 FFT tolerance, not bit-exact. |
| Ptychographic SSB preview/object steering | Done | Done | Partial | CUDA and MPS are implemented; WebGPU source exists through `quantem.gpu.webgpu` and widget bundling. |
| Ptychographic SSB optimizer/free-fit | Done | Done | Partial | MPS supports current parity shapes but large exact phase/loss remains slower than CUDA. |
| GIF/MP4 movie rendering | Done | Done | NA | CUDA/NVENC and Metal/VideoToolbox paths live here; widget owns buttons/export UI. |
| Browser source ownership | Done | Done | Done | Reusable TypeScript/WGSL sources live in `quantem.gpu.webgpu`; widget bundles them. |

Before adding another custom kernel, run `virtual_image_kernel_support(...)` for
the target shape/dtype and update the maintainer matrices with the same backend,
shape, dtype, parity metric, timing split, and memory footprint. The supported
kernel families are:

| Kernel family | CUDA source | MPS source | WebGPU source | Required gate |
|---|---|---|---|---|
| HDF5 bitshuffle/LZ4 decode | `quantem.gpu.io.backends.cuda` | `quantem.gpu.io.backends.mps` | `quantem.gpu.webgpu.bslz4` and `local-h5.ts` | Corrected-frame checksum parity and load-stage timing. |
| BF/DF/ADF masked sums | `quantem.gpu.compute.cuda` / `detector` | `quantem.gpu.compute.mps` | `quantem.gpu.webgpu.compute` / `local-h5.ts` | Exact integer product parity and first/warm interaction timing. |
| CoM/DPC | `quantem.gpu.compute.cuda` / `dpc` | `quantem.gpu.compute.mps` / `dpc` | `quantem.gpu.webgpu.compute` | Row/col CoM and centered DPC parity within `1e-5`. |
| SSB object, phase, loss | `quantem.gpu.ssb.cuda` | `quantem.gpu.ssb.mps` | `quantem.gpu.webgpu.showptycho-ssb` | Same BF policy, same aberrations, phase/loss parity, and interactive redraw timing. |
| Movie rendering | `quantem.gpu.movie.cuda` | `quantem.gpu.movie.mps` | NA | Frame parity and encoded movie smoke tests. |

### Backend performance snapshot

These public-safe numbers summarize the current full-size Show4DSTEM load and
browser product work without raw file paths or project-specific dataset names.
The full-stack rows use `512x512x192x192` HDF5 evidence. CUDA reference timing
was measured in `cuda-env` on an NVIDIA RTX PRO 6000 Blackwell GPU. WebGPU
timing used real Chrome WebGPU on Apple Metal or NVIDIA Blackwell as listed,
with software adapters rejected.

| Path | Backend / hardware | Evidence shape | Median | Parity / notes |
|---|---|---:|---:|---|
| HDF5 load/decompress | CUDA, RTX PRO 6000 Blackwell | `512x512x192x192` | `450 ms` over 946 runs | Reference warm load; min `408 ms`, max `1159 ms`, resident stack `9.66 GB`. |
| HDF5 load/decompress | CUDA, RTX PRO 6000 Blackwell | true `1024x1024x192x192` | `4.704 s` | Real acquisition, no bin/crop, `uint16` output, selected corrected frames bit-exact, resident stack `77.31 GB`. |
| HDF5 load/decompress | MPS, Apple Metal | true `1024x1024x192x192` | `4.617 s` | Real acquisition, no bin/crop, chunk-backed `uint16` output, selected corrected frames bit-exact, resident stack `77.31 GB`. |
| Local HDF5 full-stack load | WebGPU, Chrome Apple Metal | `512x512x192x192` | `772 ms` over 946 runs | Corrected-frame checksum parity versus CUDA; min `726 ms`, max `879 ms`; full path still materializes the `9.7 GB` browse cube. |
| Local HDF5 full-stack load | WebGPU, Chrome NVIDIA Blackwell | true `1024x1024x192x192`, no crop/bin | Rejected | Attempt reached about `97.2 GB` GPU memory and failed before publishing a load profile/checksum readback. Do not count strict full-stack browser browse as signed off for 1024; use product-first, true crop, or explicit detector-bin paths. |
| Local HDF5 detector-bin load | WebGPU, Chrome NVIDIA Blackwell | full `512x512x192x192` and true `256x256` crop, `detBin=2/4/8` | full page profiles `1199/1212/1106 ms`; crop p95 `798/813/775 ms` | Corrected-frame checksum parity exact versus zero-bad-before-bin reference; crop medians `774/755/733 ms`; native non-low8 `uint16` `detBin=2` also exact at `2651 ms`. |
| Local HDF5 scan crop | WebGPU, Chrome Apple Metal | true `256x256x192x192` crop | `338 ms` over 946 runs | Corrected-frame checksum parity versus CUDA; min `316 ms`, max `464 ms`. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | true `256x256`, BF radius `30` | `210 ms` over 946 runs | Product max/mean abs error `0` versus CUDA; min `185 ms`, max `246 ms`. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | full `512x512`, BF radius `30` | `378 ms` over 945 successful runs | Product max/mean abs error `0` versus CUDA; min `358 ms`, max `473 ms`. |
| Product-first BF selected-block sidecar | WebGPU, Chrome NVIDIA Blackwell | true `1024x1024`, BF radius `30` | `4.92 s` wall; `1.56 s` product stage | True real-acquisition product-first BF signoff; selected compressed payload `6.88 GB`, output `4.19 MB`, max/mean abs error `0` versus an independent Python reference. This is not full-stack no-bin browse/load signoff. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | `1024x1024` repeat-stress, BF radius `30` | `1170 ms` over 944 successful runs | Product max/mean abs error `0`; this is four repeats of real `512` evidence, not a true 1024 acquisition signoff. |
| Visible Show4DSTEM interaction | WebGPU, Chrome Apple Metal | full `512x512x192x192` local HDF5 | full load `933 ms`; drag frames `0.5-0.9 ms` | BF/ADF/DPC display interactions stay GPU-resident after load; warm cached BF/ADF/DPC hits were `0.1-0.5 ms`. |
| DPC/iDPC display | WebGPU, Chrome NVIDIA Blackwell | full `512x512x192x192` no-bin | DPC row/col/iDPC display medians `14.9/13.2/13.2 ms` | Headed real-adapter signoff after FFT command batching; full recompute medians `13.7/19.3/22.7 ms`; corrected-frame parity passed; DPC max abs error `7.63e-6`; iDPC mean abs error `4.70e-6`, max `3.05e-5` from float32 FFT order; idle RAF `60 FPS`. Local-file timing harness runs use `--require-local-profile` so URL fallback is rejected. |

Across the 8-hour browser soak there were 5 transient Chrome/CDP socket or
timeout harness failures among 5676 recorded rows. Successful parity rows had no
numeric mismatch.

### Native SSB kernel tracking

The native SSB live-redraw target is tracked as a 12-cell backend matrix:
`cuda`, `mps`, and `webgpu` across `128x128`, `256x256`, `512x512`, and
`1024x1024` scan sizes. Detailed timing, reference-check status, and known bottlenecks
live in `docs/maintainer/ssb-performance.md`.

Current status:

| Backend | `128` | `256` | `512` | `1024` | Status |
|---|---:|---:|---:|---:|---|
| CUDA object / phase / loss | object `4.83 ms`; phase+loss `9.65 ms` | object `2.17 ms`; phase+loss `20.89 ms` | real full-BF phase+loss `31.27 ms` / `32.0 FPS`; synthetic phase+loss `27.46 ms` | object `40.90 ms`; phase+loss `190.88 ms` / `5.2 FPS` | CUDA 512 full-BF real-field phase/loss passes 30 FPS on GPU1. 1024 exact phase/loss uses split-512 row/column FFTs and is about `2x` faster than the old exact path, but still misses the 10/30 FPS target. |
| MPS Hermitian preview/free-fit | object `2.45 ms`; phase+loss `~8.3 ms` | object `8.62 ms`; phase `32.75 ms`; phase+loss `~34-35 ms` | radius-30 real field: object `10.86 ms`, phase+loss `76.28 ms`; full active real field: object `55.20 ms`, phase+loss `528.90 ms` | object `143 ms`; exact phase+loss `~669 ms` for full-BF-sized synthetic `G_qk` | Implemented on a Mac MPS machine for prepared Hermitian `G_qk`. Full-BF 128 is real-time, 256 phase-only reaches 30 FPS, 512 radius-30 object-wave steering is real-time, and larger exact phase/loss remains much slower than CUDA. |
| WebGPU phase/loss widget path | supported | supported | supported | supported | Browser runtime bundled by `quantem.widget`; reusable TypeScript/WGSL source is shipped in `quantem.gpu.webgpu`. |

Do not treat this table as a reason to downsample or crop. Full-resolution
claims must keep the BF policy, scan size, and scientific objective unchanged.

## Phase-1 Status

Implemented in this package:

- `import quantem.gpu`
- `quantem.gpu.device_report()` and `quantem.gpu.select_device()`
- `quantem.gpu.io.hdf5.load()`, copied from the proven `quantem.widget` HDF5
  loader and kept API-compatible for the migrated slice
- `quantem.gpu.io.hdf5.load(..., scan_region=(row_start, row_stop, col_start,
  col_stop))` for CUDA and MPS scan-ROI HDF5 loading without materializing the
  full scan first.
- `quantem.gpu.io.hdf5.load(..., scan_indices=...)` and
  `load_scan_indices()` for PyTorch/DataLoader-style stochastic scan batches:
  random positions are returned in requested order, while compressed HDF5 chunks
  are sorted and de-duplicated before CUDA/MPS GPU decompression.
- `quantem.gpu.io.hdf5.load(..., random_positions=...)` and
  `random_scan_indices()` for one-line global random HDF5 minibatches with
  reproducible seeds and explicit multi-file HDF5 preparation workers when a
  measured storage path benefits from concurrent readers.
- CUDA bitshuffle/LZ4 kernels and pinned-buffer HDF5 master load path
- MPS Metal bitshuffle/LZ4 kernels, chunk-backed zero-copy load path, memory
  guard, crop-first sparse decode, lazy multi-dataset loader, and
  `load_mps_4dstem`
- `quantem.widget.io.hdf5` shim in the migration worktree, re-exporting the new
  `quantem.gpu.io.hdf5` API for one release
- `quantem.gpu.detector` BF/DF/ADF, `mean_dp`, `masked_sum`, `dp_mean`,
  `virtual_image`, and BF disk detection copied from the widget/live paths with
  reference checks
- `quantem.gpu.dpc` CoM/DPC/iDPC copied from the widget path with reference checks
- `quantem.gpu.ssb` SSB engine/API copied from the live compute path, with
  real-data reference agreement and speed tests against the legacy live implementation
- `quantem.gpu.compute` MPS chunk-backed virtual-image and CoM/DPC compute
  copied from widget Metal kernels; Linux CI has dispatch guardrails, and true
  Metal runtime reference agreement must run on macOS
- `quantem.gpu.ssb.mps.ssb_preview` / `quantem.gpu.ssb_preview_mps` and
  `quantem.gpu.ssb.mps.ssb_fit` / `quantem.gpu.ssb_fit_mps`, optional
  MLX-backed MPS SSB preview/free-fit paths for chunk-backed Mac data.
- Active `quantem.gpu`, `quantem.widget`, and `quantem.live` source trees route
  migrated load and compute paths through `quantem.gpu`. The remaining unique
  ptychography CLI/fused-kernel internals still need a separate backend-folding
  pass.

Out of scope for phase 1:

- Ptychographic SSB UI work beyond the compute handoff
- Full `quantem.live` CLI/dashboard migration, though SSB/detector production
  call sites have started routing directly through `quantem.gpu`
- Rewriting every `quantem.widget.io` helper
- Paper scripts or denoise sweeps

## Next Phases

- Phase 2: complete product migration coverage, including macOS MPS runtime
  reference agreement and broader real-data product reference agreement.
- Phase 3: broaden SSB real-data reference agreement, including full CUDA-engine optimizer
  reference agreement and dashboard integration for the MPS MLX fit path.
- Phase 4: move live CLI/dashboard callers onto `quantem.gpu`.
- Phase 5: fold remaining ptychography internals under `quantem.gpu` backends.
