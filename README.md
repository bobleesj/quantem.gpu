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

Compute common BF, DF, ADF, and DPC images directly through `quantem.gpu`:

```python
from quantem.gpu import adf, bf, df, dpc, virtual

bright = bf(data)
annular = adf(data, inner=40, outer=90, unit="px")
dark = df(data)
dpc_result = dpc(data)
custom = virtual(data, mode="BF")
```

## Documentation

The docs site lives in `docs/` and mirrors the `quantem.widget` documentation
shape at a smaller compute-package scale:

- install and backend checks
- HDF5 loading and scan-region tutorials
- BF/DF/ADF, DPC, ptychographic SSB, and movie tutorials
- display-with-widget notes

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
  BF/DF/DPC, Metal bitshuffle/LZ4 IO, SSB preview/free-fit, and movie rendering
  run through Apple GPU paths where implemented.
- `cpu`: h5py/hdf5plugin reference decode for availability and reference agreement.

### Coverage snapshot

| Area | CUDA | MPS | Further work |
|---|---|---|---|
| HDF5 load/decompress | Implemented | Implemented | More held-out real-data reference agreement. |
| `load(..., scan_region=...)` | Implemented | Implemented | More malformed-file and dataset coverage. |
| BF/DF/ADF, mean DP, DPC/iDPC | Implemented | Implemented | Broader visual reference agreement reports. |
| Ptychographic SSB | Reference path implemented | MPS preview/free-fit implemented | More datasets, scan sizes, and temporal/joint SSB validation. |
| GIF/MP4 movie rendering | CUDA/NVENC MP4 | Metal render plus ffmpeg/VideoToolbox MP4 | Larger export benchmark matrix and widget button wiring. |

### Native SSB kernel tracking

The native SSB live-redraw target is tracked as a 12-cell backend matrix:
`cuda`, `mps`, and `webgpu` across `128x128`, `256x256`, `512x512`, and
`1024x1024` scan sizes. Detailed timing, reference-check status, and known bottlenecks
live in `docs/maintainer/ssb-performance.md`.

Current status:

| Backend | `128` | `256` | `512` | `1024` | Status |
|---|---:|---:|---:|---:|---|
| CUDA object / phase / loss | object `4.83 ms`; phase+loss `9.65 ms` | object `2.17 ms`; phase+loss `20.89 ms` | real full-BF phase+loss `31.27 ms` / `32.0 FPS`; synthetic phase+loss `27.46 ms` | object `40.90 ms`; phase+loss `190.88 ms` / `5.2 FPS` | CUDA 512 full-BF real-field phase/loss passes 30 FPS on GPU1. 1024 exact phase/loss uses split-512 row/column FFTs and is about `2x` faster than the old exact path, but still misses the 10/30 FPS target. |
| MPS Hermitian preview/free-fit | object `2.45 ms`; phase+loss `~8.3 ms` | object `8.62 ms`; phase `32.75 ms`; phase+loss `~34-35 ms` | object `37.65 ms`; exact phase+loss warm `~170 ms` / `~6 FPS` | object clean `~156 ms`; exact phase/loss warm `~0.8-1.0 s` / `~1 FPS` | Implemented on a Mac MPS machine for prepared Hermitian `G_qk`. Full-BF 128 is real-time, 256 phase-only reaches 30 FPS while phase+loss remains just over budget, 512 object-wave steering is usable, and 1024 exact phase/loss is about `2-2.6x` faster than the old MLX path but still not live-interactive. |
| WebGPU phase/loss widget path | supported | supported | supported | supported | Excluded from the 2026-07-17 CUDA/MPS checkpoint. 1024 range-index HDF5 workflow works in `quantem.widget`; migration to `quantem.gpu` remains pending. |

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
  full scan first. `load_scan_region()` is kept as a compatibility helper for
  existing callers.
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

- Full SSB UI inside Show4DSTEM
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
