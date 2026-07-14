# quantem.gpu

`quantem.gpu` is the multi-backend accelerated STEM package for QuantEM.
The public brand is `quantem.gpu`, not `quantem.cuda`.

## Quick Start

Install the current release candidate from TestPyPI:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc4"
```

For CUDA machines, install the CUDA extra in an environment with a matching
CUDA runtime:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda]==0.0.1rc4"
```

For Apple Silicon MPS testing:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps]==0.0.1rc4"
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
  "quantem.gpu>=0.0.1rc4"
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
- `mps`: Apple Silicon device selection is reported. Chunk-backed virtual image
  and CoM/DPC product dispatch now lives in `quantem.gpu.compute`. Metal
  bitshuffle/LZ4 chunk IO and zero-copy chunk assembly now live in
  `quantem.gpu.io.backends.mps`. SSB has MLX-backed fixed-aberration preview
  (`ssb_preview_mps`) and C10/C12/phi12 free-fit (`ssb_fit_mps`) APIs that run
  on Apple GPU without Torch. Movie rendering can use Metal for grayscale
  frame assembly before ffmpeg/H.264 encoding.
- `cpu`: h5py/hdf5plugin reference decode for availability and parity.

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
  parity tests
- `quantem.gpu.dpc` CoM/DPC/iDPC copied from the widget path with parity tests
- `quantem.gpu.ssb` SSB engine/API copied from the live compute path, with
  real-data parity and speed tests against the legacy live implementation
- `quantem.gpu.compute` MPS chunk-backed virtual-image and CoM/DPC compute
  copied from widget Metal kernels; Linux CI has dispatch guardrails, and true
  Metal runtime parity must run on macOS
- `quantem.gpu.ssb.mps.ssb_preview` / `quantem.gpu.ssb_preview_mps` and
  `quantem.gpu.ssb.mps.ssb_fit` / `quantem.gpu.ssb_fit_mps`, optional
  MLX-backed MPS SSB preview/free-fit paths for chunk-backed Mac data.
- Active `quantem.gpu`, `quantem.widget`, and `quantem.live` source trees no
  longer import the legacy `quantem.cuda` package at runtime. The remaining
  unique legacy code is primarily ptychography CLI/fused-kernel internals, which
  still need a separate backend-folding pass.

Out of scope for phase 1:

- Full SSB UI inside Show4DSTEM
- Full `quantem.live` CLI/dashboard migration, though SSB/detector production
  call sites have started routing directly through `quantem.gpu`
- Rewriting every `quantem.widget.io` helper
- Paper scripts or denoise sweeps

## Next Phases

- Phase 2: complete product migration coverage, including macOS MPS runtime
  parity and broader real-data product parity.
- Phase 3: broaden SSB real-data parity, including full CUDA-engine optimizer
  parity and dashboard integration for the MPS MLX fit path.
- Phase 4: move live CLI/dashboard callers onto `quantem.gpu`.
- Phase 5: fold/rename remaining `quantem.cuda` ptychography internals under
  `quantem.gpu` backends.
