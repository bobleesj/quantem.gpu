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

backend = qgpu.device.detect()
print(backend)
```

Load a scan crop from an HDF5 master file. On CUDA this returns a CuPy array
without loading the full scan first.

```python
from quantem.gpu.io import load

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
from quantem.gpu.io import load

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
)
print(series.data.shape)  # (40, 1000, 192, 192)

random_series = load(
    master_paths[:40],
    random_positions=1000,
    scan_shape=(512, 512),
    seed=42,
)
```

Multi-master stochastic loading uses an internal bounded preparation scheduler.
The public API intentionally does not expose storage-worker tuning.

This sparse path is designed for no-bin ptychography on modest VRAM. A full
`1024x1024x192x192 uint16` acquisition is about `77 GB` and cannot be resident
on a 24 GB GPU, but a stochastic `1000x192x192 uint16` batch is only about
`74 MB` before ptychography working buffers. Build BF/DF/CoM/rotation products
once as a small cache, then decode random HDF5 batches into VRAM for the
optimizer step and release them.

For screen-style launch, do not recompute BF/DF/DPC from the raw HDF5 every
time. Use the cached product API:

```python
from quantem.gpu import screening

products = screening.prepare(
    "scan_master.h5",
    backend="auto",
)

print(products.from_cache, products.elapsed_s)
print(products.bright_field.shape, products.dark_field.shape)
print(products.rotation_deg)
```

On a cache miss this streams the raw HDF5 once with GPU bitshuffle/LZ4 decode
and backend-native BF/DF/CoM kernels. CUDA uses the optimized RawKernel path;
MPS uses chunk-backed Metal reductions and the same crop-first row streaming
policy. The default cache build estimates the BF disk from the first decoded row
chunk so it does not pay a second HDF5 pass before the streaming reduction; pass
`sample_positions>0` only when an explicit random probe sample is needed. On a
cache hit it reads only the small derived arrays, so UI launch can be well below
the `0.5 s` target. Cache hits are backend-neutral.

By default, `screening.prepare()` inspects current free CUDA VRAM on
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

## HDF5 IO Backend Status

`quantem.gpu` treats Arina-style HDF5 as a backend contract, not just a file
extension. A standard 4D-STEM save has a `*_master.h5` file with external
`*_data_000001.h5` links, payload at `/entry/data/data`, one diffraction
pattern per chunk `(1, det_row, det_col)`, and Bitshuffle/LZ4 filter `32008`.

Current real-data status for full no-bin `512x512x192x192` detector data:

| Workflow | CUDA | MPS / Apple GPU | CPU / HDF5 filter | Notes |
| --- | ---: | ---: | ---: | --- |
| Load/decode `uint16` Arina H5 | Done | Done | Reference | CUDA and MPS preserve native integer evidence. |
| Save Arina H5, `uint16` | Done | Done, `~1.69-1.96 s`; default-path check `1.753 s` | Done, `~30.4 s` | Chunk-backed MPS loads use native Metal Bitshuffle/LZ4 directly from the Metal buffers plus async HDF5 `write_direct_chunk`; random-sample agreement vs decoded reference was exact. The fastest path avoids a second full raw copy, with a modest file-size tradeoff. |
| Save Arina H5, `float32` | Done | Done, `~6.8 s` | Done | MPS stores lossless float32 Bitshuffle+LZ4 chunks; synthetic round trip is exact. |
| Save Arina H5, `uint8` | Display-only GPU path | Done, `~1.36-1.57 s` | Done, `~19.3 s` | MPS/CUDA use GPU Bitshuffle/LZ4 for clipped display exports; full real-data MPS sampled agreement was exact against `min(uint16, 255)`. It is not scientific agreement when counts exceed 255. |
| 1-2 s full save target | Partial | Done for `uint8` and `uint16` | Gap | MPS async write overlap plus native Metal chunk-buffer compression put full no-bin display and exact-count saves inside the target band. |

Use `save()` for portable master/data-file exports:

```python
from quantem.gpu import io

io.save(
    "merged_master.h5",
    merged_mps_tensor,
    scan_shape=(512, 512),
    dtype="u16",
    backend="auto",  # infer CUDA or MPS/Metal from the resident array
)
```

NumPy reference writes require `backend="cpu"` explicitly; `auto` never
silently selects the CPU scientific path.

Compute common BF, DF, ADF, and DPC images through their scientific domains:

```python
from quantem.gpu import detector, dpc

bright = detector.bf(data)
annular = detector.adf(data, inner=40, outer=90, unit="px")
dark = detector.df(data)
dpc_result = dpc.run(data)
custom = detector.virtual(data, mode="BF")
```

Kernel authors can audit large virtual-image plans with the private coverage
helper described in the
[maintainer checklist](docs/maintainer/virtual-image-kernel-checklist.md).
Scientist-facing code should stay on `quantem.gpu.detector`.

## Documentation

The docs site lives in `docs/` and mirrors the `quantem.widget` documentation
shape at a smaller compute-package scale:

- install and backend checks
- HDF5 loading and scan-region tutorials
- BF/DF/ADF, DPC, ptychographic SSB, and movie tutorials
- display-with-widget notes

Reusable browser compute is owned beside its scientific domain:
`device/webgpu.ts`, `io/backends/webgpu`, `detector/compute/webgpu`,
`dpc/compute/webgpu`, and `ssb/compute/webgpu`. `quantem.widget` bundles these
canonical sources verbatim for browser and offline HTML execution.

The shipped Show4DSTEM WebGPU source covers GPU-resident BF/DF/ADF masked
reductions, DPC row/col reducers, and fixed-rotation iDPC; `quantem.widget`
bundles these sources for browser/offline HTML use while keeping the widget
package focused on UI.

`quantem.gpu` is intentionally a compute and IO library, not the user-facing
Show4DSTEM command package. Use `quantem.widget` for CLI launchers such as:

```bash
quantem show4dstem /data/session --backend mps --count 7 --bin 1 --dtype u8
quantem show4dstem /data/session --backend cuda --count 7 --devices 0,1 --bin 1 --dtype u8
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype u8
```

`--devices 0,1` is CUDA placement. WebGPU runs inside the browser on one
selected adapter and consumes the domain-owned sources bundled by
`quantem.widget`.

Build it locally with:

```bash
python -m pip install -e ".[docs]"
jupyter-book build docs
```

Install the matching GPU release candidate before testing widget integration:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[movie]>=0.0.1rc5"
```

Then existing widget calls continue to work while the heavy load and compute
paths route through `quantem.gpu`:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

result = load("scan_master.h5", scan_region=(0, 32, 0, 32))
viewer = Show4DSTEM(result.data)
viewer
```

## Charter

`quantem.gpu` owns:

- GPU IO and decompression: bitshuffle/LZ4 HDF5 chunk decode, chunk assembly,
  load-to-device, pinned/zero-copy host transfer paths, and detector masking
  during decode.
- Heavy compute: virtual images, BF/DF/DPC, reductions, and SSB fitting,
  reconstruction, and preview.
- Device policy: explicit `cuda`, `mps`, and browser `webgpu` selection with
  no silent CPU scientific fallback.

`quantem.widget` owns frontend behavior: anywidget UI, export, interaction, and
display. It should call `quantem.gpu` for accelerated load and compute.

`quantem.live` apps, CLI, and dashboard call these public APIs rather than
keeping second permanent copies of load/decompress/math paths.

Dependency arrow:

```text
file -> quantem.gpu (load + decompress + to_device) -> arrays
     -> quantem.gpu (BF/DF/DPC / SSB / movies) -> quantem.widget (display)
```

## Backends

- `cuda`: CuPy RawKernel bitshuffle/LZ4 decompression, GPU arrays, and
  CUDA/NVENC MP4 rendering.
- `mps`: Apple Silicon Metal/MLX paths for MacBook-scale 4D-STEM. The raw
  Metal loaders keep data chunk-backed and avoid materializing one giant
  Torch-MPS tensor, which matters because Torch-MPS can hit 32-bit indexing /
  `>2^31` element limits and unified-memory pressure on full 4D-STEM stacks.
  BF/DF/DPC, Metal bitshuffle/LZ4 IO, MPS `uint8` browse and native `uint32`
  loads, SSB fit/reconstruction/preview, and movie rendering run through Apple
  GPU paths where
  implemented.
- `cpu`: test-only h5py/hdf5plugin reference decode.
- `webgpu`: domain-owned browser sources bundled by the widget into anywidget
  JavaScript and exported HTML.

### Count dtypes

Use explicit names for detector-count storage:

```python
from quantem.gpu.io import load

native_u32 = load("scan_master.h5", dtype="uint32").data  # four-byte source counts
packed_u4 = load("scan_master.h5", dtype="u4").data       # CUDA only, values 0..15
```

`dtype="u4"` never means NumPy's four-byte `<u4` dtype. It means packed
4-bit detector counts, two pixels per byte, and CUDA raises if any corrected
count exceeds 15. Use `dtype="uint32"` or `dtype="u32"` for native four-byte
detector sources.

### Feature matrix

Status terms: `Done` means implemented with real-data parity and performance
evidence; `Partial` means source exists but the full signoff matrix is not
complete; `Gap` means the backend does not implement that capability yet.

| Capability | CUDA | MPS | WebGPU | Notes |
|---|---|---|---|---|
| Device report and explicit selection | Done | Done | NA | WebGPU adapter selection happens in the browser; software adapters are rejected for timing claims. |
| HDF5 master metadata and discovery | Done | Done | Done | One shared API should serve widget and live callers. |
| Full HDF5 bitshuffle/LZ4 load/decompress | Done | Done | Done | CUDA uses CuPy/CUDA kernels; MPS uses Metal chunk-backed unified memory; WebGPU uses browser local-file HDF5 plus WGSL decode. Native `uint8`/`uint16`/`uint32` detector sources are supported; `load(..., dtype='uint32')` or `dtype='u32'` requests four-byte unsigned output. `load(..., dtype='u4')` means true packed 4-bit counts (`0..15`), not NumPy `<u4`; CUDA returns a packed two-counts-per-byte array after an exact range audit. MPS/WebGPU HDF5 packed `u4` output is a named gap and raises honestly. WebGPU strict full-stack no-bin `1024x1024x192x192` browse is intentionally rejected as a memory-policy path; use product-first, crop, or explicit bin. |
| `load(..., scan_region=...)` crop-first IO | Done | Done | Done | CUDA/MPS crop during load; WebGPU slices frame windows before upload/decode. |
| Detector bin during load, min-memory | Done | Done | Done | WebGPU has an explicit count-preserving `detBin` load option in the local-H5 source; full `512x512x192x192` `detBin=2/4/8` headed parity is exact on a real NVIDIA WebGPU adapter, including native non-low8 `uint16` `detBin=2`. |
| BF/DF/ADF resident kernels | Done | Done | Done | CUDA RawKernel, MPS Metal, and WebGPU WGSL selected reducers are implemented for `uint8`/`uint16`/`uint32` resident data; CUDA also has packed `uint4` selected/dense reducers and CoM kernels. |
| Dense DF/ADF strategy | Done | Done | Done | Uses cached full-detector total minus complement when that is cheaper than scanning dense masks. |
| CoM/DPC resident kernels | Done | Done | Done | CUDA and MPS have fused moment kernels; WebGPU row/col DPC has full no-bin headed signoff on real hardware. |
| iDPC | Done | Done | Done | WebGPU has a fixed-rotation browser iDPC solver using paired DPC buffers and a dual-real FFT. It matches the Python reference within float32 FFT tolerance, not bit-exact. |
| Ptychographic SSB preview | Done | Done | Partial | CUDA and MPS are implemented; WebGPU SSB source lives under `quantem.gpu.ssb.compute.webgpu` and is bundled by the widget. |
| Ptychographic SSB fit/reconstruction | Done | Done | Partial | MPS supports current parity shapes but large exact phase/loss remains slower than CUDA. |
| GIF/MP4 movie rendering | Done | Done | NA | CUDA/NVENC and Metal/VideoToolbox paths live here; widget owns buttons/export UI. |
| Browser source ownership | Done | Done | Done | TypeScript/WGSL source lives beside each device, IO, detector, DPC, or SSB domain; widget bundles it. |

Before adding another custom kernel, follow the private coverage check in the
maintainer checklist and update the maintainer matrices with the same backend,
shape, dtype, parity metric, timing split, and memory footprint. The supported
kernel families are:

| Kernel family | CUDA source | MPS source | WebGPU source | Required gate |
|---|---|---|---|---|
| HDF5 bitshuffle/LZ4 decode | `quantem.gpu.io.backends.cuda` | `quantem.gpu.io.backends.mps` | `quantem.gpu.io.backends.webgpu` and `local-h5.ts` | Corrected-frame checksum parity and load-stage timing. |
| BF/DF/ADF masked sums | `quantem.gpu.detector.compute.cuda` | `quantem.gpu.detector.compute.mps` | `quantem.gpu.detector.compute.webgpu` | Exact integer product parity and first/warm interaction timing. |
| CoM/DPC | `quantem.gpu.dpc.compute.cuda` | `quantem.gpu.dpc.compute.mps` | `quantem.gpu.dpc.compute.webgpu` | Row/col CoM and centered DPC parity within `1e-5`. |
| SSB object, phase, loss | `quantem.gpu.ssb.compute.cuda` | `quantem.gpu.ssb.compute.mps` | `quantem.gpu.ssb.compute.webgpu` | Same BF policy, same aberrations, phase/loss parity, and interactive redraw timing. |
| Movie rendering | `quantem.gpu.movie.cuda` | `quantem.gpu.movie.mps` | NA | Frame parity and encoded movie smoke tests. |

### Backend performance snapshot

These public-safe numbers summarize the current full-size Show4DSTEM load and
browser product work without raw file paths or project-specific dataset names.
The full-stack rows use `512x512x192x192` HDF5 evidence. CUDA reference timing
was measured in an isolated environment on an NVIDIA RTX PRO 6000 Blackwell GPU. WebGPU
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
| Product-first BF selected-block cache | WebGPU, Chrome Apple Metal | true `256x256`, BF radius `30` | `210 ms` over 946 runs | Product max/mean abs error `0` versus CUDA; min `185 ms`, max `246 ms`. |
| Product-first BF selected-block cache | WebGPU, Chrome Apple Metal | full `512x512`, BF radius `30` | `378 ms` over 945 successful runs | Product max/mean abs error `0` versus CUDA; min `358 ms`, max `473 ms`. |
| Product-first BF selected-block cache | WebGPU, Chrome NVIDIA Blackwell | true `1024x1024`, BF radius `30` | `4.92 s` wall; `1.56 s` product stage | True real-acquisition product-first BF signoff; selected compressed payload `6.88 GB`, output `4.19 MB`, max/mean abs error `0` versus an independent Python reference. This is not full-stack no-bin browse/load signoff. |
| Product-first BF selected-block cache | WebGPU, Chrome Apple Metal | `1024x1024` repeat-stress, BF radius `30` | `1170 ms` over 944 successful runs | Product max/mean abs error `0`; this is four repeats of real `512` evidence, not a true 1024 acquisition signoff. |
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
| CUDA object / phase / loss | object `4.83 ms`; phase+loss `9.65 ms` | object `2.17 ms`; phase+loss `20.89 ms` | real full-BF phase+loss `31.27 ms` / `32.0 FPS`; synthetic phase+loss `27.46 ms` | object `40.90 ms`; phase+loss `190.88 ms` / `5.2 FPS` | CUDA 512 full-BF real-field phase/loss passes 30 FPS on the reference GPU. 1024 exact phase/loss uses split-512 row/column FFTs and is about `2x` faster than the old exact path, but still misses the 10/30 FPS target. |
| MPS Hermitian preview | object `2.45 ms`; phase+loss `~8.3 ms` | object `8.62 ms`; phase `32.75 ms`; phase+loss `~34-35 ms` | radius-30 real field: object `10.86 ms`, phase+loss `76.28 ms`; full active real field: object `55.20 ms`, phase+loss `528.90 ms` | object `143 ms`; exact phase+loss `~669 ms` for full-BF-sized synthetic `G_qk` | Implemented on an Apple Silicon MPS machine for prepared Hermitian `G_qk`. Full-BF 128 is real-time, 256 phase-only reaches 30 FPS, 512 radius-30 object-wave steering is real-time, and larger exact phase/loss remains much slower than CUDA. |
| WebGPU phase/loss widget path | supported | supported | supported | supported | Browser runtime bundled by `quantem.widget`; reusable TypeScript/WGSL source is owned by each scientific domain, with SSB implementation files under `quantem.gpu.ssb.compute.webgpu`. |

Do not treat this table as a reason to downsample or crop. Full-resolution
claims must keep the BF policy, scan size, and scientific objective unchanged.

## Current implementation status

Implemented in this package:

- `import quantem.gpu`
- `quantem.gpu.device.detect()` and `quantem.gpu.device.resolve()`
- `quantem.gpu.io.load()` as the single CUDA/MPS HDF5 load entry point.
- `quantem.gpu.io.load(..., scan_region=(row_start, row_stop, col_start,
  col_stop))` for scan-ROI loading without materializing the full scan first.
- `quantem.gpu.io.load(..., scan_indices=...)` for
  PyTorch/DataLoader-style stochastic scan batches:
  random positions are returned in requested order, while compressed HDF5 chunks
  are sorted and de-duplicated before CUDA/MPS GPU decompression.
- `quantem.gpu.io.load(..., random_positions=...)` for one-line global random
  HDF5 minibatches with
  reproducible seeds and an internal bounded multi-file preparation scheduler.
- CUDA bitshuffle/LZ4 kernels and pinned-buffer HDF5 master load path
- MPS Metal bitshuffle/LZ4 kernels, chunk-backed zero-copy load path, memory
  guard, crop-first sparse decode, and lazy multi-dataset loader behind
  `quantem.gpu.io.load()`
- `quantem.gpu.io.inspect()` and `quantem.gpu.io.discover()` for header-only
  readiness checks and acquisition-folder discovery.
- `quantem.gpu.detector` BF/DF/ADF, `mean_dp`, `masked_sum`, `virtual`, and
  automatic BF disk detection with reference checks
- `quantem.gpu.dpc` CoM/DPC/iDPC with reference checks
- `quantem.gpu.SSB`, the single backend-neutral SSB workflow above private
  CUDA, MPS, and WebGPU compute implementations
- domain-owned CUDA, MPS, and WebGPU compute implementations; Linux CI has
  dispatch guardrails, and true Metal runtime reference agreement runs on macOS
- MLX/Metal SSB compute for chunk-backed Mac data, returning the same single
  `SSBResult` contract as CUDA
- Active `quantem.gpu`, `quantem.widget`, and `quantem.live` source trees route
  migrated load and compute paths through `quantem.gpu`; ptychography callers
  use the shared workflow contract.

Widget and live packages may retain UI and acquisition orchestration, but new
backend compute belongs here. Remaining work broadens real-data parity coverage
and WebGPU fit capability without adding a second public API.
