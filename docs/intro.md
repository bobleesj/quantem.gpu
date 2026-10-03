# quantem.gpu

GPU-accelerated 4D-STEM analysis for scientists using Python. Load an
acquisition, select diffraction patterns, create BF/ADF and DPC images, and
reconstruct phase with single-sideband ptychography (SSB). Start with the
Python workflow below; choose Apple Silicon MPS or NVIDIA CUDA for execution.

```{admonition} Living pre-release draft
:class: important
This site documents the current source checkout. APIs and runtime coverage
may change; use the [source installation instructions](install.md) and record
`git rev-parse HEAD` with your results. The version field still reads
`0.0.1rc8`, but the current examples require changes newer than that published
candidate.

The documentation is a draft, but retained performance and parity rows are not
draft estimates: each is a frozen historical measurement tied to its stated
date, source revision, device, data plan, cache state, and acceptance rule. A
newer revision does not automatically inherit those measurements.
```

## Install for your computer

| Computer | Python installation |
|---|---|
| Apple Silicon Mac | {ref}`MPS/Metal setup <apple-silicon-mac-mps-metal>` |
| NVIDIA GPU on Linux | {ref}`CUDA setup <nvidia-gpu-cuda-linux>` |

Both backends use the public Python API. Supported options differ by operation;
check the relevant guide before choosing one. The Mac backend uses MLX/Metal
for computation; array indexing returns PyTorch tensors on the selected GPU.

## Python quick start

Use your own acquisition in place of `gold_master.h5`; data are not bundled.
The [installation guide](install.md) includes QuantEM for plotting.

```python
from functools import partial

from quantem.gpu import detector, io
from quantem.core.visualization import show_2d

show_2d = partial(show_2d, cmap="inferno")
data = io.load("gold_master.h5")
show_2d([detector.bf(data), detector.adf(data)], title=["BF", "ADF"], norm="power_sqrt")
```

The detector fits the bright-field disk automatically. `data[row, column]`
returns a GPU Torch tensor for one diffraction pattern; `data.metadata`
contains calibration and provenance. Close the acquisition with `data.close()`
after the final calculation or viewer. Inferno is a notebook-local default;
one plot can override it with `cmap="gray"`.

For the complete scientist workflow, use the
[README tutorials](https://github.com/bobleesj/quantem.gpu#load-diffraction-patterns).
The detailed guides cover [I/O and calibration](api/io.md),
[automatic BF/ADF and overrides](api/images_dpc.md),
[QEM import/export](api/qem-python.md), and [SSB](api/ssb.md).

## What would you like to do next?

| Task | Guide |
|---|---|
| Load a file, select patterns, or crop detector pixels | [I/O and array indexing](api/io.md) |
| Make BF/ADF images or change the detector radius | [Virtual detectors and DPC](api/images_dpc.md) |
| Find aberrations and reconstruct SSB phase | [SSB](api/ssb.md) |
| Save an acquisition with its calibration | [QEM import/export](api/qem-python.md) and [notebook example](examples/qem_portable.ipynb) |
| Check available Python entry points | [Python API guide](api/index.md) |

## The shared coordinate contract

Data use `(scan_rows, scan_cols, detector_rows, detector_cols)` order,
written $I[R_r,R_c,k_r,k_c]$ in the scientific guides: $R$ identifies scan
coordinates and $k$ identifies detector coordinates.
`data[row, column]` selects a scan position; the final two axes describe its
pattern. Read `.sampling`, `.units`, and `.origin` for axis calibration,
and `.metadata` for the complete record. Unknown physical calibration remains
explicit; do not interpret a pixel distance as a physical length without it.

See [Data model and coordinates](kernels/data-model.md) for the full contract.

## Implementation and benchmark overview

For implementation work, start with [Scientific kernels](kernels/index.md),
[backend guides](platforms/index.md), or the [developer guide](developer/index.md).
The [implementation overview](dashboard.md) records current coverage. Numerical
results are maintained in two places only:

- the [implementation dashboard](dashboard.md) is the friendly, module-first
  view of current support, representative timing, memory, and parity state; and
- [verified benchmark results](performance/results.md) is the authoritative
  row-level ledger with revision, fixture, cache state, load plan, distribution,
  memory observation, and numerical gate.

| If you are… | Start here |
|---|---|
| choosing a runtime or checking current support | [Implementation dashboard](dashboard.md) |
| comparing a measured configuration | [Verified benchmark results](performance/results.md) |
| adding or rerunning a benchmark | [Benchmark methodology](performance/methodology.md) and [continuous profiling](performance/continuous-profiling.md) |
| investigating an older or rejected experiment | [Optimization ledger](maintainer/backend-optimization-matrix.md) and [historical experiments](maintainer/history/index.md) |

```{admonition} One claim, one owner
:class: important
The landing page does not copy benchmark tables. Current overview values belong
on the dashboard; complete provenance belongs in the results ledger; superseded
or rejected experiments belong in the maintainer archive. A cached reopen is
not a first source load, and a cropped or explicitly binned result is never
reported as native resolution.
```

::::{dropdown} How loading becomes a usable product — implementation details

## How loading becomes a usable product

```text
START WALL CLOCK
      │
      ▼
HDF5 master + compressed shards
      │  open, metadata, source identity, prepared index lookup/build
      ▼
Verified source geometry: scan shape, detector shape, dtype, calibration
      │  estimate resident + scratch + product memory
      ▼
Explicit load plan: full scan; no automatic real-space crop;
                    detector bin; source/accumulation/output dtype; reason
      │  plan source-aligned chunks and reusable buffers
      ▼
Storage read ──overlap──► GPU bitshuffle/LZ4 decode
                              │  bad-pixel policy + dtype conversion
                              │  + exact detector sum/bin when selected
                              ▼
ANS-encoded acquisition + complete provenance (Python CUDA/MPS)
      │  fused/reused GPU reductions
      ▼
Mean diffraction, BF/ADF/DF, CoM, DPC, iDPC
      │
      ▼
FIRST COMPLETE USABLE PRODUCT  ← STOP WALL CLOCK
      │
      └── optional cache/finalization, reported separately
```

Detector binning is exact block summation, not interpolation or cropping. It
may run while chunks are decoded so the unbinned 4D volume is never
materialized unnecessarily. The metadata still reports the original detector
shape, selected bin, output shape, accumulation/output dtype, memory estimate,
and policy reason. See [Load, decode, and bin](kernels/load-decode-bin.md) for
the mathematical contract and [Benchmark methodology](performance/methodology.md)
for every timed stage.

::::

## Citing and support

If quantEM's interactive widgets, GPU-accelerated I/O, or data processing
and reconstruction tools contributed to your research, please consider citing:

> Sangjoon Lee et al., “Interactive Framework for Real-Time 4DSTEM Analysis
> and Reconstruction,” *Microscopy and Microanalysis* 32 (Supplement 1),
> ozag053.941 (2026). https://doi.org/10.1093/mam/ozag053.941

Questions and bug reports belong in the
[issue tracker](https://github.com/bobleesj/quantem.gpu/issues).
