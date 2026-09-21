# quantem.gpu

`quantem.gpu` is a reusable scientific IO and compute layer for 4D-STEM. It
defines shared scientific contracts for loading, detector reductions, CoM/DPC,
display math, and single-sideband ptychography. Implementations target NVIDIA
CUDA, Python MPS, native Swift/Metal, WebGPU, native Vulkan, and an explicit CPU
reference. Operation and representation coverage differs by runtime; a backend
name alone is not a claim of feature or performance parity.

## Documentation

Visit the **[quantem.gpu documentation](https://bobleesj.github.io/quantem.gpu/)**
for the scientific equations, kernel source maps, runtime implementation guides,
API contracts, benchmark methodology, and cross-backend parity evidence.

> [!IMPORTANT]
> **Pre-release documentation:** `quantem.gpu` and this site are an evolving
> draft during the `0.0.1` release-candidate series. APIs and support guidance
> may change between candidates. This README describes the current source tree,
> which can be ahead of the TestPyPI candidate (`quantem.gpu==0.0.1rc6`). Pin an
> exact verified Git revision when using the new count-ANS workflows; installing
> the older candidate does not establish support for those changes. Native Swift
> clients should also pin an exact verified Git revision.

Opt-in [experimental resident ANS](docs/integrations/experimental-resident-ans.md)
connects the package count codec to downstream viewers, with explicit format and
performance limits.

[Scaled uint16 storage](docs/integrations/scaled-uint16.md) provides automatic
calibration, packed residency and calibrated reads on CUDA and Python MPS.

Choose your entry point:

- [Implementation overview](docs/dashboard.md): the dense one-page map
  of scientific operations, runtime coverage, parity gates, and the latest
  revision-pinned measurements.
- [Scientific kernels](docs/kernels/index.md): equations, coordinates,
  optimization topology, sources, and parity gates by operation.
- [Kernel architecture](docs/concepts/kernel-architecture.md): how the
  domain-first source tree and cross-language contracts fit together.
- [Kernel implementations](docs/platforms/index.md): CUDA, Python MPS, native
  Swift/Metal, WebGPU, Vulkan, and CPU reference internals.
- [Dense, packed, and ANS data](docs/api/representations.md): representation,
  dtype, ownership, and current operation support for each runtime.
- [Resident analysis entry points](docs/developer/reproducing-resident-analysis.md):
  load and query retained representations, pin implementations, and reproduce
  scientific results without copying backend code.
- [QuantEM.GPU Remote](docs/remote/index.md): deploy the CUDA engine as a
  loopback service and connect locally or through SSH.
- [Verified performance](docs/performance/results.md): dated, revision-pinned
  measurements with hardware, data shape/dtype, cache state, and load plan.
- [Benchmark coverage and runbooks](docs/performance/coverage.md): filterable
  measured, partial, pending, refuted, and unsupported configurations with
  stable commands for closing each open gate.
- [Developer guide](docs/developer/index.md): adding and reviewing kernels.
- [Migration status](docs/maintainer/migration.md): retained dense/packed
  paths, experimental backends, breaking receipt changes, and remaining work.
- [Count-IO roadmap](https://github.com/bobleesj/quantem.gpu/issues/7): remaining
  implementation, physical-device, and consumer gates with agent checklists.

The README is deliberately a doorway. Detailed implementation notes and
historical measurements remain in the documentation site, where they can keep
their provenance without obscuring the public entry points.

## Scientific contract

All runtimes use

$$
I[R_r,R_c,k_r,k_c],
\qquad (\mathrm{row},\mathrm{column})\equiv(r,c),
$$

where $\mathbf R=(R_r,R_c)$ is the real-space probe/scan coordinate and
$\mathbf k=(k_r,k_c)$ is the detector coordinate.

Backends may change layout, tiling, fusion, queueing, buffer reuse, and kernel
topology. They may not silently change scan coverage, detector sampling,
binning, masks, precision, calibration, objective, or provenance. A cropped or
binned result is never represented as native resolution.

## Install

For current source development, clone the repository, record its revision, and
install the extras for your runtime (for example, `mps` on Apple Silicon):

```bash
git clone https://github.com/bobleesj/quantem.gpu.git
cd quantem.gpu
git rev-parse HEAD
python -m pip install -e ".[mps]"
```

Keep the recorded revision with your results; a moving `main` is not a
reproducibility pin. For the release-candidate baseline instead:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc6"
```

Choose only the runtime extras you need:

| Extra | Purpose |
|---|---|
| `cuda` | CUDA IO, decompression, and compute |
| `mps` | Apple Silicon Python MPS/Metal paths |
| `remote` | service dependencies for QuantEM.GPU Remote |
| `movie` | GIF/MP4 rendering dependencies |
| `dev` | Python tests and development tools |
| `docs` | documentation build dependencies |

See [Install](docs/install.md) for complete commands and runtime verification.

## Quick start

In a notebook, start with `data = io.load("acquisition.qem")`; keep it open
across cells and call `data.close()` when finished. For scripts, use `with`
for automatic cleanup:

```python
from quantem.gpu import detector, io

with io.load("acquisition.qem") as loaded:
    session = detector.prepare(loaded)
    diffraction = session.frame(0)
    print(loaded.shape, loaded.dtype, loaded.representation)
    print(loaded.logical_bytes, loaded.resident_bytes)
```

No storage option is needed. Supported original HDF5, NumPy, DM3/DM4 and
EMPAD float acquisitions use bounded ANS ingestion on CUDA or MPS; `.qem`
reopens its saved encoding. Only requested diffraction patterns or reduced
products are decoded, not the complete acquisition. Automatic backend selection
never silently falls back to CPU. See the [IO contract](docs/api/io.md) for
format, calibration and native Swift coverage limits.

## Count representations and file compression

These controls answer different questions:

| Control | Values | Meaning |
|---|---|---|
| `format` on save | `"arina"`, `"quantem"` | Acquisition/container layout |
| `compression` on save | `"auto"` or a supported codec for that format | Disk encoding; Arina retains bitshuffle/LZ4, QuantEM currently uses ANS |
| `representation` on load | Omit for normal acquisition loading | Expert layout selection; inspect the returned storage metadata |
| `backend` | `"auto"`, `"cuda"`, `"mps"`, explicit `"cpu"` | Python execution runtime, subject to operation support |

Loading detects format and compression from file contents. Supported originals
default to ANS (`representation="encoded"` in result metadata). Existing
prepared packed sources retain their declared layout; they are not advertised
as ANS. Unsupported inputs fail explicitly instead of expanding a dense cube.

Save and reopen a supported acquisition without choosing its codec:

```python
from quantem.gpu import detector, io

with io.load("acquisition.npy") as data:
    io.save("acquisition.qem", data)
```

For portable conversion, metadata inspection and lossless round trips, see
the [QEM Python guide](docs/api/qem-python.md) and
[synthetic notebook](docs/examples/qem_portable.ipynb). The reference CPU path
is explicit; supported dtypes and detector geometries are documented there.

Saving a supported encoded resident preserves its encoded measurements and
scientific metadata without materializing the acquisition. Original tags can
contain identifying information: review metadata before sharing a `.qem` file.

| New count-ANS workflow | Implementation | Current qualification |
|---|---|---|
| Save/reopen dense native counts | CPU reference | Bounded exact uint8/uint16 tests |
| File to ANS or packed; DP and exact mask sums | Python MPS | Small physical integer tests |
| File to ANS or packed; DP and exact mask sums | CUDA | Host oracle and compilation; physical GPU pending |
| ANS arrays/files to DP and exact mask sums | Native Swift/Metal | Small physical integer tests; `.qem` file reader, bounded real-data reopen, and 512/1024 geometry smoke pass |
| New ANS mean-DP, moments, and reverse GPU conversions | Pending | Not qualified |

This table covers the new ANS profile, not the separate retained dense/packed
workflows. Neither these checks nor an ANS-to-packed conversion establish
full-volume real-time performance or SSB support for the new profile. See the
[representation contract](docs/api/representations.md) for per-profile operation
support, required source authentication, conversion examples, and ownership.

`io.load` returns `io.FourDSTEMData` for one source. `residency` reports where
the payload is retained. `logical_bytes` is the dense-equivalent size;
`resident_bytes` is reported owned storage, **not peak process or GPU memory**.
Measure peak RSS, accelerator allocation/reserve, and conversion scratch
separately. `detector_bin=1` keeps native detector sampling. See
[Load, decode, and bin](docs/kernels/load-decode-bin.md),
[BF/DF/ADF](docs/kernels/virtual-detectors.md), and
[CoM/DPC/iDPC](docs/kernels/com-dpc-idpc.md).

## Architecture

The repository separates library code, documentation, tests, and retained
experiments:

```text
quantem.gpu/
├── src/quantem/gpu/     # reusable scientific library and runtime resources
├── docs/               # API contracts, source maps, dashboard, and runbooks
├── tests/              # contracts, parity, hardware, e2e, and infrastructure
├── benchmarks/         # benchmark/profile registries and retained artifacts
├── scripts/            # reproducible checks, profiling, and fixture preparation
├── experiments/        # historical investigations; not public API entry points
├── Package.swift       # native Swift products and bundled Metal resources
└── pyproject.toml      # Python packaging and runtime extras
```

The library is organized by **scientific domain first** and runtime second.
This is an abbreviated map of actual paths, not a promise of every possible
domain/backend combination:

```text
src/quantem/gpu/
├── device/                         # explicit backend selection
├── io/                             # load.py, save.py, models.py, representation.py
│   ├── _ans_contract.py, _ans_dispatch.py  # count-rANS validation and dispatch
│   └── backends/{cpu,cuda,mps,webgpu}/
├── screening/                      # bounded load/reduction orchestration
├── detector/backends/{cuda,mps,webgpu}/
├── dpc/backends/{cuda,mps,webgpu}/
├── display/backends/               # cpu.py, cuda.py, webgpu/, direct3d/
├── ssb/backends/{cuda,mps,webgpu}/
├── geometry/                       # scan geometry and transforms
├── optics/                         # calibration and aberrations
├── remote/                         # exact scientific-array transport
├── webgpu/index.ts                 # browser consumer exports, no kernels
├── swift/{Sources,Tests,Benchmarks}/
├── vulkan/{include,src,shaders,tests,benchmarks}/
└── android/                        # compatibility CMake entry and headers only
```

Swift has a separate build tree because Swift Package Manager needs
target-oriented sources, tests, Metal resources, and native libraries. It does
not have separate scientific ownership. Every implementation shares the same
shape, dtype, coordinate, crop/bin, accumulation, and provenance rules.

IO separates result models, selection rules, metadata readers, representation
dispatch, and staging-memory ownership from load orchestration. For Python MPS,
`io/backends/mps/dense.py`, `packed.py`, and `_ans.py` own their runtime paths;
their Metal resources live in `kernels/`. CUDA owns its decoder, packed, and ANS
implementations in `io/backends/cuda/`. ANS is not yet a WebGPU/Vulkan IO path.
Older `compute` imports
remain as import-only compatibility files, not duplicate kernels. Browser
builds export the complete source graph with `quantem.gpu.webgpu.export_sources`.
See the [source map and reproducible checks](docs/maintainer/backend-layout-and-parity.md)
for canonical entry points and consumer migration details.

Tests are grouped into `contracts`, `parity`, `hardware`, `e2e`, and
`infrastructure`. Run `python scripts/run_tests.py --list` to find the suites;
the runner also translates old test-file paths without changing their gates.

## Native Swift and Metal

The repository-root Swift package exposes reusable, UI-independent products:

- `Native4DSTEMIO`
- `Metal4DSTEMKernels`
- `Metal4DSTEMStreamingIO`
- `MetalDisplayKernels`
- `MetalImageFFT`
- `MetalImageRuntime`
- `MetalScientificNumerics`
- `MetalSSBKernels`

```bash
swift test
swift run -c release metal-display-benchmark 512
swift test -c release --filter MetalSSBKernelsTests
```

See [Native Swift and Metal](docs/platforms/swift-metal.md) for the source map,
memory model, profiling approach, and parity gates.

## Development

Read [Contributing](CONTRIBUTING.md) before changing a public API, kernel, or
evidence file.

```bash
python -m pip install -e ".[dev,docs]"
PYTHONPATH=src python -m pytest -q
python scripts/benchmark_registry.py validate
python scripts/backend_status.py check
python scripts/check_profile_registry.py
swift test
jupyter-book build docs
python scripts/check_docs_links.py --html-root docs/_build/html
```

Every new backend or optimization must add scientific parity and physical
performance evidence. Cold source load, warm source load, resident interaction,
and saved-result reopen are different benchmark states and are reported
separately.

## Evidence boundary

The README does not copy benchmark results. Current overview values belong in
the [implementation dashboard](docs/dashboard.md); exact revisions, fixtures,
cache states, load plans, timing distributions, memory observations, and
parity gates belong in [Verified performance](docs/performance/results.md).
Historical and rejected experiments remain in the maintainer ledgers and are
never promoted as current results.

One retained WebGPU acceptance case covers product-first BF on a
true real-acquisition `1024x1024x192x192` source with max/mean abs error `0`
against its independent reference.
This is not full-stack no-bin browse/load signoff; that separate gate remains
pending in the
[load acceptance record](docs/maintainer/backend-4dstem-load-checklist.md).

## Citing quantem.gpu

If the quantEM interactive framework, including `quantem.gpu` accelerated IO,
detector or DPC analysis, display math, SSB reconstruction, or
CUDA/MPS/Metal/WebGPU workflows, contributed to your research, please consider
citing:

> Sangjoon Lee et al., “Interactive Framework for Real-Time 4DSTEM Analysis
> and Reconstruction,” *Microscopy and Microanalysis* 32 (Supplement 1),
> ozag053.941 (2026). https://doi.org/10.1093/mam/ozag053.941

## Package boundary

`quantem.gpu` owns reusable accelerated IO, math, kernels, result contracts,
and backend/resource estimation. Consuming applications own presentation,
interaction, acquisition lifecycle, cache policy, and orchestration.

The package is distributed under the [MIT License](LICENSE).
