# quantem.gpu

`quantem.gpu` is a reusable scientific IO and compute layer for 4D-STEM. It
defines one backend-neutral contract for loading, detector reductions, CoM/DPC,
display math, and single-sideband ptychography across NVIDIA CUDA, Apple
MPS/Metal, native Swift, WebGPU, and an explicit CPU reference.

## Documentation

Visit the **[quantem.gpu documentation](https://bobleesj.github.io/quantem.gpu/)**
for the scientific equations, kernel source maps, runtime implementation guides,
API contracts, benchmark methodology, and cross-backend parity evidence.

> [!IMPORTANT]
> **Pre-release documentation:** `quantem.gpu` and this site are an evolving
> draft during the `0.0.1` release-candidate series. APIs and support guidance
> may change between candidates. For reproducible Python testing, pin the exact
> TestPyPI candidate shown here (`quantem.gpu==0.0.1rc6`); native Swift clients
> should pin an exact verified Git revision. A newer candidate is adopted only
> after its compatibility, parity, and performance gates are repeated.

Choose your entry point:

- [Implementation overview](docs/dashboard.md): the dense one-page map
  of scientific operations, runtime coverage, parity gates, and the latest
  revision-pinned measurements.
- [Scientific kernels](docs/kernels/index.md): equations, coordinates,
  optimization topology, sources, and parity gates by operation.
- [Kernel architecture](docs/concepts/kernel-architecture.md): how the
  domain-first source tree and cross-language contracts fit together.
- [Kernel implementations](docs/platforms/index.md): CUDA, Python MPS, native
  Swift/Metal, WebGPU, and CPU reference internals.
- [QuantEM.GPU Remote](docs/remote/index.md): deploy the CUDA engine as a
  loopback service and connect locally or through SSH.
- [Verified performance](docs/performance/results.md): dated, revision-pinned
  measurements with hardware, data shape/dtype, cache state, and load plan.
- [Developer guide](docs/developer/index.md): adding and reviewing kernels.

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

Install the documentation's exact release-candidate pin from TestPyPI:

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

```python
from quantem.gpu import detector, dpc, io

loaded = io.load(
    "scan_master.h5",
    backend="auto",
    dtype="u16",
    det_bin=1,
)

bright_field = detector.bf(loaded.data)
annular_dark_field = detector.adf(
    loaded.data,
    inner=40,
    outer=90,
    unit="px",
)
dpc_result = dpc.run(loaded.data)

print(loaded.data.shape, loaded.data.dtype)
print(loaded.metadata)
```

`det_bin=1` keeps native detector sampling. See
[Load, decode, and bin](docs/kernels/load-decode-bin.md),
[BF/DF/ADF](docs/kernels/virtual-detectors.md), and
[CoM/DPC/iDPC](docs/kernels/com-dpc-idpc.md).

## Architecture

The repository is organized by **scientific domain first** and runtime second:

```text
src/quantem/gpu/
├── device/                         # explicit backend selection
├── io/backends/{cpu,cuda,mps,webgpu}/
├── detector/compute/{cuda,mps,webgpu}/
├── dpc/compute/{cuda,mps,webgpu}/
├── display/                        # shared display math and GPU sources
├── ssb/compute/{cuda,mps,webgpu}/
├── remote/                         # exact scientific-array transport
└── swift/{Sources,Tests,Benchmarks}/
```

Swift has a separate build tree because Swift Package Manager needs
target-oriented sources, tests, Metal resources, and native libraries. It does
not have separate scientific ownership. Every implementation shares the same
shape, dtype, coordinate, crop/bin, accumulation, and provenance rules.

## Native Swift and Metal

The repository-root Swift package exposes reusable, UI-independent products:

- `Native4DSTEMIO`
- `Metal4DSTEMKernels`
- `Metal4DSTEMStreamingIO`
- `MetalDisplayKernels`
- `MetalImageFFT`
- `MetalImageRuntime`
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
swift test
jupyter-book build docs
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

If `quantem.gpu` accelerated IO, detector or DPC analysis, display math, SSB
reconstruction, or CUDA/MPS/Metal/WebGPU workflows contributed to your
research, please consider citing:

> Sangjoon Lee et al., “Interactive Framework for Real-Time 4DSTEM Analysis
> and Reconstruction,” *Microscopy and Microanalysis* 32 (Supplement 1),
> ozag053.941 (2026). https://doi.org/10.1093/mam/ozag053.941

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
The package is distributed under the [MIT License](LICENSE).
