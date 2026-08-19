# quantem.gpu

`quantem.gpu` is a reusable scientific IO and compute layer for 4D-STEM. It
defines one backend-neutral contract for loading, detector reductions, CoM/DPC,
display math, and single-sideband ptychography across NVIDIA CUDA, Apple
MPS/Metal, native Swift, WebGPU, and an explicit CPU reference.

## Documentation

Visit the **[quantem.gpu documentation](https://bobleesj.github.io/quantem.gpu/)**
for the scientific equations, kernel source maps, backend implementation guides,
API contracts, benchmark methodology, and cross-backend parity evidence.

Choose your entry point:

- [Scientific kernels](docs/kernels/index.md): equations, coordinates,
  optimization topology, sources, and parity gates by operation.
- [Kernel architecture](docs/concepts/kernel-architecture.md): how the
  domain-first source tree and cross-language contracts fit together.
- [Backend implementation](docs/platforms/index.md): CUDA, Python MPS, native
  Swift/Metal, WebGPU, remote CUDA, and CPU reference guidance.
- [Verified performance](docs/performance/results.md): dated, revision-pinned
  measurements with hardware, data shape/dtype, cache state, and load plan.
- [Developer guide](docs/developer/index.md): adding and reviewing kernels.

The README is deliberately a doorway. Detailed implementation notes and
historical measurements remain in the documentation site, where they can keep
their provenance without obscuring the public entry points.

## Scientific contract

All runtimes use

$$
I[r_y,r_x,q_y,q_x],
\qquad (\mathrm{row},\mathrm{column})\equiv(y,x),
$$

where $\mathbf r=(r_y,r_x)$ is the scan coordinate and
$\mathbf q=(q_y,q_x)$ is the detector coordinate.

Backends may change layout, tiling, fusion, queueing, buffer reuse, and kernel
topology. They may not silently change scan coverage, detector sampling,
binning, masks, precision, calibration, objective, or provenance. A cropped or
binned result is never represented as native resolution.

## Install

Install the current release candidate from TestPyPI:

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
| `remote` | loopback CUDA service used through an SSH tunnel |
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
- `MetalDisplayKernels`
- `MetalImageFFT`
- `MetalImageRuntime`

```bash
swift test
swift run -c release metal-display-benchmark 512
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

## Current evidence boundary

The true real-acquisition `1024×1024×192×192` product-first BF has
max/mean abs error `0` against its frozen reference.
This is not full-stack no-bin browse/load signoff. The complete source shape,
dtype, selected payload, cache
state, hardware, revision, and timing remain in
[Verified performance](docs/performance/results.md), not in this landing page.

## Citing quantem.gpu

If `quantem.gpu` accelerated IO, detector or DPC analysis, display math, SSB
reconstruction, or CUDA/MPS/Metal/WebGPU workflows contributed to your
research, please consider citing:

> Sangjoon Lee et al., “Interactive Framework for Real-Time 4DSTEM Analysis
> and Reconstruction,” *Microscopy and Microanalysis* 32 (Supplement 1),
> ozag053.941 (2026). https://doi.org/10.1093/mam/ozag053.941

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
The package is distributed under the [MIT License](LICENSE).
