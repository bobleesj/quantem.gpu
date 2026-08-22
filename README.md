# quantem.gpu

`quantem.gpu` is the accelerated scientific IO and compute layer for QuantEM.
It provides one backend-neutral contract for 4D-STEM loading, detector
reductions, DPC, display math, SSB, and reusable GPU kernels across NVIDIA CUDA,
Apple MPS/Metal, native Swift, and browser WebGPU.

Application packages consume these APIs. They do not keep private copies of
the kernels or silently change scan coverage, detector sampling, precision, or
scientific provenance.

## Documentation

Visit the [quantem.gpu documentation](https://bobleesj.github.io/quantem.gpu/)
for installation, scientific workflows, CUDA and Apple GPU setup, API
references, remote CUDA operation, benchmark methodology, and cross-backend
parity evidence. The [documentation source](docs/intro.md) remains available
in the repository before the GitHub Pages deployment is enabled.

`quantem.gpu` connects scientific detector data to reusable accelerated
algorithms. Agent-assisted contributions are welcome when they preserve the
scientific contract, add parity evidence, and keep application UI outside this
package. Start with [Contributing](CONTRIBUTING.md), the
[developer guide](docs/developer/index.md), or
[adding a backend](docs/developer/adding-backend.md).

## Architecture

The repository is organized by **scientific domain first** and accelerator
second. Each domain owns one public workflow and result contract; backend
folders contain private implementations.

~~~text
quantem.gpu/
├── pyproject.toml                  # Python package and optional backends
├── Package.swift                   # native package entry point
├── src/quantem/gpu/
│   ├── device/                     # explicit backend detection/selection
│   ├── io/backends/{cpu,cuda,mps,webgpu}/
│   ├── detector/compute/{cuda,mps,webgpu}/
│   ├── dpc/compute/{cuda,mps,webgpu}/
│   ├── display/                    # shared display math and GPU sources
│   ├── ssb/compute/{cuda,mps,webgpu}/
│   ├── remote/                     # exact scientific-array transport
│   └── swift/{Sources,Tests,Benchmarks}/
└── tests/parity/backend_matrix.json
~~~

Swift has a separate **build tree**, not separate scientific ownership. Swift
Package Manager requires target-oriented sources, tests, Metal resources, and
native libraries; Python and browser sources use different packaging systems.
All implementations still share the same shape, dtype, `(row, column)`,
crop/bin, accumulation, and provenance rules.

See the [backend layout and parity contract](docs/maintainer/backend-layout-and-parity.md)
for the target structure, compatibility-shim policy, and migration gates.

## Install

Install the current release candidate from TestPyPI:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc6"
```

Choose extras for the runtime you actually need:

| Extra | Purpose |
|---|---|
| `cuda` | CUDA IO, decompression, and compute |
| `mps` | Apple Silicon Python MPS/Metal paths |
| `remote` | loopback service used through an SSH tunnel |
| `movie` | GIF/MP4 rendering dependencies |
| `dev` | Python tests and development tools |
| `docs` | documentation build dependencies |

For example:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda,remote]==0.0.1rc6"
```

See [Install](docs/install.md) for CUDA, MPS, movie, and widget combinations.

## Quick start

Load the full source with native detector sampling and precision, then compute
products through their scientific domains:

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

`det_bin=1` means no detector binning. If detector binning or a scan region is
requested, it must be explicit and recorded in the output provenance. A binned
or cropped result is never reported as native resolution.

The public IO surface is `quantem.gpu.io.discover`, `inspect`, `load`, and
`save`. See the [IO API](docs/api/io.md), [HDF5 tutorial](docs/tutorials/load_hdf5.md),
and [detector/DPC tutorials](docs/tutorials/bf_df_adf.md) for complete workflows.

## Native Swift and Metal

Native macOS and iOS clients consume the repository-root Swift package at an
exact revision. Its reusable products include:

- `Native4DSTEMIO` for source inspection, QH5 indexing, cache integrity, and
  native HDF5 access;
- `Metal4DSTEMKernels` for load plans, decode, detector products, CoM, DPC, and
  iDPC primitives;
- `MetalDisplayKernels` for range, histogram, colormap, and display kernels;
- `MetalImageFFT` and `MetalImageRuntime` for already-transferred 2D products.

The native package contains no SwiftUI, AppKit, UIKit, or application state.
Live4DSTEM owns UI and scheduling and must not copy `.metal` sources.

```bash
swift test
swift run -c release metal-display-benchmark 512
```

See [Native 4D-STEM IO](docs/api/native_4dstem_io.md) and
[native Metal image endpoints](docs/api/metal_image.md).

## Backends and parity

Python uses `cuda`, `mps`, and explicit `cpu` reference backends. WebGPU is a
browser runtime, not a Python device name. Native Swift/Metal is a compiled
client library, not another public scientific workflow.

Backend support is accepted in layers:

1. public contract, shapes, dtypes, provenance, and honest failure behavior;
2. synthetic numerical tests, including odd shapes and incomplete edge bins;
3. frozen cross-backend fixtures;
4. full real-data evidence without hidden crop, bin, or precision changes; and
5. physical-device end-to-end tests with memory and timing evidence.

Integer decode, binning, masks, sums, histograms, and RGBA outputs are expected
to be byte-exact. Floating operations use frozen, operation-specific metrics;
tolerances are not widened to make a new implementation pass. CPU is a
reference backend and is never a silent production fallback.

The canonical coverage map is
[`tests/parity/backend_matrix.json`](tests/parity/backend_matrix.json). See
[Backends](docs/backends.md) for the capability matrix and measured summary,
and the [optimization matrix](docs/maintainer/backend-optimization-matrix.md)
for retained and rejected performance work.

One important browser boundary remains visible here: true real-acquisition `1024x1024`
product-first BF has max/mean abs error `0` against its reference,
but strict full-stack no-bin browser loading is not signed off at that scale.
This is not full-stack no-bin browse/load signoff.

## Scaling the repository

When adding a domain, backend, or optimized kernel:

1. place reusable work in the scientific domain that owns it;
2. keep backend modules private behind that domain's public API;
3. reuse backend-neutral models for geometry, dtype, provenance, and results;
4. add the implementation and its gates to the parity matrix;
5. compare every backend with the same fixture, mask, parameters, and output
   contract;
6. record physical correctness, peak memory, and cold/warm performance
   separately from source-presence and compile tests;
7. migrate one domain at a time with import-only compatibility shims; and
8. pin consumers to the verified package revision before deleting a shim.

Reusable GPU math, kernels, resource estimation, and scientific provenance
belong here. UI, view state, user resource-policy choices, cache scheduling,
and reconstruction orchestration belong in consuming applications.

## Remote CUDA

The optional remote service keeps raw data and CUDA compute on the workstation
and sends exact scientific arrays through an authenticated SSH tunnel. It binds
to loopback by default and does not contain a web frontend or application UI.

See [Connect a native client to remote CUDA](docs/tutorials/remote_cuda.md) for
installation, Live4DSTEM connection, multi-GPU placement, memory admission, and
the transport provenance contract.

## Development

Read [Contributing](CONTRIBUTING.md) before changing a public API, backend,
kernel, or evidence file. Documentation starts at [docs/intro.md](docs/intro.md):

- [installation](docs/install.md)
- [backend capability and evidence](docs/backends.md)
- [verified benchmark results and provenance](docs/performance/results.md)
- [Python API](docs/api/index.md)
- [tutorials](docs/tutorials/load_hdf5.md)
- [native Swift/Metal APIs](docs/api/native_4dstem_io.md)
- [maintainer architecture and parity](docs/maintainer/backend-layout-and-parity.md)

Install a development checkout and run both language suites:

```bash
python -m pip install -e ".[dev,docs]"
PYTHONPATH=src python -m pytest -q
swift test
```

Build the documentation with:

```bash
jupyter-book build docs
```

## Citing quantem.gpu

If the quantEM interactive framework—including `quantem.gpu`, GPU-accelerated
I/O, analysis, or reconstruction workflows on MPS or CUDA—contributed to your
research, please consider citing Lee et al., *Interactive Framework for
Real-Time 4DSTEM Analysis and Reconstruction*, *Microscopy and Microanalysis*
32 (Supplement 1), ozag053.941 (2026),
[https://doi.org/10.1093/mam/ozag053.941](https://doi.org/10.1093/mam/ozag053.941).

## Package boundaries

- `quantem.gpu` owns reusable accelerated IO, math, kernels, result contracts,
  and backend/resource estimation.
- `quantem.widget` owns browser UI, interaction, export, and display
  orchestration while bundling canonical WebGPU sources from this package.
- Live4DSTEM and `quantem.live` own application UI, acquisition lifecycle,
  cache policy, and reconstruction orchestration while consuming exact
  `quantem.gpu` revisions.

The package is distributed under the [MIT License](LICENSE).
