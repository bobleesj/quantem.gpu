# quantem.gpu

One scientific GPU contract for electron microscopy—from compressed detector
data to exact products—across NVIDIA CUDA, Apple MPS/Metal, native Swift, and
browser WebGPU.

`quantem.gpu` owns the reusable, performance-critical layer shared by QuantEM
applications: 4D-STEM IO, bitshuffle/LZ4 decode, detector reductions, CoM/DPC,
display math, SSB, resource estimation, and scientific provenance. User
interfaces call this package rather than copying kernels or changing the
scientific workflow for each platform.

## Start with a full-resolution load

After [installing](install.md), load one HDF5 master with native detector
sampling and precision:

```python
from quantem.gpu import detector, dpc, io

loaded = io.load(
    "scan_master.h5",
    backend="auto",
    dtype="u16",
    det_bin=1,
)

bright_field = detector.bf(loaded.data)
dpc_result = dpc.run(loaded.data)

print(loaded.data.shape, loaded.data.dtype)
print(loaded.metadata)
```

`det_bin=1` means no detector binning. Any detector bin, scan bin, detector
region, or scan region must be explicit and preserved in provenance. A reduced
result is never represented as native-resolution evidence.

Continue with [loading HDF5](tutorials/load_hdf5.md),
[virtual detectors](tutorials/bf_df_adf.md), or [DPC](tutorials/dpc.md).

## One contract, several runtimes

| Runtime | Primary use | Implementation boundary |
|---|---|---|
| CUDA | Linux workstations, servers, and large-memory GPUs | Python API with CuPy and CUDA kernels |
| Python MPS | Apple Silicon scripts and shared Python workflows | Python API with MLX/PyObjC/Metal implementations |
| Native Swift/Metal | macOS and iOS applications | SwiftPM libraries with Metal resources; no Python runtime |
| WebGPU | Browser and exported-HTML compute | Domain-owned TypeScript/WGSL bundled by `quantem.widget` |
| CPU reference | Portable adjudication and small tests | Explicit reference only; never a silent accelerated fallback |

Choose a runtime in [Platforms](platforms/index.md). The public scientific
workflow, coordinate order, dtype meaning, and provenance remain consistent.

## Scientific capabilities

| Domain | Public entry point | What stays shared |
|---|---|---|
| IO | `quantem.gpu.io` | discovery, inspection, load/save, decode, crop/bin geometry, provenance |
| Detector | `quantem.gpu.detector` | mean diffraction, BF/ABF/ADF/DF, masks, exact integer reductions |
| DPC | `quantem.gpu.dpc` | CoM row/column convention, rotation, centering, iDPC normalization |
| Display | `quantem.gpu.display` and Swift products | range, histogram, colormap, log transform, FFT conventions |
| SSB | `quantem.gpu.SSB` | BF selection, aberrations, precision, objective, results |
| Remote | `quantem.gpu.remote` | exact array transport, source identity, device admission telemetry |

The [API reference](api/index.md) documents the public surface. Backend modules
are implementation details.

## Performance numbers are evidence

Performance is a first-class part of this documentation—not a marketing
footnote. The dedicated [Performance & parity](performance/index.md) section
keeps current numbers, hardware, shapes, dtypes, parity metrics, memory, cold
versus warm state, and rejected experiments easy to navigate.

Start with:

- [Current verified results](backends.md) for the capability and timing summary;
- [Benchmark methodology](performance/methodology.md) for what each number means;
- [Cross-backend parity](performance/parity.md) for exactness and tolerances;
- [Optimization ledger](maintainer/backend-optimization-matrix.md) for accepted
  and rejected paths; and
- [SSB performance evidence](maintainer/ssb-performance.md) for the full kernel
  history.

Numbers are retained with exact source and hardware context. A cache reopen is
not called a cold source load, a reduced fixture is not called full resolution,
and a software adapter is not called GPU evidence.

## Repository and application boundaries

```text
file -> quantem.gpu IO/decode -> backend-resident arrays
     -> quantem.gpu detector/DPC/SSB/display math -> consumer UI

file -> Native4DSTEMIO / Metal4DSTEMKernels -> resident products
     -> MetalImageFFT / MetalImageRuntime -> Live4DSTEM UI
```

- `quantem.gpu` owns reusable accelerated IO, math, kernels, result contracts,
  resource estimation, and scientific provenance.
- `quantem.widget` owns browser UI, interaction, and export while bundling this
  package's canonical WebGPU sources.
- Live4DSTEM and `quantem.live` own application UI, acquisition lifecycle,
  cache policy, and reconstruction orchestration while consuming exact package
  revisions.

Read [Repository architecture](maintainer/backend-layout-and-parity.md) before
adding or moving a backend.

## Documentation map

- **New users:** [Install](install.md) → [Choose a platform](platforms/index.md)
  → [What the scientific products measure](tutorials/workflow_math.md)
  → [Load an HDF5 master](tutorials/load_hdf5.md)
- **Scientists validating results:** [Performance & parity](performance/index.md)
  → [Current verified results](backends.md)
- **API users:** [API reference](api/index.md)
- **Kernel developers:** [Developer guide](developer/index.md)
- **Release and migration owners:** [Maintainer guide](maintainer/index.md)

## Citing and getting help

If the quantEM interactive framework—including `quantem.gpu`, GPU-accelerated
I/O, analysis, or reconstruction workflows on MPS or CUDA—contributed to your
research, please consider citing Lee et al., *Interactive Framework for
Real-Time 4DSTEM Analysis and Reconstruction*, *Microscopy and Microanalysis*
32 (Supplement 1), ozag053.941 (2026),
[https://doi.org/10.1093/mam/ozag053.941](https://doi.org/10.1093/mam/ozag053.941).

Questions and bug reports belong in the
[quantem.gpu issue tracker](https://github.com/bobleesj/quantem.gpu/issues).
The package is maintained by the Ophus group and distributed under the MIT
License.
