# quantem.gpu

GPU-accelerated 4D-STEM analysis in Python: load acquisitions, explore
virtual detectors, calculate CoM/DPC, and reconstruct phase with
single-sideband ptychography (SSB). Use NVIDIA CUDA on Linux or Python MPS
on Apple Silicon.

[Documentation](https://bobleesj.github.io/quantem.gpu/) ·
[API guide](docs/api/io.md) · [Contributing](CONTRIBUTING.md) ·
[Issues](https://github.com/bobleesj/quantem.gpu/issues)

## Install

Python 3.11 or newer is required. For the current ANS loading and SSB workflows,
install from source:

**NVIDIA GPU — CUDA (Linux)**

```bash
git clone https://github.com/bobleesj/quantem.gpu.git
cd quantem.gpu
python -m pip install -e ".[cuda]"
```

**Apple Silicon Mac — MPS/Metal**

```bash
git clone https://github.com/bobleesj/quantem.gpu.git
cd quantem.gpu
python -m pip install -e ".[mps]"
```

The Mac backend uses MLX and Metal; PyTorch is not required for these workflows.
For DM3/DM4 files, add the `dm` extra: `".[cuda,dm]"` or `".[mps,dm]"`.
Record `git rev-parse HEAD` with your results to reproduce the exact version.

This is pre-release software. The older TestPyPI candidate
`quantem.gpu==0.0.1rc8` does not include all current source features.
See [installation and runtime checks](docs/install.md) for details.

## Load and inspect diffraction patterns

```python
from quantem.gpu import detector, io

with io.load("acquisition.qem") as data:
    session = detector.prepare(data)
    diffraction = session.frame(0)
    print(data.shape, data.dtype)
```

In a notebook, you can keep `data = io.load("acquisition.qem")` open across
cells; call `data.close()` after the last use.

Supported original HDF5, NumPy, DM3/DM4 and EMPAD float acquisitions are ingested
into ANS storage on the GPU; `.qem` reopens its saved encoding. Only requested
patterns or reduced products are decoded. Automatic backend selection chooses
CUDA or MPS and never silently falls back to CPU. Format and geometry limits
are documented in the [I/O guide](docs/api/io.md).

Save a reusable acquisition with the same entry point:

```python
with io.load("acquisition.npy") as data:
    io.save("acquisition.qem", data)
```

See the [pattern-selection tutorial](docs/tutorials/select-diffraction.md) for
one pattern, multiple positions, detector crops, and acquisitions in a 5D series.

## Reconstruct phase with SSB

Supply the microscope calibration and fit defocus and astigmatism:

```python
from quantem.gpu import SSB

with SSB.open(
    "acquisition.qem",
    voltage_kV=300,
    semiangle_mrad=30,
    scan_sampling_A=0.264,
) as workflow:
    result = workflow.fit(save_to="results/ssb")
    phase = result.phase
```

The calibration values above are examples; use your acquisition's values.
`phase` is in radians. Use `reconstruct()` for known aberrations, or `preview()`
for interactive parameter changes. The [SSB guide](docs/api/ssb.md) covers
fitting, tilt correction, saved results, and finer CUDA preview sampling.

## What can I do?

| Workflow | Guide |
|---|---|
| Load, inspect, and save acquisitions | [I/O](docs/api/io.md) |
| BF, DF, ADF and other virtual detectors | [Detector reductions](docs/kernels/virtual-detectors.md) |
| CoM, DPC and integrated DPC | [CoM/DPC](docs/kernels/com-dpc-idpc.md) |
| Phase reconstruction and aberration fitting | [SSB](docs/api/ssb.md) |
| Run CUDA computation remotely | [Remote compute](docs/remote/index.md) |

Arrays use `I[R_r,R_c,k_r,k_c]`: scan row, scan column, detector row,
detector column. Coordinates are `(row, column)`. Supported operations differ
between runtimes; see the [Implementation overview](docs/dashboard.md) and
[verified performance](docs/performance/results.md) for measured coverage.

Backend implementation details, including native and browser integrations,
belong in the [developer documentation](docs/developer/index.md).

## Citing quantem.gpu

If the quantEM interactive framework contributed to your research, please
consider citing:

> Sangjoon Lee et al., “Interactive Framework for Real-Time 4DSTEM Analysis
> and Reconstruction,” *Microscopy and Microanalysis* 32 (Supplement 1),
> ozag053.941 (2026). https://doi.org/10.1093/mam/ozag053.941

[MIT License](LICENSE).
