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

The Mac SSB backend uses MLX and Metal. Bounded `data.read()` access uses
PyTorch; install a GPU-enabled PyTorch build to use those examples.
For DM3/DM4 files, add the `dm` extra: `".[cuda,dm]"` or `".[mps,dm]"`.
Record `git rev-parse HEAD` with your results to reproduce the exact version.

This is pre-release software. The older TestPyPI candidate
`quantem.gpu==0.0.1rc8` does not include all current source features.
See [installation and runtime checks](docs/install.md) for details.

## Load diffraction patterns

The examples below use a gold acquisition: a 512 × 512 scan with a
192 × 192 detector. Replace `gold_master.h5` with your local master file and
keep its companion HDF5 files beside it. The data are not bundled here.

Supported acquisitions use ANS storage on CUDA or MPS; `.qem` reopens its
saved encoding. Automatic backend selection never silently falls back to CPU.
Install `quantem` for `show_2d`, and GPU-enabled PyTorch for `data.read()`.

### One diffraction pattern

```python
from quantem.gpu import detector, io
from quantem.core.visualization import show_2d

# Keep data open while working through these examples.
data = io.load("gold_master.h5")
session = detector.prepare(data)

row, column = 10, 12
pattern = session.frame(row * session.scan_shape[1] + column)
show_2d(pattern, norm="power_sqrt", title="Gold: scan (10, 12)")
```

`frame()` takes a row-major **scan index** and returns one NumPy pattern.
Use `output="native"` to keep a supported result on the GPU. The gold source
flags four detector pixels; default loading applies median hot-pixel
correction. Use `hot_pixel_correction="none"` when loading to retain the
original measurements.

### Several diffraction patterns

```python
positions = [(10, 12), (8, 10), (0, 0)]
selected = [
    session.frame(row * session.scan_shape[1] + column)
    for row, column in positions
]
show_2d(
    selected, norm="power_sqrt", axsize=(3, 3),
    title=[f"Gold: scan {position}" for position in positions],
)
```

![Three gold diffraction patterns](docs/_static/gold-multiple-patterns.png)

For a rectangular patch of **scan positions**, read a GPU tensor:

```python
patterns = data.read(scan_region=(8, 12, 10, 16))
print(patterns.shape)  # (4, 6, 192, 192): 24 diffraction patterns
```

Regions are `(row_start, row_stop, column_start, column_stop)`, with exclusive
stops. Indices start at zero. `data.read()` returns a PyTorch tensor on the
source GPU; omitting the region requests the whole decoded acquisition.

### A region inside a diffraction pattern

```python
patch = data.read(
    scan_region=(10, 11, 12, 13),
    detector_region=(64, 128, 64, 128),
)[0, 0]
print(patch.shape)  # (64, 64)
show_2d(
    [pattern, patch.cpu().numpy()], norm="power_sqrt", axsize=(3.5, 3.5),
    title=["Full detector", "Detector crop: [64:128, 64:128]"],
)
```

![Gold diffraction pattern and detector crop](docs/_static/gold-pattern-crop.png)

The red box marks the crop in this saved figure. Crop coordinates start at
zero locally; add 64 to recover original detector coordinates. Square-root
contrast is display-only, scaled independently per panel. The crop matched
the full loaded pattern's pixels exactly on CUDA.

Combine a larger `scan_region` with the same `detector_region` to crop several
patterns. Selection does not bin or interpolate pixels, though decoding may
require whole frames internally before cropping.

```python
data.close()  # after the last read or viewer using this acquisition
```

### One acquisition from a 5D-STEM series

The conceptual axes are `(acquisition, scan_row, scan_column, detector_row,
detector_column)`. The acquisition axis may represent tilt, time, or repeats.
If each acquisition is a separate file, load only the one you need:

```python
paths = ["tilt_00.qem", "tilt_01.qem", "tilt_02.qem"]
with io.load(paths[1]) as acquisition:
    pattern = detector.prepare(acquisition).frame(0)
```

To keep several acquisitions resident without a dense 5D stack:

```python
series = io.load(paths, stack=False)
try:
    acquisition = series[1]  # second acquisition, in paths order
    pattern = detector.prepare(acquisition).frame(0)
finally:
    for acquisition in series:
        acquisition.close()
```

`series[1]` selects an acquisition; `frame(1)` selects a scan position.
Call `read()` on the chosen 4D acquisition. For separate 4D datasets within
an EMD file, select the actual stored dataset path instead:

```python
with io.load("experiment.emd", dataset_path="experiment/acquisition/data") as data:
    pattern = detector.prepare(data).frame(0)
```

This is not arbitrary indexing into a single on-disk 5D tensor; there is no
general `acquisition_index=` argument. See the [I/O guide](docs/api/io.md)
for supported container layouts.

### Save a reusable acquisition

```python
with io.load("acquisition.npy") as data:
    io.save("acquisition.qem", data)
```

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
