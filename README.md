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

The Mac SSB backend uses MLX and Metal. Array indexing uses
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
Install `quantem` for `show_2d`, and GPU-enabled PyTorch for array indexing.

The indexing order stays the same for every selection:

```text
data[scan_row, scan_column, detector_row, detector_column]
```

### Load and look at one pattern

```python
from quantem.gpu import io
from quantem.core.visualization import show_2d

data = io.load("gold_master.h5")
show_2d(data[10, 12], norm="power_sqrt")  # pattern at scan position (10, 12)
```

Indexing returns a GPU tensor; `show_2d` handles it directly. The gold source
flags four detector pixels; default loading applies median hot-pixel
correction. Use `hot_pixel_correction="none"` when loading to retain the
original measurements.

Inspect the acquisition without decoding the whole array:

```python
data.shape     # (512, 512, 192, 192): scan axes, then detector axes
data.dtype     # measurement dtype
data.metadata  # retained calibration and source information
```

### Compare several positions

```python
show_2d([data[10, 12], data[8, 10], data[0, 0]], norm="power_sqrt")  # three positions
```

![Three gold diffraction patterns](docs/_static/gold-multiple-patterns.png)

For a rectangular patch of **scan positions**, read a GPU tensor:

```python
patterns = data[8:12, 10:16]  # 4 × 6 scan positions; full detector
print(patterns.shape)  # (4, 6, 192, 192): 24 diffraction patterns
```

Indices start at zero; slice stops are excluded. Integers, slices and ellipsis
are supported, including negative indices and steps. Boolean masks and index
arrays are not supported. `data[:]` requests the entire decoded acquisition;
select a smaller region to limit memory. Metadata stays in `data.metadata`.

### One detector pixel across the scan

```python
show_2d(data[:, :, 95, 100], norm="power_sqrt")  # one detector pixel, all positions
```

This returns a scan image: the value at detector pixel `(95, 100)` at every
specimen position. `:` means keep every value along that axis.

### A region inside a diffraction pattern

```python
show_2d(data[10, 12, 64:128, 64:128], norm="power_sqrt")  # crop at one position
```

![Gold diffraction pattern and detector crop](docs/_static/gold-pattern-crop.png)

The red box marks the crop in this saved figure. Crop coordinates start at
zero locally; add 64 to recover original detector coordinates. Square-root
contrast is display-only, scaled independently per panel. The crop matched
the full loaded pattern's pixels exactly on CUDA.

Apply the same detector crop at several scan positions:

```python
patches = data[8:12, 10:16, 64:128, 64:128]  # crop at 24 positions
print(patches.shape)  # (4, 6, 64, 64)
```

Selection does not bin or interpolate pixels. CUDA streamed integer ANS data decode
only the selected detector streams; other storage profiles may decode whole
frames. Strided slices decode their bounding region before selecting values.

### Bright-field and annular dark-field images

A virtual detector sums selected detector pixels at each scan position.
Bright field (BF) uses a disk around the transmitted beam; annular dark field
(ADF) uses a ring outside it. Compute the mean diffraction pattern and fit
the beam disk once, then reuse its center and radius:

```python
from quantem.gpu import detector

mean_dp = detector.mean(data)                        # average over scan positions
center, radius = detector.fit_probe(mean_dp)         # (row, column), radius in pixels
bf = detector.bf(data, center=center, radius=radius)
adf = detector.adf(data, center=center, radius=radius)
show_2d([bf, adf], title=["BF", "ADF"], norm="power_sqrt")
```

![Gold bright-field and annular dark-field scan images](docs/_static/gold-bf-adf.png)

`fit_probe` estimates the bright-field disk geometry using thresholding and
its centroid; it does not recover probe phase or aberrations. BF includes
pixels through the fitted radius. The default ADF ring spans one to two
times that radius, limited by the detector. Both images show the full gold
scan with independent square-root display contrast. The acquisition remains
ANS encoded; only the reduced images are returned as NumPy arrays.

To choose collection angles, use `detector.adf(data, inner=40, outer=90,
unit="px", center=center, radius=radius)`. Use `unit="mrad"` when convergence
semi-angle calibration is available in the metadata. Omitting the geometry
from BF/ADF/DF calls estimates it again, so supply it when making several images.

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
    pattern = acquisition[0, 0]
```

To keep several acquisitions resident without a dense 5D stack:

```python
series = io.load(paths, stack=False)
show_2d(series[1][10, 12], norm="power_sqrt")  # second acquisition, one position
```

When finished with the series:

```python
for acquisition in series:
    acquisition.close()
```

`series[1]` selects an acquisition; `series[1][10, 12]` selects its pattern
at scan position `(10, 12)`. For separate 4D datasets within
an EMD file, select the actual stored dataset path instead:

```python
with io.load("experiment.emd", dataset_path="experiment/acquisition/data") as data:
    pattern = data[0, 0]
```

This is not arbitrary indexing into a single on-disk 5D tensor; there is no
general `acquisition_index=` argument. See the [I/O guide](docs/api/io.md)
for supported container layouts.

## Why QEM? One research format across detector vendors

Different microscopes and detectors write different file layouts, metadata
names, and units. Researchers should be able to load an acquisition and analyze
it without rewriting those conventions for every instrument. **QEM is our open,
research-oriented format for keeping diffraction measurements and their
scientific meaning together.** We recommend it for reusable QuantEM datasets.
It is an evolving format, not an established industry standard.

### What does a `.qem` file contain?

| Content | Why it matters |
|---|---|
| Encoded diffraction measurements, shape and dtype | Retain the data without storing a full uncompressed cube |
| Named scan and detector axes | Make `(row, column)` ordering explicit |
| Calibration values with units and provenance | Interpret distances, angles and microscope settings consistently |
| Reader-retained source metadata | Keep the original instrument context alongside normalized fields |
| Recorded corrections and precision choices | Know what happened before analysis |
| Format/codec versions, indexes and integrity checks | Locate, decode and validate the saved measurements |

Available calibration is expressed in microscopy units: scan sampling in Å,
detector sampling in mrad or Å⁻¹, accelerating voltage in kV, convergence
semiangle in mrad, and dwell time in µs. Angular and reciprocal sampling are
different quantities; the format preserves that distinction. Missing values
remain unknown. Importers retain supported source metadata, not necessarily
every proprietary field.

QEM uses a versioned binary container with a JSON metadata header and encoded
payloads. ANS compression supports compact storage and selected-pattern decoding.
The file is **not HDF5**; use the QEM reader. The
[container specification](docs/api/qem-format.md),
[codec definitions](docs/api/qem-codecs.md), and
[Python format guide](docs/api/qem-python.md) document how it is written and read.

### Import, export, and reopen

```python
from quantem.gpu import io

with io.load("gold_master.h5") as data:
    io.save("gold.qem", data)

data = io.load("gold.qem")
show_2d(data[10, 12], norm="power_sqrt")
data.metadata
```

Save once, then use the same indexing workflow when reopening. The saved file
contains its encoded measurements; decoding does not require the original
vendor files. Keep those originals as the acquisition archive. Saving preserves
the loaded measurements and their recorded processing: compression does not
undo hot-pixel corrections or an explicitly requested lossy precision change.
An existing destination is not overwritten.

```python
data.close()
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
