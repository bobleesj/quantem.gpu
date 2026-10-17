# Save and share your data

QEM stores measurements and their scientific metadata together. It provides a
common Python workflow across supported detector inputs, so a saved acquisition
can be reopened without the original vendor files. It is an experimental open
format; use the [current source installation](../install.md) and record its
Git revision.

## Save a loaded acquisition

```python
from quantem.gpu import io

data = io.load("gold_master.h5")
io.save("gold.qem", data)
```

Keep `data` open while using it, then call `data.close()`. Saving creates a new
file and refuses to overwrite an existing destination. No codec configuration
is needed: supported GPU acquisitions use ANS storage, and saving an encoded
acquisition retains that storage without expanding the full array.

## What is in the file?

| Content | Why it matters |
|---|---|
| Detector measurements, shape, and dtype | Reopen the stored values and their array geometry |
| Sampling and units | Interpret scan and detector coordinates physically |
| Microscope fields retained by the reader | Keep voltage, angles, and other available acquisition context |
| Source metadata and processing history | Know where values came from and which corrections were applied |
| Integrity checks | Detect damaged or incomplete stored data |

Lossless storage preserves the values being saved. Loading defaults to
`hot_pixel_correction="none"`: original measurements, dtype and detector flags
survive encoding and saving. This includes uint32 flag markers. Scientific
detector products exclude invalid pixels using the separate validity mask.
Requesting `hot_pixel_correction="median"` or `"zero"` explicitly replaces
flagged pixels and records that change. Saving cannot undo acquisition-time
processing or an explicitly requested replacement.
Missing calibration stays unknown. Retaining vendor tags does not mean that
every tag has been interpreted or that every proprietary object is archived.

QEM is not an HDF5 container. Open it with QuantEM rather than `h5py`.

## Reopen and inspect

```python
from quantem.gpu import detector

saved = io.load("gold.qem")
bf = detector.bf(saved)
saved.shape, saved.dtype, saved.sampling, saved.units
saved.metadata
```

The same indexing and detector calls work after reopening. `saved[10, 12]`
returns a GPU Torch tensor; `bf` is a reduced NumPy image. Close `saved` when
finished. Read sampling together with its unit; do not assume an unlabelled
number is in angstrom, nm, or mrad.

The Python GPU path retains signed and unsigned 8/16/32/64-bit integers and
float16/32/64 bits, including signed zero and NaN payloads. Storage support does
not imply identical calculation precision on every GPU. Integer detector sums
use checked 64-bit accumulation; a sum that cannot fit raises instead of wrapping.
Torch MPS does not support float64 tensors. For exact float64 reads on Metal:

```python
pattern = saved.read(scan_region=(10, 11, 12, 13), output="native")
try:
    original_pattern = pattern.get()  # one exact pattern, copied to host
finally:
    pattern.release()  # Metal result ownership ends here
```

CUDA native reads return CuPy arrays. Metal display and SSB products round to
float32 on the GPU without modifying the saved float64 measurements.

## Review calibration once

`data.calibration` shows voltage, convergence angle, scan sampling and detector
sampling, with units and Recorded / Overridden / Missing status. The notebook
summary and `io.inspect(path)` expose these quantities without decoding counts.
Unknown calibration does not prevent raw inspection or saving.

```python
data.calibration
data.set_calibration(
    semiangle_mrad=30,
    scan_sampling_A=(0.4, 0.4),
    evidence="Session calibration, measured scan step and aperture",
)
io.save("calibrated.qem", data)
```

Use measured values for your acquisition. Edits are separate from recorded
metadata and survive save, move, reopen and resave. Nothing saves automatically.
Recreate an existing reconstruction session after editing its calibration.
SSB needs voltage and scan sampling, plus detector sampling or a calibrated
convergence angle from which the disk radius can determine detector sampling.
It cannot determine an absolute angular scale from the disk radius alone.

For a header-only check without a GPU, use `io.inspect("gold.qem")`.
For a stored-file integrity check:

```bash
python -m quantem.gpu.formats.qem.validation gold.qem
```

Integrity checks detect storage errors; they do not establish the physical
accuracy of a calibration or the correctness of the original measurements.

## Continue to DPC and SSB

The [main Gold workflow](../python-workflow.md) puts loading, DPC and SSB on
one page. Saving as QEM is optional; those operations also accept the original
loaded acquisition.

The {ref}`advanced Gold notebook <gold-advanced>` demonstrates saving the
calibration, reopening the file, and showing a selected diffraction pattern
before saving and after reopening. It then inspects the mean pattern, fitted
aberrations and model probe, and compares native and 4× phase.

## Which files can I convert?

Use `io.load(source_path)` followed by `io.save("copy.qem", data)`.
Supported layouts include qualified HDF5, NCEM EMD, DM3/DM4, NumPy arrays, and
specified EMPAD float exports. Support depends on dtype and geometry; see the
[I/O input table](io.md) before converting a new source. DM3/DM4 need the `dm`
extra. Raw EMPAD2 sensor words are different from calibrated float exports.
Python MPS/CUDA support does not imply that every native application build
can open the same geometry or codec.

## Share a reproducible file

Retain the source license, conversion revision, calibration provenance, and
processing history with the data. Review retained source tags before publishing;
they may identify an acquisition or contain local paths. These functions save
locally and do not upload files.

For existing tutorial inputs, follow the {ref}`pinned download workflow <shared-tutorial-data>`:
retrieve the dataset at a fixed revision, then open its local acquisition with
`quantem.gpu.io`. Reuse that input across tutorials. The shared data repository
contains data and descriptive metadata; tutorial code, notebooks, models and
result reports remain in their source repositories or local runs.

For format developers, the
[file-format guide](../developer/file-formats.md) links the byte layout,
metadata schema, and independent conformance checks.
