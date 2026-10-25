# I/O API

Start with [From acquisition to images](../python-workflow.md) or the [README examples](https://github.com/bobleesj/quantem.gpu/blob/main/README.md#load-diffraction-patterns)
for individual patterns, scan patches, detector crops, and acquisition series.

`quantem.gpu.io` has four public operations:

```python
from quantem.gpu import io

files = io.discover("/data/session")
with io.load(files[0]) as data:
    pattern = data[10, 12]
    metadata = data.metadata
    io.save("copy.qem", data)
```

For ordinary work, load once and read metadata from the returned acquisition.
`inspect` is optional: use it for header-only checks, acquisition-readiness
polling, or metadata access without an accelerator.

Leave the loaded dataset as the last expression in a notebook cell to show
its calibration table before starting SSB, ptychography, or another algorithm:

```python
data = io.load("acquisition.qem")
data
```

`data.metadata_table()` returns the same table as a pandas DataFrame. It shows
the recorded geometry, physical values and units, saved calibration overrides,
and pixel-correction metadata without decoding detector values. Missing
rotation is shown as **Not recorded**, not as a measured zero. `data.calibration`
(and `data.calibration.table()`) shows the four microscope calibrations alone,
with **Value**, **Unit** and **Source** columns. Every calibration table,
including `SSB.calibration()`, uses one label set
(`quantem.gpu.io.metadata_display.CALIBRATION_LABELS`): **Accelerating
voltage** (kV), **Convergence semiangle** (mrad), **Scan step (row, col)**
(angstrom), **Detector step (row, col)** (mrad per pixel, or 1/angstrom per
pixel), **Scan-detector rotation** (deg) and **Rotation flipped**, with the
sources **Recorded**, **Overridden**, **Recorded + overridden** and **Not
recorded**. A zero or negative physical value is **Not recorded**. Complete source
records remain available in `data.metadata`. Header-only `io.inspect(path)`
shows the same calibration fields before loading; header inspection does not
authenticate the detector payload.

Metadata parsing may run on the host, but detector decoding and compression do
not silently fall back to CPU. `backend="auto"` selects CUDA or MPS and raises
with a corrective message when neither accelerated backend is available. The
explicit `backend="cpu"` path exists for reference and parity tests, and for
computers without a GPU: `io.load(path, device="cpu")` is the same request and
returns a dense host acquisition whose indexing gives CPU tensors and which
`numpy.asarray` converts. `device=` takes the names used across QuantEM:
`"auto"` (CUDA, then MPS, then CPU, printing which), `"cuda"`, `"cuda:N"`,
`"mps"` and `"cpu"`.

(shared-tutorial-data)=
## Shared tutorial data

`quantem.data.download` retrieves a public input from the
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data)
dataset repository; `quantem.gpu.io` then opens the local acquisition file.
`quantem.data` is a separate package and is not needed for your own local
files. `io.discover` searches local folders, not Hugging Face.

```{admonition} The quantem.data form below is not in a release yet
:class: warning
Qualified names such as `denoise/simulated_au_nanobeam_pg_dose1` and the
`revision` argument are newer than quantem.data 0.0.1rc4 on TestPyPI, whose
`download` raises `TypeError` when given `revision`. Until a quantem.data
release includes them, use the `huggingface_hub` form that follows it.
```

This example uses a synthetic Au input at a fixed data revision. A qualified
name selects one input even when another folder has the same short name:

```python
from quantem.data import download
from quantem.gpu import io

folder = download(
    "denoise/simulated_au_nanobeam_pg_dose1",
    revision="5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20",
)
master = folder / "truth-events-dose-1.0_master.h5"
data = io.load(master)
```

With released packages, `huggingface_hub` fetches the same folder; `master`
and `io.load` then follow as above:

```python
from pathlib import Path

from huggingface_hub import snapshot_download

root = snapshot_download(
    "bobleesj/quantem-data", repo_type="dataset",
    revision="5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20",
    allow_patterns="denoise/simulated_au_nanobeam_pg_dose1/*", token=False,
)
folder = Path(root) / "denoise/simulated_au_nanobeam_pg_dose1"
```

Keep `data` open while analyzing it or using a viewer, then call `data.close()`.
A pixel the stored detector mask flags holds no measurement, and its value
would bias every sum over the detector. The default `hot_pixel_correction="median"`
replaces it by the median of the valid pixels of its 3 × 3 window (or of the
smallest larger window that holds one, inside a flagged cluster), rounded half
to even, and the load line says how many. `hot_pixel_correction="none"` keeps
the stored values. This master flags no pixels, so both settings return the
same values here.
Neither setting fits a detector noise model. Use `io.inspect(master)` for an
optional header-only check before loading.

Either download returns the dataset folder with its HDF5 shard, metadata and
checksums. Keep the master and shard together. `io.load` does not read the
adjacent `geometry.json` or `noise-recipe.json`.

| Input | What `quantem.gpu.io` does with it |
| --- | --- |
| `4dstem/gold_128_npy_bin8` | `io.load(folder / "data.npy")` returns uint16 data shaped `(128, 128, 24, 24)`. The sampling is recorded in the adjacent `meta.json`, which `io.load` does not read. |
| `denoise/simulated_au_nanobeam_pg_dose1` | Synthetic master and shard shown above. |
| `widget/gold_session` | Not a 4D-STEM acquisition: 26 survey images for an image viewer. |
| `widget/ducky` | Not a 4D-STEM acquisition: a synthetic reconstruction monitor and snapshots for a viewer. |

Every path in the table exists at the one data revision used above,
`5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20`.

The Au input is a noiseless simulated expectation, float32, shaped
`(32, 32, 192, 192)`, stored with an assumed noise recipe. It contains no
experimental ARINA measurements and is not a validated ARINA noise calibration.
Experimental inspiration:
[Au liquid-cell data](https://doi.org/10.5281/zenodo.18167694).
The dataset repository holds input data, data cards, metadata and checksums;
scientific code, notebooks, weights and reports stay in their owning
repositories or local run directories. Downloading does not upload, move or
delete data.

## `discover`

Find candidate HDF5 masters before inspecting or loading them:

```python
masters = io.discover(
    "/data/session",
    pattern="*_master.h5",
    recursive=True,
    scan_shape=(512, 512),
)
```

The optional scan shape uses the public `(row, col)` convention and filters by
frame count without decoding detector pixels.

## `inspect`

Read headers and external-link metadata without loading the 4D array:

```python
report = io.inspect("scan_master.h5", scan_shape=(512, 512))
if not report.ready:
    raise RuntimeError(f"{report.reason} Next step: {report.action}")
```

The report includes the frame count, detector `(row, col)` shape, dtype, source
layout, and a source signature suitable for acquisition-readiness polling.
For `.qem`, inspection rejects an incomplete file length but does not read and
authenticate the entire payload. `io.load` verifies payload checksums before
exposing the resident measurements.

## `load`

### Array indexing

```python
with io.load("acquisition.qem") as data:
    pattern = data[10, 12]
    region = data[8:12, 10:16, 64:128, 64:128]
    metadata = data.metadata
```

`io.load` returns a `Dataset4dstemGPU` acquisition handle. QuantEM.GPU owns
its storage, decoding and metadata; QuantEM's core data classes are unchanged.

| Attribute or expression | Meaning |
| --- | --- |
| `data.shape` | Scan row, scan column, detector row, detector column sizes |
| `data.ndim` | Number of logical axes |
| `data.size` | Total number of logical detector values, not bytes |
| `len(data)` | Size of the first scan axis |
| `data.dtype` | Stored scientific dtype |
| `data.sampling` | Per-axis spacing; `None` where unknown |
| `data.units` | Per-axis physical units; `None` where unknown |
| `data.origin` | First-pixel coordinates; `None` where unknown |
| `data.metadata` | Calibration, provenance, and storage metadata |
| `data.logical_bytes` | Size of the complete uncompressed array |
| `data.resident_bytes` | Size reported by the resident storage owner |

Indexing returns an ordinary PyTorch tensor on the source GPU.
Install PyTorch for this indexing interface. The calibration properties are
read-only tuples in axis order: scan row, scan column, detector row, detector
column. Scan lengths use angstrom; detector spacing retains its recorded
angular or reciprocal-length unit. Missing calibration is never invented.
Selected Torch tensors do not carry the owner's calibration automatically;
pass the appropriate sampling and units explicitly when adding scale bars.
A single scalar selection returns a scalar tensor. Integers remove axes;
slices and one ellipsis support negative indices and steps. Index arrays, Boolean masks, and new axes are not supported.
This selects measurements without interpolation. Strides may decode a bounding
region. Request only the working region that fits the GPU memory budget.

Iterating over `data` yields scan rows, as array iteration does; it is not a
way to retrieve storage fields. Use `.data` only when an advanced operation
needs the encoded storage object. `np.asarray(data)` on encoded storage is rejected to prevent an
accidental whole-acquisition host copy. If a host array is needed, explicitly
select a bounded region first: `data[10, 12].cpu().numpy()`.
Use `data[10, 12]` directly for PyTorch operations.
For several acquisitions, select the owner first: `series[1][10, 12]`.
The explicit `read(scan_region=..., detector_region=...)` operation is also
available for region-based pipeline code.

### Notebooks and scripts

In a notebook, load with one assignment and keep the owner available across cells:

```python
data = io.load("acquisition.qem")
```

Explore with `detector.bf(data)` or array indexing such as `data[10, 12]`. Run `data.close()`
after the last use, including any viewer using its buffers. Close an old owner
before replacing it by rerunning a loading cell.

In scripts and batch jobs, prefer automatic cleanup, including on exceptions:

```python
with io.load("acquisition.qem") as data:
    pattern = data[0, 0]
```

Neither spelling changes storage, precision or backend selection. Do not use
an operation handle or viewer after closing its acquisition.

### Inspect the loaded acquisition

Every supported source uses the same entry point. It returns
`Dataset4dstemGPU`, which keeps backend-native data and its scientific/storage
metadata together:

```python
loaded = io.load("scan-lossless.h5", backend="auto")

print(loaded.shape)
print(loaded.dtype)
print(loaded.representation)
print(loaded.metadata["residency"])
print(loaded.logical_bytes, loaded.resident_bytes)
```

For ordinary acquisition use, omit `representation` and
`backend`. Supported originals become ANS-resident on the selected CUDA or MPS
device; saved `.qem` files retain their declared encoded layout.
Unknown formats are rejected, not silently expanded or relabeled as ANS.

### Load original arrays into ANS and save a `.qem` copy

NumPy arrays, EMPAD-G1 RAW/XML and calibrated EMPAD2 float32 exports now use
the same encoded-device workflow as supported HDF5 sources:

```python
from quantem.gpu import detector, io

with io.load("scan.npy", backend="mps") as loaded:  # or backend="cuda"
    dp = loaded[10, 20]
    mean_dp = detector.mean(loaded)
    bright = detector.bf(loaded)
    io.save("scan.qem", loaded)
```

`dp` is a Torch tensor on the source GPU. `mean_dp` and `bright` are reduced
NumPy images; only those small products are transferred to the host. The complete
acquisition stays ANS-encoded. The original-array ingestion path uses at most 32 MiB per input
window, with separate bounded encoder scratch. Saving copies encoded bytes,
original metadata and normalized scientific fields without expanding the cube.

| Original source | Encoded loading |
| --- | --- |
| NumPy `.npy` | Four axes; signed/unsigned 8, 16, 32, 64-bit integers or float16/32/64 |
| EMPAD-G1 `.raw` / `.xml` | Float32 records; 130×128 storage, 128×128 detector |
| EMPAD2 `.xml` | Calibrated 128×128 float32 exports, not encoded sensor words |
| DigitalMicrograph `.dm3` / `.dm4` | One calibrated four-axis real numeric image; reader must support its vendor dtype |
| NCEM EMD `.emd` | Four-axis arrays; `dataset_path` selects among multiple acquisitions |
| HDF5 | 4D datasets or flattened 3D frames, including contiguous and gzip layouts |

Float32 ANS retains the source detector geometry, including rectangular detectors,
without cropping or rounding to integer counts. A single float32 frame must fit
the 32 MiB working-window limit; larger detectors use fewer frames per window.
Headerless EMPAD RAW needs
`scan_shape=(rows, columns)`. XML calibration is retained. Raw EMPAD2 detector
words still require matching sensor calibration and a qualified decoder;
this does not establish EMPAD-G3 or arbitrary EMD support. The native
bitshuffle/LZ4 HDF5 layout retains its direct GPU decoder. Other supported
HDF5 layouts use bounded storage-library reads before GPU ANS encoding.

NCEM EMD coordinate vectors follow the [EMD specification](https://emdatasets.com/format/).
Regular scan sampling is normalized to Å and reciprocal sampling to Å⁻¹
(angular sampling stays in mrad). Original coordinate values, labels and units
are retained; nonuniform coordinates and unknown units are not guessed.
The reader retains coordinate vectors up to 4,096 values and records omitted
larger vectors explicitly. Arrays use scan-row, scan-column, detector-row,
detector-column order; arbitrary Velox event/image layouts are not implied.
Use `io.inspect(path, dataset_path="experiment/acquisition/data")` and the same
`dataset_path` in `io.load` when an EMD contains multiple acquisitions.

GPU NumPy/EMPAD loading rejects a dense residency override. Explicit
`backend="cpu", representation="dense"` remains available for reference access,
not as an automatic fallback. Unsupported dtypes/layouts raise an actionable
error before allocating a resident cube.

NumPy simulations stored as `int32`, such as those in
[the SrTiO3 dislocation dataset](https://doi.org/10.5281/zenodo.7464234),
retain int32, including negative values and counts above 65,535. GPU encoded
loading of arrays retains the original dtype and every measurement bit; it does
not narrow after a range audit. On MPS, an Arina bitshuffle master's uint16
or uint32 counts are held in the narrowest unsigned type that holds every
value after any requested bad-pixel correction: uint8 when every value is at
most 255, else uint16, else uint32 (one value above 65,535, such as a kept
0xffffffff bad-pixel marker, keeps the whole scan uint32). The counts read back
unchanged and use the fast detector sums; `io.load` names the stored type in
one line, and `dtype="native"` reads them in the file's dtype, a cast on read.
`.qem` copies (`convert`, `io.save`) store that narrow type and record the
source dtype; `convert(..., verify=True)` verifies every value against the source. `auto_narrow`
applies only to the explicit dense reader, and defaults to false. EMPAD RAW loading finds a sibling XML that
names the RAW file, retaining its scan dimensions, instrument fields, and
independently indexed virtual-detector regions. If multiple XML files name the
same RAW, open the intended XML explicitly. The archive's separate `para.txt`
is not automatically interpreted: keep it with the original acquisition and explicitly record any
calibration taken from it. Its three-dimensional `potential.npy` is a simulated
object, not a four-dimensional detector acquisition.

The [TCMEP dataset](https://doi.org/10.5281/zenodo.15084123) contains prepared
`/dp` HDF5 stacks and MATLAB 7.3 `/cbed` simulation arrays. Keep each prepared
stack beside its `params_backup.mat` and, when supplied, `data_position.hdf5`.
The Python reader validates the recorded raster positions or inclusive crop
bounds rather than guessing a square scan. It retains the source parameters,
normalizes voltage to kV, convergence semi-angle to mrad, and diffraction
sampling to Å⁻¹. Object sampling `dx` is not scan sampling; position units that
are not documented in the companion remain unspecified.

```python
from quantem.gpu import detector, io

with io.load("data_roi0_Ndp128_dp.hdf5", backend="mps") as loaded:
    pattern = loaded[0, 0]
    io.save("acquisition.qem", loaded)
```

Use `backend="cuda"` for an NVIDIA device. Gzip HDF5 uses bounded host reads
followed by GPU ANS encoding; this is not GPU gzip decompression. Float64 input
retains all original IEEE bits in typed ANS on both CUDA and MPS. Torch MPS
cannot represent float64 tensors: use `loaded.read(output="native")` for an
owned Metal array, and call its `release()` after use. Display and SSB products
use float32 arithmetic on Metal; that conversion leaves stored values intact.
Reconstruction objects, probes and result TIFFs are not diffraction acquisitions.
These Python paths do not certify the native Swift reader, which still limits
float acquisition geometry to 128 × 128 and does not read these gzip stacks.

(original-fp8-input)=
### Original FP8 input

CUDA and MPS preserve these original FP8 variants without quantization:
`float8_e4m3fn`, `float8_e5m2`, `float8_e4m3fnuz`, `float8_e5m2fnuz`, and
`float8_e8m0fnu`. Pass an already typed four-dimensional NumPy array
(`ml_dtypes`) or Torch tensor directly:

```python
from quantem.gpu import detector, io

with io.load(original_fp8_array, backend="mps") as loaded:
    original_pattern = loaded[10, 12]  # original FP8 Torch tensor on the GPU
    image = detector.bf(loaded)       # float32 calculation on the GPU
    io.save("acquisition.qem", loaded)
```

Use `backend="cuda"` on NVIDIA. Default reads retain every FP8 byte; inspect
their bits through `original_pattern.view(torch.uint8)` when the caller's
tensor library lacks FP8 arithmetic. Detector products and SSB interpret the
original values in float32 with CUDA/Metal kernels. They do not change stored
precision. The E8M0 format includes no zero or negative finite values; it is
not a signed measurement substitute. External tensor or block scales are not
inferred or applied.

Ordinary NumPy and HDF5 saves can lose the FP8 variant name. For HDF5, retain
one original byte per measurement as `uint8` or opaque `V1` and set the
measurement dataset's `quantem_dtype` attribute to its exact FP8 name:

```python
import h5py

with h5py.File("acquisition.h5", "w") as handle:
    data = handle.create_dataset("data", data=original_fp8_array.view("uint8"))
    data.attrs["quantem_dtype"] = original_fp8_array.dtype.name

loaded = io.load("acquisition.h5", backend="cuda")
io.save("acquisition.qem", loaded)
loaded.close()
```

For a three-dimensional frame stack, also pass `dataset_path` and the complete
`scan_shape`. Untagged `uint8` remains integer counts; an untagged opaque array
is not interpreted as FP8. Use `io.load` followed by optional `io.save` for
this workflow. Tagged external detector chunks also use the existing collection
converter, with optional GPU comparison of every original measurement byte (`verify=True`).
Original input files remain unchanged. A fill is defined for uint8, uint16
and uint32 counts: a source of another dtype whose mask flags pixels is
refused until it is loaded with `hot_pixel_correction="none"`, which excludes
the recorded bad pixels from products while their stored bytes remain intact. QEM preserves calibration, overrides,
and the exact variant on reopen. Additional `io.save(..., metadata={...})`
fields augment a loaded acquisition's metadata.

FP8 requires the Python CUDA/MPS reader with `ml_dtypes`. Default typed reads
also require a Torch build that defines the requested FP8 dtype; an older
build reports an actionable error. Native byte-preserving reads remain
available through `read(output="native")`. Native Swift and
WebGPU do not yet read this typed profile. Older readers reject unknown dtype
names instead of treating the bytes as integer counts.

(cuda-h5-encoded-residency)=
### Stream complete H5 counts into CUDA encoded residency

```python
from quantem.gpu import io, detector

loaded = io.load("scan_master.h5", backend="cuda")
pattern = loaded[0, 0]
```

This default CUDA path streams bounded chunks of a complete H5 acquisition
and preserves every stored count. uint8/uint16 use exact spatial sums;
uint32 uses typed bit lanes without narrowing or spatial indexes. The library's default H5 representation on
accelerator backends is encoded. Existing encoded files keep their original
buffers. Keep acquisitions separately with `io.load(paths, stack=False)` and select
an owner for each detector or reconstruction operation.

The resident's `resident_profile`, `physical_resident_bytes`, `index_bytes`
and `load_timings` metadata describe the actual loaded representation. The
runtime H5 resident and the integer `.qem` codec share the
`runtime-column-rans-spatial-v2` profile, so native streamed residents are
saved as `.qem` copies with the workflow below without re-encoding. Complete-series
120 Hz throughput is not established by the bounded CUDA parity tests.

### Open native DigitalMicrograph camera counts

Install the `dm` extra (`pip install "quantem.gpu[dm]"`) to read DM3/DM4
metadata. Open a complete calibrated 4D diffraction image directly:

```python
from quantem.gpu import detector, io

loaded = io.load("STEM SI.dm4", backend="cuda")
pattern = loaded[0, 0]
bright_field = detector.bf(loaded, bright_field_center_px=(431.5, 431.5), bright_field_radius_px=126)
# Close after all operations and viewers finish using the acquisition.
loaded.close()
```

The reader selects the unique 4D image and excludes embedded thumbnails and
survey images. Native uint8/uint16 counts stream through bounded pinned staging
into lossless CUDA ANS residency with exact spatial sums. Scan tails need not
be multiples of 512. No crop, binning, clipping, detector masking, or intensity
normalization is applied. Geometry and calibration use `(row, col)` order;
metadata retains axis units, sampling, pixel origins and microscope voltage.
Unsupported axis layouts and ambiguous multiple 4D images raise actionable
errors. `backend="cpu", representation="dense"` explicitly opens a read-only
memory map for reference access.

Save this encoded CUDA resident once to reopen it without re-encoding counts
or rebuilding spatial indexes:

```python
loaded = io.load("STEM SI.dm4", backend="cuda")
io.save("STEM SI.qem", loaded, format="quantem", backend="cuda")
loaded.close()
reopened = io.load("STEM SI.qem", backend="cuda")
```

This writes a `QEMDATA1` (`quantem.qem`) copy of the exact ANS bytes, spatial
indexes, detector validity and calibration. Reopening verifies the header and
every 64 MiB block with SHA-256 while uploading through two bounded pinned
buffers. It does not require the original DM file. Writes are atomic and reject
existing destinations. Saved copies are detected by magic regardless of
extension; the only accepted extension is `.qem`.
DM selection/conversion options remain unsupported. Native uint8/uint16 DM
counts also have an MPS encoded loading path; DM4 tests alone do not qualify all
DM3 variants. Load differently shaped acquisitions separately; a list can use
`stack=False` to return independent residents.

### Reopen float32 `.qem` on CUDA or MPS

For `float32-bit-lanes-rans-v1` files, use the same
public API on either accelerator:

```python
from quantem.gpu import detector, io

with io.load("measurements.qem", backend="cuda") as loaded:  # or "mps"
    pattern = loaded[0, 0]
    mean_pattern = detector.mean(loaded)
    io.save("measurements-copy.qem", loaded)
```

The acquisition remains encoded on-device. The selected pattern is a PyTorch
tensor; `pattern.cpu().numpy()` copies that selection to the host. The mean pattern is returned as NumPy.
There is no CPU scientific fallback or complete dense-cube
allocation. Copying retains original IEEE float bits, calibration, source
documents and the saved background recipe. The output must not already exist.
See the [float codec contract](qem-codecs.md)
for supported geometry, reduction precision and memory limits.

(converting-a-collection-from-the-command-line)=
### Converting a collection from the command line

For one acquisition, manual calibration, or declared median correction, start
with [Convert one HDF5 acquisition](qem-python.md#convert-one-hdf5-acquisition).
The CLI and `quantem.gpu.io.convert.convert` use the same conversion method.

```bash
quantem-gpu convert /data/arina/session --dry     # GPU payload-size estimate; no copies written
quantem-gpu convert /data/arina/session           # writes name.qem beside each name_master.h5
quantem-gpu convert /data/arina/session --out /archive/session
```

Each published copy keeps master-file fields and attributes (units included) under
its HDF5 path, embeds the master file itself so that long tables such as the
flatfield survive (`quantem.gpu.io.convert.restore_master` writes it back byte for
byte), records the name and size of its source files, and fills the scientific metadata (source format, accelerating
voltage, dwell time) from the Arina master.

Routine conversion defaults to `verify=False, source_hashes=False`. QEM header
and body checksums, file synchronization and no-overwrite publication remain
enabled. The embedded master has its own checksum computed from the bytes
already read for embedding. Add `--source-hashes` (`source_hashes=True`) to read
every original file and include its SHA-256. Add `--verify` (`verify=True`) to
compare every saved value against the source. The options are independent;
`--verify --source-hashes` enables both for validation.

An Arina master records neither the
probe semi-angle nor the scan step; when the acquisition's folder has a session
`dataset.yaml` (or `dataset.yml`) of schema `quantem.dataset/2`, its `microscope` voltage and
semi-angle and the `[row, col]` scan step become calibration overrides whose
evidence is the session file's SHA-256
(`quantem.gpu.io.convert.session_calibration`). The step is the one QuantEM Live
uses: the `files` entry whose `master` is this file, its own `scan_sampling_A`,
else the `calibrations` row of its `mag`, else `calibrations.default`. A
`dataset.yaml` of another schema is not read; QuantEM Live's
`tools/rewrite_dataset_yaml.py` rewrites it once, and the copy is written
without it with the reason in the conversion notes. Readers then report them as the
effective `semiangle_mrad`, `voltage_kV` and `scan_sampling_A`, so the copy
needs no sidecar. The session's values override what the master records (an Arina's
`photon_energy` field can hold a stale voltage). Only the fields used are attached (`dataset.json`), not the
session notes.

`microscope.detector_sampling_mrad` supplies detector sampling;
a matching `files` entry's `detector_sampling_mrad` takes precedence. This field
accepts one positive value or a `[row, column]` pair. Explicit `--voltage-kv`,
`--semiangle-mrad`, `--scan-sampling-a`, and `--detector-sampling-mrad` override
session calibration, with parameter provenance retained in QEM. Both session
filenames at once are an error; choose one. The command line names the calibrated
fields, values, units, and sources for each copy.

With `--verify`, every value is compared with independent detector-file reads
before publication, using the selected GPU path when supported.
Flagged pixels are kept and included in the requested comparison: conversion disables display-time
hot-pixel correction by default, and the detector pixel mask is saved with the copy. CUDA and
MPS retain `uint32` files as uint32, including flagged 0xFFFFFFFF markers and
valid counts above 65,535. Raw reads and source verification include every
pixel. Detector products exclude invalid pixels through the separate validity
mask. The collection converter also supports the real numeric dtypes listed
above; unsupported vendor layouts still fail explicitly.

For uint8/uint16/uint32 acquisitions, `--hot-pixel-correction median` or `zero`
explicitly requests replacement of stored-mask pixels. Verification compares
against an independent reference of that correction and checks all originally
valid counts unchanged. The method, coordinates, original flags and processing
history remain in QEM; reopening does not repeat correction.
Use `--backend cuda` or `--backend mps` to select a device backend. An
acquisition whose copy would be larger than its HDF5 files is reported and left
as HDF5. Oversized masters that cannot be embedded are not converted, so long
metadata tables cannot be silently lost. Source files are never modified or
removed; existing copies are never overwritten. The existing `--no-verify`
option remains accepted and selects the new default. A conversion without
full comparison reports `result.verified is None`. Dry-run size is the encoded payload estimate, not final file
size, and still performs GPU loading and encoding.

### Load an existing saved copy directly into native Metal

The native Swift reader accepts the same `quantem.qem` files written by the
Python MPS/CUDA and native exporters:

```swift
let snapshot = try NativeANSSnapshot(url: qemURL)
let source = try MetalRuntimeANSResidentSource.load(
  snapshot: snapshot, device: device, maximumAdditionalBytes: budget
)
let pattern = try source.extractRawDiffraction(scanRow: 0, scanColumn: 0)
```

It validates typed section bounds and checksums, streams bounded file ranges
into private Metal buffers, and preserves native uint8/uint16 counts without a
dense allocation. The path is independent of scan shape, including 512×512,
1024×1024, and non-square scans. This is an encoded-file reopen path; the macOS
original HDF5 loader still performs bitshuffle/LZ4 decode on cold HDF5 input.
The same handoff has been checked on bounded real 4D-STEM HDF5 slices; see the
local experiment record for the retained parity evidence.
Qualification of this saved-file reader does not establish cold original-HDF5
latency or application scheduling. Measure the selected original-file ingestion
path separately from `.qem` reopen.

(cuda-h5-paired-residency)=
### Stream complete uint16 H5 counts into the paired CUDA layout

This is an advanced CUDA integration path with reusable native output buffers.
For ordinary acquisitions, use `io.load(path)` and the automatic detector
functions without a session or representation option.

```python
from quantem.gpu import io, detector

series = io.load(masters, backend="cuda", representation="paired",
                 dtype="native", apply_mask=False)
session = detector.prepare(series)
images = session.masked_sum(detector_mask, output="native")
preview = session.masked_sum(detector_mask, output="native", out=images, block_stride=4)   # every 4th scan row, 1/4 of the time
```

`representation="paired"` accepts one master or a list. Each acquisition
becomes its own exact `PairedCounts` source; a list is loaded by one streaming
pipeline whose shard reads run ahead across file boundaries, so the series
proceeds at the drive's rate. The layout needs complete uint16 acquisitions
whose scan count is a multiple of 512. Per-source `load_timings` report the
direct-read, header-parse, encode and index seconds separately from the
resident-ready wall time.

Save the resident arrays once and reopen them without decoding:

```python
series[0].data.save("acquisition.paired")
reopened = io.load("acquisition.paired", backend="cuda", dtype="native", apply_mask=False)
```

The saved form starts with the fixed `QGPUPAIR` magic, so `io.load` selects
`"paired"` for it automatically; asking for another representation on that file
raises. Applications that admit acquisitions against a memory budget can drive
`quantem.gpu.io.PairedLoader.load_many(paths, admit=...)` directly and stop the
series while earlier files are still streaming.

### Representation

See [Count representations](representations.md) for per-backend
operation support, exactness, and ownership. Dense array algorithms are
retained, but dense is not a public GPU acquisition loading mode.

`representation` describes how the complete logical array is retained. It has
the following layout names; the loader admits only backend-appropriate choices:

| Representation | Meaning |
|---|---|
| `"dense"` | Every logical value occupies its ordinary dense array element |
| `"encoded"` | Exact integer counts remain entropy-coded with the tables needed for decoding |
| `"paired"` | Exact integer counts in the CUDA paired-count tANS layout with a polar interaction index; explicit for original HDF5, detected for saved paired resident forms |

These are the only representation names. The authenticated storage schema
selects the precise decoder within a representation; users do not select an
internal codec profile through this argument.

Native Swift/Metal can reopen a saved `.qem` copy directly, with a bounded
physical geometry gate. That gate does not qualify complete-series loading,
peak memory, or interactive throughput. The explicit CPU reference can decode encoded data to dense.
Accelerated HDF5-to-ANS ingestion is implemented on CUDA and MPS. Native
float32 coverage remains narrower than Python coverage. Do not infer support for every
source/representation/backend combination from the selector names.

Representation is independent of dtype and residency. An encoded `uint8`
source and an encoded `uint16` source have the same representation but
different scientific dtypes. CUDA device memory, Apple
unified memory, and host memory are residency locations, not representations.

The shortest call is source-native:

```python
loaded = io.load("scan-lossless.h5", backend="auto")
```

Ordinary HDF5 uses ANS residency on accelerator backends. A saved `.qem`
source remains encoded. Dense GPU loading is rejected.

ARINA bitshuffle + LZ4 masters use the optimized loading path automatically.
CUDA overlaps bounded compressed reads with GPU decoding and encoding; MPS
overlaps loading with Metal work and reuses bounded staging buffers. Native
Swift/Metal uses the same bounded-window approach through its native loader.
No performance switch or pre-conversion is required. Keep the master and all
external detector files together. To retain every original detector value,
including flagged pixels, use `hot_pixel_correction="none"`; the ordinary
detector-correction default is unchanged.

QEM loading authenticates its encoded buffers before returning them. It avoids
HDF5 decoding and re-encoding, but still reads the complete saved representation
and checks every payload checksum. Cache state and storage bandwidth therefore
affect loading time even when the GPU has available compute capacity.

Read only the working region needed by the calculation:

```python
data = io.load("scan_master.h5")
pattern = data[0, 0]
```

Tiny explicit CPU references may use `backend="cpu", representation="dense"`.
They are not a production loading fallback.

### Selection

Keep the acquisition compressed and request bounded working regions:

```python
data = io.load("scan_master.h5")
crop = data[32:160, 48:176, :192, :192]
```

Indices follow `(scan_row, scan_column, detector_row, detector_column)` and
slice stops are excluded. This selection does not bin or alter stored samples. Additional transformations
belong to the requested working array, not an implicit loading policy.

### Dtype selection

Preserve original counts or float32 measurements by omitting `dtype`:

```python
exact = io.load("scan_master.h5")
```

| Selector | Meaning | Scientific boundary |
|---|---|---|
| `dtype="native"` or `None` | Preserve the backend-native source dtype | Preferred when the source precision must remain unchanged |
| `dtype="scaled_uint16"` | Explicit calibrated approximate intensity storage | Quantization is reported; not a lossless substitute for float32 |

The ordinary ANS count loader rejects dtype narrowing (`u8`, `u16`, `auto`)
rather than silently changing scientific values. `dtype="float16"` is also
refused: `scaled_uint16` is the only stored precision. Omit `dtype` to keep
float32 values and save a lossless `.qem` copy.

Native `uint8` input and a `uint16` input converted to `uint8` are different
provenance. A lossless conversion requires a retained source identity, bad-pixel
policy, maximum count, and `pixelsAbove255 == 0`. Otherwise retain the
saturation count and label the result browse-only. Reconstruction workflows
should retain the raw-count precision required by their objective.
In particular, values above 255 cannot be preserved in an unsigned 8-bit array.

The resident payload is not peak memory. Record the requested, source,
working, accumulation, and output dtypes; original/output shapes; bin/crop;
payload bytes; predicted peak; measured process/accelerator peak; pressure or
swap; and the resource-policy reason. See the
{ref}`memory rules of the benchmark methodology <benchmark-memory>`.

For a series, retain separate encoded acquisitions:

```python
acquisitions = io.load(master_paths, stack=False)
# Close each owner after its last queued scientific operation.
```

`io.load` has no joint-series sampling, drift, or whole-acquisition tensor
output. Request bounded patterns from each owner and pass explicit probe
coordinates to the reconstruction workflow.

## `save`

Save backend-resident arrays without routing through a host reference writer:

```python
saved = io.save(
    "processed_master.h5",
    data,
    backend="auto",
    dtype="u16",
    metadata={"scan_sampling_A": [0.264, 0.264]},
)
saved.wait()
```

`backend="auto"` infers CUDA from a CuPy array and MPS from an MPS tensor.
A NumPy array requires `backend="cpu"` explicitly; this makes reference/test
writes visible rather than accidental.

The default file contract remains an Arina-style master with external data
files and lossless bitshuffle/LZ4 storage for integer detector counts.
`save` always returns a completion handle. With the default `wait=True`, the
handle is already complete; with `wait=False`, call `saved.wait()` before using
the output.

(io-file-format-compression)=
### File format, compression, and resident representation

These are independent decisions, not different names for the same setting:

| Option | Decision | Current examples |
|---|---|---|
| `format` on save | File layout | `"arina"`, `"quantem"` |
| `compression` on save | Lossless file encoding | `"bitshuffle_lz4"` for Arina; `"ans"` for QuantEM |
| `representation` on load | In-memory count layout | GPU: `"encoded"`, explicit CUDA `"paired"`; CPU reference: `"dense"` |

For example, save and reopen an ANS-resident acquisition:

```python
# Step 1. Save a copy of a complete encoded CUDA or Metal resident.
# Nothing is re-encoded; the resident's exact bytes and indexes are stored.
io.save("experiment.qem", resident)

# Step 2. Load the copy and select the representation used by GPU operations.
with io.load("experiment.qem") as data:
    print(data.shape, data.dtype, data.representation)
```

Loading detects the decoder from file contents. There is no `decompression=`
argument, and changing a filename extension does not change the encoding.
`compression="auto"` selects bitshuffle/LZ4 for Arina and ANS for QuantEM.
Only `format="arina"` and `format="quantem"` are accepted: removed format
aliases raise instead of redirecting the call. Incompatible format/compression
pairs also raise. ANS is not an implemented HDF5 filter here. This API change
does not change the bytes of the supported file formats.

The standalone writer is transactional and never overwrites an existing file.
Source/working shape, dtype, and calibration are preserved. Saving and reopening
an encoded acquisition is distinct from low-level representation conversion.
Use `io.inspect` for supported-container metadata and readiness; successful
inspection is not payload-integrity or full-volume performance qualification.

### EMPAD correction provenance

The reference exporter rejects an explicitly already-background-corrected float
array until that correction state is qualified across readers. Keep that array
and its metadata, or export the original measurements. Decoding does not perform
an additional background subtraction. Original XML is retained; storage footer
words are not detector measurements. See [QEM interoperability](qem-interoperability.md)
for independent validation and the [metadata mapping](qem-metadata-mapping.md) for
which source fields are normalized or merely retained.

Portable SSB export also accepts a `.qem` acquisition as its source. Its source
identity hashes the validated QEM bytes and is distinct from the original HDF5
identity. Re-encoding the acquisition changes that identity; the exported phase
values are unchanged. Keep the conversion record when relating the QEM working
counts to an archived raw acquisition.

The loaded acquisition's metadata table also separates **QEM file size** on
disk, **ANS resident size** (or **Resident size** for other representations),
and **Uncompressed size** from shape and dtype. All three use decimal GB
(1 GB = 1,000,000,000 bytes). Resident storage includes the encoded representation
and its indexes, not temporary SSB workspaces or total device memory. Inspecting
these sizes does not decode the acquisition.
