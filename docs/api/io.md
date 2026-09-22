# I/O API

`quantem.gpu.io` has four public operations:

```python
from quantem.gpu import io

files = io.discover("/data/session")
readiness = io.inspect(files[0])
with io.load(files[0]) as loaded:
    saved = io.save("copy.qem", loaded)
```

Metadata parsing may run on the host, but detector decoding and compression do
not silently fall back to CPU. `backend="auto"` selects CUDA or MPS and raises
with a corrective message when neither accelerated backend is available. The
explicit `backend="cpu"` path exists for reference and parity tests.

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

### Notebooks and scripts

In a notebook, load with one assignment and keep the owner available across cells:

```python
data = io.load("acquisition.qem")
```

Explore with `detector.prepare(data)` or `data.read(...)`. Run `data.close()`
after the last use, including any viewer using its buffers. Close an old owner
before replacing it by rerunning a loading cell.

In scripts and batch jobs, prefer automatic cleanup, including on exceptions:

```python
with io.load("acquisition.qem") as data:
    pattern = data.read(scan_region=(0, 1, 0, 1))
```

Neither spelling changes storage, precision or backend selection. Do not use
an operation handle or viewer after closing its acquisition.

### Inspect the loaded acquisition

Use the same entry point for complete fields, scan crops, detector crops, and
stochastic scan batches. It returns `FourDSTEMData`, which keeps backend-native
data and its scientific/storage metadata together:

```python
loaded = io.load("scan-lossless.h5", backend="auto")

print(loaded.shape)
print(loaded.dtype)
print(loaded.representation)
print(loaded.residency)
print(loaded.logical_bytes, loaded.resident_bytes)
```

For ordinary acquisition use, omit `representation`, `compression`, and
`backend`. Supported originals become ANS-resident on the selected CUDA or MPS
device; saved `.qem` files retain their declared encoded layout. Existing
prepared packed containers remain a separate explicit storage contract.
Unknown formats are rejected, not silently expanded or relabeled as ANS.

### Load original arrays into ANS and save a `.qem` copy

NumPy arrays, EMPAD-G1 RAW/XML and calibrated EMPAD2 float32 exports now use
the same encoded-device workflow as supported HDF5 sources:

```python
from quantem.gpu import detector, io

with io.load("scan.npy", backend="mps") as loaded:  # or backend="cuda"
    dp = loaded.read(scan_region=(10, 11, 20, 21))[0, 0]
    session = detector.prepare(loaded)
    mean_dp = session.mean_dp(output="native")
    io.save("scan.qem", loaded)
```

`dp` is a Torch tensor on the source GPU. `output="native"` keeps point and
mean diffraction products on that GPU; the default `output="numpy"` copies
only the requested small product to the host. The complete acquisition stays
ANS-encoded. The original-array ingestion path uses at most 32 MiB per input
window, with separate bounded encoder scratch. Saving copies encoded bytes,
original metadata and normalized scientific fields without expanding the cube.

| Original source | Encoded loading |
| --- | --- |
| NumPy `.npy` | Four axes, uint8/uint16 or float32; wider integer counts require an exact range audit |
| EMPAD-G1 `.raw` / `.xml` | Float32 records; 130×128 storage, 128×128 detector |
| EMPAD2 `.xml` | Calibrated 128×128 float32 exports, not encoded sensor words |
| DigitalMicrograph `.dm3` / `.dm4` | Native uint8/uint16 or float32; one calibrated four-axis image |
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

GPU NumPy/EMPAD loading rejects dense or packed residency overrides. Explicit
`backend="cpu", representation="dense"` remains available for reference access,
not as an automatic fallback. Unsupported dtypes/layouts raise an actionable
error before allocating a resident cube.

NumPy simulations stored as `int32`, such as those in
[the SrTiO3 dislocation dataset](https://doi.org/10.5281/zenodo.7464234),
are audited in bounded windows before allocation. With `auto_narrow=True`
(the default), nonnegative counts up to 65,535 use uint8 or uint16 ANS
without changing any value. The original dtype, complete count range, and
exact-narrowing provenance survive `.qem` export. Negative or larger values
are rejected rather than clipped. EMPAD RAW loading finds a sibling XML that
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
    pattern = detector.prepare(loaded).frame(0)
    io.save("acquisition.qem", loaded)
```

Use `backend="cuda"` for an NVIDIA device. Gzip HDF5 uses bounded host reads
followed by GPU ANS encoding; this is not GPU gzip decompression. Float64 input
is accepted only when `auto_narrow=True` and a complete bounded bitwise
float64 → float32 → float64 audit proves every value unchanged. Otherwise it
fails before resident allocation. Exact narrowing and the original dtype are
recorded in `.qem` provenance; arbitrary float64 support is not implied.
Reconstruction objects, probes and result TIFFs are not diffraction acquisitions.
These Python paths do not certify the native Swift reader, which still limits
float acquisition geometry to 128 × 128 and does not read these gzip stacks.

(cuda-h5-encoded-residency)=
### Stream complete H5 counts into CUDA encoded residency

```python
from quantem.gpu import io, detector

loaded = io.load("scan_master.h5", backend="cuda")
session = detector.prepare(loaded)
pattern = session.frame(0, output="native")
```

This default CUDA path streams bounded chunks of a complete uint8/uint16 H5
acquisition, preserves every stored count, and builds exact spatial sums while
those chunks are available. The library's default H5 representation on
accelerator backends is encoded. Existing encoded files keep their original
buffers. Prepare a list
of equally shaped acquisitions for joint native DP and detector queries:
`detector.prepare([first, second])`.

The runtime H5 resident uses a separate internal encoded profile from the portable
encoded file format. Its `resident_profile`, `physical_resident_bytes`, `index_bytes`
and `load_timings` metadata describe the actual loaded representation. Native
streamed residents can be saved as CUDA snapshots using the workflow below;
transcoding to portable ANS is not implemented. Complete-series
120 Hz throughput is not established by the bounded CUDA parity tests.

### Open native DigitalMicrograph camera counts

Install the `dm` extra (`pip install "quantem.gpu[dm]"`) to read DM3/DM4
metadata. Open a complete calibrated 4D diffraction image directly:

```python
from quantem.gpu import detector, io

loaded = io.load("STEM SI.dm4", backend="cuda")
session = detector.prepare(loaded)
pattern = session.frame(0, output="native")
mask = detector.detector_mask((431.5, 431.5), 0, 126, loaded.shape[-2:])
bright_field = session.masked_sum(mask, output="native")
# Finish using the session before releasing its source.
session.close()
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
    session = detector.prepare(loaded)
    pattern = session.frame(0)
    mean_pattern = session.mean_dp()
    io.save("measurements-copy.qem", loaded)
```

The acquisition remains encoded on-device. Only requested small products
return to NumPy; there is no CPU scientific fallback or complete dense-cube
allocation. Copying retains original IEEE float bits, calibration, source
documents and the saved background recipe. The output must not already exist.
See the [float codec contract](qem-codecs.md)
for supported geometry, reduction precision and memory limits. This does not
extend the collection converter below to raw float32 HDF5 inputs.

### Converting a collection from the command line

```bash
quantem-gpu convert /data/arina/session --dry     # GPU payload-size estimate; no copies written
quantem-gpu convert /data/arina/session           # writes name.qem beside each name_master.h5
quantem-gpu convert /data/arina/session --out /archive/session
```

Each published copy keeps master-file fields and attributes (units included) under
its HDF5 path, embeds the master file itself so that long tables such as the
flatfield survive (`qem_conversion.restore_master` writes it back byte for
byte), records the name, size and SHA-256 of its source files, and fills the scientific metadata (source format, accelerating
voltage, dwell time) from the Arina master. After writing, every value is
compared with the detector files read through h5py before the copy is published.
Flagged pixels are preserved and checked too: conversion disables display-time
hot-pixel correction. CUDA `uint32` files are stored as `uint16` only after every
stored value is shown to fit, with the original dtype retained in metadata.
Larger values are refused, even at flagged pixels. `float32` acquisitions are
not supported by this collection command (other `.qem` writers support them).
Use `--backend cuda` or `--backend mps` to select a device backend; the Metal
collection loader currently accepts uint8/uint16, not uint32. An
acquisition whose copy would be larger than its HDF5 files is reported and left
as HDF5. Oversized masters that cannot be embedded are not converted, so long
metadata tables cannot be silently lost. Source files are never modified or
removed; existing copies are never overwritten. `--no-verify` explicitly skips
the comparison. Dry-run size is the encoded payload estimate, not final file
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

```python
from quantem.gpu import io, detector

series = io.load(masters, backend="cuda", representation="paired",
                 dtype="native", apply_mask=False)
session = detector.prepare([item.data for item in series])
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
operation support, exactness, and ownership. Dense array algorithms and low-level
packed paths are retained, but neither is a public GPU acquisition loading mode.

`representation` describes how the complete logical array is retained. It has
the following layout names; the loader admits only backend-appropriate choices:

| Representation | Meaning |
|---|---|
| `"dense"` | Every logical value occupies its ordinary dense array element |
| `"packed"` | Exact integer counts use compact storage consumed by a matching kernel |
| `"encoded"` | Exact integer counts remain entropy-coded with the tables needed for decoding |
| `"paired"` | Exact integer counts in the CUDA paired-count tANS layout with a polar interaction index; explicit for original HDF5, detected for saved paired resident forms |

These are the only representation names. The authenticated `storage_schema`
selects the precise decoder within a representation; users do not select an
internal bitpacking or block-compression profile through this argument.

Low-level encoded-to-packed conversion has bounded physical integer-parity
evidence; it is not a public acquisition-loading override. Native Swift/Metal can reopen a
saved `.qem` copy directly, with the same bounded physical geometry gate.
These results do not qualify complete-series loading, peak memory, or
interactive throughput. The explicit CPU reference can decode encoded data to dense.
Accelerated HDF5-to-ANS ingestion is implemented on CUDA and MPS. Native
float32 coverage remains narrower than Python coverage. Do not infer support for every
source/representation/backend combination from the selector names.

Representation is independent of dtype and residency. A lossless-packed
`uint8` source and a lossless-packed `uint16` source have the same
representation but different scientific dtypes. CUDA device memory, Apple
unified memory, and host memory are residency locations, not representations.

The shortest call is source-native:

```python
loaded = io.load("scan-lossless.h5", backend="auto")
```

Ordinary HDF5 uses ANS residency on accelerator backends. A saved `.qem`
source remains encoded. Dense/packed GPU loading is rejected, including older
packed files: reopen their original acquisition and save a new `.qem` copy.
Read only the working region needed by the calculation:

```python
loaded = io.load("scan_master.h5")
pattern = loaded.read(scan_region=(0, 1, 0, 1))
```

Tiny explicit CPU references may use `backend="cpu", representation="dense"`.
They are not a production loading fallback.

### Selection and exact detector binning

Keep the acquisition compressed and request bounded working regions:

```python
full = io.load("scan_master.h5")
crop = full.read(
    scan_region=(32, 160, 48, 176),
    detector_region=(0, 192, 0, 192),
)

```

`scan_region` and `detector_region` are always
`(row_start, row_stop, col_start, col_stop)`.
This read does not bin or alter stored samples. Additional transformations
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
rather than silently changing scientific values. The retired bit-packed
`float16` acquisition profile is also rejected; reopen its original float32
source and save a lossless `.qem` copy.

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
{ref}`dtype and peak-memory dashboard <dtype-support-and-peak-memory>`.

For a series, retain separate encoded acquisitions:

```python
acquisitions = io.load(master_paths, stack=False)
# Close each owner after its last queued scientific operation.
```

The former dense joint-series options (`random_positions`, `scan_indices`,
`drift`, and whole-acquisition `output="torch"`) are not supported by the ANS
loader. Request bounded patterns from each owner and pass explicit probe
coordinates to the reconstruction workflow. Do not use a dense-loading
fallback to recover the old combined interface.

## `save`

Save backend-resident arrays without routing through a host reference writer:

```python
saved = io.save(
    "processed_master.h5",
    data,
    backend="auto",
    dtype="u16",
    metadata={"scan_sampling_A": 0.264},
)
saved.wait()
```

`backend="auto"` infers CUDA from a CuPy array and MPS from an MPS tensor or
chunk-backed MPS array. A NumPy array requires `backend="cpu"` explicitly;
this makes reference/test writes visible rather than accidental.

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
