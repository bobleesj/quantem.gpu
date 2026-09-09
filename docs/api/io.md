# I/O API

`quantem.gpu.io` has four public operations:

```python
from quantem.gpu import io

files = io.discover("/data/session")
readiness = io.inspect(files[0])
loaded = io.load(files[0], backend="auto", dtype="u16")
saved = io.save("copy_master.h5", loaded.data, backend="auto", dtype="u16")
saved.wait()
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

## `load`

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

(cuda-h5-ans-residency)=
### Stream complete H5 counts into CUDA ANS residency

```python
from quantem.gpu import io, detector

loaded = io.load("scan_master.h5", backend="cuda", representation="ans",
                 dtype="native", apply_mask=False)
session = detector.prepare(loaded)
pattern = session.frame(0, output="native")
```

This opt-in CUDA path streams bounded chunks of a complete uint8/uint16 H5
acquisition, preserves every stored count, and builds exact spatial sums while
those chunks are available. The library's default H5 representation remains
dense. Existing ANS files keep their original encoded buffers. Prepare a list
of equally shaped acquisitions for joint native DP and detector queries:
`detector.prepare([first, second])`.

The runtime H5 resident uses a separate internal ANS profile from the portable
ANS file format. Its `resident_profile`, `physical_resident_bytes`, `index_bytes`
and `load_timings` metadata describe the actual loaded representation. Saving
or transcoding this new resident is not yet implemented. Complete-series
120 Hz throughput is not established by the bounded CUDA parity tests.

(cuda-h5-paired-residency)=
### Stream complete uint16 H5 counts into the paired CUDA layout

```python
from quantem.gpu import io, detector

series = io.load(masters, backend="cuda", representation="paired",
                 dtype="native", apply_mask=False)
session = detector.prepare([item.data for item in series])
images = session.masked_sum(detector_mask, output="native")
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
operation support, exactness, and ownership. Dense and packed paths are retained;
packed storage is not a replacement for algorithms that require dense arrays.

`representation` describes how the complete logical array is retained. It has
the following public selectors on this integration branch:

| Representation | Meaning |
|---|---|
| `"dense"` | Every logical value occupies its ordinary dense array element |
| `"packed"` | Exact integer counts use compact storage consumed by a matching kernel |
| `"ans"` | Exact integer counts remain entropy-coded with the tables needed for decoding |
| `"paired"` | Exact integer counts in the CUDA paired-count tANS layout with a polar interaction index; explicit for original HDF5, detected for saved paired resident forms |

These are the only representation names. The authenticated `storage_schema`
selects the precise decoder within a representation; users do not select an
internal bitpacking or block-compression profile through this argument.

The new ANS-to-packed file workflow is available on Python MPS and CUDA, with
bounded physical integer-parity evidence. That evidence does not qualify
complete-series loading, peak memory, or interactive throughput. The explicit
CPU reference can
decode ANS to dense. GPU dense materialization, reverse conversions, and native
file-reader integration remain pending. Do not infer support for every
source/representation/backend combination from the selector names.

Representation is independent of dtype and residency. A lossless-packed
`uint8` source and a lossless-packed `uint16` source have the same
representation but different scientific dtypes. CUDA device memory, Apple
unified memory, and host memory are residency locations, not representations.

The shortest call is source-native:

```python
loaded = io.load("scan-lossless.h5", backend="auto")
```

An existing Lossless Pack Format source stays packed. Ordinary HDF5 follows the
current dense path. A standalone ANS source stays ANS unless a
supported conversion is requested. Loading never silently creates or evicts a
cache because those are consumer-policy decisions. Ask for dense explicitly
when an algorithm truly requires it:

```python
loaded = io.load(
    "scan_master.h5",
    backend="cuda",
    representation="dense",
    dtype="u16",
)
```

Requesting `representation="packed"` for an ordinary HDF5 source
fails with the preparation step instead of claiming that the source is packed.

### Selection and exact detector binning

The dense path also supports regions and stochastic batches:

```python
full = io.load(
    "scan_master.h5",
    backend="auto",
    representation="dense",
    dtype="u16",
)

crop = io.load(
    "scan_master.h5",
    backend="mps",
    scan_region=(32, 160, 48, 176),
    detector_region=(0, 192, 0, 192),
)

batch = io.load(
    masters,
    backend="cuda",
    random_positions=1000,
    scan_shape=(512, 512),
    seed=42,
)
```

`scan_region` and `detector_region` are always
`(row_start, row_stop, col_start, col_stop)`.
Use `detector_bin=1` to retain native detector sampling or a larger explicit
factor for exact detector-space sum binning. The former `det_bin` spelling is a
deprecated compatibility alias.

### Dtype selection

Keep native or unsigned 16-bit counts for an exact raw-count workflow, and make
an unsigned 8-bit browse representation explicit:

```python
exact = io.load("scan_master.h5", backend="auto", dtype="u16")
browse = io.load("scan_master.h5", backend="auto", dtype="u8")
```

| Selector | Meaning | Scientific boundary |
|---|---|---|
| `dtype="native"` or `None` | Preserve the backend-native source dtype | Preferred when the source precision must remain unchanged |
| `dtype="u16"` | Request unsigned 16-bit resident counts | Exact only while every corrected or binned value fits `uint16`; exact detector sums widen when required |
| `dtype="u8"` | Decode directly to unsigned 8-bit and saturate values above 255 | Browse/screening representation unless a complete source audit proves zero saturation |
| `dtype="auto"` | Use the loader's advisory compact-dtype selection | Convenience only; do not cite it as a complete-source losslessness audit |

Native `uint8` input and a `uint16` input converted to `uint8` are different
provenance. A lossless conversion requires a retained source identity, bad-pixel
policy, maximum count, and `pixelsAbove255 == 0`. Otherwise retain the
saturation count and label the result browse-only. Reconstruction workflows
should retain the raw-count precision required by their objective.

The resident payload is not peak memory. Record the requested, source,
working, accumulation, and output dtypes; original/output shapes; bin/crop;
payload bytes; predicted peak; measured process/accelerator peak; pressure or
swap; and the resource-policy reason. See the
{ref}`dtype and peak-memory dashboard <dtype-support-and-peak-memory>`.

For stochastic loading, `random_positions=` asks QuantEM to select positions,
while `scan_indices=` accepts positions selected by an external sampler. The
loader sorts and de-duplicates storage reads, decodes on the GPU, and restores
the requested stochastic order.

For joint time-series ptychography, keep one shared random batch and attach the
per-frame drift vectors without resampling the raw diffraction patterns:

```python
batch = io.load(
    master_paths,                       # e.g. 40 frame masters
    random_positions=1000,
    same_random_positions=True,
    scan_shape=(512, 512),
    drift=drift_fields,                 # shape (40, 512, 512, 2)
    output="torch",
)
positions = batch.metadata["drift_batch"]["corrected_positions"]
```

`drift_fields[f, r, c]` supplies the row/column shift for frame `f` at scan
position `(r, c)`. `positions` remains float32 for fractional shifts such as
`0.4` or `-0.6`.
The detector patterns are unchanged; the reconstruction forward model consumes
these corrected probe positions. Integer and fractional drift use the same
API. Use `scan_shift_row_col=` with `scan_region=` only when an explicitly
resampled scan-space stack is desired.

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
| `representation` on load | In-memory count layout | `"dense"`, `"packed"`, `"ans"` |

For example, save an ANS-compressed QuantEM file, then use bitpacking in memory:

```python
# Step 1. Write native four-dimensional NumPy uint8/uint16 counts exactly.
# The CPU encoder is an explicit reference; accelerated ANS saving is pending.
io.save(
    "experiment.qgpu", native_counts,
    format="quantem", compression="ans", backend="cpu",
)

# Step 2. Load the file and select the representation used by GPU operations.
with io.load("experiment.qgpu", representation="packed", backend="mps") as data:
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
Source/working shape, dtype, and calibration are preserved. ANS-to-packed does
not materialize a full dense tensor, but both encoded representations coexist
during conversion. This is not yet incremental file-shard streaming or a
full-volume memory/performance qualification. `discover` and `inspect` support
for this standalone envelope is also pending; use an explicit file path.
