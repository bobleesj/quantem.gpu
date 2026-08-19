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
stochastic scan batches:

```python
full = io.load("scan_master.h5", backend="auto", dtype="u16")

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
