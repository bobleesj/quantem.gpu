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
`(row_start, row_stop, col_start, col_stop)`. `dtype="u8"` is appropriate only
after the count range proves it is lossless; use `u16` when that audit is absent
or any count exceeds 255. Reconstruction workflows should retain the raw-count
precision they require.

For stochastic loading, `random_positions=` asks QuantEM to select positions,
while `scan_indices=` accepts positions selected by an external sampler. The
loader sorts and de-duplicates storage reads, decodes on the GPU, and restores
the requested stochastic order.

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
