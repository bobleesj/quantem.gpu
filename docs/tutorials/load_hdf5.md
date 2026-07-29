# Load an HDF5 master

Use `quantem.gpu.io.load()` for bitshuffle/LZ4 HDF5 4D-STEM masters. The returned
object is a `LoadResult` with `data` and `metadata`.

```python
from quantem.gpu.io import load

result = load(
    "scan_master.h5",
    backend="auto",
    det_bin=1,
)

data = result.data
metadata = result.metadata
print(data.shape, data.dtype)
print(metadata.get("scan_shape"), metadata.get("detector_shape"))
```

`det_bin=1` keeps native detector sampling. Use `det_bin=2` or `4` only when
you intentionally want a detector-reduced preview or a smaller resident array;
record that choice in reports.

On CUDA, `data` is a device array. On MPS, data may be a chunk-backed object
that keeps the heavy detector frames device-owned for product computation. CPU is
available as a reference path, but it is not the target for large interactive
work.

Display is a widget concern:

```python
from quantem.widget import Show4DSTEM

Show4DSTEM(result.data)
```

## Stochastic ptychography batches

For iterative ptychography, load only the scan positions needed for the next
optimizer step:

```python
from quantem.gpu.io import load

batch = load(
    master_paths,
    random_positions=1000,
    scan_shape=(512, 512),
    seed=123,
    backend="cuda",
)

print(batch.data.shape)  # (n_files, 1000, 192, 192)
print(batch.metadata["sample"])
```

Use `scan_indices=` instead when your sampler already chose the positions:

```python
batch = load(
    master_paths,
    scan_indices=per_frame_indices,
    scan_shape=(512, 512),
    backend="cuda",
)
```

The sampler is global over the full scan, not a localized scan tile. The loader
sorts and de-duplicates HDF5 frame indices for compressed reads, runs the GPU
bitshuffle/LZ4 decompressor, and restores the requested stochastic order before
returning data to the solver.

Multi-file sparse batches use an internal bounded preparation scheduler. The
public workflow stays focused on scientific selection rather than storage-worker
tuning.

This lets reconstruction code run no-bin `192x192` detector ptychography on
24 GB GPUs by keeping only mini-batches in VRAM. A full
`1024x1024x192x192 uint16` scan is about `77 GB`, but one
`1000x192x192 uint16` batch is about `74 MB` before float/complex working
buffers.

## Cached products for fast screen launch

Use `load_calibration_products()` when the user-facing path needs BF, DF, CoM,
and DPC/rotation products immediately:

```python
from quantem.gpu import load_calibration_products

products = load_calibration_products(
    "scan_master.h5",
    backend="auto",
    memory_budget_gb=12,
)

print(products.loaded_from_cache)
print(products.bf.shape, products.df.shape, products.rotation_deg)
```

The first cache build still reads the raw HDF5 evidence and streams the detector
volume in bounded chunks. CUDA builds with RawKernel reductions; MPS builds with
chunk-backed Metal reductions. The default BF-disk estimate comes from the first
decoded row chunk; set `sample_positions>0` only when a separate random probe
sample is worth the extra HDF5 pass. That build step is not the interactive
launch path. After the cache exists, the UI reads small BF/DF/CoM arrays and
fitted parameters from the `.npz` product cache, which is the path intended for
sub-`0.5 s` screen opens. This cache is derived metadata and reduced images; it
is not the normal Show4DSTEM WebGPU HDF5 folder export. Existing caches can be
read from CUDA, MPS, or CPU-facing code.

Keep load parameters explicit in reports:

- `backend`
- `loaded_from_cache` for calibration products
- `det_bin`
- `dtype`
- `scan_region`, if used
- `scan_indices` or `random_positions`, if used
- public-safe file label, scan shape, detector shape, and timing
