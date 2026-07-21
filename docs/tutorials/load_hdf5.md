# Load an HDF5 master

Use `quantem.gpu.load()` for bitshuffle/LZ4 HDF5 4D-STEM masters. The returned
object is a `LoadResult` with `data` and `metadata`.

```python
from quantem.gpu import load

result = load(
    "scan_master.h5",
    backend="auto",
    det_bin=4,
)

data = result.data
metadata = result.metadata
print(data.shape, data.dtype)
print(metadata.get("scan_shape"), metadata.get("detector_shape"))
```

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
from quantem.gpu import load

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

For multi-file sparse batches, keep `prep_workers` explicit. More workers can
help on storage that scales with concurrent payload reads, but they can also be
slower for scattered compressed HDF5 sidecars. On one real-data 40-master
`512x512x192x192` test with 1000 random positions per master, `prep_workers=1`
was fastest for true cold scattered reads (`8.90 s`), while `2`, `4`, and `8`
workers measured `8.98 s`, `9.47 s`, and `9.97 s`. Warm-cache repeats were much
faster at about `1.0-1.6 s`, but `8` workers still regressed.

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
    backend="cuda",
    memory_budget_gb=12,
)

print(products.loaded_from_cache)
print(products.bf.shape, products.df.shape, products.rotation_deg)
```

The first cache build still reads the raw HDF5 evidence and streams the detector
volume in bounded chunks. The default BF-disk estimate comes from the first
decoded row chunk; set `sample_positions>0` only when a separate random probe
sample is worth the extra HDF5 pass. That build step is not the interactive
launch path. After the cache exists, the UI reads small BF/DF/CoM arrays and
fitted parameters from the `.npz` sidecar, which is the path intended for
sub-`0.5 s` screen opens. Existing caches can be read from Mac MPS or CPU-facing
code; CUDA is currently the cache-generation backend for the full raw stream.

Keep load parameters explicit in reports:

- `backend`
- `loaded_from_cache` for calibration products
- `det_bin`
- `dtype`
- `scan_region`, if used
- `scan_indices` or `random_positions`, if used
- `prep_workers`, if not default
- public-safe file label, scan shape, detector shape, and timing
