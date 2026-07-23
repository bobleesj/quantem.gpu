# IO API

Primary imports:

```python
from quantem.gpu import (
    load,
    load_calibration_products,
    load_scan_indices,
)
from quantem.gpu.io import discover_masters, get_metadata, is_master_ready
```

## `load`

Use `load()` for full-field or crop-first HDF5 loading:

```python
full = load("scan_master.h5", backend="auto", det_bin=4)
crop = load("scan_master.h5", backend="cuda", scan_region=(0, 32, 0, 32))
```

Important keyword arguments:

| Argument | Meaning |
|---|---|
| `backend` | `"auto"`, `"cuda"`, `"mps"`, or `"cpu"` |
| `det_bin` | detector binning factor |
| `dtype` | optional browse dtype such as `"u8"` when supported |
| `scan_region` | `(row_start, row_stop, col_start, col_stop)` crop |
| `scan_indices` | caller-provided stochastic scan positions |
| `random_positions` | number of global random scan positions to sample |
| `prep_workers` | explicit HDF5 preparation worker count for sparse multi-file loads |

## Stochastic scan batches

Use `random_positions=` when QuantEM should sample global scan positions, and
use `scan_indices=` when an external sampler already chose the positions:

```python
batch = load(
    master_paths[:40],
    random_positions=1000,
    scan_shape=(512, 512),
    seed=42,
    backend="cuda",
)

explicit = load(
    master_paths[:40],
    scan_indices=per_frame_indices,
    scan_shape=(512, 512),
    backend="cuda",
)
```

The returned data keeps stochastic order. Internally, each file's positions are
converted to HDF5 frame indices, sorted, de-duplicated, GPU-decompressed, and
then gathered back to the requested order.

`prep_workers=` can prepare several HDF5 masters concurrently before GPU
decompression, but it is not automatically faster. On one real-data 40-master
`512x512x192x192` benchmark with 1000 random scan positions per master, a true
cold scattered read measured `8.90 s` with `prep_workers=1`, `8.98 s` with `2`,
`9.47 s` with `4`, and `9.97 s` with `8`. With the same data hot in the OS page
cache, loads were about `1.0-1.6 s`, and `8` workers still regressed. Keep the
default single worker unless local storage measurements show that more readers
help.

This is the preferred IO shape for no-bin iterative ptychography on smaller
GPUs. A full `1024x1024x192x192 uint16` scan is about `77 GB`, but a
`1000x192x192 uint16` mini-batch is about `74 MB` before solver working buffers.
Compute COM/rotation and other calibration products as chunked reductions,
discard the raw chunk, then feed stochastic HDF5 batches to the solver.

## Cached calibration products

Use `load_calibration_products()` for screen launch, Show4DSTEM sidecar setup,
or ptychography calibration setup when BF/DF/CoM/rotation should appear
immediately:

```python
products = load_calibration_products(
    "scan_master.h5",
    backend="auto",
)

print(products.loaded_from_cache, products.elapsed_s)
print(products.bf.shape, products.df.shape, products.com_row.shape)
```

On a cache miss, this streams the full raw HDF5 once with GPU bitshuffle/LZ4
decode and backend-native BF/DF/CoM kernels. CUDA uses RawKernel reductions; MPS
uses chunk-backed Metal reductions and crop-first scan-row streaming. When
`memory_budget_gb` is omitted, the planner inspects current free CUDA VRAM on
CUDA machines and otherwise uses a conservative streaming plan. Full-region
CUDA requests are routed through the optimized full-master loader instead of the
crop loader. For chunked builds, the default BF-disk probe estimate uses the
first decoded row chunk; `sample_positions>0` opts into a separate random
scan-position sample when that is preferred over minimum latency. On a cache
hit, it loads only the small `.npz` product cache and is the path expected to
meet a sub-`0.5 s` screen/UI launch budget. Cache hits are backend-neutral;
existing caches can be loaded from CUDA, MPS, or CPU-facing code without probing
the build backend.

For a real `1024x1024x192x192 uint16` compressed master, representative
cache-miss timings on one 96 GB NVIDIA workstation GPU were:

| Memory budget | Plan | Cache build |
|---|---:|---:|
| `24 GB` | 7 row chunks | `7.08 s` |
| `48 GB` | 4 row chunks | `7.31 s` |
| `96 GB` | 1 full-master chunk | `3.81 s` |

The rotation search runs after BF/DF/CoM have been reduced to `1024x1024` maps;
on cached products, repeated `find_optimal_rotation()` calls measured about
`0.027 s` median. The slow part of a cache miss is therefore raw HDF5 streaming,
not rotation. For ptychography sweeps, build the product cache once per master
and reuse the cached center/radius/rotation metadata for subsequent trials.

## Metadata and readiness

Use metadata/readiness helpers before launching a large decode:

```python
from quantem.gpu.io import discover_masters, get_metadata, is_master_ready

masters = discover_masters("/data/session")
for master in masters:
    print(master, is_master_ready(master), get_metadata(master))
```
