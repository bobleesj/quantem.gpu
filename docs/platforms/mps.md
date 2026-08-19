# Python MPS

Use Python MPS when a Python workflow runs locally on Apple Silicon. Select it
explicitly for tests and benchmarks:

```python
from quantem.gpu import io

loaded = io.load(
    "scan_master.h5",
    backend="mps",
    dtype="u16",
    det_bin=1,
)
```

Implementations may use MLX, PyObjC/Metal, or chunk-backed unified-memory
objects behind the public API. The backend preserves the same `(row, column)`,
shape, dtype, mask, bin, and provenance contract as CUDA and the CPU reference.

Large results remain chunk-backed where required. Memory guards inspect planned
decoded, scratch, resident, and product bytes rather than compressed file size.
An unsafe plan fails before allocation or returns an explicit resource-policy
choice to the consuming application; the backend does not crop the scan.

Physical performance evidence identifies the Mac model, chip and GPU cores,
unified memory, pressure and swap state, source/cache state, and end-to-end wall
time. See [benchmark methodology](../performance/methodology.md) and
[M2 Air Metal evidence](../maintainer/m2-air-lz4-match-unroll-2026-08-18.md).
