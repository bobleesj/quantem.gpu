# Scaled uint16 resident intensities

On CUDA and Python MPS, `dtype="scaled_uint16"` now calibrates bounded regions
automatically. There is no scaling-mode keyword. Algorithms keep computing in
float32; the GPU converts each finished region once, measures its storage error,
and packs its uint16 codes. Ordinary reads and detector products restore the
region's physical intensity calibration.

```python
from quantem.gpu import io

loaded = io.load("float32_master.h5", dtype="scaled_uint16")
patch = loaded.read(scan_region=(0, 8, 0, 8))
io.save("scaled_master.h5", loaded)
reopened = io.load("scaled_master.h5")
```

A GPU tensor or generated source with `shape`, `dtype`, and ordered `blocks()`
can use the same load call. All logical frames must be yielded exactly once.
Generated sources retain their scientific algorithm; IO owns conversion,
calibration, packing, reductions and persistence. `io.save(path, source,
dtype="scaled_uint16")` streams regions without keeping the full packed output.
The caller must close loaded residents when done.

The version-2 precision record stores contiguous frame bounds, scale/offset,
geometry, source dtype and GPU-measured errors. Codes use nearest rounding with
ties to even; restored intensities are float32. Reloads retain the recorded
calibration, including for cross-region crops, instead of recalibrating codes.
Old globally scaled files retain their original version-1 behavior. Older
readers reject version 2, and the native Swift global file reader does not yet
support regional files. This change does not claim Live4DSTEM UI integration.

The brief message gives packed size, RMSE, maximum error and overflow. Detailed
region reports are in `loaded.metadata["precision"]`. Error is relative to the
float32 source, not experimental ground truth. A cropped reload reports saved
source-region metrics, not newly measured crop error. Float32 remains the choice
for an archive without this additional storage rounding. MPS rejects float32
subnormal inputs explicitly. The existing HDF5 writer requires detector element
counts divisible by eight for partial bitshuffle blocks.

The shared hardware qualification checks NumPy code/restoration parity,
rounding ties, signed/constant and wide-range values, cross-region reads,
detector crops, mean/BF/CoM products, owned Torch reads, one-pass generated saves,
save/reload equality, and legacy files. Run on the corresponding physical GPU:

```sh
QUANTEM_REGIONAL_BACKEND=cuda PYTHONPATH=src pytest tests/hardware/test_regional_precision.py -q
QUANTEM_REGIONAL_BACKEND=mps PYTHONPATH=src pytest tests/hardware/test_regional_precision.py -q
```

Large scientific operations remain GPU-only. Small independent NumPy fixtures
are arithmetic oracles; host filesystem work and reduced scalar metadata do not
perform MAPED computations. The existing CUDA global encoder remains available
for old behavior; version-2 CUDA encoding uses double-precision calibration,
matching the NumPy contract and Metal's compensated arithmetic on the tested
fixtures. Backend MAPED float32 rounding and storage-conversion parity are
separate claims.
