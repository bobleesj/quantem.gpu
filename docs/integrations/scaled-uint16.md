# Scaled uint16 resident intensities

On CUDA and Python MPS, `dtype="scaled_uint16"` now calibrates bounded regions
automatically. There is no scaling-mode keyword. Algorithms keep computing in
float32; the GPU converts each finished region once, measures its storage error,
and retains its uint16 codes in lossless ANS storage. Ordinary reads and detector products restore the
region's physical intensity calibration.

```python
from quantem.gpu import io

loaded = io.load("float32_master.h5", dtype="scaled_uint16")
patch = loaded.read(scan_region=(0, 8, 0, 8))
io.save("scaled_master.h5", loaded)
reopened = io.load("scaled_master.h5")
```

Scaled results report `representation="encoded"` and
`metadata["resident_codec"] == "ans"`. Omit `representation` for the default;
explicit `representation="encoded"` is also accepted for scaled precision.
Existing globally scaled files also reopen into ANS residency. The HDF5 file
writer still uses GPU bitshuffle/LZ4 for disk compression; reopening encodes
the saved uint16 codes into ANS without recalibration. ANS residency is not a
claim that the file itself is an ANS archive. Float16 residency remains bit-packed.

## What `dtype` means

`dtype` selects the stored numerical representation at the IO boundary. It does
not select the MAPED calculation precision, GPU backend, or compression codec.
`scaled_uint16` is the only scaled-storage spelling; `uint16_scaled` is not an alias.

| Choice | Stored values | Values returned by precision reads | Scientific consequence |
|---|---|---|---|
| `float32` export | Original float32 intensities | Float32 intensities | No additional precision reduction |
| `float16` | Half-precision floating-point intensities | Float32 reconstruction of the stored float16 values | Reduced precision; spacing grows with magnitude; no intensity scale/offset |
| `scaled_uint16` | Unsigned 16-bit codes with saved scale and offset per region | Float32 calibrated intensities | Uniform intensity step within each region; rounding introduces measured storage error |

Both float16 and scaled uint16 storage are supported by CUDA and Python MPS IO.
ANS preserves scaled uint16 codes exactly; float16 currently uses bit packing,
which also preserves its stored bit patterns exactly. Compression does not recover
precision removed by conversion. Physical resident size depends on the encoded
codes and metadata, not only the nominal two bytes per stored value.
Plain `uint16` is an ordinary integer conversion, not calibrated scaled storage.
Do not substitute it for `scaled_uint16` when preserving fractional intensities.

On loading an existing precision file, omit `dtype` to reuse its saved values
and calibration. This does not redo the original conversion. Explicitly choosing
a different reduced precision can add rounding; prefer converting from the
original float32 archive. Reloading with `dtype="float32"` is not a way to undo
precision loss and is rejected for these calibrated files; use ordinary reads
to obtain their float32 reconstructed intensities.

For scaled storage, reconstruction is `intensity = code * scale + offset`,
rounded to float32. RMSE and maximum error compare that reconstruction with the
pre-conversion source. They are storage error, not a measure of MAPED's physical
accuracy. The precision report and calibration are saved with the data.

A GPU tensor or generated source with `shape`, `dtype`, and ordered `blocks()`
can use the same load call. All logical frames must be yielded exactly once.
Generated sources retain their scientific algorithm; IO owns conversion,
calibration, encoding, reductions and persistence. `io.save(path, source,
dtype="scaled_uint16")` streams regions without keeping the full encoded output.
The caller must close loaded residents when done.

The version-2 precision record stores contiguous frame bounds, scale/offset,
geometry, source dtype and GPU-measured errors. Codes use nearest rounding with
ties to even; restored intensities are float32. Reloads retain the recorded
calibration, including for cross-region crops, instead of recalibrating codes.
Old globally scaled files retain their original version-1 behavior. Older
readers reject version 2, and the native Swift global file reader does not yet
support regional files. This change does not claim Live4DSTEM UI integration.

The brief message gives ANS resident size, RMSE, maximum error and overflow. Detailed
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
