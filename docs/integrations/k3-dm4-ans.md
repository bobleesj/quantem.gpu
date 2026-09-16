# K3 DM4 and GPU ANS snapshots

Calibrated DigitalMicrograph DM4 acquisitions can be opened directly on CUDA,
Python MPS and native Swift/Metal. The loader selects the unique four-dimensional
uint8/uint16 image, excluding survey images. Scan and detector axes retain their
native `(row, column)` order and calibration. Encoding uses bounded scan windows;
no scan binning, detector binning, crop or intensity scaling is introduced.

The shared [native acquisition layout protocol v1.1](../api/native-acquisition-formats.md)
defines K3 identification, exact tag paths, normalized fields, units and snapshot
metadata placement. K3 is identified from the camera model, not a filename;
generic DM4 remains generic. In Live4DSTEM, **Save Compressed Copy…** creates an
`.ans` copy; reopening retains the K3 identity, recorded acquisition processing,
scan sampling, reciprocal sampling and voltage. No original file is modified.

## Save once and reopen

```python
from quantem.gpu import io

with io.load("STEM SI.dm4", backend="mps") as acquisition:
    io.save("acquisition.ans", acquisition, format="quantem", backend="mps")

with io.load("acquisition.ans", backend="mps") as acquisition:
    print(acquisition.shape, acquisition.dtype)
```

Use `backend="cuda"` on NVIDIA. Both backends write and read the same
`QGPUSTRM` container with profile `runtime-column-rans-spatial-v2`.
Reopening restores encoded bytes and spatial indexes directly, with header and
body SHA-256 verification; it does not re-encode the original acquisition.
Saving refuses to overwrite an existing destination. The original DM4 is not
needed for reopening a complete snapshot.

This runtime snapshot is distinct from the portable `QGANS` reference container.
The `.ans` extension alone does not identify a codec. These source changes need a
matching development revision; they are not a claim about an older PyPI release.
Python DM4 metadata reading needs the `dm` extra (`ncempy`); native Swift does not.

## Exact detector products

```python
import numpy as np
from quantem.gpu import detector, io

with io.load("acquisition.ans", backend="mps") as acquisition:
    session = detector.prepare(acquisition)
    try:
        row, column = np.indices(acquisition.shape[2:])
        center = (np.array(acquisition.shape[2:]) - 1) / 2
        squared_radius = (row - center[0])**2 + (column - center[1])**2
        adf = session.masked_sum_exact(
            (squared_radius >= 180**2) & (squared_radius <= 360**2)
        )
    finally:
        session.close()
```

Stored detector validity applies to products, while raw diffraction reads retain
file counts. The Metal implementation uses exact packed 8/32-pixel spatial sums
for mask interiors and ANS decoding for boundary residuals. Small mask changes
use signed delta updates. Python MPS sums use uint64, including carry and borrow
across the uint32 boundary. Native viewer product storage is uint32; camera
loaders reject uint16 detector geometries whose possible sums exceed that range
and direct the caller to the Python MPS path.

## Native macOS integration

`NativeDM4Source` inspects native DM4 metadata. `NativeANSSnapshot` inspects the
saved container and verifies its checksums before GPU consumption.
`MetalRuntimeANSResidentSource.load(camera:device:)` and
`load(snapshot:device:)` create the same native resident owner.
`MetalRuntimeANSSeries` owns the reusable diffraction and detector output buffers.
Call `release()` on the series and `releaseResidentStorage()` on its source when
finished. The UI owns selection, scheduling and presentation; the package owns
IO, codecs and scientific kernels.

The Live4DSTEM macOS integration discovers HDF5, DM4 and runtime ANS files in
mixed folders. Finder/Open With can open `.dm4` and `.ans` documents directly.
HDF5 keeps its existing original/packed loading routes and correction policy.

## Reproduce correctness and performance

On an Apple Silicon Mac, build the existing `metal-runtime-ans-benchmark` product
and run it with `--camera /absolute/path/acquisition.ans` or a DM4 path.
`K3_AUDIT_DM4=/absolute/path/original.dm4` additionally compares every decoded
count with native file bytes. This exhaustive audit is separate from timing a
single interactive request. `K3_BENCH_TRIALS` controls translated-mask trials.

The focused tests are `NativeCameraSourceTests` and
`tests/hardware/mps/test_camera_spatial_ans.py`; CUDA interchange tests are in
`tests/contracts/io/test_digitalmicrograph.py` with `QUANTEM_TEST_CUDA=1`.
Record the exact revision, hardware, file shape/dtype, cache state, first query,
steady-state queries and physical presentation separately. A kernel completing
within 8.33 ms does not establish 120 Hz on a 60 Hz display. Large arbitrary masks,
first-use compilation and unrelated GPU work can have different costs.

For the bounded native workflow check, run:

```sh
K3_REOPEN_TRIALS=5 bash scripts/check_metal_camera.sh original.dm4 saved.ans
# Also encode, save and verify every retained native metadata field:
bash scripts/check_metal_camera.sh original.dm4 saved.ans new-copy.ans
```

The first command reports header-through-resident wall time, including body
checksum verification and GPU setup. Repeated loads reuse filesystem pages;
these are not cold-storage or UI-presented-frame measurements. The check compares
seven original diffraction frames and BF/ADF/DF values at those scan positions
for four translated masks. It is not an exhaustive every-count audit. Raw DM4
loading additionally reads and encodes the original volume and must be timed
separately; a fast compressed reopen is not a one-second original-load claim.

Native Swift writers can call `source.saveSnapshot(to: destination)` on a live,
exclusively owned resident with spatial indexes. To create those indexes during
original HDF5 loading, pass `includeSpatialIndex: true` to
`MetalRuntimeANSResidentSource.load(source:device:...)`. DM4 residents already
include them. Run encoding and saving on a worker, keep the owner alive until
completion, and use `shouldCancel` for cooperative cancellation. Saving keeps
existing destinations and publishes completed files atomically. The saved
`source_metadata` retains native metadata; this path preserves stored counts
without median correction.
