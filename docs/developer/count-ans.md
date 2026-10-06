# Exact count-ANS codec and retained experiment formats

This is an experimental opt-in API. Start with
[experimental resident ANS](../integrations/experimental-resident-ans.md) for current ownership,
downstream viewer usage, supported formats, and qualification limits.

The canonical count-ANS v1 representation preserves every native uint8/uint16
count, including hardware sentinels. It uses independent detector-column byte
rANS streams within scan blocks, a model selector for every stream, and an
explicit uint16 literal profile for incompressible columns. Dimensions are
`(scan_row, scan_column, detector_row, detector_column)`.

## Package ownership

`io.save(path, counts, format="quantem", compression="ans", backend="cpu")` is the explicit
reference encoder. It accepts four-dimensional NumPy arrays or memory maps,
uses bounded scan blocks, records source geometry and 64 MiB body-block checksums,
and publishes a new file atomically without overwriting an existing file.
No quantization, crop, binning or invalid-pixel replacement occurs. This is a
correctness reference, not an accelerated full-acquisition encoder.

```python
from quantem.gpu import io

with io.load("acquisition.h5", backend="cuda", representation="encoded") as resident:
    io.save("acquisition.qem", resident, format="quantem", backend="cuda")
```

`io.qem` validates the `quantem.qem` container and returns one
backend-neutral array contract.
`quantem.gpu.resident.cuda.counts.StreamedCounts` holds the saved integer codec
on CUDA. `io.load` verifies every body-block checksum while it copies the encoded
arrays into one device buffer; the decode kernels check each stream's terminal
state when they decode it. Block decode, point-pattern gather and binary mask
sums retain integer exactness; mask sums are uint64. No full dense source is
constructed.

The container contract is shared with the existing CUDA and MPS implementations.
The native `MetalRuntimeANSResidentSource.load(snapshot:device:)` reader accepts
the same `quantem.qem` file directly. It validates the envelope and typed
section bounds, streams bounded ranges into private Metal buffers, and
validates every encoded stream before publication. It is geometry-general and
does not allocate the logical dense cube. This is a file-load and parity
qualification; it is not a claim that a cold HDF5 source has already been
transcoded to ANS.

The bounded real-data handoff is retained in the local experiment ledger:
64-frame slices from two externally chunked HDF5 acquisitions were encoded
exactly, authenticated, and reopened by native Metal. The complete logical
volume was not materialized. This validates the workflow boundary, not a
full-acquisition or first-load performance claim.

The public loader supports exact CUDA and MPS residency and explicitly requested
CPU reference materialization. The returned dataset keeps its encoded buffers
until `close()`. Indexing returns a device tensor, and
`detector.prepare(...).masked_sum_exact(...)` returns exact `uint64` sums. No
dense acquisition is allocated when `representation="encoded"` is selected:

```python
from quantem.gpu import detector, io

with io.load(saved.path, backend="cuda", representation="encoded", device=0) as loaded:
    pattern = loaded[0, 0]
    image = detector.prepare(loaded).masked_sum_exact(binary_detector_mask)

reference_counts = io.load(saved.path, backend="cpu", representation="dense").data
```

Downstream browser viewers consume canonical files through the package WebGPU
count codec.
Existing HDF5 loading defaults are unchanged.

Body-block checksums detect corruption. Compare an independently retained source
digest when source authentication is required. A valid container checksum
alone does not establish acquisition identity.

## Verification

`tests/test_qem_reference.py` and `tests/test_qem_publication.py` check
native uint8/uint16 round trips, tail blocks, sentinel counts and checksummed
publication with integer equality. CUDA residents are exercised by the hardware
tests, which skip without a CUDA device:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python -m pytest -q tests/hardware/cuda/test_streamed_h5.py
```

## Native Metal file loading and geometry

The native reader uses the same dimension order as the Python contract:
`(scan_row, scan_column, detector_row, detector_column)`. It accepts any
positive four-dimensional uint8/uint16 geometry that fits the count-ANS stream
index range; 512×512 and 1024×1024 square scans and non-square scans use the
same code path.

```swift
let snapshot = try NativeANSSnapshot(url: qemURL)
let source = try MetalRuntimeANSResidentSource.load(snapshot: snapshot, device: device)
let pattern = try source.extractRawDiffraction(scanRow: 0, scanColumn: 0)
source.releaseResidentStorage()
```

The native Metal smoke harness is `metal-runtime-ans-benchmark`. The current
physical gate covers 512×512×1×1 uint16, 1024×1024×1×1 uint16, and a non-square
63×512×2×3 uint8 file. Cold HDF5→`.qem` encoding remains separate work: `io.save`
copies an encoded resident byte-for-byte, so the first load still builds the
resident from the source, and the macOS original HDF5 path remains the exact
bitshuffle/LZ4 decoder.

For production, select the path by source state rather than by detector size:

```text
HDF5 first open  → exact indexed BSLZ4 decode → packed Metal resident
copy build      → encoded CUDA/Metal resident → saved .qem copy, no re-encoding
.qem reopen     → bounded pread/upload → private Metal encoded resident
```

The same metadata-driven contract covers 512×512, 1024×1024, and non-square
scan geometries. Fixed-geometry diagnostic kernels are not part of this ANS
workflow.

## Remaining unification work

The browser loads retained detector-rANS manifests through
`detector/webgpu/rans.ts` and `.qem` files through `RansResidentSet.loadQemFile` /
`loadQemFiles` (`detector/webgpu/qem-source.ts`). Downstream viewers open exported
`.qem` files through the same reader. Changing production sources requires integer parity first. No
latency claim follows from this codec correctness work.
