# Count representations: dense, packed, and ANS

Dense and lossless-packed representations remain supported parts of the library.
Dense arrays remain
the ordinary input for algorithms that require them; packed sources let
compatible kernels address exact integer counts without expanding the complete
4D array. Choosing packed storage must not silently choose another dtype,
mask, scan selection, detector bin, or calibration.

The three selectors are `"dense"`, `"packed"`, and `"ans"`. They describe
in-memory layout, separately from file `format` and `compression`. There are no
representation-name aliases. Within `packed`, the authenticated storage schema
selects the matching decoder; different profiles do not share a decoder merely
because they share this public name.
See {ref}`file format and compression <io-file-format-compression>`
for the canonical save/load workflow and its current limits.

| New count workflow | Implementation | Qualification |
|---|---|---|
| QuantEM/ANS file to dense counts | Explicit CPU reference | Bounded exact integer tests |
| QuantEM/ANS file to ANS resident | Python MPS | Physical small-file integer parity |
| QuantEM/ANS file to packed resident | Python MPS | Physical small-file integer parity |
| QuantEM/ANS file to ANS or packed resident | CUDA | Host oracle and compilation; physical GPU pending |
| ANS arrays to exact DP and mask sums | Native Swift/Metal | Small physical integer tests; file reader pending |
| GPU dense materialization and reverse conversions for the new profile | Pending | Not qualified |

For these new profiles, `detector.prepare(data).frame(...)` and
`masked_sum_exact(...)` consume resident counts. Mean DP, moments, and SSB are
not qualified by those tests. Same-representation conversion returns the same
owner; ANS-to-packed creates an independent owner without closing the source.
Both encoded forms coexist at conversion peak, without full dense expansion.
The following sections document the retained dense/`packed` baseline,
not a promise that its operations automatically work on the new ANS profile.

## Choose the representation

```python
from quantem.gpu import io

# Step 1. Retain an original HDF5 source as ordinary dense counts.
dense = io.load("scan_master.h5", representation="dense", dtype="native")

# Step 2. Load an already prepared Lossless Pack Format source directly.
packed = io.load("scan-lossless.h5", representation="packed")

# Step 3. Check scientific geometry separately from physical storage.
print(packed.shape, packed.dtype)
print(packed.logical_bytes, packed.resident_bytes)
```

The default is currently **source-native**, not automatic transcoding. An
ordinary HDF5 source follows the existing dense path; a prepared Lossless Pack
Format source stays packed; a standalone ANS source stays ANS. ANS-to-packed is
an explicit implemented conversion on Python MPS/CUDA, while unsupported
conversions raise with a corrective next step. Automatic original-HDF5 packing
and prepared-packed to dense materialization are not implemented by Python
`io.load` yet. Native
preparation is a separate, authenticated
[producer lifecycle](native_lossless_pack_v1_producer.md).

For portable direct-bitpacked CUDA loads, pass `expected_source_sha256=` from
the trusted producer receipt. It is the **whole prepared container** SHA-256,
not the original HDF5 source-identity field. Python MPS accepts the same check.
Do not substitute a newly computed hash of an untrusted file for retained
producer evidence. Header-only `io.inspect` reports
`index_complete_payload_unverified`; loading authenticates the payload.

## One scientific contract, separate storage facts

| Field | Meaning |
|---|---|
| `representation` | `dense`, `packed`, or `ans` |
| `source_dtype` | Original detector-count type |
| `working_dtype` | Exact type exposed to scientific operations after the declared mask policy |
| `source_shape` | Original scan and detector geometry |
| `working_shape` | Returned logical geometry, in scan-row, scan-column, detector-row, detector-column order |
| `scan_bin`, `detector_bin`, `crop` | Explicit scientific transformations; never implied by representation |
| `residency` | Host or device location, independent of representation |
| `logical_bytes` | Bytes an equivalent dense working tensor would occupy |
| `resident_bytes` | Reported payload/allocation bytes, not peak process or accelerator memory |
| `storage_schema` | Internal encoding profile required to interpret the source |

For Python, the two byte counts above are result properties backed by
`working_logical_tensor_bytes` and `physical_resident_bytes` in metadata.
Source-derived calibration and authenticated detector exclusions are retained.
Excluded raw values are not discarded merely because products mask them out.
The public packed loader rejects older mask-only containers lacking exact raw
reconstruction. Its working-array operations still apply the declared mask.

Dense dtype conversion can be lossy even though the destination is dense.
`loaded.lossless` means exactness is established by its metadata; `False` also
includes unverified narrowing. Prefer `dtype="native"`; request integer
sum-binning explicitly and prove the output range before narrowing it.

## Current backend boundary

This table describes code paths, not performance qualification. Each platform
still needs paired real-fixture, peak-memory, load, and interaction evidence.

| Platform | Dense path retained | Packed path | Current packed boundary |
|---|---|---|---|
| CUDA / Python | `io.load(..., backend="cuda", representation="dense")` | Same verb with `representation="packed"` | Both current profiles; direct-bitpacked input requires an external container seal |
| MPS / Python | `io.load(..., backend="mps", representation="dense")` | Same verb with `representation="packed"` | Direct-bitpacked profile only; uint16/LZ4 profile remains native-Metal-only on Apple |
| Swift / Metal | Native indexed and resident loaders | Native packed loader and producer | Both profiles; explicit resource plan, source authentication, and owned resident lifetime |
| WebGPU | `loadLocalH5Master` | `loadCompactH5WebGPU` | Both readers packaged; different existing source/lifecycle interfaces; exact receipt required for raw-lossless admission |
| Vulkan / native | Bounded dense staging and selected-frame decode | `PackedDetectorSession` | Expanded width 0–16 descriptors; compact headers currently width 0–8 only; not full dense residency |
| CPU reference | `io.load(..., backend="cpu", representation="dense")` | Explicit reference decoder for tests | No accelerated/public packed `io.load` backend |

Python detector calls recognize package-owned CUDA/MPS packed sources before
array conversion. They reuse the existing resident kernels:

```python
from quantem.gpu import detector

# Step 1. Prepare a lightweight operation handle, not another 4D array.
session = detector.prepare(packed)

# Step 2. Read one requested diffraction pattern.
diffraction = session.frame(23)

# Step 3. Compute a binary-mask image with an exact integer result.
counts = session.masked_sum_exact(detector_mask)
```

Here `detector_mask` is a caller-defined boolean array with the working
detector shape. BF/DF/ADF use the same resident binary-mask path when the
caller supplies a center and radius. The ordinary public `masked_sum` keeps
its existing float32 output; use `masked_sum_exact` when exact integer totals
are required. Python MPS exposes resident mean diffraction and CoM from
authenticated prepared moments. CUDA exposes unmasked CoM when the source
contains authenticated prepared moments, and `SSB.open` routes qualified
packed sources through the same `io.load` owner. See the [SSB API](ssb.md) for
its calibration and native-shape requirements. CUDA packed mean diffraction,
masked CoM, and multi-frame scan reductions remain unsupported. Missing
operations raise explicitly; they do not silently allocate a dense volume.

Python packed loading currently accepts one complete source at a time, native
working dtype, row-major order, and its recorded mask. Selections, additional
binning, series stacking, Torch conversion, and unmasked raw access remain
separate gates. The existing dense workflows for selection, binning,
reconstruction, and `io.save` remain available. `io.save` is not a packed-source
transcoder.

## Ownership and regression checks

For the staged integration of ANS count encodings, detector indexes, and
compressed final Fourier fields, see the
[representation integration guide](../developer/representation-integration.md).
The count-ANS integration described above does not add cross-backend
compressed-Fourier support or qualify retained research codecs automatically.

Keep the source alive until the last queued operation completes. Python packed
sources and directly owned MPS buffers support `loaded.close()` or a context
manager. Ordinary NumPy, CuPy, and Torch arrays retain their normal reference
ownership; `close()` does not destroy borrowed array references. WebGPU display
consumers distinguish borrowed buffers from owned uploads and never destroy a
borrowed scientific buffer. Native Vulkan sessions own their final buffers;
their loaders borrow bounded spans during ingestion.

Regression entry points:

```bash
python scripts/run_tests.py tests/contracts/io/test_representation.py tests/hardware/mps/test_mps_compact_v3.py -q
PYTHONPATH=src pytest -q tests/contracts/test_webgpu_*lifetime.py tests/contracts/test_webgpu_resident_contract.py
QGPU_RUN_PYTHON_PARITY=1 QGPU_PYTHON="$(command -v python)" swift test --package-path src/quantem/gpu/swift
```

Use the {download}`Vulkan build guide <../../src/quantem/gpu/android/README.md>` for host
contract tests and physical Android dispatch. Host/Node tests and small Metal
fixtures do not establish real-device throughput or app frame rate. Report
cold original source, prepared creation, prepared reopen, resident interaction,
process RSS, accelerator allocation, and peak memory separately for **each**
implemented representation. See [parity methodology](../performance/parity.md).
