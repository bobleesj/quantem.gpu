# Count representations: dense, packed, and encoded

For normal acquisition loading, use `io.load(path)` without a representation
option. Supported originals default to ANS on CUDA/MPS; saved files retain
their authenticated layout. The selectors below document advanced and retained
reference paths, not options a new user must choose. No format-support claim
follows from a selector alone: see the [acceptance matrix](../maintainer/ans-io-acceptance.md).

Dense and lossless-packed representations remain supported parts of the library.
Dense arrays remain
the ordinary input for algorithms that require them; packed sources let
compatible kernels address exact integer counts without expanding the complete
4D array. Choosing packed storage must not silently choose another dtype,
mask, scan selection, detector bin, or calibration.

The four selectors are `"dense"`, `"packed"`, `"encoded"`, and `"paired"`. They
describe in-memory layout, separately from file `format` and `compression`. There are no
representation-name aliases. Within `packed`, the authenticated storage schema
selects the matching decoder; different profiles do not share a decoder merely
because they share this public name.
See {ref}`file format and compression <io-file-format-compression>`
for the canonical save/load workflow and its current limits.
Prepared CUDA series also report `representation="encoded"`, with the fixed
`resident_profile` and `resident_codec="tans"` distinguishing their paired-tANS
dense component and sparse-count layout from portable or runtime rANS sources.
See the [reproduction guide](../developer/reproducing-resident-analysis.md) for
the entry-point and implementation map.

The following table records lower-level representation qualification. Rows for
packed residents describe conversion/decoder support, not permission to load
packed acquisitions through the public GPU loader.

| Count representation workflow | Implementation | Qualification |
|---|---|---|
| QuantEM encoded file to dense counts | Explicit CPU reference | Bounded exact integer tests |
| QuantEM encoded file to encoded resident | Python MPS | Physical small-file integer parity |
| QuantEM encoded file to packed resident | Python MPS | Physical small-file integer parity |
| QuantEM encoded file to encoded or packed resident | CUDA | Bounded physical GPU integer parity |
| Complete H5 to runtime encoded with spatial indexes | CUDA | Bounded real and adversarial count parity; full-66 throughput unqualified |
| Complete uint16 H5 to paired-count tANS resident with polar index (`"paired"`) | CUDA | Synthetic exact sums and frames; frozen full-array digests on one native acquisition; 69-acquisition series load measured on one device |
| Saved paired resident form to paired resident | CUDA | Byte-identical reopen on synthetic and native sources |
| Encoded arrays/files to exact DP and mask sums | Native Swift/Metal | Small physical integer tests; bounded `.qem` file-load parity |
| GPU dense materialization and reverse conversions for the new profile | Pending | Not qualified |

For these new profiles, `detector.prepare(data).frame(...)` and
`masked_sum_exact(...)` consume resident counts. Mean DP, moments, and SSB are
not qualified by those tests. Same-representation conversion returns the same
owner; encoded-to-packed creates an independent owner without closing the source.
Both compact forms coexist at conversion peak, without full dense expansion.
For measured codec tradeoffs and the distinction between payload, scratch and
complete-process peak, see the
[CUDA count-codec investigation](../performance/cuda-count-codecs.md). Its
experimental kernels do not change the qualification table above.
The following sections document the retained dense/`packed` baseline,
not a promise that its operations automatically work on every encoded profile.

## Choose the representation

```python
from quantem.gpu import io

# Step 1. Retain an original HDF5 source as compact encoded counts.
encoded = io.load("scan_master.h5")

# Step 2. Save an independently reopenable compressed copy.
io.save("scan.qem", encoded)

# Step 3. Check scientific geometry separately from physical storage.
print(encoded.shape, encoded.dtype)
print(encoded.logical_bytes, encoded.resident_bytes)
```

Acquisition loading uses **ANS residency** on CUDA and MPS. Original supported
files are ingested in bounded blocks, and `.qem` copies reopen compressed.
Dense/packed overrides and older prepared-packed acquisitions are rejected by
the public GPU loader. Reopen the original acquisition and save a new `.qem`
copy instead. Low-level conversion APIs are separate from this loading policy.
Explicit
`io.load(..., backend="cuda", representation="encoded", apply_mask=False)` now
streams complete uint8/uint16 H5 acquisitions into a runtime encoded resident with
exact spatial indexes. Use `io.save` for the supported `.qem` round trip.
`representation="paired"` is the second opt-in CUDA layout for complete uint16
acquisitions; it streams whole shards with direct I/O, keeps every count, saves
its resident arrays once (`loaded.data.save(path)`) and reopens that file under
the same selector without decoding. See the
[paired resident layout](../developer/paired-resident.md) for its contract.
Automatic original-HDF5 packing and prepared-packed to dense materialization
remain unimplemented. Native
preparation is a separate, authenticated
[producer lifecycle](native_lossless_pack_v1_producer.md).

For retained low-level portable direct-bitpacked CUDA readers, use the container seal from
the trusted producer receipt. It is the **whole prepared container** SHA-256,
not the original HDF5 source-identity field. Python MPS accepts the same check.
Do not substitute a newly computed hash of an untrusted file for retained
producer evidence. Header-only `io.inspect` reports
`index_complete_payload_unverified`; loading authenticates the payload.

## One scientific contract, separate storage facts

| Field | Meaning |
|---|---|
| `representation` | `dense`, `packed`, or `encoded` |
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
The low-level packed reader rejects older mask-only containers lacking exact raw
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
| CUDA / Python | Explicit arrays and bounded reads, not acquisition loading | Low-level conversion only | Public acquisition loading requires ANS |
| MPS / Python | Explicit arrays and bounded reads, not acquisition loading | Low-level conversion only | Public acquisition loading requires ANS |
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
contains authenticated prepared moments. `SSB.open` follows the same ANS-only
acquisition loading policy. See the [SSB API](ssb.md) for
its calibration and native-shape requirements. CUDA packed mean diffraction,
masked CoM, and multi-frame scan reductions remain unsupported. Missing
operations raise explicitly; they do not silently allocate a dense volume.

The public GPU loader no longer admits prepared-packed acquisitions. Re-export
the original source as `.qem`. Use bounded reads from the encoded owner for
selected diffraction patterns; explicit array algorithms remain separate.
For multiple acquisitions, `io.load(paths, stack=False)` returns independent
encoded owners. Dense stacking, whole-acquisition Torch conversion, and combined
selection/resampling options are not implied by encoded loading. Unsupported
combinations raise rather than materializing a complete 4D array.

Native Swift/Metal float32 NumPy ingestion and `.qem` reopen carry the actual
detector `(row, column)` geometry, including rectangular and non-power-of-two
shapes. The native acceptance gate verifies original ingestion, cross-language
reopen, native export, exact diffraction bits, mask products, and region means.
This backend gate does not certify application UI integration or every native
file reader; see the [acceptance matrix](../maintainer/ans-io-acceptance.md).

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
QGPU_RUN_PYTHON_PARITY=1 QGPU_PYTHON="$(command -v python)" swift test
```

Use the {download}`Vulkan build guide <../../src/quantem/gpu/android/README.md>` for host
contract tests and physical Android dispatch. Host/Node tests and small Metal
fixtures do not establish real-device throughput or app frame rate. Report
cold original source, prepared creation, prepared reopen, resident interaction,
process RSS, accelerator allocation, and peak memory separately for **each**
implemented representation. See [parity methodology](../performance/parity.md).
