# Count representations: dense, encoded, and paired

For normal acquisition loading, use `io.load(path)` without a representation
option. Supported originals default to ANS on CUDA/MPS; saved files retain
their authenticated layout. The selectors below document advanced and retained
reference paths, not options a new user must choose. No format-support claim
follows from a selector alone: see the [acceptance matrix](../maintainer/ans-io-acceptance.md).

The three selectors are `"dense"`, `"encoded"`, and `"paired"`. They describe
in-memory layout, separately from file `format` and `compression`. There are no
representation-name aliases. Within `encoded`, the authenticated storage schema
selects the matching decoder; different profiles do not share a decoder merely
because they share this public name. Choosing a representation never silently
chooses another dtype, mask, scan selection, detector bin, or calibration.
See {ref}`file format and compression <io-file-format-compression>`
for the canonical save/load workflow and its current limits.
See the [reproduction guide](../developer/reproducing-resident-analysis.md) for
the entry-point and implementation map.

| Count representation workflow | Implementation | Qualification |
|---|---|---|
| QuantEM encoded file to dense counts | Explicit CPU reference | Bounded exact integer tests |
| QuantEM encoded file to encoded resident | Python MPS | Physical small-file integer parity |
| QuantEM encoded file to encoded resident | CUDA | Bounded physical GPU integer parity |
| Complete H5 to runtime encoded with spatial indexes | CUDA | Bounded real and adversarial count parity |
| Complete uint16 H5 to paired-count tANS resident with polar index (`"paired"`) | CUDA | Synthetic exact sums and frames; frozen full-array digests on one native acquisition; 69-acquisition series load measured on one device |
| Saved paired resident form to paired resident | CUDA | Byte-identical reopen on synthetic and native sources |
| Encoded arrays/files to exact DP and mask sums | Native Swift/Metal | Small physical integer tests; bounded `.qem` file-load parity |
| GPU dense materialization and reverse conversions | Pending | Not qualified |

For these profiles, `detector.prepare(data).frame(...)` and
`masked_sum_exact(...)` consume resident counts. For measured codec tradeoffs
and the distinction between payload, scratch and complete-process peak, see the
[CUDA count-codec investigation](../performance/cuda-count-codecs.md). Its
experimental kernels do not change the qualification table above.

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
Dense overrides are rejected by the public GPU loader; use bounded reads from
the encoded acquisition instead. Explicit
`io.load(..., backend="cuda", representation="encoded", apply_mask=False)`
streams complete uint8/uint16 H5 acquisitions into a runtime encoded resident with
exact spatial indexes. Use `io.save` for the supported `.qem` round trip.
`representation="paired"` is the second opt-in CUDA layout for complete uint16
acquisitions; it streams whole shards with direct I/O, keeps every count, saves
its resident arrays once (`loaded.data.save(path)`) and reopens that file under
the same selector without decoding. See the
[paired resident layout](../developer/paired-resident.md) for its contract.

The Lossless Pack Format containers written by the native Swift
[producer](native_lossless_pack_v1_producer.md) are read only by the native
Swift/Metal loader. `io.load` does not open them; reopen the original
acquisition and save a `.qem` copy instead.

## One scientific contract, separate storage facts

| Field | Meaning |
|---|---|
| `representation` | `dense`, `encoded`, or `paired` |
| `source_dtype` | Original detector-count type |
| `working_dtype` | Exact type exposed to scientific operations after the declared mask policy |
| `source_shape` | Original scan and detector geometry |
| `working_shape` | Returned logical geometry, in scan-row, scan-column, detector-row, detector-column order |
| `scan_bin`, `detector_bin`, `crop` | Explicit scientific transformations; never implied by representation |
| `residency` | Host or device location, independent of representation |
| `logical_bytes` | Bytes an equivalent dense working tensor would occupy |
| `resident_bytes` | Reported payload/allocation bytes, not peak process or accelerator memory |
| `resident_profile` | Internal encoding profile of the loaded resident, when reported |

For Python, the two byte counts above are result properties backed by
`working_logical_tensor_bytes` and `physical_resident_bytes` in metadata.
Source-derived calibration and authenticated detector exclusions are retained.
Excluded raw values are not discarded merely because products mask them out.

Dense dtype conversion can be lossy even though the destination is dense.
`loaded.lossless` means exactness is established by its metadata; `False` also
includes unverified narrowing. Prefer `dtype="native"`, and prove the output
range before narrowing it.

## Current backend boundary

This table describes code paths, not performance qualification. Each platform
still needs paired real-fixture, peak-memory, load, and interaction evidence.

| Platform | Dense path retained | Encoded path |
|---|---|---|
| CUDA / Python | Explicit arrays and bounded reads, not acquisition loading | Public acquisition loading |
| MPS / Python | Explicit arrays and bounded reads, not acquisition loading | Public acquisition loading |
| Swift / Metal | Native indexed and resident loaders | Native encoded sources; native packed loader and producer |
| WebGPU | `loadLocalH5Master` | Browser `.qem` count acquisitions (`RansResidentSet.loadQemFiles`) |
| CPU reference | `io.load(..., backend="cpu", representation="dense")` | Explicit `.qem` reference decoder |

Python detector calls reuse the resident kernels of the loaded source:

```python
from quantem.gpu import detector

# Step 1. Prepare a lightweight operation handle, not another 4D array.
session = detector.prepare(encoded)

# Step 2. Read one requested diffraction pattern.
diffraction = session.frame(23)

# Step 3. Compute a binary-mask image with an exact integer result.
counts = session.masked_sum_exact(detector_mask)
```

Here `detector_mask` is a caller-defined boolean array with the working
detector shape. BF/DF/ADF use the same resident binary-mask path when the
caller supplies a center and radius. The ordinary public `masked_sum` keeps
its existing float32 output; use `masked_sum_exact` when exact integer totals
are required. `SSB.open` follows the same ANS-only acquisition loading policy.
See the [SSB API](ssb.md) for its calibration and native-shape requirements.
Missing operations raise explicitly; they do not silently allocate a dense volume.

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

Keep the source alive until the last queued operation completes. Encoded
sources and directly owned MPS buffers support `loaded.close()` or a context
manager. Ordinary NumPy, CuPy, and Torch arrays retain their normal reference
ownership; `close()` does not destroy borrowed array references. WebGPU display
consumers distinguish borrowed buffers from owned uploads and never destroy a
borrowed scientific buffer.

Regression entry points:

```bash
python scripts/run_tests.py tests/contracts/io/test_representation.py -q
swift test
```

Host/Node tests and small Metal fixtures do not establish real-device
throughput or app frame rate. Report cold original source, prepared reopen,
resident interaction, process RSS, accelerator allocation, and peak memory
separately for **each** implemented representation. See
[parity methodology](../performance/parity.md).
