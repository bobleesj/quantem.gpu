# Migration notes

`quantem.gpu` exists to remove permanent duplicate accelerated code from
`quantem.widget`, `quantem.live`, and the legacy `quantem.cuda` name.

## Current ownership

`quantem.gpu` owns:

- GPU IO and decompression.
- Chunk assembly and load-to-device.
- Device selection and backend errors.
- Heavy BF/DF/DPC image compute.
- SSB compute APIs.

`quantem.widget` owns:

- anywidget UI.
- Interaction state.
- HTML/notebook export.
- Display wrappers around arrays and reduced images from `quantem.gpu`.

`quantem.live` calls `quantem.gpu` for product and SSB compute instead of
keeping second copies.

## Dense, packed, and experimental status

Dense loading remains supported, including host arrays and accelerator-resident
arrays where the backend supports them. Representation (`dense` or
`packed`), location (host or device), and scientific dtype are separate
choices. Original compressed HDF5 does not silently become a prepared packed
file, and packed input does not silently expand into a dense volume.

The following summarizes the [representation contract](../api/representations.md),
not a new qualification registry. Exact gates remain in
`tests/parity/backend_matrix.json`; measured and pending performance remain in
`benchmarks/profile_matrix.json`.

| Runtime | Implemented entry points | Remaining or experimental scope |
|---|---|---|
| Python CUDA | Dense and lossless-packed `io.load`; detector reductions; prepared CoM and packed SSB within their recorded contracts | Packed mean-DP, masked CoM, and arbitrary packed scan reductions are not general public operations. |
| Python MPS | Dense loading and direct-bitpacked loading; detector reductions and prepared products | The packed uint16/LZ4 profile is native-Metal-only on Apple. |
| Native Swift/Metal | Indexed dense and both packed profiles; source inspection, admission, authenticated loads, detector and prepared products | App adoption and physical end-to-end qualification must use an exact package revision. |
| WebGPU | Dense and packed readers, batched detector updates, resident display and lifetime handling | Experimental consumer integration. The held uint16 DPC/iDPC numerical candidate is not promoted; device-specific parity and presentation gates remain open. |
| Android/Vulkan | Native packed detector session, BF/DF/ADF, selected diffraction, bounded dense decode/staging | Experimental. Compact headers support widths 0–8, expanded descriptors 0–16. No full dense-volume residency, shared SSB, or general 1024 FFT claim. |
| Native Direct3D | Caller-owned D3D11 FFT implementation and tests | Experimental, not release-qualified or a general packed/IO backend. |
| CPU reference | Explicit dense reference workflows and test decoders | Not an automatic fallback or a public accelerated packed loader. |

These source changes do **not** establish full-file cold loading in 1–2 seconds
or 120 presented scientific updates per second. Preparation, authentication,
source reads, device residency, reconstruction, and actual presentation must be
measured separately on the target device. Physical phone acceptance remains a
consumer task after repinning; a host test or a resident kernel benchmark is not
its substitute.

## What the refactor changes

The [repository architecture](backend-layout-and-parity.md) defines one owner
per implementation. IO models, metadata, selection, pinned staging, and packed
dispatch have separate modules. Scientific backend code now lives under each
domain's `backends/`; native Android code is owned by `vulkan/`.

- Retain the import-only `compute/` and earlier IO compatibility files while
  consumers migrate. They are live compatibility boundaries, not dead kernels.
- Retain the Android CMake/header forwarding entry and native library names.
- Remove unused private helpers only after checking callers. The reviewed
  cleanup removes obsolete loading, detector, and screening helpers.
- Common imports no longer import CuPy or replace the caller's pinned-memory
  allocator. CUDA allocation still uses CuPy's configured allocator; the
  existing bounded host-registration pool remains in `io/_memory.py`.
- Dense streaming orchestration still occupies `io/load.py`. Further splitting
  remains work, with source ordering, cancellation, and failure cleanup frozen.

## Canonical representation names and receipt v3

The resident receipt is now `quantem.gpu.4dstem-resident-receipt/v3`.
`representation` is `dense`, `packed`, or `ans`; the separate `storage_encoding`
field is removed. `storage_schema`, source/working dtype, geometry, hashes, and
byte counts retain the detailed scientific meaning. This is an explicit schema
change, not wire compatibility with v1 or v2. Existing sealed results keep their
original versions; do not rewrite old evidence to make it look like a new run.

Consumers of the earlier Python MPS representation enum must use
`DataRepresentation.DENSE`, `.PACKED`, or `.ANS`. Swift clients use `.dense`,
`.packed`, or `.ans`. The former `ResidentStorageEncoding` and
`MPSResidentRepresentation` type aliases and the
`lossless_packed` selector are removed. Update receipt parsers deliberately;
unknown schema versions must fail closed. Apple capability records are v4 and
publication records are v2. WebGPU and Vulkan share the three-value vocabulary;
that does not qualify ANS loading or kernels on those backends.

The remote `storage_kind` field also reports `packed`; its separate
`storage_schema` continues to identify the decoder. Save calls accept only
`format="arina"` or `format="quantem"`. Replace `format="ans"` with
`format="quantem", compression="ans"`, and remove the old HDF5 format aliases.
File encodings themselves are unchanged. Ordinary native HDF5 now loads into
packed storage by default on CUDA:

```python
tilts = io.load(files, stack=False)
```

Preparation decodes and packs one complete acquisition at a time; all packed
sources remain resident when the call returns. Original uint8/uint16 counts,
full geometry, and detector-mask metadata are retained. Masks are applied by
scientific consumers, not by modifying the packed counts. Each result is
caller-owned and must be closed after its final consumer. The existing dense
result also supports `to_representation("packed")` on CUDA. This does not create
a packed file or imply MPS/WebGPU support for this conversion.

## Next migration steps

1. Pin each consumer to a reviewed package revision. Export the complete
   WebGPU source graph with `webgpu.export_sources(...)`; build native clients
   from SwiftPM or the Vulkan CMake entry, without copying kernels.
2. Adapt receipt parsers to v3 and Apple capability controls to v4. Keep unsupported
   operations unavailable rather than expanding or downcasting implicitly.
3. Verify original compressed HDF5 and prepared packed inputs separately,
   including 512 and 1024 scans where admitted, file A–B–A switching,
   cancellation, replacement release, and relaunch.
4. Run real detector translation and resizing, DF/ADF rings, DPC, colormaps,
   contrast, and FFT-off behavior in the actual app. Record scientific update
   cadence and presentation independently, with no hidden binning.
5. Close the held WebGPU numerical gates and missing backend operations before
   enabling them. Remove compatibility files only after every consumer has
   migrated and been tested.

Native macOS Live4DSTEM calls the Swift package products instead of a local
Python backend:

| Client need | Endpoint |
|---|---|
| Browser FFT of BF/ADF/custom | `MetalImageFFT.logMagnitude` |
| Histogram / contrast window | `MetalImageRuntime` |
| HDF5/EMD catalog | `Native4DSTEMIO` |
| Decode / detector / CoM | `Metal4DSTEMKernels` |
| Remote raw 4D on a CUDA workstation | Python `quantem.gpu` CUDA service only |

Do not restore an app-local FFT, histogram, or HDF5 parser. Do not bundle
Python in the signed Mac app. Pin Live4DSTEM to an exact `quantem.gpu`
revision after this package is published; a local path override is only for
integration worktrees.

## Release checks

Before publishing an rc:

1. Run focused GPU parity tests.
2. Build wheel and sdist into a temporary directory.
3. Run `twine check`.
4. Inspect package contents for private data or generated reports.
5. Install from TestPyPI and verify:

   ```python
   import importlib.metadata as md
   import quantem.gpu

   assert md.version("quantem.gpu") == quantem.gpu.__version__
   ```

## Do not regress

- Do not move GPU decompression back into widget.
- Do not make SSB depend on anywidget.
- Do not use `quantem.cuda` as the public package name.
- Do not treat CPU fallback speed as acceptable for GPU workflows.
- Do not use fast-mode SSB as parity evidence.
- Do not copy `MetalImageFFT` or `Native4DSTEMIO` source into Live4DSTEM.
- Do not add a local Python FFT or HDF5 helper to the Mac app.


## Python HDF5 loading defaults to packed storage

`io.load(path)` now preserves complete native uint8/uint16 HDF5 counts in
lossless packed GPU storage. Backend selection remains automatic. For multiple
acquisitions use `io.load(paths, stack=False)` to retain separate packed owners.
Saved packed, ANS and paired sources continue to reopen their recorded layouts.

Code requiring dense tensors, selection, binning, or dtype conversion must request
`representation="dense"` explicitly. Packed counts retain detector-mask metadata;
apply that mask when calculating products rather than changing stored counts.

Ordinary HDF5 packing is currently implemented for CUDA. Unsupported dtypes and
backends raise with corrective guidance; there is no implicit dense or CPU fallback.
Python MPS can reopen supported prepared packed sources; the native Metal HDF5
loader remains a separate path. CUDA conversion currently stages one complete
native acquisition, so packed output size alone is not a loading peak-memory bound.
