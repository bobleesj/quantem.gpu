# Android NDK and Vulkan

The native backend lives in `src/quantem/gpu/vulkan`. It provides exact
bounded streaming, GPU decode, packed residency, selected diffraction,
virtual-detector updates, and prepared DPC. Android application code and
physical presentation remain outside this package.

New CMake consumers can use `src/quantem/gpu/vulkan` as the build entry point.
It delegates to these same native sources and preserves the old target names,
header paths, and ABI. Host tests remain portable reference/contract checks;
the GPU implementation still requires the Android NDK toolchain. This is not
an added Linux Vulkan runtime. See the
[layout migration guide](../maintainer/backend-layout-and-parity.md).

## Resident representation

`DataRepresentation` distinguishes `dense` from `lossless_packed`. `LoadPlan`
reports dense decoded staging; `PackedDetectorSessionAdmission` reports packed
resident storage. These fields do not change scientific dtype or turn bounded
staging into a claim of full-volume residency. See the shared
[representation contract](../api/representations.md).

For each detector pixel, successive scan positions are grouped into tiles.
Each tile uses only the bits required by its largest unsigned value. A zero
tile has no payload. A shader looks up the relevant descriptor and extracts
the original integer without expanding the complete four-dimensional array.

The expanded format uses one uint32 descriptor per detector-pixel/scan-tile
pair. The low five bits store width 0–16 and the high 27 bits store a uint32
payload offset. A 128-scan tile occupies `16 * width` bytes. The host packer,
validator, and reference unpacker are in `packed_detector.cpp`.

The resident session also accepts 32-scan compact nibble headers with
checkpoints. **This Vulkan compact-header path currently admits only widths
0–8.** The shared compact uint16 encoding is not yet supported here; do not
infer cross-backend format parity from the branch merge.

A dense `512 × 512 × 192 × 192` uint8 array is 9 GiB; uint16 is 18 GiB.
Packed size depends on the complete source's value distribution plus headers.
All source shards, staging, outputs, and allocation overhead must pass memory
admission. A file fitting in storage does not establish GPU residency.

## Loading and interaction

`PackedDetectorSession` owns final Vulkan allocations. Its caller supplies
the complete source plan, authenticated exclusions, and a bounded loader that
fills borrowed final-buffer spans. Raw-LZ4 payloads can decode directly on the
GPU into those buffers. The loader owns source identity verification and HDF5
metadata interpretation. General HDF5 discovery and one-time preparation are
not hidden inside a GPU decode timing.

Once admitted, selected diffraction and detector reconstruction read the
resident packed source without reopening HDF5. Circular masks use zero-based
`(row, column)` and inner-inclusive, outer-exclusive radii. The detector planner
chooses between an exact full reconstruction and a signed mask difference;
it always applies differences to the last successfully committed result.

Prepared source-bound detector maps accelerate only exact matching masks.
Prepared full-detector moments support GPU DPC priming and rotations. These
products do not replace the full resident source or authorize interpolation.
An empty FFT output disables FFT execution; FFT currently supports only
512-square scans. Resident detector requests also accept 1024-square scans,
subject to admission, without an implicit crop or resize.

## Indexed streaming and ABI

The separate versioned C ABI accepts prepared contiguous uint8/uint16 sources
or original uint16 bitshuffle/LZ4 HDF5 payloads with metadata-only
`QH5IDX01` indexes. It provides exact streamed products, events, cancellation,
descriptor ownership, and GPU selected-frame decoding. It is not a claim of
dense residency.

`scripts/audit_android_vulkan_fixture.py` audits source identity and value
range. `scripts/build_android_qh5_indexes.py` builds byte-window indexes from
that audit. Their generated manifests contain private source paths and must
remain outside public version control. The paired-trial summary helper is
`scripts/summarize_android_vulkan_benchmark.py`.

The native application owns storage permissions, generation scheduling,
touch input, presentation, and lifecycle. It should link the package library,
not fork the shaders.

## Verification boundary

Use the {download}`native build instructions <../../src/quantem/gpu/vulkan/README.md>`
for portable reference tests and Android cross-compilation.
The tests cover packing, mask planning, source/descriptor validation, selected
diffraction, and ABI ownership. Android-only tests exercise actual dispatch
when run on a physical Vulkan device.

A build or host test does not establish phone performance. Acceptance requires
the exact installed artifact, source identity and mask, cold-state definition,
full file-to-first-presentation timing, memory accounting, and measured
detector-center and radius gestures on the physical device. Universal
120 FPS, two-second switching, general uint16 compact-header support, and a
fully unified cross-backend resident API remain open gates.
