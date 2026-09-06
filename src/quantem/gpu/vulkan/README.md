# Android Vulkan backend

This is the UI-free native Android backend for QuantEM.GPU. It contains the
C ABI for indexed/streamed sources and a C++ resident packed-detector session.
The consuming app owns HDF5 metadata discovery, source authentication,
Android storage grants and lifecycle, UI scheduling, and presentation.

## Source map

| Directory | Responsibility |
|---|---|
| [include](include) | Public C ABI and C++ contract/session headers |
| [src](src) | Source lifecycle, validation, dispatch, and resident ownership |
| [shaders](shaders) | Native Vulkan decode, detector, and FFT implementations |
| [tests](tests) | Portable reference contracts and Android-only GPU admission |
| [benchmarks](benchmarks) | Native benchmark entry points |

The adjacent [android entry](../android/README.md) forwards to this CMake target;
it is not another kernel implementation. The Python count-ANS integration does
not add ANS loading to Vulkan. This backend's indexed/dense-staging and packed
contracts remain distinct from that pending feature.

## Exact resident path

`PackedDetectorSession` accepts a complete 512-square or 1024-square scan
through an authenticated, bounded shard loader. It allocates the final
device-readable packed buffers and validates the full source plan before
publishing a session. It does not allocate a dense 4D array.

- Expanded descriptors preserve unsigned integer values with widths 0–16 and
  admit either 32- or 128-scan tiles. The portable owning packer remains 128-scan.
- Compact 32-scan nibble headers currently admit widths 0–8 only. The shared
  v3 compact profile also requires a uint8 working array; a full uint16 compact
  profile needs a separate shared encoding contract. Admission rejects unused
  payload prefixes, noncanonical checkpoints, and nonzero unused width nibbles.
- Raw-LZ4 decoding can write directly into the final packed payload.
- Selected diffraction is decoded on the GPU from resident data.
- BF/DF/ADF movement and radius changes use exact full sums or signed mask
  differences against the last successfully committed detector result.
- Prepared, source-bound detector maps and full-detector DPC moments can be
  supplied during admission. A prepared map is used only for its exact mask.
- Passing an empty FFT destination performs no FFT. FFT requests are currently
  limited to 512-square scans; 1024-square scans must omit FFT.

Coordinates are zero-based `(row, column)`. Packed circular masks use
`distance_squared >= inner_squared && distance_squared < outer_squared`.
Nonzero entries in an explicitly authenticated exclusion mask are excluded.
Losslessness refers to the declared working array and mask policy, not the
compressed HDF5 container bytes. No implicit crop, binning, or narrowing is
performed. Memory admission is source- and device-specific.

The synchronous loader must authenticate every borrowed destination before
returning, join any workers, and never retain borrowed storage. Session
requests serialize; selected diffraction does not change the detector's
committed base. Device/fence failure invalidates the session.

## Original indexed source path

`qgpu_vulkan_open_v1` accepts ordered, identity-bound source descriptors and
metadata-only `QH5IDX01` indexes for original uint16 bitshuffle/LZ4 HDF5
payloads. It also supports prepared contiguous uint8/uint16 segments. Full
products use bounded streaming and uint64 integer accumulation rather than
claiming dense residency.

`qgpu_vulkan_read_selected_diffraction_gpu_v1` decodes one indexed uint16
frame on Vulkan without traversing full products. The corresponding function
without `_gpu` is a host reference, not the Android interaction path.
The streaming mask contract in `contract.hpp` is separate from the packed
circular-mask contract; clients must use the matching interface.

The C ABI duplicates borrowed file descriptors and exposes lifecycle events,
generation cancellation, and immutable result views. Release/close functions
take pointer-to-pointer arguments and clear them idempotently.

## Build and test

Portable contract and reference tests:

```bash
cmake -S src/quantem/gpu/vulkan -B build/android-contract
cmake --build build/android-contract --parallel
ctest --test-dir build/android-contract --output-on-failure
```

Android libraries, shaders, test binaries, and benchmark:

```bash
cmake -S src/quantem/gpu/vulkan -B build/android-vulkan \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/android-vulkan --parallel
```

Set `ANDROID_NDK` to an installed Android NDK before configuring. The build
uses its `glslc` or an explicit `QUANTEM_GPU_GLSLC` path. Link
`quantem_gpu_android_vulkan` with its transitive contract library and Android
platform dependencies.

Host tests and cross-compilation are not physical-device acceptance.
The Android-only `quantem_gpu_android_packed_admission_tests` target adds
expanded 32-scan uint16 and compact-header corruption regression cases; its
synthetic codec fixtures are not full real-source or application qualification.
Run device tests only with the application's physical-device owner. Record
source identity, exact shape/dtype/mask, artifact hashes, memory, real-file
open-to-presentation, and actual detector-center/radius gestures separately.
Neither universal 120 FPS nor a two-second file switch is guaranteed.

See [Android Vulkan documentation](../../../../docs/platforms/android-vulkan.md)
for the package boundary and remaining integration gates.
