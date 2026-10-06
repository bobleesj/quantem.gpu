# Backend layout and parity contract

`quantem.gpu` is organized by scientific domain first and accelerator second.
This is deliberate: IO, detector reductions, DPC, display, and SSB each expose
one scientific contract, while CUDA, Python MPS, native Swift/Metal, and WebGPU
provide implementations of that contract. A backend must not invent a second
public workflow or result type.

## Ownership rules

- Public Python callers enter through `quantem.gpu.io`, `quantem.gpu.detector`,
  `quantem.gpu.dpc`, `quantem.gpu.display`, and `quantem.gpu.ssb`.
- Backend modules implement those contracts and are not consumer APIs.
- Native clients consume the repository-root Swift package. Swift and
  Swift-only Metal sources live in `native/swift`, outside the Python wheel;
  Metal sources and data Python also reads live once in the Python package,
  and `Package.swift` copies them into the resource bundles.
- Browser consumers enter through `src/quantem/gpu/webgpu/index.ts`; its
  exports point to the existing domain-owned implementations.
- Native Vulkan builds enter through `native/vulkan/CMakeLists.txt`, outside
  the Python package; C ABI and library/target names are unchanged.
- Browser kernels remain beside their scientific domain. WebGPU is a browser
  runtime, not a Python device name and not a synonym for Metal.
- CPU/NumPy is an explicit reference backend for tests. Production scientific
  work must never silently fall back to it.
- UI, view state, cache scheduling, and resource-policy choices remain in the
  consuming application. Reusable math, kernels, resource estimation, and
  scientific provenance remain in `quantem.gpu`.

## IO representation layout

The migration separates representation-specific implementation names
without changing the public Python verbs, load defaults, metadata, or kernels.
The Python files below were moved, not copied. Browser exports refer to the
existing source modules, so their file registry and GPU lifetimes remain shared.

| Backend | HDF5 decode implementation |
|---|---|
| Python CUDA | `io/hdf5/cuda/decode.py` kernels, consumed by `io/encoded.py`, `io/paired.py`, and `io/precision.py` |
| Python MPS | `io/hdf5/mps/decode.py`, consumed by `io/encoded.py` and `io/precision.py` |
| Explicit CPU reference | `io/hdf5/cpu.py` (dense) |
| WebGPU | `io/hdf5/webgpu/dense.ts` exports `local-h5.ts` (dense) |

Python callers continue to use `io.load(..., representation=...)`.
The [representation contract](../api/representations.md) remains authoritative
for supported formats and operations. Representation names do not imply that
every backend accepts every source or offers every operation.

The WebGPU entry provides `loadLocalH5Master`, `loadLocalH5MaskedSum`,
`setLocalFiles`, `clearLocalFiles`, `DetectorCompute`, and
`GPUColormapEngine`. Register the selected HDF5 master/chunk files with
`setLocalFiles` before using the dense loader. `clearLocalFiles` releases
registry references, not GPU buffers.

Python backend imports use the canonical modules above. The old
`cpu.reference` and `mps.decoder` compatibility modules are removed, and the
lossless-pack readers (`cuda.packed`, `mps.packed`) were deleted with their
container format. Mutable backend state has one owner; tests
that inject allocation, cache, or device state target that canonical module.
Browser build scripts must export the complete dependency graph.

The loader separates result/ownership types (`io/dataset.py`), bounded
region reads (`io/read.py`), CPU scan-order unflattening (`io/selection.py`),
header readers (`formats/hdf5/master.py`) and pinned staging ownership
(`device/cuda_runtime.py`).

`load.py` keeps the public route choice. Bounded HDF5-to-ANS streaming lives
in `io/encoded.py`, the CUDA paired-count layout in `io/paired.py`, scaled
precision storage in `io/precision.py`, and the dense CPU reference in
`io/hdf5/cpu.py`; GPU dense loading is not a route. Further decomposition must
preserve source-range, cancellation, and failure behavior.

## Current source tree

Only implemented directories are shown. A directory does not imply every
operation is supported by every backend:

~~~text
src/quantem/gpu/                  # layers import downward only:
                                  # device < formats < resident < io < detector, geometry
                                  # < dpc, parallax < screening < ssb < remote < cli
  device/                         # device detection, CUDA and Metal runtime plumbing
  formats/                        # files on disk, GPU-free: hdf5/, qem/, emd, empad, ...
  resident/                       # acquisitions held on the GPU and their exact reductions
    cuda/, mps/                   # backend residents; kernels/ beside the Python that reads them
  io/
    __init__.py                   # public API only
    load.py                       # public load and route choice
    dataset.py                    # loaded data and ownership contract
    selection.py                  # scan-order unflattening for the CPU reference
    representation.py             # representation is independent of dtype
    hdf5/{cpu.py,cuda,mps,webgpu} # bitshuffle+LZ4 codecs and browser sources
  detector/
    __init__.py                   # public BF/ABF/ADF/DF/mean-DP API
    cuda/, mps/, webgpu/
  dpc/
    __init__.py                   # public CoM, rotation, and iDPC API
    results.py
    webgpu/
  display/
    __init__.py                   # shared display-math contract
    colormaps.json                # colormap control points for every backend
    metal/display.metal           # native Metal display shader
    cpu.py, webgpu/
  ssb/
    __init__.py                   # one SSB workflow and result contract
    contract.py, cuda/, mps/, webgpu/
  geometry/{cuda.py,webgpu/}
  parallax/cuda/
  optics/cuda.py
  movie/{export,cuda,mps}.py
  remote/                         # transport of exact scientific arrays
  webgpu/                         # TypeScript exports and the sources.json build manifest

native/                           # not part of the Python wheel
  swift/
    Sources/                      # SwiftPM products mirroring the domains
    Tests/
    Benchmarks/
    Vendor/                       # CHDF5.xcframework
  vulkan/{include,src,shaders,tests,benchmarks}/

tests/
  contracts/                      # public API, provenance, and failure rules
  parity/
    backend_matrix.json           # machine-readable required coverage
    fixtures/                     # retained frozen scientific evidence
    webgpu/                       # executable TypeScript checks
  hardware/{cuda,mps,metal}/       # device-dependent gates
  e2e/                            # consumer-local override and real-data gates
  infrastructure/                 # docs, packaging, benchmark checks

benchmarks/
  benchmark_registry.json         # accepted immutable measurements
  profile_matrix.json             # measured/pending/unsupported run matrix
~~~

`mps` remains the Python backend selector for compatibility. Its accelerated
implementation may use MLX or Metal. Native Swift products use `Metal` in their
names because they expose Metal buffers and command encoding directly.

## Canonical owners

Implementations have one canonical owner. Backend code lives in a `cuda/`,
`mps/` or `webgpu/` directory (or one `cuda.py`, `mps.py` or `cpu.py` module)
directly under the package that owns the science; there is no `backends/` or
`compute/` level, and no compatibility shims.

Do not create a second support or benchmark registry for this migration.
Extend the existing matrices and keep historical result paths and trial IDs
intact. Moving source code does not make pending hardware cells pass.

## Browser build integration

Do not maintain a consumer-side list of individual kernel files. The package
ships `webgpu/sources.json`, the complete package-owned dependency graph of
`webgpu/index.ts` as paths relative to `quantem.gpu`. Build scripts read that
manifest (for example through `importlib.resources.files("quantem.gpu")`),
copy the listed files into a generated directory, and point the TypeScript
bundler at `webgpu/index.ts`. Qualify the complete consumer bundle against an
exact package revision before adoption. Manifest completeness alone is not
application compatibility.

## Reproduce the layout checks

Run from the repository root. These checks validate imports, numerical
regressions, package resources, native contracts, and documentation. They do
not replace real-device timing or app acceptance.

```bash
# Python: imports, lazy accelerators, dense and encoded behavior.
python scripts/run_tests.py --list
python scripts/run_tests.py contracts parity infrastructure -q
python scripts/run_tests.py hardware/cuda -q
python scripts/run_tests.py hardware/mps -q
python scripts/run_tests.py e2e -q

# Suite names map to test directories; pytest paths and node IDs pass through.
python scripts/run_tests.py tests/contracts/io/test_load.py -q

# Browser: install the development tools, then bundle and execute import checks.
npm install --no-save --package-lock=false esbuild jsfive typescript @webgpu/types
python scripts/run_tests.py tests/contracts/test_webgpu_entrypoint.py -q
npx --no-install tsc --noEmit --skipLibCheck --target ES2022 \
  --module ESNext --moduleResolution bundler --types @webgpu/types \
  src/quantem/gpu/webgpu/index.ts

# Native Vulkan: portable host contracts only; no GPU performance claim.
cmake -S native/vulkan -B /tmp/qgpu-vulkan-host
cmake --build /tmp/qgpu-vulkan-host
ctest --test-dir /tmp/qgpu-vulkan-host --output-on-failure

# Native Apple: unchanged SwiftPM entry.
swift test -j 4

python scripts/check_profile_registry.py
jupyter-book build docs --warningiserror --nitpick
python scripts/check_docs_links.py --html-root docs/_build/html
```

When tooling lives outside the checkout, set `NODE_PATH` to its `node_modules`
directory for the bundle test. A missing Node/jsfive dependency is an explicit
skip, not a passing WebGPU contract. Type-checking still needs the appropriate
TypeScript module/type search paths for that external directory.

## One cross-language scientific contract

Every backend result bundle must record:

- source identity or fixture hash;
- source scan and detector shape and dtype;
- half-open scan and detector regions;
- scan bin, detector bin, output shape, output dtype, and accumulation dtype;
- bad-pixel policy and detector-mask definition;
- `(row, column) ≡ (r, c)` coordinate convention;
- backend, device, source revision, and kernel revision;
- whether the result is native resolution, explicitly binned, or explicitly
  cropped; and
- output array hashes plus the parity metric used.

Real-space crop is never an implicit memory or speed policy. Detector binning
must be explicit and count-preserving, including incomplete edge bins. A
binned array must never be described as native-resolution evidence.

## Parity layers

Parity is cumulative. A source-presence or compile test does not replace a
hardware or real-data gate.

1. **Contract:** imports, signatures, shapes, dtypes, coordinate order,
   provenance, and honest unsupported/failure behavior.
2. **Synthetic numerical:** small odd and rectangular arrays; partial edge
   bins; masks; nonfinite display inputs; deterministic seeds.
3. **Frozen cross-backend:** every backend reads the same input fixture and
   writes the same versioned result bundle. Integer decode, bin, masks, sums,
   and RGBA/histogram outputs are byte-exact.
4. **Real-data:** full source shape and dtype, no unreported crop/bin/precision
   change, exact source hashes, peak memory, and output hashes.
5. **Physical end-to-end:** the real consumer uses a local package override or
   exact revision pin; cold first-source, warm source, and saved-result reopen
   are reported separately.

Floating-point operations use a frozen, operation-specific metric. CoM, DPC,
FFT, and SSB must report maximum and high-percentile error as applicable; a
tolerance may not be widened to make a new backend pass. Goldens are generated
only by an explicit recapture command and never by the backend being adjudicated.

## Migration gate

Move one domain at a time in this order: contract/fixtures, IO, detector/DPC,
display, then SSB. For each move:

1. freeze the old output bundles;
2. move the module and update every caller in the same coordinated change:
   this repository, quantem.widget, quantem.live, and denoise. No import-only
   compatibility shim or alias keeps the old path;
3. run CPU-reference, CUDA, MPS/Metal, Swift, and WebGPU gates listed in
   `tests/parity/backend_matrix.json`;
4. test supported native and browser clients through a local package override;
   and
5. commit the move and pin consumers to that exact revision.

No folder cleanup is complete merely because unit tests pass on one host.
