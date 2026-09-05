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
- Native clients consume the repository-root Swift package. Swift and Metal
  sources remain in `src/quantem/gpu/swift`; they are not copied into an app.
- Browser consumers enter through `src/quantem/gpu/webgpu/index.ts`; its
  exports point to the existing domain-owned implementations.
- Native Vulkan builds enter through `src/quantem/gpu/vulkan/CMakeLists.txt`.
  Native sources now live there. The `android` CMake entry and headers forward
  to the canonical tree; C ABI and existing library/target names stay unchanged.
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

| Backend | Dense implementation | Lossless-packed implementation | Retained compatibility path |
|---|---|---|---|
| Python CUDA | `io/load.py` orchestration and `io/backends/cuda/decoder.py` kernels | `io/backends/cuda/packed.py` | `cuda/compact_h5.py` |
| Python MPS | `io/backends/mps/dense.py` | `io/backends/mps/packed.py` | `mps/decoder.py`, `mps/compact_v3.py` |
| Explicit CPU reference | `io/backends/cpu/dense.py` | Test-only reference decoder, not public packed loading | `cpu/reference.py` |
| WebGPU | `io/backends/webgpu/dense.ts` exports `local-h5.ts` | `io/backends/webgpu/packed.ts` exports `compact-h5.ts` | Existing TypeScript imports stay valid |

Python callers continue to use `io.load(..., representation=...)`.
The [representation contract](../api/representations.md) remains authoritative
for supported formats and operations. Representation names do not imply that
every backend accepts every source or offers every operation.

The WebGPU entry provides `loadLocalH5Master`, `loadLocalH5MaskedSum`,
`setLocalFiles`, `clearLocalFiles`, `loadCompactH5WebGPU`, resident/qualification
types, `DetectorCompute`, and `GPUColormapEngine`. Register the selected HDF5
master/chunk files with `setLocalFiles` before using the dense loader.
`clearLocalFiles` releases registry references, not GPU buffers. The packed
loader accepts its own source directly. This migration does not unify those
different lifecycles or add automatic transcoding.

Compatibility files contain imports only. Callable/class imports retain object
identity; mutable backend state has one owner in the canonical module. Tests
that inject private allocation, cache, or device state must target that module,
not assign globals on a compatibility file. Public Python call sites remain
unchanged; browser build scripts must export the complete dependency graph.

The loader separates result/ownership types (`models.py`), exact scan indexing
(`_selection.py`), header readers (`_metadata.py`), pinned staging ownership
(`_memory.py`), and packed dispatch/provenance (`_packed.py`). Their existing
callable imports remain available through `io.load`.

Dense streaming orchestration still lives in `load.py`. It has not been
renamed and described as a small new CUDA engine. Further decomposition must
preserve source-range, cancellation, and failure behavior.

## Current source tree

Only implemented directories are shown. A directory does not imply every
operation is supported by every backend:

~~~text
src/quantem/gpu/
  device/                         # device detection and explicit selection
  io/
    __init__.py                   # public API only
    load.py                       # public load and dense orchestration
    models.py                     # loaded data and ownership contract
    _selection.py                 # exact scan indexing and ordering
    _metadata.py                  # acquisition/header readers
    _memory.py                    # pinned host-buffer ownership
    _packed.py                    # packed dispatch and provenance
    representation.py             # representation is independent of dtype
    resident_contract.py          # source/working geometry and provenance
    backends/{cpu,cuda,mps,webgpu}/
      dense.py                    # only where a distinct implementation exists
      packed.py                   # TypeScript uses corresponding .ts entries
  detector/
    __init__.py                   # public BF/ABF/ADF/DF/mean-DP API
    geometry.ts                   # shared browser row/column geometry
    backends/{cuda,mps,webgpu}/
  dpc/
    __init__.py                   # public CoM, rotation, and iDPC API
    results.py
    backends/{cuda,mps,webgpu}/
  display/
    __init__.py                   # shared display-math contract
    backends/{cpu.py,cuda.py,webgpu/,direct3d/}
  ssb/
    __init__.py                   # one SSB workflow and result contract
    backends/{cuda,mps,webgpu}/
  geometry/backends/webgpu/
  parallax/backends/cuda/
  remote/                         # transport of exact scientific arrays
  webgpu/                         # TypeScript exports and build resource API
  swift/
    Sources/                      # SwiftPM products mirroring the domains
    Tests/
    Benchmarks/
  vulkan/{include,src,shaders,tests,benchmarks}/
  android/                        # compatibility CMake entry and headers

tests/
  contracts/                      # public API, provenance, and failure rules
  parity/
    backend_matrix.json           # machine-readable required coverage
    fixtures/                     # retained frozen scientific evidence
    webgpu/                       # executable TypeScript checks
  hardware/{cuda,mps}/             # device-dependent gates
  e2e/                            # consumer-local override and real-data gates
  infrastructure/                 # docs, packaging, benchmark checks
  direct3d/                       # native .NET test project
  path_migrations.json            # exact old-to-new test paths

benchmarks/
  benchmark_registry.json         # accepted immutable measurements
  profile_matrix.json             # measured/pending/unsupported run matrix
~~~

`mps` remains the Python backend selector for compatibility. Its accelerated
implementation may use MLX or Metal. Native Swift products use `Metal` in their
names because they expose Metal buffers and command encoding directly.

## Compatibility map

Implementations have one canonical owner. These retained boundaries keep
existing imports and native build entries available:

| Current path | Target | Migration rule |
|---|---|---|
| `io/load.py` helper imports | Models and private responsibility modules | Original callable imports; private state belongs to the canonical module. |
| `{detector,dpc,ssb,geometry,parallax}/compute/` | Corresponding `backends/` | Import-only Python and TypeScript compatibility files. |
| `detector/compute/backends.py` | `detector/backends/dispatch.py` | Original callable imports. |
| `display/cuda.py`, `display/reference.py`, `display/webgpu/` | `display/backends/...` | Original callable imports and TypeScript exports. |
| `display/direct3d/` | `display/backends/direct3d/` | Original project compiles the same canonical C# source. |
| Earlier Python test paths | Organized test directories | `scripts/run_tests.py` translates file paths and pytest node IDs. |
| `android/{include,src,shaders,...}` | `vulkan/{include,src,shaders,...}` | CMake and header forwarding; unchanged library names. |

`io/backends` is the naming reference for new Python backend directories.
Existing `compute` imports remain valid until all consumers are tested against
an exact local package revision. Compatibility shims must contain imports only,
not a second implementation.

Do not create a second support or benchmark registry for this migration.
Extend the existing matrices and keep historical result paths and trial IDs
intact. Moving source code does not make pending hardware cells pass.

## Browser build integration

Do not maintain a consumer-side list of individual kernel files. Export the
complete package-owned dependency graph into a fresh generated directory:

~~~python
from quantem.gpu import webgpu

# Step 1: export sources, compatibility imports, and required JSON resources.
generated = webgpu.export_sources("build/generated/quantem-gpu")

# Step 2: point your TypeScript bundler at generated / "webgpu/index.ts".
# Existing domain imports are also present in the exported tree.
~~~

`source_names()` and `source_text(name)` expose the same graph to build tools
that own file writing. Export refuses a nonempty directory; it never erases a
consumer source tree. These are build-time APIs, not a Python GPU executor.
Qualify the complete consumer bundle against an exact package revision before
adoption. Export completeness alone is not application compatibility.

## Reproduce the layout checks

Run from the repository root. These checks validate imports, numerical
regressions, package resources, native contracts, and documentation. They do
not replace real-device timing or app acceptance.

```bash
# Python: new/old import identity, lazy accelerators, dense and packed behavior.
python scripts/run_tests.py --list
python scripts/run_tests.py contracts parity infrastructure -q
python scripts/run_tests.py hardware/cuda -q
python scripts/run_tests.py hardware/mps -q
python scripts/run_tests.py e2e -q

# Earlier paths and pytest node IDs are translated without changing gates.
python scripts/run_tests.py tests/contracts/io/test_load.py -q

# Browser: install the development tools, then bundle and execute import checks.
npm install --no-save --package-lock=false esbuild jsfive typescript @webgpu/types
python scripts/run_tests.py tests/contracts/test_webgpu_entrypoint.py -q
npx --no-install tsc --noEmit --skipLibCheck --target ES2022 \
  --module ESNext --moduleResolution bundler --types @webgpu/types \
  src/quantem/gpu/webgpu/index.ts

# Native Vulkan: portable host contracts only; no GPU performance claim.
cmake -S src/quantem/gpu/vulkan -B /tmp/qgpu-vulkan-host
cmake --build /tmp/qgpu-vulkan-host
ctest --test-dir /tmp/qgpu-vulkan-host --output-on-failure

# Native Apple: unchanged SwiftPM entry and explicit Python parity opt-in.
QGPU_RUN_PYTHON_PARITY=1 QGPU_PYTHON="$(command -v python)" swift test -j 4

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

1. freeze the old import paths and output bundles;
2. add the new internal path plus import-only compatibility shims;
3. run CPU-reference, CUDA, MPS/Metal, Swift, and WebGPU gates listed in
   `tests/parity/backend_matrix.json`;
4. test supported native and browser clients through a local package override;
5. commit the move independently and pin consumers to that exact revision; and
6. delete shims only in a later reviewed change after all consumers migrate.

No folder cleanup is complete merely because unit tests pass on one host.
