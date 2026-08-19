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
- Browser kernels remain beside their scientific domain. WebGPU is a browser
  runtime, not a Python device name and not a synonym for Metal.
- CPU/NumPy is an explicit reference backend for tests. Production scientific
  work must never silently fall back to it.
- UI, view state, cache scheduling, and resource-policy choices remain in the
  consuming application. Reusable math, kernels, resource estimation, and
  scientific provenance remain in `quantem.gpu`.

## Canonical target tree

The current tree is already mostly domain-first. New work should converge on
this shape without a repository-wide rename:

~~~text
src/quantem/gpu/
  device/                         # device detection and explicit selection
  io/
    __init__.py                   # public API only
    models.py                     # backend-neutral results and provenance
    planning.py                   # crop/bin/dtype/resource plans
    backends/{cpu,cuda,mps,webgpu}/
  detector/
    __init__.py                   # public BF/ABF/ADF/DF/mean-DP API
    geometry.py                   # backend-neutral row/column geometry
    backends/{cpu,cuda,mps,webgpu}/
  dpc/
    __init__.py                   # public CoM, rotation, and iDPC API
    models.py
    backends/{cpu,cuda,mps,webgpu}/
  display/
    __init__.py                   # shared display-math contract
    backends/{cpu,cuda,mps,webgpu}/
  ssb/
    __init__.py                   # one SSB workflow and result contract
    models.py
    backends/{cuda,mps,webgpu}/
  remote/                         # transport of exact scientific arrays
  swift/
    Sources/                      # SwiftPM products mirroring the domains
    Tests/
    Benchmarks/

tests/
  contract/                       # public API, provenance, and failure rules
  parity/
    backend_matrix.json           # machine-readable required coverage
    fixtures/                     # small redistributable source evidence
    goldens/                      # frozen outputs plus provenance
    runners/                      # common result-bundle writers/comparators
  hardware/{cuda,mps,webgpu}/     # device-required checks
  e2e/                            # consumer-local override and real-data gates
~~~

`mps` remains the Python backend selector for compatibility. Its accelerated
implementation may use MLX or Metal. Native Swift products use `Metal` in their
names because they expose Metal buffers and command encoding directly.

## Current-tree migration map

The following inconsistencies are migration work, not reasons for an immediate
bulk move:

| Current path | Target | Migration rule |
|---|---|---|
| `detector/compute/{cuda,mps,webgpu}` | `detector/backends/...` | Move only after public detector parity is frozen; leave import shims for one compatibility cycle. |
| `dpc/compute/{cuda,mps,webgpu}` | `dpc/backends/...` | Move with the same row/column and dtype contract; do not duplicate DPC math. |
| `ssb/compute/{cuda,mps,webgpu}` | `ssb/backends/...` | Move last because retained exact-performance evidence names these modules. |
| `display/cuda.py`, `display/reference.py`, `display/webgpu/` | `display/backends/...` | Preserve packaged resource paths until browser-client source-sync tests accept the new manifest. |
| Root-level parity tests and split fixture folders | `tests/parity/...` | Centralize by moving tests only after their node IDs and external harnesses are mapped. |

`io/backends` is the naming reference for new Python backend directories.
Existing `compute` imports remain valid until all consumers are tested against
an exact local package revision. Compatibility shims must contain imports only,
not a second implementation.

## One cross-language scientific contract

Every backend result bundle must record:

- source identity or fixture hash;
- source scan and detector shape and dtype;
- half-open scan and detector regions;
- scan bin, detector bin, output shape, output dtype, and accumulation dtype;
- bad-pixel policy and detector-mask definition;
- `(row, column) ≡ (y, x)` coordinate convention;
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
