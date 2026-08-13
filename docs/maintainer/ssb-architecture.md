# SSB architecture

## Public contract

`quantem.gpu.SSB` is the only scientist-facing SSB compute entry point. CUDA,
MPS, and WebGPU are execution backends, not separate APIs.

```python
from quantem.gpu import SSB

with SSB.open(
    "scan_master.h5",
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=30,
    scan_sampling_A=(0.5, 0.5),
    rotation_angle_deg=-8.2,
) as ssb:
    result = ssb.fit(save_to="results/ssb")
```

The public units are fixed: kV, mrad, angstrom, nanometres for C10/C12,
radians for phi12, degrees only for scan rotation, and `(row, col)` for all
coordinates. Missing required data raises a deterministic error; adapters do
not probe loosely named attributes.

There are exactly two scientist-facing SSB symbols: `SSB` and `SSBResult`.
`SSBResult` owns optimizer metadata and the final complex object wave, and
derives object phase and amplitude from that wave. There is no separate fit or
phase-evaluation result class.

Backend geometry and browser transport use private typed records such as
`BrightfieldDisk` and `SSBExportState`. They are integration details and are
not exported from `quantem.gpu` or `quantem.gpu.ssb`.

The invariant scientific defaults are 200 trials, exact full-BF
phase-variance loss, Nelder-Mead refinement, float32 real values, and complex64
object values. A backend may reject a job it cannot execute, but may not switch
to CPU, crop or subsample the BF evidence, quantize values, or change the
objective silently.

`fit()` and `reconstruct()` own saved-result reuse through the same `save_to`
and `force` controls. Reuse requires an exact scientific signature and restores
one `SSBResult`; there is no cache manager, persistence session, or second
workflow API. The NPZ owns the complex object wave, and its JSON companion owns
readable provenance and result metadata. Serialization occurs only at this
artifact boundary and never changes GPU computation.

## Ownership boundaries

| Layer | Owns | Must not own |
| --- | --- | --- |
| `quantem.gpu.SSB` | validation, public units, backend selection, fit/reconstruct/preview lifecycle, shared results | device kernels, widget rendering, CLI parsing |
| CUDA/MPS backend modules | preparation, device buffers, exact objective, reconstruction kernels | public parameter names, result schemas, CLI behavior |
| WebGPU adapter | browser transport and WebGPU execution of the same serialized scientific plan | silent preview substitution or different units |
| `quantem.widget` | `quantem showptycho` parser, HTML/widget rendering, provenance presentation | backend-specific fit orchestration |
| `quantem.live` | acquisition integration and calls into `quantem.gpu.SSB` | copied SSB optics, optimizers, or result classes |

Python CUDA and MPS run in process. WebGPU runs asynchronously in a browser, so
transport cannot be identical. Its serialized plan, parameter names, defaults,
precision contract, detector evidence, and result metadata must be identical.
Until browser fitting implements and passes the shared 200-trial plus
Nelder-Mead parity suite, a WebGPU fit request must fail explicitly rather than
fall back to a fixed reconstruction.

## Canonical CLI

There is one command and no compatibility alias:

```text
quantem showptycho SOURCE \
  --backend auto|cuda|mps \
  --trials 200 \
  --refinement nelder-mead \
  --out OUTPUT_FOLDER
```

Microscope calibration may come from required source metadata or explicit
flags. If a required field is absent, the CLI names that field and the
corrective flag; it does not guess through `getattr` chains. Full automatically
detected bright field is the default. An intentionally approximate preview must
use a separately labelled preview option and its output cannot be accepted as a
fit result.

The CLI flow is fixed:

1. parse and validate one backend-neutral request;
2. call `SSB.open()`, which automatically chooses the fastest exact storage;
3. detect the complete bright-field disk once, or load its exact persisted
   `(row, col)` selection from a BF-column companion;
4. call `fit()`, which optimizes and reconstructs;
5. save the shared result and provenance;
6. hand the result to ShowPtycho for rendering.

The parser must not import concrete CUDA/MPS engines or call MPS-specific fit
functions. Backend tuning knobs belong in backend benchmarks, not the public
scientist CLI.

## Required parity gates

Every backend release must run the same fixtures at 128 and 256 scan sizes,
plus 512 where hardware permits. Tests compare BF `(row, col)` indices,
float32/complex64 dtypes, aberrations, full-BF loss, object phase, and saved
provenance. WebGPU tests run in a real browser; CUDA and MPS tests run on their
native devices. Performance signoff is recorded only after parity passes.

## Compute package convention

Every backend has the same discoverable scan-size layout:

```text
quantem/gpu/ssb/compute/
├── protocol.py
├── cuda/kernels/{fft128,fft256,fft512,fft1024}.py
├── mps/kernels/{fft128,fft256,fft512,fft1024}.py
└── webgpu/kernels/{fft128,fft256,fft512,fft1024}.ts
```

Shared generators remain in each backend's `kernels/common` module, but one
deterministic registry selects the size implementation. Shape dispatch must not
be scattered through UI or optimizer code. SSB-specific WebGPU code lives under
`ssb/compute/webgpu`; generic browser device utilities live under `device`, and
HDF5/decoder utilities live under `io/backends/webgpu`.
