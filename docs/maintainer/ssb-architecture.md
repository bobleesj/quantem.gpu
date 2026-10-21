# SSB architecture

## Public contract

`quantem.gpu.SSB` is the scientist-facing SSB compute entry point. CUDA and
MPS are its execution backends, not separate APIs. The browser engine in
`ssb/webgpu` is a separate runtime that applications bundle; it is not an
`SSB(backend=...)` option.

```python
from quantem.gpu import SSB

with SSB.open(
    "scan_master.h5",
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=30,
    scan_sampling_A=(0.5, 0.5),
    scan_detector_rotation_deg=351.8,
) as ssb:
    result = ssb.find_aberrations(save_dir="results/ssb")
```

The public units are fixed: kV, mrad, angstrom, nanometres for `C10_nm` and
`C12_nm`, degrees for `phi12_deg` and the scan-detector rotation, and
`(row, col)` for all coordinates. Missing required data raises a deterministic error; adapters do
not probe loosely named attributes.

`SSBResult` owns optimizer metadata and the final complex object wave, and
derives object phase and amplitude from that wave. There is no separate fit or
phase-evaluation result class.

Backend geometry and browser transport use private typed records such as
`BrightfieldDisk` and `SSBExportState`. They are integration details and are
not exported from `quantem.gpu` or `quantem.gpu.ssb`.

The scientific defaults are 200 trials, the least-squares agreement of the SSB
model with the full-BF evidence (the joint tilt fit by default; the MPS
standard fit, `tilt=False`, holds the sample tilt and depth spread at 0), Newton
refinement, float32 real values, and complex64 object values. CUDA's standard
fit still minimizes the full-BF phase variance with Nelder-Mead. The phase
variance is the reported diagnostic `result.loss` everywhere
(`2026-10-20-ssb-standard-fit-objective.md`). A
backend may reject a job it cannot execute, but may not switch to CPU, crop or
subsample the BF evidence, quantize values, or change the objective silently.

`find_aberrations()` and `reconstruct()` own saved-result reuse through the same `save_dir`
and `force` controls. Reuse requires an exact scientific signature and restores
one `SSBResult`; there is no cache manager, persistence session, or second
workflow API. The NPZ owns the complex object wave, and its JSON companion owns
readable provenance and result metadata. Serialization occurs only at this
artifact boundary and never changes GPU computation.

## Backend differences

The same call does not return the same phase image on CUDA and MPS:

- the default phase estimator is `phase_of_mean` on CUDA and `mean_phase` on
  MPS;
- `reconstruct(..., upsampling_factor=...)` above 1 is CUDA only;
- the bright-field disk is selected by the same rule with different rounding
  (CUDA device float32, MPS host float32), so edge pixels can differ; and
- the standard fit's objective differs: agreement at depth 0 on MPS, phase
  variance on CUDA; and the Nelder-Mead tolerances and the loss reduction order
  differ.

Each result records its backend, estimator and output factor
(`result.phase_estimator`, `result.upsampling_factor`).

## Ownership boundaries

| Layer | Owns | Must not own |
| --- | --- | --- |
| `quantem.gpu.SSB` | validation, public units, backend selection, fit/reconstruct/preview lifecycle, shared results | device kernels, viewer rendering |
| CUDA/MPS backend modules | preparation, device buffers, exact objective, reconstruction kernels | public parameter names, result schemas |
| `ssb/webgpu` browser engine | WebGPU execution of an exported bright-field plan with the same units and parameter names | aberration search, silent preview substitution |
| Applications and viewers | rendering, export folders, provenance presentation, acquisition integration | copied SSB optics, optimizers, or result classes |

Python CUDA and MPS run in process. The browser engine runs asynchronously and
reconstructs from exported bright-field columns and calibration; it does not
fit aberrations. Fitting stays on CUDA or MPS until a browser fit passes the
shared 200-trial plus Nelder-Mead parity suite.

## Required parity gates

Every backend release runs the same fixtures at 128 and 256 scan sizes, plus
512 where hardware permits. Tests compare BF `(row, col)` indices,
float32/complex64 dtypes, aberrations, full-BF loss, object phase, and saved
provenance. WebGPU tests run in a real browser on a hardware adapter; CUDA and
MPS tests run on their native devices. Performance signoff is recorded only
after parity passes.

## Kernel selection by scan size

Each backend selects its kernels by square scan size (128, 256, 512 and 1024)
in one place: `ssb/cuda/kernels`, `ssb/mps/kernels` and
`ssb/webgpu/backend.ts`. Shape dispatch is not scattered through optimizer or
viewer code. SSB-specific WebGPU code lives under `ssb/webgpu`; generic browser
device utilities live under `device`, and HDF5 decoder utilities live under
`io/hdf5/webgpu`.
