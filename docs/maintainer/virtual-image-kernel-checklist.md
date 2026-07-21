# Virtual-Image Kernel Checklist

Target: BF/DF/ADF, CoM/DPC, and arbitrary detector ROI dragging should use a
custom GPU kernel on every production backend. Do not use detector binning,
scan cropping, or CPU fallback as evidence for this checklist unless the row
says so.

## Why This Matters

The future large browse target is `1024x1024x192x192 uint8`, which is 36.0 GiB
resident. Old CuPy/Torch gather paths allocate large transient selected-pixel
slabs during every drag. A BF disk at radius 30 reads about 2,828 detector
pixels per scan position; a dense DF mask can touch most of the 36,864 detector
pixels. That is why the standard must be one custom kernel per product over
resident data, plus a dense-mask strategy such as `total - complement`.

## Support Probe

Use the shape-only probe before allocating huge arrays:

```python
from quantem.gpu.compute import virtual_image_kernel_support

support = virtual_image_kernel_support(
    backend="cuda",
    shape=(1024, 1024, 192, 192),
    dtype="uint8",
    bf_radius=30,
)
print(support.custom_kernel, support.resident_gib, support.mask_paths)
```

The CUDA `1024x1024x192x192 uint8` target should report:

| Product | Expected path |
|---|---|
| BF | `cuda_rawkernel_selected` |
| ADF | `cuda_rawkernel_selected` |
| DF | `cuda_rawkernel_total_minus_complement` |
| CoM/DPC | `cuda_rawkernel_com` |

The MPS `1024x1024x192x192 uint8` target should report the same formulation:

| Product | Expected path |
|---|---|
| BF | `mps_metal_selected` |
| ADF | `mps_metal_selected` |
| DF | `mps_metal_total_minus_complement` |
| CoM/DPC | `mps_metal_com` |

The Show4DSTEM WebGPU browser runtime is widget-bundled, but the reusable
source now belongs in `quantem.gpu.webgpu`. The parity/performance standard is
the same:

| Product | Expected path |
|---|---|
| BF | `webgpu_wgsl_masked_sum_buffer` |
| ADF | `webgpu_wgsl_masked_sum_buffer` |
| DF | `webgpu_wgsl_masked_sum_buffer` plus dense-mask cache if needed |
| CoM/DPC | `webgpu_wgsl_masked_dpc_buffer` / `webgpu_wgsl_masked_com_buffer` |
| iDPC | `webgpu_wgsl_masked_idpc_buffer` with paired DPC buffers and dual-real FFT |

## Backend Checklist

| Backend path | Current status | Required tests | Performance gate |
|---|---|---|---|
| Show4DSTEM Python CUDA, resident CuPy | Implemented in `CudaKernelCompute`; widget must preserve the CuPy source for compute while keeping Torch for existing display code. Uses warp-shuffle selected reducers, a custom total-count reducer, fused dense `total - complement`, a fused CoM/DPC reducer, a cached full-detector CoM field, and a small per-viewer detector-index cache. | Exact parity vs old CuPy selected-pixel sum for BF/ADF/DF and old CuPy CoM for DPC; widget smoke must report `CudaKernelCompute`; compare-grid path must use the CUDA backend. | 512x512x192x192 uint16 no-bin BF/ADF/DF/DPC faster than old widget path; 1024x1024x192x192 uint8 shape probe must pass before real-data allocation tests. |
| Public `quantem.gpu.detector` CUDA helpers | Implemented through `cuda_masked_sum`. | Exact parity for `masked_sum`, `virtual_image`, BF/ADF/DF helper outputs. | Same or faster than old CuPy gather path, with lower transient memory. |
| Show4DSTEM MPS chunk-backed data | Implemented for uint8/uint16 through `MetalVirtualImage`; dense DF uses cached `total - complement`; CoM/DPC uses raw Metal `com_u8`/`com_u16`; no Torch-MPS giant tensor for full no-bin browse loads. | Mac runtime parity vs NumPy/reference on BF/ADF/DF and CoM/DPC; widget smoke must report `MetalRawBackend`; no silent CPU fallback. | 512 no-bin interaction should use the Metal path or fast sidecar; 1024 uint8 requires an explicit memory policy before real allocation. |
| Show4DSTEM WebGPU browser | Implemented in canonical `quantem.gpu.webgpu` sources. BF/DF/ADF uses `maskedSumBuffer`; DPC row/col uses WGSL CoM, global mean reduction, one-ULP mean-side correction, and centered component output through `maskedDpcBuffer`; iDPC uses paired DPC buffers plus a dual-real FFT. Readback wrappers remain for widget model compatibility and parity tests. | Source contract test plus headed Chrome test with a real adapter, not SwiftShader; BF/ADF/DF, CoM, DPC, and iDPC parity against NumPy/Python reference. | Drag path should keep VI, DPC, and iDPC GPU-resident where the display pipeline can accept GPU buffers; widget model-byte shims remain for current anywidget compatibility. |
| Multi-tilt/series compare grid | CUDA path tested for seven 512 panels at detector bin 2; full no-bin panels are one-at-a-time unless sharded. | Compare-grid BF/ADF/DF parity and timing for 7 panels; verify per-panel backend path. | Refresh all visible panels without falling back to per-panel Torch gather. |

## Done For This Branch

- CUDA RawKernel selected-pixel reducer for resident CuPy `uint8` and `uint16`.
- CUDA selected reducers use warp shuffle instead of repeated block-wide shared
  reductions.
- CUDA dense DF uses cached integer `total - complement` and fuses complement
  reduction, subtraction, and final float32 output in one kernel.
- CUDA total-count maps use a custom row reducer instead of CuPy's generic
  `sum(axis=1)` path.
- CUDA DPC/CoM uses a fused raw kernel that accumulates total intensity,
  detector-row moment, and detector-column moment in one pass over each
  diffraction pattern. The full-detector CoM field is cached per backend, so
  repeated DPC/iDPC requests do not reread the resident 4D block.
- Shape-only support check covers `1024x1024x192x192 uint8` for CUDA and MPS.
- MPS Metal `uint8` masked-sum, detector-sum, mean-DP, bin-sidecar,
  radial-cache, and CoM kernels are present; `dtype="u8"` full-master loads stay
  chunk-backed instead of materializing a Torch-MPS tensor.
- MPS dense dark-field masks use the cached `total - complement` path, matching
  the CUDA dense-mask strategy.
- WebGPU source ownership is scaffolded in `quantem.gpu.webgpu` with the
  Show4DSTEM engine and ShowPtycho SSB engine copied as canonical source
  package data. Widget build/export syncs these sources before bundling.
  BF/DF/ADF has a GPU-resident buffer path; DPC row/col now uses a WGSL CoM
  reducer, WGSL mean reducer, one-ULP mean-side correction, and WGSL
  centered-component pass. Browser iDPC uses paired DPC buffers and a dual-real
  FFT before the Poisson integration.
- Private full 512x512x192x192 no-bin WebGPU DPC/iDPC browser signoff on a real
  NVIDIA Blackwell adapter:
  - corrected-frame load parity: passed
  - DPC row/col max abs error: `7.63e-6`
  - iDPC mean abs error: `4.70e-6`; max abs error: `3.05e-5` from float32 FFT order
  - DPC row/DPC col/iDPC display medians: `14.9/13.2/13.2 ms`
  - DPC row/col/iDPC recompute medians: `13.7/19.3/22.7 ms`
  - idle RAF: `60 FPS`
  - local-file timing reruns use `--require-local-profile` so the browser URL
    fallback cannot be recorded as a local-file benchmark
- Private full 512x512x192x192 and true crop-256 WebGPU detector-bin local-H5
  signoff on a real NVIDIA Blackwell adapter:
  - `detBin=2/4/8` corrected-frame checksums: exact against the
    zero-bad-before-bin reference
  - full-load low8 page profiles: `1.199/1.212/1.106 s`
  - crop-256 20-repeat medians: `0.774/0.755/0.733 s`
  - crop-256 p95: `0.798/0.813/0.775 s`
  - native non-low8 `uint16` `detBin=2`: exact at `2.651 s`
- Private true 1024x1024x192x192 WebGPU product-first selected-block BF
  signoff on a real NVIDIA Blackwell adapter:
  - BF radius: `30`
  - selected compressed payload: `6.88 GB`
  - output image: `4.19 MB`
  - 4-run median wall/profile/product: `4.92/4.85/1.56 s`
  - max/mean absolute error: `0/0` against an independent Python reference
  - this is product-first BF evidence, not full-stack no-bin browse/load
- Private real-data WebGPU browser stress on a 128x128 scan, 96x96 detector,
  uint8 sidecar, real NVIDIA Vulkan adapter:
  - mount/decode to interactive: `1.94 s`
  - BF warm recompute: `2.7-3.7 ms`
  - DPC row warm recompute floor: `6.4-8.7 ms` with browser scheduling outliers
  - DPC col warm recompute floor: `8.3-9.7 ms` with browser scheduling outliers
  - idle RAF: `60.0 FPS`
- Exact focused parity tests for CUDA selected and dense-mask paths.
- Private full 512x512x192x192 real-data benchmark:
  - BF median `4.96 ms -> 1.35 ms`
  - ADF median `16.16 ms -> 3.86 ms`
  - DF median `62.64 ms -> 1.84 ms`
  - max absolute error `0` for every row.
- Private seven-tilt detector-bin2 real-data benchmark:
  - BF median `1.50 ms -> 0.54 ms`
  - ADF median `3.83 ms -> 1.35 ms`
  - DF median `15.88 ms -> 0.53 ms`
  - max absolute error `0` for every row.
- Private seven-tilt Show4DSTEM compare-grid method benchmark after widget
  backend reuse:
  - BF full grid `3.97 ms` (`0.57 ms/panel`)
  - ADF full grid `9.56 ms` (`1.37 ms/panel`)
  - DF full grid `4.00 ms` (`0.57 ms/panel`)
  - max absolute error `0` after the widget's detector-area normalization.
- Private DPC/CoM real-data benchmark:
  - full 512x512x192x192 uint16: `200.42 ms -> 12.39 ms`, max absolute error `0`
  - seven detector-bin2 panels: old summed median `373.14 ms`; uncached CUDA
    summed median `24.06 ms`; first backend-filled grid `24.63 ms`; cached
    repeat path returns the stored CoM arrays without launching a GPU kernel.

## Failed Hypotheses

- Replacing the full-detector CoM loop's per-pixel detector-coordinate division
  with incremental row/column bookkeeping was exact but slower: the seven-panel
  DPC grid regressed from about `27.3 ms` to about `34.5 ms`. Keep the simpler
  division form unless a profiler points elsewhere.

## Next Checklist

- Run Mac MPS `uint8` runtime parity/performance on full no-bin data.
- Run Mac MPS CoM/DPC runtime parity/performance on full no-bin data and record
  the first-click vs cached-repeat timing.
- Add a headed WebGPU test that records the real adapter and verifies BF/ADF/DF
  parity plus `maskedSumBuffer` drag behavior.
- Keep WebGPU CoM/DPC/iDPC headed parity/performance tests in the release gate,
  including the FFT command-batching path that keeps iDPC median redraw under
  the 30 FPS budget.
- Remove any remaining widget-local permanent backend copies after each synced
  `quantem.gpu.webgpu` source is covered by build and browser parity tests.
- Run a real WebGPU `1024x1024x192x192` full-stack no-bin browse/load
  allocation/performance test when enough free browser GPU memory is available,
  or document the memory-policy rejection. Product-first BF for true 1024 is
  already signed off.
