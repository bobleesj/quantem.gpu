# SSB performance notes

This page records the current native CUDA SSB live-redraw contract so future
work starts from measured behavior, not from memory.

## User-facing target

Microscopists need to steer aberration controls while viewing the same
full-BF reconstruction used for the final result. For native full-resolution
live display, the current targets are:

- `512x512`: at least 30 FPS (`<=33.3 ms/redraw`).
- `1024x1024`: at least 10 FPS (`<=100 ms/redraw`).

Do not claim these numbers from detector binning, scan cropping, fewer BF
pixels, or saved complex64 caches. Those are separate preview/export choices.

## Exact object redraw path

`SSB.result()` displays the complex object wave and then exposes its phase.
For that object path, the inverse FFT can move outside the BF average:

```text
mean_bf(ifft2(corrected_bf)) == ifft2(mean_bf(corrected_bf))
```

The CUDA object redraw path now uses this identity for large native scans:

1. Compute the same per-BF `pk` correction as the full custom SSB path.
2. Sum corrected Fourier-domain terms in BF groups on the GPU.
3. Reduce those group sums to one corrected Fourier image.
4. Run one `ifft2` for the final object.

This changes the memory/computation topology only. It does not change the BF
selection, scan size, precision type, or object definition.

The path is intentionally separate from the phase-variance optimizer path.
`reconstruct()` and `reconstruct_with_loss()` average per-BF phase and still
need their own optimized kernels because `angle(mean(object))` is not the same
operation as `mean(angle(object))`.

## Current measured baseline

Hardware: RTX PRO 6000 Blackwell-class GPU on `mjgoat`.

Input: synthetic complex64 `G_qk`, fitted 192-pixel detector BF disk radius
`53 px`, `8809` BF pixels, native scan size, no crop, no binning.

Benchmark: `SSBEngine.reconstruct_object(C10, C12, phi12)` after cache warmup.

| Scan | Mean | p50 | p95 | FPS | VRAM pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128x128` | `0.80 ms` | `0.80 ms` | `0.81 ms` | `1247.9` | `2.3 GB` |
| `256x256` | `3.97 ms` | `3.37 ms` | `5.55 ms` | `251.6` | `9.4 GB` |
| `512x512` | `12.29 ms` | `12.30 ms` | `12.53 ms` | `81.4` | `19.1 GB` |
| `1024x1024` | `56.22 ms` | `55.54 ms` | `62.20 ms` | `17.8` | `76.2 GB` |

This meets the current live-object target for both sizes. Treat it as a kernel
microbenchmark, not as complete scientist-workflow signoff. Real-data HDF5
load, hot-pixel filtering, BF-mask setup, browser/widget interaction, and
display readback still need end-to-end checks for every public workflow.

The same run measured the existing phase-mean and phase+loss paths. Those
paths are different scientific quantities and remain the next optimization
target:

| Mode | `128x128` | `256x256` | `512x512` | `1024x1024` |
| --- | ---: | ---: | ---: | ---: |
| Object redraw | `0.80 ms / 1247.9 FPS` | `3.97 ms / 251.6 FPS` | `12.29 ms / 81.4 FPS` | `56.22 ms / 17.8 FPS` |
| Phase redraw | `10.05 ms / 99.5 FPS` | `22.49 ms / 44.5 FPS` | `72.39 ms / 13.8 FPS` | `342.71 ms / 2.9 FPS` |
| Phase+loss | `9.17 ms / 109.1 FPS` | `19.55 ms / 51.2 FPS` | `69.48 ms / 14.4 FPS` | `326.71 ms / 3.1 FPS` |

## 12-cell backend tracking matrix

Track native SSB live-redraw work as a 12-cell backend matrix: three GPU
backends by four native scan sizes. Each cell must record implementation
status, parity status, and the best measured performance before it is treated
as a supported scientist workflow.

| Backend / size | `128x128` | `256x256` | `512x512` | `1024x1024` |
| --- | --- | --- | --- | --- |
| CUDA object redraw | Implemented. Mean `0.80 ms`, p95 `0.81 ms`, `1247.9 FPS`. | Implemented. Mean `3.97 ms`, p95 `5.55 ms`, `251.6 FPS`. | Implemented. Mean `12.29 ms`, p95 `12.53 ms`, `81.4 FPS`. | Implemented. Mean `56.22 ms`, p95 `62.20 ms`, `17.8 FPS`. |
| MPS object redraw | Pending object Fourier-sum port. | Pending object Fourier-sum port. | Pending object Fourier-sum port. | Pending object Fourier-sum port. |
| WebGPU phase/loss path | Implemented in `quantem.widget`; migration pending. Synthetic browser parity passed. | Implemented in `quantem.widget`; migration pending. Synthetic browser parity passed. | Implemented in `quantem.widget`; migration pending. Real Samsung 512 full-BF drive measured mean `31.4 ms` GPU and `41.8 ms` UI for C10 changes at `9070/9070` BF. | Implemented in `quantem.widget`; migration pending. Real Berk 1024 range-index load passes. Live redraw works but remains about `150-200 ms` UI for control nudges, below the 30 FPS target. |

Interpretation:

- CUDA is the only backend with all four object-redraw cells implemented and
  parity-tested against the previous per-BF IFFT object path.
- MPS has useful SSB preview/free-fit infrastructure, but it has not received
  the exact object Fourier-sum topology. Do not claim MPS parity from image
  agreement or reduced optimizer settings.
- WebGPU currently lives in `quantem.widget` because it is bundled for browser
  export. The maintenance target is to move reusable kernel source, shape
  guards, and parity fixtures into `quantem.gpu`, then let `quantem.widget`
  import/build from that source for display.

### WebGPU 1024 status from 2026-07-16

The browser kernel was extended to support `1024x1024` by using a 256-thread
WGSL topology with looped row/column load-store and looped butterflies. This
avoids relying on a 1024-thread workgroup, which common WebGPU limits reject.

Headed Chrome/CDP evidence on NVIDIA Blackwell:

| Case | Data | Result |
| --- | --- | --- |
| Synthetic shape matrix | `128/256/512/1024`, 8 BF pixels | Browser parity passed for phase and FFT log-magnitude at every size. |
| Synthetic stress | `1024x1024`, 64 BF pixels | WGSL compute mean `8.2 ms`; page wall mean about `503 ms` because the standalone parity/demo page repaints and compares too much on the CPU. |
| Real Samsung full BF | `512x512`, `9070/9070` BF | C10 keyboard drive mean `31.4 ms` GPU, mean `41.8 ms` UI, about `23.9 FPS`; screenshot/report under `/tmp/showptycho-webgpu-size-matrix/real_samsung_fullbf_c10_keys/`. |

Real `1024x1024` data target used for browser signoff:

```text
/home/owner/ssd/data/berk_tomo_20260716_one_tilt/pos_38_tilt0.h5
dataset: /entry/data/data
native shape: (1024, 1024, 192, 192) via flattened (1048576, 192, 192)
dtype: uint16 on disk; exact max count 12, so uint8 is lossless for browsing/load
wrapper: /home/owner/data/reports/berk_tomo_20260716_one_tilt_ssb/pos_38_tilt0_master_wrapper.h5
```

Headed Chrome result on mjgoat after adding the range-index HDF5 source path:

| Step | Result |
| --- | --- |
| Initial compressed-source load | Reducer ready in `15.6 s` wall; profile `parse 103 ms`, `decode 5013 ms`, `gather 2286 ms`, `fft 65 ms`, total setup `13.05 s`. |
| Network shape | One small master fetch, one `16 MB` chunk-index fetch, then `206` byte-range reads for the `2.7 GB` compressed HDF5 data file. The previous single `200` full-file fetch failed in Chrome before WebGPU work started. |
| 0.3 BF interaction | `542/1805` selected BF, `379` active aperture BF. C10/C12/phi12/scan-rotation coordinate drives updated live, with UI readouts `148-200 ms` and GPU readouts `134-189 ms`. |
| Near-full BF setup | `1767/1805` selected BF, `1382` active aperture BF. HDF5 setup completed in `14.1 s` wall; profile total `13.11 s`. |
| Near-full BF interaction | C10 drive updated live at about `141 ms` GPU and `152 ms` UI. |

Interpretation for the microscopist: the full native Berk field can now be
opened from the compressed HDF5 source without saving `g_bf.c64`, and the
controls do update the scientific image and FFT at 1024. It is not yet a
30 FPS steering experience. The next WebGPU work is reducing redraw latency
for the 1024 phase/loss path, not further reducing or binning the dataset.

This is the correct real-data target for WebGPU 1024 workflow testing. Do not
substitute a synthetic 1024 page for final signoff.

## Parity evidence

Focused CUDA parity tests live in `tests/test_ssb_cuda_128.py`.

The object Fourier-sum path is compared against the previous per-BF chunked
IFFT object path for `128x128`, `256x256`, `512x512`, and `1024x1024`:

- `p99.9(abs_err) < 5e-9`
- `p99.9(rel_err) < 1e-4`

The `1024x1024` reconstruct-with-loss path is compared to an explicit CuPy
reference with a tolerance that allows rare `atan2` branch-cut pixels while
requiring the scalar objective and 99.9% of phase pixels to match.

Run before changing SSB kernels:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/owner/repos/quantem.gpu/src \
pytest -q /home/owner/repos/quantem.gpu/tests/test_ssb_cuda_128.py
```

## Repeat the native benchmark

Use the local Codex skill benchmark for synthetic kernel timing:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/owner/repos/quantem.gpu/src \
python /home/owner/.codex/skills/quantem-ssb-kernel-optimization/scripts/ssb_native_bench.py \
  --repo /home/owner/repos/quantem.gpu \
  --sizes 128,256,512,1024 \
  --num-bf 8809 \
  --iters 4 \
  --mode object
```

Repeat with `--mode phase` and `--mode loss` for the full 12-run matrix. Do not
compare object-mode FPS to phase-mode optimizer FPS without saying which
scientific quantity is being drawn.

## Next performance work

Problem: the live object redraw target is met in the synthetic native-kernel
benchmark, but real-data workflow signoff is still incomplete.

Action: run the Samsung/BTO HDF5 path end to end, including load/decode,
hot-pixel filtering, BF-mask formation, Nelder-Mead/SSB setup, live controls,
FFT display, and browser/widget reporting.

Problem: the phase-variance optimizer path has not received the same topology
breakthrough.

Action: keep optimizing `reconstruct()` and `reconstruct_with_loss()` with
dedicated 1024 variance kernels or an equivalent exact reformulation; do not
reuse the object-path claim for the optimizer objective.

Problem: `1024x1024` batched optimizer variance is still disabled.

Action: implement and parity-test a dedicated 1024 batch variance kernel before
enabling batch trials at that size.

Problem: MPS and WebGPU are not yet at parity with the CUDA native-size matrix.

Action: run and update the 12-cell backend matrix above. MPS has
`quantem.gpu.ssb.mps` and existing CUDA-reference tests, but the object
Fourier-sum topology has not been ported there. WebGPU SSB currently lives in
`quantem.widget`; 1024 support exists there, but reusable WGSL kernels and
parity fixtures still need to move into `quantem.gpu`.
