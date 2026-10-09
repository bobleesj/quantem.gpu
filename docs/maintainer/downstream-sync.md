# Downstream sync after the restructure (2026-10-05)

Which revision of each repository goes with the restructured quantem.gpu, and how other code
moves to the new import paths. Each synced commit is on its repository's `main`.

## The synced set

| Repository | Synced commit | How it pins quantem.gpu |
|---|---|---|
| quantem.gpu | f9185fb4 (the public history starts here) | release as tag `gpu-v0.0.1rc13` (the release workflow stamps the version from the tag) |
| quantem.widget | `main` after the rc13 pin | `quantem.gpu[movie]>=0.0.1rc13` |
| quantem.live | `main` after the rc13 pin | `quantem.gpu>=0.0.1rc13` (pyproject and both environment files) |
| Live4DSTEM | b6f2c30 | `Package.swift` revision f9185fb4 (`Package.resolved` originHash updated) |
| Live4DSTEM-linux | a792926 | `Package.swift` revision and `linux/pyproject.toml` git pin f9185fb4 |
| quantem.thick | 7044371 | imports `quantem.gpu.io.convert` (unpinned) |
| denoise | `main` after the rc13 pin | `fourdstem = ["quantem.gpu>=0.0.1rc13", ...]` |

The Python pins resolve to the `gpu-v0.0.1rc13` release on TestPyPI, which declares torch;
the Swift and Linux pins name f9185fb4. An editable environment must move quantem.gpu,
quantem.widget, quantem.live, quantem.thick and denoise together: the callers import modules
that only the restructured quantem.gpu has.

## Behaviour changes that reach callers

- `io.load` on CUDA and MPS returns ANS-encoded acquisitions; bounded views come from
  `loaded.read(scan_region=..., detector_region=...)`, products from `detector.prepare(loaded)`.
- SSB automatic detector sampling is `semiangle / disk-edge radius` (it was twice that); fits that
  relied on the automatic value change. See `maintainer/2026-10-05-ssb-detector-sampling.md`.
- Mean diffraction patterns are the exact total divided once in float64; frame sums and maxima of
  integer counts are exact uint64 on every backend.
- The retired `.ans` snapshot format is gone everywhere; `.qem` is the saved encoded format.
- The WebGPU `.qem` reader is `RansResidentSet.loadQemFile` / `loadQemFiles`.

## Other repositories that import quantem.gpu

myptycho, quantem.live-localtilt, amorphoscopy-diffraction and
2026-tilt-aberration-ssb import modules that moved or were deleted. The table below maps every
module of quantem.gpu at 3a8abad6 whose names moved, to where its public names live now
(generated from the two source trees; "deleted" means none of its public names remain).

| Old module (3a8abad6) | Now | Note |
|---|---|---|
| `quantem.gpu._compact` | `quantem.gpu.resident.cuda.paired` | 3 of 4 names deleted |
| `quantem.gpu._compact.groups` | deleted | 2 of 2 names deleted |
| `quantem.gpu._compact.interaction` | `quantem.gpu.detector.cuda.streamed_series` |  |
| `quantem.gpu._compact.layout` | deleted | 7 of 7 names deleted |
| `quantem.gpu._compact.load` | `quantem.gpu.io.load` | 2 of 3 names deleted |
| `quantem.gpu._compact.mask_plan` | `quantem.gpu.detector.cuda.mask_plan` |  |
| `quantem.gpu._compact.metadata` | deleted | 1 of 1 names deleted |
| `quantem.gpu._compact.paired` | `quantem.gpu.resident.cuda.paired`, `quantem.gpu.detector.cuda.paired_series`, `quantem.gpu.detector.cuda.dense` | 1 of 16 names deleted |
| `quantem.gpu._compact.planner` | deleted | 1 of 1 names deleted |
| `quantem.gpu._compact.source` | deleted | 2 of 2 names deleted |
| `quantem.gpu._compact.streamed` | `quantem.gpu.resident.cuda.counts`, `quantem.gpu.formats.qem.snapshot`, `quantem.gpu.detector.cuda.dense` |  |
| `quantem.gpu._cuda_libraries` | deleted | 1 of 1 names deleted |
| `quantem.gpu.detector.backends` | deleted |  |
| `quantem.gpu.detector.backends.bounded` | `quantem.gpu.detector.bounded` |  |
| `quantem.gpu.detector.backends.counts` | `quantem.gpu.detector.counts` | 1 of 2 names deleted |
| `quantem.gpu.detector.backends.cuda` | deleted |  |
| `quantem.gpu.detector.backends.cuda.kernels` | deleted | 19 of 19 names deleted |
| `quantem.gpu.detector.backends.cuda.probe` | `quantem.gpu.detector.cuda.probe` |  |
| `quantem.gpu.detector.backends.cuda.series` | `quantem.gpu.detector.cuda.series` |  |
| `quantem.gpu.detector.backends.dispatch` | `quantem.gpu.detector.cuda.dense`, `quantem.gpu.detector.mps.dense`, `quantem.gpu.detector.tensors` | 2 of 5 names deleted |
| `quantem.gpu.detector.backends.float_ans` | `quantem.gpu.detector.float_ans` |  |
| `quantem.gpu.detector.backends.mps` | deleted |  |
| `quantem.gpu.detector.backends.mps.ans_series` | deleted | 1 of 1 names deleted |
| `quantem.gpu.detector.backends.mps.kernels` | `quantem.gpu.resident.mps.frames`, `quantem.gpu.resident.mps.virtual_image` | 1 of 4 names deleted |
| `quantem.gpu.detector.backends.packed` | deleted | 2 of 2 names deleted |
| `quantem.gpu.detector.backends.protocol` | deleted | 3 of 3 names deleted |
| `quantem.gpu.detector.backends.support` | deleted | 3 of 3 names deleted |
| `quantem.gpu.detector.backends.webgpu` | deleted |  |
| `quantem.gpu.detector.compute` | deleted |  |
| `quantem.gpu.detector.compute.backends` | deleted |  |
| `quantem.gpu.detector.compute.cuda` | deleted |  |
| `quantem.gpu.detector.compute.cuda.kernels` | deleted |  |
| `quantem.gpu.detector.compute.cuda.probe` | deleted |  |
| `quantem.gpu.detector.compute.mps` | deleted |  |
| `quantem.gpu.detector.compute.mps.kernels` | deleted |  |
| `quantem.gpu.detector.compute.packed` | deleted |  |
| `quantem.gpu.detector.compute.protocol` | deleted |  |
| `quantem.gpu.detector.compute.support` | deleted |  |
| `quantem.gpu.detector.compute.webgpu` | deleted |  |
| `quantem.gpu.detector.workflow` | `quantem.gpu.detector.workflow`, `quantem.gpu.detector.session` |  |
| `quantem.gpu.device._cupy` | `quantem.gpu.device.cuda_runtime` |  |
| `quantem.gpu.device.backend` | `quantem.gpu.device.select` |  |
| `quantem.gpu.display.backends` | deleted |  |
| `quantem.gpu.display.backends.cpu` | deleted |  |
| `quantem.gpu.display.backends.cuda` | deleted |  |
| `quantem.gpu.display.backends.webgpu` | deleted |  |
| `quantem.gpu.display.cuda` | deleted |  |
| `quantem.gpu.display.reference` | deleted |  |
| `quantem.gpu.display.webgpu` | deleted |  |
| `quantem.gpu.dpc.backends` | deleted |  |
| `quantem.gpu.dpc.backends.cuda` | deleted |  |
| `quantem.gpu.dpc.backends.cuda.backend` | `quantem.gpu.dpc.workflow` |  |
| `quantem.gpu.dpc.backends.mps` | deleted |  |
| `quantem.gpu.dpc.backends.mps.backend` | deleted | 1 of 1 names deleted |
| `quantem.gpu.dpc.backends.webgpu` | deleted |  |
| `quantem.gpu.dpc.compute` | deleted |  |
| `quantem.gpu.dpc.compute.cuda` | deleted |  |
| `quantem.gpu.dpc.compute.cuda.backend` | deleted |  |
| `quantem.gpu.dpc.compute.mps` | deleted |  |
| `quantem.gpu.dpc.compute.mps.backend` | deleted |  |
| `quantem.gpu.dpc.compute.webgpu` | deleted |  |
| `quantem.gpu.geometry.workflow` | `quantem.gpu.geometry.rotation` |  |
| `quantem.gpu.io._ans_contract` | deleted |  |
| `quantem.gpu.io._ans_legacy` | deleted |  |
| `quantem.gpu.io._array_resident` | `quantem.gpu.io.arrays` |  |
| `quantem.gpu.io._array_sources` | `quantem.gpu.formats.empad` |  |
| `quantem.gpu.io._camera_mps` | deleted | 3 of 3 names deleted |
| `quantem.gpu.io._compact_h5` | deleted | 5 of 5 names deleted |
| `quantem.gpu.io._digitalmicrograph` | `quantem.gpu.io.digitalmicrograph` |  |
| `quantem.gpu.io._emd_metadata` | `quantem.gpu.formats.emd` |  |
| `quantem.gpu.io._float_ans` | `quantem.gpu.resident.float_ans` | 3 of 5 names deleted |
| `quantem.gpu.io._hdf5_array_resident` | `quantem.gpu.io.arrays` |  |
| `quantem.gpu.io._hdf5_chunk_index` | deleted |  |
| `quantem.gpu.io._hot_pixels` | `quantem.gpu.resident.hot_pixels` |  |
| `quantem.gpu.io._memory` | deleted |  |
| `quantem.gpu.io._metadata` | `quantem.gpu.formats.emd`, `quantem.gpu.formats.hdf5.master` |  |
| `quantem.gpu.io._native_packed` | deleted | 1 of 1 names deleted |
| `quantem.gpu.io._packed` | deleted |  |
| `quantem.gpu.io._paired` | `quantem.gpu.io.paired` |  |
| `quantem.gpu.io._precision` | `quantem.gpu.io.precision`, `quantem.gpu.formats.precision` |  |
| `quantem.gpu.io._prepared_series` | deleted | 1 of 1 names deleted |
| `quantem.gpu.io._prepared_stack_metadata` | `quantem.gpu.formats.prepared_stack` |  |
| `quantem.gpu.io._publish` | `quantem.gpu.formats.publish` |  |
| `quantem.gpu.io._qem_metadata` | `quantem.gpu.formats.qem.metadata` |  |
| `quantem.gpu.io._qem_reference` | `quantem.gpu.formats.qem.reference`, `quantem.gpu.formats.qem.snapshot` |  |
| `quantem.gpu.io._read` | `quantem.gpu.io.read` |  |
| `quantem.gpu.io._resident` | deleted | 2 of 2 names deleted |
| `quantem.gpu.io._selection` | `quantem.gpu.io.selection` | 1 of 2 names deleted |
| `quantem.gpu.io._source112_archive` | deleted |  |
| `quantem.gpu.io._source112_archive_browser` | deleted |  |
| `quantem.gpu.io._streamed` | `quantem.gpu.io.encoded` |  |
| `quantem.gpu.io._streamed_file` | `quantem.gpu.io.qem`, `quantem.gpu.formats.qem.snapshot` | 5 of 8 names deleted |
| `quantem.gpu.io.backends` | deleted |  |
| `quantem.gpu.io.backends.cpu` | deleted |  |
| `quantem.gpu.io.backends.cpu.dense` | `quantem.gpu.io.hdf5.cpu` |  |
| `quantem.gpu.io.backends.cuda` | deleted |  |
| `quantem.gpu.io.backends.cuda._ans` | deleted | 2 of 2 names deleted |
| `quantem.gpu.io.backends.cuda.decoder` | deleted |  |
| `quantem.gpu.io.backends.cuda.float_ans` | `quantem.gpu.resident.cuda.float_ans` |  |
| `quantem.gpu.io.backends.cuda.hot_pixels` | `quantem.gpu.resident.cuda.hot_pixels` |  |
| `quantem.gpu.io.backends.cuda.packed` | deleted | 6 of 6 names deleted |
| `quantem.gpu.io.backends.cuda.precision` | `quantem.gpu.resident.cuda.precision` |  |
| `quantem.gpu.io.backends.mps` | deleted |  |
| `quantem.gpu.io.backends.mps._ans` | deleted | 3 of 3 names deleted |
| `quantem.gpu.io.backends.mps._spatial` | `quantem.gpu.resident.mps.spatial` |  |
| `quantem.gpu.io.backends.mps._streamed` | `quantem.gpu.resident.mps.counts` |  |
| `quantem.gpu.io.backends.mps.consumer` | deleted | 12 of 12 names deleted |
| `quantem.gpu.io.backends.mps.dense` | `quantem.gpu.io.hdf5.mps.decode`, `quantem.gpu.io.hdf5.cpu` | 6 of 9 names deleted |
| `quantem.gpu.io.backends.mps.float_ans` | `quantem.gpu.resident.mps.float_ans` |  |
| `quantem.gpu.io.backends.mps.hot_pixels` | `quantem.gpu.resident.mps.hot_pixels` |  |
| `quantem.gpu.io.backends.mps.packed` | deleted | 15 of 15 names deleted |
| `quantem.gpu.io.backends.mps.precision` | `quantem.gpu.resident.mps.precision`, `quantem.gpu.resident.cuda.precision`, `quantem.gpu.resident.mps.arrays` | 2 of 18 names deleted |
| `quantem.gpu.io.backends.mps.qh5` | deleted | 6 of 6 names deleted |
| `quantem.gpu.io.backends.mps.resident_dpc` | deleted | 4 of 4 names deleted |
| `quantem.gpu.io.backends.mps.series` | deleted | 2 of 2 names deleted |
| `quantem.gpu.io.backends.protocol` | `quantem.gpu.detector.session` | 1 of 2 names deleted |
| `quantem.gpu.io.backends.webgpu` | deleted |  |
| `quantem.gpu.io.constants` | deleted |  |
| `quantem.gpu.io.integrity` | deleted | 1 of 1 names deleted |
| `quantem.gpu.io.load` | `quantem.gpu.io.selection`, `quantem.gpu.io.load` | 6 of 8 names deleted |
| `quantem.gpu.io.models` | `quantem.gpu.io.dataset`, `quantem.gpu.formats.hdf5.readiness` |  |
| `quantem.gpu.io.qem_conversion` | `quantem.gpu.io.convert` |  |
| `quantem.gpu.io.qem_validation` | `quantem.gpu.cli`, `quantem.gpu.formats.qem.validation` |  |
| `quantem.gpu.io.readiness` | `quantem.gpu.formats.hdf5.readiness` |  |
| `quantem.gpu.io.representation` | `quantem.gpu.io.representation`, `quantem.gpu.io.convert` | 1 of 3 names deleted |
| `quantem.gpu.io.resident_contract` | deleted | 2 of 2 names deleted |
| `quantem.gpu.io.save` | `quantem.gpu.io.hdf5.write`, `quantem.gpu.io.save` | 2 of 7 names deleted |
| `quantem.gpu.io.ssb_result` | `quantem.gpu.formats.ssb_phase`, `quantem.gpu.cli` |  |
| `quantem.gpu.io.uint4` | deleted | 6 of 6 names deleted |
| `quantem.gpu.movie` | `quantem.gpu.movie.export`, `quantem.gpu.movie.cuda` |  |
| `quantem.gpu.movie.cuda_mp4` | `quantem.gpu.movie.cuda` |  |
| `quantem.gpu.movie.mps_mp4` | `quantem.gpu.movie.cuda` |  |
| `quantem.gpu.optics.aberration` | deleted | 5 of 5 names deleted |
| `quantem.gpu.optics.aberration_fitting` | `quantem.gpu.optics.cuda` | 4 of 5 names deleted |
| `quantem.gpu.optics.physics` | `quantem.gpu.optics.physics` | 5 of 12 names deleted |
| `quantem.gpu.parallax.backends` | deleted |  |
| `quantem.gpu.parallax.backends.cuda` | deleted |  |
| `quantem.gpu.parallax.backends.cuda.alignment` | `quantem.gpu.parallax.cuda.alignment` | 4 of 5 names deleted |
| `quantem.gpu.parallax.backends.cuda.backend` | `quantem.gpu.parallax.cuda.reconstruction` | 2 of 3 names deleted |
| `quantem.gpu.parallax.backends.cuda.correlation` | `quantem.gpu.parallax.cuda.correlation` | 2 of 3 names deleted |
| `quantem.gpu.parallax.compute` | deleted |  |
| `quantem.gpu.parallax.compute.cuda` | deleted |  |
| `quantem.gpu.parallax.compute.cuda.alignment` | deleted |  |
| `quantem.gpu.parallax.compute.cuda.backend` | deleted |  |
| `quantem.gpu.parallax.compute.cuda.correlation` | deleted |  |
| `quantem.gpu.parallax.results` | `quantem.gpu.parallax.results` | 1 of 2 names deleted |
| `quantem.gpu.remote.maped_api` | deleted | 8 of 8 names deleted |
| `quantem.gpu.remote.prepare` | deleted | 1 of 1 names deleted |
| `quantem.gpu.remote.server` | `quantem.gpu.remote.browse`, `quantem.gpu.remote.app`, `quantem.gpu.remote.catalog` | 7 of 12 names deleted |
| `quantem.gpu.remote.ssb_api` | `quantem.gpu.remote.catalog` | 18 of 19 names deleted |
| `quantem.gpu.screening._cuda` | deleted |  |
| `quantem.gpu.screening._memory` | deleted | 1 of 1 names deleted |
| `quantem.gpu.screening._mps` | deleted |  |
| `quantem.gpu.ssb.backends` | deleted |  |
| `quantem.gpu.ssb.backends.contract` | `quantem.gpu.ssb.contract` |  |
| `quantem.gpu.ssb.backends.cuda` | deleted |  |
| `quantem.gpu.ssb.backends.cuda.backend` | `quantem.gpu.ssb.cuda.backend` | 1 of 2 names deleted |
| `quantem.gpu.ssb.backends.cuda.engine` | `quantem.gpu.ssb.cuda.engine`, `quantem.gpu.ssb.cuda.kernels.engine` |  |
| `quantem.gpu.ssb.backends.cuda.kernels` | `quantem.gpu.ssb.cuda.kernels` |  |
| `quantem.gpu.ssb.backends.cuda.kernels.common` | `quantem.gpu.ssb.cuda.kernels.common` |  |
| `quantem.gpu.ssb.backends.cuda.kernels.fft1024` | `quantem.gpu.ssb.cuda.kernels.fft1024` |  |
| `quantem.gpu.ssb.backends.cuda.kernels.fft128` | `quantem.gpu.ssb.cuda.kernels.fft128` |  |
| `quantem.gpu.ssb.backends.cuda.kernels.fft256` | `quantem.gpu.ssb.cuda.kernels.fft256` |  |
| `quantem.gpu.ssb.backends.cuda.kernels.fft512` | `quantem.gpu.ssb.cuda.kernels.fft512` |  |
| `quantem.gpu.ssb.backends.cuda.optimizer` | `quantem.gpu.ssb.cuda.optimizer` |  |
| `quantem.gpu.ssb.backends.mps` | deleted |  |
| `quantem.gpu.ssb.backends.mps.backend` | `quantem.gpu.ssb.mps.backend` |  |
| `quantem.gpu.ssb.backends.mps.brightfield_columns` | deleted |  |
| `quantem.gpu.ssb.backends.mps.engine` | `quantem.gpu.ssb.mps.frames` | 1 of 3 names deleted |
| `quantem.gpu.ssb.backends.mps.kernels` | `quantem.gpu.ssb.mps.kernels` |  |
| `quantem.gpu.ssb.backends.mps.kernels.common` | `quantem.gpu.ssb.mps.kernels.common` |  |
| `quantem.gpu.ssb.backends.mps.kernels.fft1024` | `quantem.gpu.ssb.mps.kernels.fft1024` |  |
| `quantem.gpu.ssb.backends.mps.kernels.fft128` | `quantem.gpu.ssb.mps.kernels.fft1024` |  |
| `quantem.gpu.ssb.backends.mps.kernels.fft256` | `quantem.gpu.ssb.mps.kernels.fft1024` |  |
| `quantem.gpu.ssb.backends.mps.kernels.fft512` | `quantem.gpu.ssb.mps.kernels.fft1024` |  |
| `quantem.gpu.ssb.backends.mps.optimizer` | `quantem.gpu.ssb.cuda.optimizer`, `quantem.gpu.ssb.mps.optimizer` |  |
| `quantem.gpu.ssb.backends.mps.thick_sample` | `quantem.gpu.ssb.mps.thick_sample`, `quantem.gpu.ssb.cuda.kernels.engine` |  |
| `quantem.gpu.ssb.backends.webgpu` | deleted |  |
| `quantem.gpu.ssb.plots` | deleted | 2 of 2 names deleted |
| `quantem.gpu.ssb.temporal` | deleted | 6 of 6 names deleted |
| `quantem.gpu.ssb.torch_ssb` | deleted | 1 of 1 names deleted |
| `quantem.gpu.ssb.workflow` | `quantem.gpu.ssb.units`, `quantem.gpu.ssb.contract`, `quantem.gpu.ssb.workflow` |  |
| `quantem.gpu.swift.Benchmarks.MetalImageFFTBenchmark.compare_torch_fft` | `quantem.gpu.cli` | 6 of 7 names deleted |
| `quantem.gpu.webgpu` | deleted | 3 of 3 names deleted |
