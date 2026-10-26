# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

### Added

- Python MPS: drag windows for the regularized `phase` and `complex` methods (`ssb.mps.direct.DragWindow`), the CUDA
  `DragWindow` design. `direct_workflow.drag`, `prepare` and `advance` now serve MPS slider frames from a Chebyshev
  series of the prepared normal equations along the dragged control (29 exact preparations over C10 +-6.8 nm or
  C12 +-3.4 nm at 300 kV and 30 mrad, one per `advance`), evaluated in float64 on CPU threads and solved as
  `PreparedDirect.solve`. On two measured 512 x 512 acquisitions at 3x on an Apple M5 Max (4 CPU threads) a frame
  takes 25-43 ms (phase) and 60-81 ms (complex) against 0.33-0.63 s for an exact preparation, and agrees with the
  exact image within 1.0e-6 of its largest phase (log-amplitude within 6e-7); a window takes 11-16 s to build and
  holds 0.96 GB (phase) or 2.9 GB (complex). Previews come only from `drag`; `run`, `preview` and saved results stay
  exact. The window rule (reach, truncation, order, width, nodes) moves to `ssb.chebyshev`, shared with CUDA.

- Python MPS: the regularized `phase` and `complex` reconstructions, `DirectPtycho.run(method=...)` and
  `preview(method=...)`, on Apple GPUs (`ssb.mps.direct`), with CUDA's model, layout and regularization. MLX Metal
  kernels accumulate the normal equations from the retained G(q, k) (float32 per bright-field pixel, float-float across
  pixel batches); the damped systems are solved in float64 on CPU threads; one inverse FFT of the packed eta + i phi
  spectrum gives the wave. On five measured 512 x 512 acquisitions (1x, 2x and Auto, every regularization preset,
  456 images) the images agree with CUDA and with a float64 evaluation within 2x CUDA's own float32 error (rms;
  within 10x at the maximum): phase 2e-8 to 1.5e-7 rad rms, amplitude 0.8e-7 to 1.4e-7 of its mean. On an Apple
  M5 Max an optics change takes about 0.3-0.5 s at 3x and 0.6-0.9 s at 4x, a damping change 11-20 ms at 3x and
  26-60 ms at 4x. The post-fit three-method comparison remains CUDA-only.
- Native (Metal) SSB fifth-order fit: the higher-order box is wide by default. `SSBFitOptions.higherOrderLimit`
  bounds each of the 22 higher-order coordinates (C21 through C56) to ±20 rad of probe phase at the aperture edge
  (`wideHigherOrderLimit`, the default, which no longer binds) or ±2 (`standardHigherOrderLimit`, the earlier box,
  still an option); other values are refused, and saved options without it decode as ±2.
  `SSBModelFit.fifthOrderBounds(higherOrderLimit:)` gives the box, and `refineToFifthOrder` refines an already
  refined fit again from its kept low-order fit. A ±20 fit is checked by default (`checkHigherOrderConsistency`,
  `SSBFit.higherOrderConsistency`): it is refitted on each half of an alternating 0.05 1/A ring split of the band
  frequencies from the same low-order start, and when the two halves' probe phases differ by more than 0.5 rad rms
  over the aperture (`half_band_rms_rad`) the ±2 fit is returned instead with its higher orders marked not trusted
  (`trusted`, `shown_limit`). On measured frames of a thin two-dimensional crystal the halves agree to 0.16-0.35 rad;
  a thick crystalline acquisition (1.33 rad) falls back. The check adds two fifth-order fits (a third on fallback); `onRefinedFit` hands a viewer the
  ±20 fit first. An opt-in record (`computeHigherOrderGuard`, default false) adds `SSBFit.higherOrderGuard`
  (`SSBHigherOrderGuard`, Python `higher_order_fit.assess`): the coherence with Poisson noise floors from G(0, k) and
  each term's standard and edge-phase error from the Hessian covariance (0.65-0.9 s on a 512 x 512 scan); its
  simulation thresholds do not separate measured data, so nothing acts on it. `MetalSSBJointModel.bandPowerSummary()`
  gives P, the band weight and the noise variance.

- Native (Metal) drag previews of the regularized `phase` and `complex` reconstructions (`ssb-direct` v0.2):
  `MetalSSBJointModel.buildDragWindow(along:at:upsamplingFactor:method:regularization:nodes:)` builds a Chebyshev
  series of the prepared normal equations along one `SSBDragControl` (Python's CUDA `DragWindow`) and
  `dragPreview(_:along:upsamplingFactor:method:regularization:)` evaluates it, marked by `MetalSSBJointPhase.preview`.
  On an Apple M5 Max at 3x a preview takes 5-10 ms against 0.17-0.19 s for an exact optics change and agrees with the
  exact image to about 1e-6 of its standard deviation; a window takes about 5 s to build. Complex systems of two or
  more aliases prepare with one thread per frequency and alias (0.28 to about 0.18 s at 3x).

- Native (Metal) regularized direct reconstructions, ShowPtycho's `phase` and `complex` beside `ssb`:
  `MetalSSBJointModel.reconstruct(_:upsamplingFactor:method:regularization:)` with `SSBReconstructionMethod` and
  `SSBDirectRegularization` (phase and amplitude damping, smoothness prior, detector weighting), the same fitted
  optics, tilt, depth and Auto output sampling as SSB (`ssb-direct` v0.1). On five measured 512 x 512 acquisitions,
  1x and Auto, every regularization preset: within 2e-7 rad rms of CUDA, complex amplitude within 2e-7 of its
  mean. On an Apple M5 Max an optics change takes about 0.18 s (phase) and 0.28 s (complex) at 3x, a damping change
  1-7 ms.

### Removed

- Native (Metal) SSB: `SSBFifthOrderSearch.bounds`, the standard ±2 box that the lower-level fifth-order calls
  defaulted to while `SSBFitOptions` defaults to ±20, is removed. `SSBModelFit.refineFifthOrder`,
  `SSBFifthOrderGradientSearch.refine`, `SSBFifthOrderSearch.refine`, `SSBFifthOrderGradientSearch.projectedGradient`
  and the benchmark diagnostics `fifthOrderGradientCheck` and `fifthOrderKernelTimings` take `bounds:` explicitly;
  build it with `SSBModelFit.fifthOrderBounds(higherOrderLimit:)`. Fits are unchanged.
  `metal-ssb-fit-benchmark --fifth-only` now uses `--higher-order-limit` (default 20; it used ±2 silently), and
  `--objective-bench` clips its candidates to the run's recorded limit (±2 for runs recorded without one).
- Native (Metal) SSB is model-based only: the engine's own non-model reconstruction and the phase-variance fit are
  retired. Removed: `MetalSSBEngine.reconstruct(aberrations:scanDetectorRotationDegrees:)`, `phaseVariance`,
  `optimize` and `MetalSSBPhaseVarianceResult`; `SSBOptimizer` and `SSBOptimizationProgress`; the phase-variance
  branch of `SSBModelFit` with its `engine:` argument (`fit(model:options:)`, `findAberrations(model:...)`) and
  `SSBFit.phaseVarianceLoss`; their `ssb.metal` kernels (accumulation, finalization, chi tables, fused phase loss
  and moments); the `metal-ssb-benchmark` product, the `tests/metal` SSB harnesses with `check_metal_ssb*.sh`, and
  the frozen phase-variance fit trajectory (`ssb_fit_gate.py` and its pin). `SSBFitOptions(objective:
  .phaseVariance)` still decodes from saved runs and is refused as a fit; `MetalSSBSavedRun` runs still open, and
  `MetalSSBEngine.phase(of:)` shows them. Python's `objective="phase_variance"` is unchanged.
- Native (Metal) SSB: the `mean_phase` and `complex_wave` estimators are retired. `SSBPhaseEstimator`, the
  `estimator:` argument of `MetalSSBJointModel.reconstruct`, the mean-phase general and fused 1536-point kernels
  (`ssb-mean-phase`) and the complex-wave amplitude path are removed; the native reconstruction choices are
  ShowPtycho's three, `SSBReconstructionMethod` `.ssb`, `.phase` and `.complex`.
- Python SSB (CUDA and MPS): the `mean_phase` and `complex_wave` estimators are retired; `DirectPtycho.run` has
  ShowPtycho's three methods, `ssb` (the phase of the mean wave), `phase` and `complex`. `run`, `reconstruct` and
  `preview` raise for `phase_estimator="mean_phase"` (naming `method="phase"`) and `"complex_wave"` (naming
  `method="complex"`); `phase_estimator` accepts only `"phase_of_mean"`. Results and screening records saved with a
  retired estimator still load as saved, with a one-line notice. Removed: the backends' `reconstruct_result`,
  `preview` and `preview_sample` (and from `SSBProtocol`), the CUDA engine's mean-phase image and complex-object
  paths (`reconstruct`, `reconstruct_object`, `reconstruct_full`, `reconstruct_full_with_loss`, the sum-only
  chunked core) with their kernels (`mean_phase_kernel`, `pk_kernel_full`, the full-aberration fused row IFFTs of
  the 256/512/1024 FFT modules, `ifft1024_cols_accumulate_sum_split512_t64`, `ssb{N}_corrected_fourier_sum_t256`),
  the MPS no-loss mean-phase methods and the mean-phase upsampled paths. The phase-variance loss and
  `objective="phase_variance"` are unchanged: `ThickSample.phase_variance` and the MPS backend's `phase_variance`,
  `sample_phase_variance` and `phase_variance_higher_order` evaluate it. CUDA fits without tilt now hold their SSB
  phase directly instead of a complex object reconstructed and then replaced.
- Native (Metal) SSB: the fit objectives' v0.1 test-reference kernels are deleted (`ssb_low_order.metal`,
  `joint_objective_angle_sums`, `fifth_agreement`), with `SSBAgreementKernel` and
  `MetalSSBJointModel.agreementKernel`: the low-order agreement always uses its angle-sum kernel (the per-term
  option `joint_objective_batch` had no caller). Shared sources changed, so `ssb-joint-agreement` v0.4,
  `ssb-low-order-derivatives` v0.4, `ssb-fifth-agreement` v0.4, `ssb-fifth-gradient` v0.3, `ssb-per-term` v0.3 and
  `ssb-direct` v0.3 are registered; every output is bitwise unchanged.

### Changed

- Python SSB (CUDA and MPS): the fifth-order fit's higher-order box is wide by default, as on native:
  `find_aberrations(higher_order_limit=None)` (the default) bounds each of the 22 higher-order coordinates to ±20 rad
  of probe phase at the aperture edge, and `higher_order_limit=2` keeps the earlier box (`search="primary"` refuses
  either). A ±20 fit is checked with native's definition (`fifth_order_fit.higher_order_consistency`): the fifth
  order is refitted with exact gradients on each half of an alternating 0.05 1/A ring split of the band frequencies,
  both from the low-order fit, and when the two probe phases through fifth order (C10 and C12 included, tilt and depth
  not) differ by more than 0.5 rad rms over the aperture the ±2 fit is returned and its higher orders are not
  trusted, with one printed line. `optimizer["higher_order_consistency"]` records `half_band_rms_rad`, `trusted`,
  `shown_limit`, both halves and the checked ±20 fit; `optimizer["higher_order_limit"]` is the limit of the fit
  returned. On five measured 512 x 512 acquisitions CUDA's statistic equals native's to 0.001 rad (0.155-0.353 rad on
  four thin two-dimensional crystal frames, ±20 kept; 1.333 rad on a thick crystal, ±2 returned), and MPS gives the
  same values. The check adds 0.65-1.2 s on an RTX PRO 6000 Blackwell Max-Q (default fit 1.5-3.0 s against 0.8-1.6 s
  at ±2) and 2.2-7.4 s on an Apple M5 Max (8.1-23.6 s against 5.0-18.3 s).
- Native (Metal) SSB fifth-order fit: an exact-gradient trust-region quasi-Newton search (`ssb-fifth-gradient`
  v0.1, `SSBFitOptions.higherOrderSearch = .exactGradientCoarseToFine`) replaces the finite-difference Newton by
  default, which remains available as `.finiteDifferenceNewton`. On an Apple M5 Max the fifth-order stage takes
  0.7-1.3 s instead of 65-111 s, and the whole fit through fifth order 3.4-6.5 s instead of 74-166 s, with a
  higher Double-summed agreement on three 512 x 512 acquisitions. `SSBModelFit.refineToFifthOrder` refines a
  low-order fit without repeating its search.
- Native (Metal) SSB fit, low-order Newton (`ssb-low-order-derivatives` v0.2): the value and the derivatives are
  separate passes, so no derivatives are formed at a rejected step or for the starts the second iteration's
  pruning stops; elsewhere one fused pass forms both. The derivative sums drop from 51 to 29 by exact algebra
  (tilt enters the weights only through q.theta; the C10/C12 phase slopes are linear in k), with the aperture,
  sincos and quotient shortcuts of `ssb-joint-agreement` v0.2. On an Apple M5 Max one Newton center takes
  26-50 ms instead of 54-104 ms and the Newton 0.34-0.74 s instead of 0.91-1.83 s on three 512 x 512
  acquisitions, with the same centers, kept branch and low-order fit (within 2e-6 nm).
- Native (Metal) SSB fit, low-order stage: one 180-degree branch is searched and the other gets its exact image
  (C10 and C12 negated as even-degree terms, tilt negated, depth spread kept: the same agreement, the phase with
  opposite contrast for the branch test), and the trust-region Newton climbs on the exact gradient and Hessian of
  the agreement (`ssb-low-order-derivatives` v0.1) instead of 13- or 28-point finite-difference stencils. On an
  Apple M5 Max the stage takes 2.8-6.5 s instead of 11-35 s on three 512 x 512 acquisitions, with the same kept
  branch and an equal or higher agreement.
- `DirectPtycho.calibration_table()` is now `ptycho.calibration()`; `ptycho.calibration(to_dict=True)` returns the same rows as a dict (file value, unit, value used and source per parameter). The old name is removed.
- QuantEM Live screening records export and reopen at their tilt-corrected fit, the default SSB result: `formats.ssb_phase.export_result` takes `computed.ssb_tilt` and `ssb_phase_tilt.npy` when a record also holds an ordinary fit (`--ssb-mode both`), and the pair states the fit's `sample_tilt_mrad` and `depth_spread_nm` (both or neither; schema unchanged, earlier pairs read as before). `series.screen_record_settings` and `open_screen_record` take the same fit, and `open_screen_record` sets the session's sample tilt and depth spread (`series.screen_record_fit`, `series.screen_record_sample`).
- CUDA tilt-corrected SSB is faster with the same fits (within the cross-backend tolerances): the fused agreement
  kernel keeps its per-set sums in registers and drops three divisions per probe tilt (10.9 ms instead of 25.4 ms
  per pass of eight parameter sets, 7.2 ms instead of 24.5 ms at depth spread 0), the mean-phase loss corrects
  straight from the stored half plane (97 ms instead of 223 ms on a 512 x 512 scan) and the drag preview computes
  each bright-field pixel's geometry once per block. On a 512 x 512 scan one tilt fit takes 0.67 s instead
  of 1.62 s, and both branches 1.5 s instead of 3.6 s; `docs/performance/ssb/reports/2026-10-20-ssb-cuda-tilt-speed.md`.
- SSB fit methods have one vocabulary, `find_aberrations(tilt=..., objective=...)`. The default,
  `find_aberrations()`, is the model-based, tilt-corrected fit (least-squares model agreement, sample tilt and depth
  spread fitted); `tilt=False` holds the tilt and depth spread at 0; `tilt=False, objective="phase_variance"` is the
  older SSB search (minimum phase variance over bright-field pixels, Optuna TPE and Nelder-Mead). `tilt=True` with
  `objective="phase_variance"` raises. `result.objective` records the objective, `report()` shows it, and it is a
  saved setting. The MPS phase-variance search (`ssb/mps/loss.py`, its pair kernels, the 512 x 512 pair-pack
  policy and `scripts/benchmark_ssb_mps_scaling.py`) is restored unchanged for that option and reproduces the
  removed search bit for bit on gold; the M5 Max benchmark baseline times both searches
  (`search_candidate_phase_variance`, `find_aberrations_phase_variance`).
- On CUDA, `find_aberrations(tilt=False)` fits on the agreement objective as on MPS (`fit_standard_resident` on the
  fused CUDA agreement kernel at depth spread 0, Newton or Nelder-Mead, `fixed_aberrations` accepted); on a
  512 x 512 acquisition it gives the MPS coefficients to 4e-5 nm and 1e-4 degrees in 0.7 s, and its C10 moves
  from the phase-variance 10.05 nm to 9.90 nm. The phase-variance
  fit, `find_aberrations(tilt=False, objective="phase_variance")`, now runs the search of MPS's
  `optimize_phase_variance` in a backend-neutral module (`ssb/phase_variance_fit.py`: the start point, Optuna TPE
  asked two trials at a time, Nelder-Mead with the C12 and phi12 step floors), so the same seed gives the same
  trials and simplex on both backends; CUDA's
  phase-variance coefficients move to that search's lower minimum (C10 by 0.03 nm on that scan) and locked
  coefficients refine. `CudaSSBBackend.optimize`, `refine`, `batch_optimize` and `batch_nelder_mead` are removed.
- On MPS, `find_aberrations(tilt=False)` fits on the joint fit's objective, the least-squares agreement of the SSB
  model with G(q, k), with the sample tilt and depth spread held at 0: the same resident 200-trial search, then
  Newton (`refinement="auto"`) or Nelder-Mead. The search box is unchanged (C10 +-40 nm, C12 0 to 10 nm);
  `search_ranges`, `refine_lock` and now `fixed_aberrations` constrain it. On synthetic crystals it lands 3x nearer
  the true C10 than the phase-variance search, and a gold fit takes 7.8 s instead of about 75 s (both branches)
  with half the peak memory. Standard-fit coefficients move (gold C10 -17.756 -> -18.467 nm). The phase variance
  stays the reported `loss`.
- The detector calibration is an explicit step: `fit = data.calibration.fit_detector_sampling()` fits the
  bright-field disk of the mean pattern on the encoded data (`detector.fit_bright_field_disk`: centre, and the
  edge radius where the disk falls to half its plateau, at the convergence semiangle), records the detector step
  with the source "Fitted (bright-field disk)" together with `bright_field_center_px` and
  `bright_field_edge_radius_px`, and `fit.show()` draws the mean pattern with the fitted edge in mrad. `io.save`
  keeps the fit; a detector step set by hand wins.
- `SSB(data)` and `SSB.open` read the recorded bright-field centre and detector step. With none recorded they
  estimate the step with the same rule and print one line naming it, instead of a silent "Pending probe fit".
- On MPS, `SSB(data)` and `SSB.open` keep an integer acquisition encoded: the bright-field columns are decoded a
  few scan rows at a time while G(q, k) is built, and G(q, k) is all the session keeps. A 512 x 512 scan with 8,939
  bright-field pixels no longer holds a 6 GB decoded crop (twice) beside it; G(q, k) and every result are
  bit-identical. Keep a dataset passed to `SSB(data)` open while the session is in use.
- On Macs with 96 GB or more, the 512 x 512 MPS loss computes its row IFFT in tiles of 1024 bright-field pixels
  instead of one 4096-pixel chunk, so a ShowPtycho session keeps 3.5 GiB of loss buffers instead of 9.5 to 15 GiB
  (session peak 34.3 -> 22.2 GiB on gold) with the same loss and phase bits; loss previews take about 4 % longer.
  `quantem.gpu.ssb.mps.reconstruct.LOSS_ROW_IFFT_TILE_BF = None` keeps the whole chunk.
- An MPS region refit (`select_region`, `find_aberrations(scan_region=...)`, ShowPtycho's Fit Regions and Crop and
  Refit) frees the full field's G(q, k) first and the next full-field preview rebuilds it (about 0.8 s, the same
  bits); fits evaluate their loss in 256-pixel tiles and the standard search in 1,024-pixel packs. A ShowPtycho
  session on gold peaks at 17.4 GiB (46.1 before the memory work, 22.2 before this change).

- SSB is tilt-corrected by default: `find_aberrations()` (and the automatic fit
  of `reconstruct()`) fits the aberrations jointly with the sample tilt and
  depth spread (`tilt=True`); `tilt=False` is the explicit opt-out for
  standard SSB, and `refine_lock` belongs to it.
- The 180° branch of an estimated scan-detector rotation is settled by two
  independent fits, at the estimate and at the estimate plus 180°, with the
  same settings (tilt-corrected joint fits by default). The kept branch has
  positive phase at the atomic columns located on the scan's dark field
  (lattice-scale maxima at least 2 Å apart, top 20% by prominence; decisive
  when `|z| >= 3` and the effect mean/std `>= 0.25`); otherwise the larger
  mean-centred phase skewness decides. One line says which and why.
  A rotation given to `SSB(...)` or recorded with the acquisition is never
  moved. The fit now costs two searches.
- `SSBResult.rotation_branches` (a `RotationBranches`) records both fits and
  the evidence, replacing `column_sign`, `initial_column_sign`,
  `initial_scan_detector_rotation_deg` and `rotation_check_phases`; saved
  results are `quantem.ssb-result/9`. `result.show("rotation")` draws the
  evidence (the `histogram` keyword is gone).

### Added

- `find_aberrations(refine_rotation=True)` (also through `compare_rotation_branches`) refines an estimated
  scan-detector rotation after the 180-degree branch decision: 1-degree steps from the estimate until the agreement
  falls on both sides (at most 10 degrees), then bounded Brent to 0.05 degrees, each angle one polish of the fit from
  the nearest angle's fit with `set_rotation` (G(q, k) is not rebuilt). `result.rotation_estimate_deg`,
  `rotation_refined` and `rotation_refinement` (the agreement at both angles, the angles evaluated, the time) record
  it, are saved, and `report()` shows both angles; one line reports the change. A given or recorded rotation is never
  refined; the default fit is unchanged bit for bit. On gold: 358.31 -> 1.24 degrees, agreement +9.8%, 11 polishes in
  12.5 s on an M5 Max, the phase image nearly unchanged (correlation 0.98).

- `SSB.compare_rotation_branches(**find_aberrations_options)` fits both 180°
  branches and returns a `RotationBranches` with `show()` and `report()`; on a
  given rotation it reports without moving it unless `apply=True`.
- `quantem.gpu.ssb.results.phase_skewness`, the mean-centred third-moment
  ratio, next to `column_sign`; `quantem.gpu.ssb.branches` holds the column
  test (`locate_atomic_columns`, `column_phase_test`).

- The beginner SSB path is load, state the beam, reconstruct:
  `data.calibration.semiangle_mrad = 30`, `ssb = SSB(data)`,
  `ssb.reconstruct()`, `ssb.show()`. `SSB` uses the recorded scan-detector
  rotation, else estimates it (modulo 180 degrees) from the centre-of-mass
  curl and prints one line; `reconstruct()` fits the aberrations once when
  none are known and settles the 180-degree choice; `ssb.show()` opens
  ShowPtycho when that package is installed. The launch table names each
  rotation's source and adds a **Rotation flipped** row.

- `quantem.gpu.ssb.DEPTH_SPREAD_RANGE_NM = (2.0, 60.0)` is the depth-spread
  range the joint fit searches and its parameter slices span, for
  applications that offer the same range.
- `quantem.gpu.optics` exports `ssb_upsampling_factor`, which other packages
  imported from the private-looking `optics.physics` path.
- `quantem.gpu.optics.interaction_parameter(voltage_kV)` returns the
  relativistic interaction parameter σ in rad/(V·Å), computed with the same
  wavelength as `wavelength_A_from_kV`.

- `quantem.gpu.detector.detector_sampling_mrad(mean_pattern, semiangle_mrad)`
  returns the `(row, col)` detector sampling in mrad per pixel from the
  bright-field disk edge, the rule SSB, parallax and screening apply when no
  sampling is recorded.

- `quantem.gpu.geometry.sample_to_propagator_tilt` and
  `propagator_to_sample_tilt` convert a tilt between the SSB scan frame
  (`sample_tilt_mrad`) and the detector frame of a ptychographic propagator
  (`propagator_tilt_mrad`), honouring `scan_detector_rotation_flipped`.
  `opposite_rotation_deg`, `rotation_gap_deg` and `mean_rotation_deg` state
  the other 180-degree branch and compare or average rotations modulo 180 or
  360 degrees.

### Fixed

- Python MPS SSB: the bright-field frequencies k are formed in float64 and rounded once to float32
  (`ssb.mps.prepared.bright_field_frequencies`, used at preparation and on a rotation change, and by the direct
  methods). They were formed from a float32-rounded disk centre, which shifts every k by the same amount. Fits move by
  at most 1.1e-3 nm in C10. On five measured 512 x 512 acquisitions (24 SSB images, 1x, 2x and Auto) the mean rms
  error against a float64 evaluation drops from 1.32 to 1.11 times CUDA's own float32 error (against CUDA: 1.26 to
  1.24); the direct methods are unchanged. Another float32 rounding of k of the same size moves the SSB phase of a
  fifth-order probe by 1 to 1.6 times CUDA's own error (rms), so any one SSB image sits at that floor: one 1x image of
  a thick crystal moves from 1.77 to 2.21 times CUDA's error against CUDA (2.03 against float64), and four random
  roundings of its k of that size give 1.33 to 2.17. That image misses the 2x-of-CUDA parity gate and is accepted as
  the k-rounding floor; evaluating chi(k +- q) - chi(k) directly, as the native redraw v0.3 does, is the way past it.
  The fused depth-model test bound (n = 128, 256, 512) rises from 5e-8 to 1e-7 for the same reason: other float32
  roundings of k give 4.5e-8 to 6.8e-8 between its two summation orders.

- Python SSB (CUDA and MPS): `find_aberrations(search="broad")` with exact gradients no longer ends below
  `search="standard"` on the same data. It refined only the best TPE trial and could converge in a lower basin
  (on two measured frames of a thin two-dimensional crystal 1.7237 and 1.7183 against standard's 1.8075 and
  1.7863). The refinement now runs from the standard start and from the 8 best TPE candidates that differ from every
  earlier start (aperture phase by 0.25 rad rms, tilt by 0.5 mrad or depth spread by 1 nm) and keeps the highest
  float64-summed agreement; a later start replaces the kept one only when higher by 1e-6 relative.
  `optimizer["starts"]` lists each start's agreement and `optimizer["kept_start"]` the one kept. On seven measured
  512 x 512 acquisitions broad equals standard on six and is 0.5 % higher on one, is never below the earlier
  finite-difference broad fit, and repeats bit for bit; it takes 7.3-8.6 s (two larger acquisitions 11.7-12.9 s)
  on an RTX PRO 6000 Blackwell Max-Q instead of about 6 s, of which TPE is 5.0-6.9 s.

- Native (Metal) `phase` and `complex` with `inverse_mean` weighting no longer refuse BF pixels outside the calibrated
  aperture: as Python, every selected BF pixel enters the median floor and the normalization (those outside have no
  transfer). The k-only aperture term A(k) now uses CUDA's square-pixel probe geometry, as A(k +- q) does
  (`ssb-direct` v0.4; about 1 ulp at the aperture edge).
- Native (Metal) SSB redraw (`ssb-redraw` v0.3, replaces v0.2): the lowest scan frequencies, where the two
  sidebands nearly cancel, are resolved by evaluating chi(k +- q) - chi(k) as polynomials expanded about each BF
  pixel and forming gamma from their half sum and half difference. On five measured 512 x 512 acquisitions, 1x, 2x
  and Auto, the native phase is now within 1.08x of CUDA's own float32 error against a float64 definition (v0.2:
  up to 2.45x at 1x on 0.525 A scans); the 512 x 512 fifth-order 3x redraw keeps its speed (30.8 ms per frame on
  an Apple M5 Max).
- `find_aberrations(search_ranges=...)` searches every coefficient the ranges
  leave out over its default range on CUDA and MPS (it was held at the start).
  On MPS, pinning C12 at 0 no longer returns C10 = C12 = phi12 = 0: the start
  is no candidate once a coefficient is pinned. A fit that holds C12 at 0
  reports phi12 as 0.
- Calibration tables round values to six significant figures for reading
  (`0.603545`, not `0.6035446045631986`); sessions and records keep the full
  values. A calibration set on `data.calibration` equal to the file's record
  reads **Recorded**, not **Overridden**.
- `reconstruct()`'s automatic fit prints one line when its rotation check
  moves the rotation by 180° or cannot decide, although the fit itself runs
  quietly.
- `dpc.run` and `screening.prepare` integrate the aligned centre-of-mass
  field as it is when the detector axes are flipped. They swapped its
  components back first, so a flipped scan's iDPC phase was the integral of
  the transposed gradient (correlation 0.10 with the unflipped phase on a
  synthetic scan; now identical). The screening cache is
  `quantem.screening-cache/7`.
- On MPS, a coefficient pinned in `search_ranges` (one number, such as
  `{"C12_nm": 0.0, "phi12_deg": 0.0}`) keeps that value in the result and its
  reconstruction, as on CUDA; the result reported the search's start
  (C12 5 nm, phi12 0°) instead.
- `SSB.open` on MPS prints one line when it reads exported bright-field
  columns that were binned on the detector, naming the export and how to fit
  native detector pixels; it used them silently.
- The **Rotation flipped** row of `SSB.calibration_table()` shows the file's
  recorded flip (it always read **Not recorded**), and `data.metadata_table()`
  gains the row.
- `SSB(data)` and `SSB.open` use a recorded `detector_sampling_inv_A`
  (`data.calibration.detector_sampling_inv_A = ...`), converted to mrad per
  pixel with the electron wavelength; it was ignored and the bright-field
  disk estimate used instead, without a word.
- Flagged detector pixels are announced: `io.load`'s load line names the
  setting that does the other thing (`4 bad pixels left as recorded
  (hot_pixel_correction="median" replaces them ...)`), and `SSB.open`, which
  replaces them with the neighbor median, prints one line with the count and
  how to keep the recorded values. The default stays `"none"`.
- `find_aberrations(tilt=True, search_ranges={"C10_nm": (-50, 50)},
  fixed_aberrations={"C12_nm": 0.0})` runs: the joint fit takes a `C10_nm`
  range or value and a `C12_nm` range from 0 or `C12_nm=0`, keeps the
  rotation check (bounds follow C10's sign onto the other branch), and
  reports `phi12_deg = 0` when C12 is held at 0. A `phi12_deg` constraint, a
  nonzero `C12_nm` alone, or `refine_lock` raise with what to pass instead.
- Assigning `ssb.aberrations = {...}` states known aberrations: the next
  `reconstruct()` applies them instead of fitting again, so a restored or
  calibrated correction is no longer overwritten. `ssb.aberrations = None`
  forgets them. `SSB(aberrations=...)` and the assignment accept the
  higher-order coefficients `reconstruct` applies.
- `SSB.reconstruct()` keeps the rotation check of the fit it applies:
  `column_sign`, `initial_scan_detector_rotation_deg`, `initial_column_sign`
  and the compared phases, so `report()` shows **Suggested rotation** and
  `result.show("rotation")` works. Changed coefficients describe another
  image and carry no check.

### Changed (breaking, no aliases)

- A ShowPtycho folder's `snapshots/cal.json` is
  `quantem.ssb-browser-calibration/3`: the display-only phase negation is
  `display_negate_phase` (was `flip_phase`) and `scan_detector_rotation_flipped`
  is part of the layout (the WebGPU `BrowserCalibration` type now lists both).
  MPS and WebGPU readers refuse `/2`; export the folder again.

- One calibration label set, defined in `quantem.gpu.io.metadata_display`
  (`CALIBRATION_LABELS`, `CALIBRATION_UNITS` and the source states), names
  the rows of `data.calibration.table()`, `data.metadata_table()`,
  `io.inspect` and `SSB.calibration_table()`. `data.calibration.table()`
  changes: **Voltage** is **Accelerating voltage**, **Scan sampling
  (row, col)** is **Scan step (row, col)**, **Detector sampling (row, col)**
  is **Detector step (row, col)**, units read `mrad per pixel` and
  `1/angstrom per pixel`, values are `(row, col)` tuples and **Missing** is
  **Not recorded**. A zero or negative physical calibration reads **Not
  recorded** in every table.

- `geometry.rotate_scan(data, angle_degrees)` takes `display_rotation_deg`,
  the name of a display-only rotation, and records it under that key in
  `metadata["scan_rotation_history"]`.

- `scan_sampling_A` (and `detector_sampling_mrad`) may be one number for
  both axes at every entry point: `SSB(...)`, `SSB.open(...)`,
  `SSB.reconstruct_series(...)`, `parallax.run(...)` and
  `optics.ssb_upsampling_factor(...)`, as `data.calibration` already
  allowed. Sessions and records still store and report the `(row, col)` pair.

- `SSBResult.refine_method` is renamed `refinement`, the name of the
  `find_aberrations(refinement=...)` keyword it records. Saved result records
  are `quantem.ssb-result/8`; a `/7` record is refused with one printed line
  and computed again. Live's fit block (`quantem.ssb-fit/7`) does not carry
  the field and is unchanged.

- `SSB.reconstruct(trials=[...])` is renamed `trial_ids=[...]`, so `trials`
  never means a list of ids in one place and a count in another. The WebGPU
  protocol's trial count (`WebGPUOptimizationOptions.trials`,
  `SSBOptimizationResult.trials`) is `numTrials`.

- `save_to` is renamed `save_dir` on `SSB.find_aberrations`,
  `SSB.reconstruct` and `SSB.reconstruct_series`, matching
  `ssb.show(save_dir=...)`. It names a results directory; an `.npz` or
  `.json` file path is refused (pass its folder). The saved files are
  unchanged: `ssb-find_aberrations.npz/.json`, `ssb-reconstruct.npz/.json`.

- The scan-detector rotation fitted from the centre-of-mass curl
  (`dpc.run`, `SSB(data)` and `SSB.open` without a rotation, and
  `screening.prepare`) is the exact minimum of the mean squared curl, in
  closed form, instead of the best of a grid of angles 1.0056° apart, which
  could leave an estimate up to 0.5° off. `dpc.workflow.find_optimal_rotation` and `dpc.run` no
  longer take `rotation_steps`, nor does `screening.prepare`; the screening
  cache is `quantem.screening-cache/6`, so earlier caches are rebuilt.
  `find_optimal_rotation(..., scan_detector_rotation_flipped=...)` keeps a
  given flip and fits the angle alone, the one search SSB uses.

- `disk_edge_radius` moves from `quantem.gpu.ssb.brightfield` to
  `quantem.gpu.detector.disk_edge_radius`.

- `detector_transposed` is renamed `scan_detector_rotation_flipped`
  everywhere: `SSB`, `SSB.open`, `dpc.run`, results, screening products,
  saved records, browser state, WebGPU and Swift. Records saved with the old
  key no longer carry the flag.
- `data.set_calibration(...)` is replaced by setting fields on
  `data.calibration` (`voltage_kV`, `semiangle_mrad`, `scan_sampling_A`,
  `detector_sampling_mrad`, `detector_sampling_inv_A`); `data.calibration`
  displays the table, and `data.calibration.table()` returns it. The source
  label "Saved override" is now "Overridden".
- An omitted `scan_detector_rotation_deg` is no longer 0 degrees; pass `0.0`
  for a scan known to have no rotation.

- `SSB.find_aberrations(tilt=True, refinement="newton")` polishes the joint
  tilt fit with Newton steps on finite-difference models whose evaluations
  share passes over G: a few batched passes instead of hundreds of
  Nelder-Mead evaluations. `compare_standard=False` skips the standard-SSB
  search that only feeds `tilt_fit_gain` (then None). `num_trials=0` with
  `tilt=True` refines from the session's aberrations, `sample_tilt_mrad` and
  `depth_spread_nm` without a search.
- `SSB(dataset, scan_region=..., mean_pattern=...)` holds the scan positions
  of `scan_region` of an encoded dataset alone, with the bright-field disk of
  the whole scan; `mean_pattern` reuses the caller's `detector.mean`. A fit
  saved from an encoded dataset with `save_dir` is identified by the data
  (shape, dtype, mean-pattern hash, scan region) and reused by a later session
  on the same data.
- `SSB.preview(..., output="native")` returns the CuPy phase on CUDA without a
  host copy.
- `quantem.gpu.io.load.bin` sums (or averages) detector or scan blocks of a
  CuPy array or Torch tensor on its GPU; integer counts sum exactly.

## rc17 - 2026-10-12

### Changed

- A session's `dataset.yaml` is read only at quantem.live's schema
  `quantem.dataset/2` (`quantem.gpu.formats.dataset_yaml`), by `io.convert`
  and `SSB.reconstruct_series`; another schema raises `ValueError` naming
  quantem.live's `tools/rewrite_dataset_yaml.py` (`convert` then copies
  without it and records why). The scan step is a `[row, col]` pair: the file
  entry's own `scan_sampling_A`, else the `calibrations` row of its `mag`,
  else `calibrations.default`. The rotation is
  `files.<n>.scan_detector_rotation_deg`, else
  `microscope.scan_detector_rotation_deg` (they were `rotation_deg` and
  `screen_rotation_deg`); a per-file `0.0` counts. The older
  `reference_structure` block is no longer read as a specimen; `specimen` is.
- `.qem` flat calibration keys are specified (QEM specification 0.0.4) and
  carry their unit: `detector_sampling` with `detector_sampling_unit` becomes
  `detector_sampling_mrad` or `detector_sampling_inv_A`, each `[row, col]`,
  exactly one present. `io.save` refuses the old pair and refuses both new
  keys together. Stored `scientific_metadata` is unchanged; every schema-2
  file loads with the same calibration. A saved `origin` without `units` is
  in the unit its header states under `detector_sampling_unit` (`rad`,
  `1/nm`) and is scaled by the factor the detector step was, as before; where
  the header states none it is in the stored detector unit.
- `scan_sampling_A` in dataset metadata is a `[row, col]` pair; a single number
  raises in `Dataset4dstemGPU.sampling`.
- An HDF5 master written by an earlier version can hold calibration this
  version does not read: `detector_sampling` with `detector_sampling_unit`,
  or one number under `scan_sampling_A`. `io.load` leaves those attributes
  out of the dataset metadata, so the axes are uncalibrated, and gives one
  `UserWarning` per file that names them, whatever `verbose` is; so do
  `io.convert`, which loads quietly, and `PairedLoader.stream`. No value is
  converted. They loaded without a message, and the attributes left in the
  metadata raised when the sampling was read or the acquisition was saved as
  `.qem`. `io.inspect` returns the attributes as stored.
- `.qem` scientific metadata schema 1 is refused with a message that names the
  schema, in Python and in the WebGPU reader. It was read with unit
  conversion. Export such a file again from its source.
- A saved `.qem` stores calibration, source fields, attachments, the sample
  and calibration overrides once, in `scientific_metadata`. The retained
  `metadata` block no longer holds copies of them or loader state (shapes,
  byte counts, backend, device, timings, raw `entry/...` paths). Readers drop
  the same keys from older files.
- The hot-pixel record carries `"schema": "quantem.hot-pixel-correction/2"`
  in place of `"version": 1`, and `coordinates_row_column` becomes
  `coordinates_row_col`. A record of another version is read as absent in a
  `.qem`; a saved paired resident with one is refused.
- The DigitalMicrograph reader no longer adds `sampling`, `units`,
  `pixel_origin` and `axis_order` to dataset metadata; its calibration is
  `scan_sampling_A` and `detector_sampling_inv_A`.
- Scan-to-detector geometry is one physical angle and one flag.
  `rotation_angle_deg` with `com_reversed` becomes `scan_detector_rotation_deg`
  in [0, 360) on `SSB`, `SSB.open`, `SSB.set_rotation`,
  `SSB.reconstruct_series`, `SSBResult`, trial records, the result signature
  and the stored fit block; `SSBResult.physical_rotation_deg` is that same
  attribute. `SSBResult.initial_rotation_deg` becomes
  `initial_scan_detector_rotation_deg`, and `rotation_flipped` is removed (the
  two angles differ when the polarity check changed the branch). `SSB` and
  `SSBResult` gain `detector_transposed`; `SSB` raises `NotImplementedError`
  for True.
- Aberration mappings carry their unit in each key, with the azimuth in
  degrees: `C10`, `C12` (nm) and `phi12` (radians) become `C10_nm`, `C12_nm`
  and `phi12_deg` in the `aberrations` argument of `SSB`, `SSB.open`,
  `SSB.reconstruct` and `SSB.preview`, in `SSBResult.aberrations` and
  `SSBResult[...]`, in `ParallaxResult.aberrations`, in the saved result and
  in the stored fit block, which no longer has `aberration_unit`. The earlier
  keys raise `ValueError`.
- `find_aberrations(refine_lock=...)` names the public coefficients
  (`C10_nm`, `C12_nm`, `phi12_deg`); it took the engines' `C10`, `C12`,
  `phi12`. An unknown key in `search_ranges` or `refine_lock` raises. Inside
  the engines the search ranges and trial parameters are `C10_A` and `C12_A`:
  they held Angstrom under the names `C10_nm` and `C12_nm`. Seeded fits give
  the same numbers.
- `SSB.preview` takes higher orders as named keys of `aberrations`
  (`C21_nm` with `phi21_deg`, `C23_nm`, `C30_nm`, ... `C56_nm` with
  `phi56_deg`); the arrays `higher_order_magnitudes` and
  `higher_order_angles` (nm and radians by slot) are removed.
- `SSBSeriesResult.records` reports `phi12 (deg)`; it was `phi12 (rad)`.
- Bright-field names are spelled out with their unit: `bf_center` becomes
  `bright_field_center_px`, `bf_radius` `bright_field_radius_px`,
  `detected_bf_radius` `bright_field_edge_radius_px`, `bf_intensity_threshold`
  `bright_field_intensity_threshold` and `num_bf`
  `num_bright_field_pixels`, as arguments and attributes of `SSB`, `SSB.open`
  and `SSBResult`, in the result signature and the stored fit block.
  `ScreeningResult.probe_center` and `probe_radius` become
  `bright_field_center_px` and `bright_field_edge_radius_px`, as do `center`
  and `radius_px` in the screening cache.
- `detector.bf(data, center=, radius=)` becomes
  `detector.bf(data, bright_field_center_px=, bright_field_radius_px=)`.
  `detector.adf` and `detector.df` take keyword-only bounds with the unit in
  the name, `inner_angle_mrad` / `outer_angle_mrad` or `inner_radius_px` /
  `outer_radius_px`, in place of `inner`, `outer` and `unit`, and
  `bright_field_center_px` / `bright_field_edge_radius_px` in place of
  `center` / `radius`. `detector.detector_mask(center, lo_px, hi_px,
  det_shape)` becomes `detector_mask(center_px, inner_radius_px,
  outer_radius_px, detector_shape)`.
- `parallax.run(center=, bf_radius=, sampling_radius=)` becomes
  `parallax.run(bright_field_center_px=, bright_field_radius_px=)`; the one
  radius selects the pixels used and defaults to the detected disk. It is
  used as given: a fractional radius selects the pixels inside it, as
  `sampling_radius` did.
- `ParallaxResult.rotation_angle_deg` becomes `scan_detector_rotation_deg`
  and holds the physical rotation in [0, 360), as every other result does.
  Only the representation of a fitted angle changes: a fit that read `-20`
  reads `340`, the same rotation (the recovery test's expected value moved
  from -20 to 340 for that reason alone). `fit_aberrations_from_shifts`
  still returns the angle within 90 degrees of zero.
- Elapsed times carry their unit: `SSBResult.elapsed` and `refine_elapsed`,
  `DPCResult.elapsed` and `ParallaxResult.elapsed` become `elapsed_s` and
  `refine_elapsed_s`; `SaveResult.elapsed_seconds` becomes `elapsed_s`.
- The axis word in arguments is `col`: `extract_diffraction_device(scan_row,
  scan_column)` of encoded residents becomes `(scan_row, scan_col)`, and
  regions are documented as `(row_start, row_stop, col_start, col_stop)`.
- The WebGPU SSB engine (`ssb/webgpu`) follows the same vocabulary. It reads
  a ShowPtycho folder's `snapshots/cal.json` as quantem.widget writes it,
  `"schema": "quantem.ssb-browser-calibration/2"`: `num_bf` becomes
  `num_bright_field_pixels`, `rotation_angle_deg` becomes
  `scan_detector_rotation_deg` (the `rotation_angle_rad` fallback is removed),
  `kx_bf`, `ky_bf` become `bright_field_k_row`, `bright_field_k_col`, `qx_1d`,
  `qy_1d` become `scan_q_row`, `scan_q_col`, and `semiangle_rad`,
  `angular_sampling_rad` become `semiangle_mrad`, `detector_sampling_mrad`
  (converted to radians once on entry). The bright-field source it reads is
  the folder manifest's `source.bf_columns`: `numBf`, `scanShape`,
  `bytesPerBf`, `bitsPerValue` become `num_bright_field_pixels`, `scan_shape`,
  `bytes_per_bright_field_pixel`, `bits_per_value`.
  `reconstruct(c10, c12, phi12)` names its engine units `(c10A, c12A,
  phi12Rad)`; options `upsample`, `bfCount`, `rotationDeg` become
  `upsamplingFactor`, `numBrightFieldPixels`, `scanDetectorRotationDeg`;
  `higherOrder` keys `<name>_mag` and `<name>_angle` become `<name>_A` and
  `phi<nm>_deg`, and the two terms without an azimuth, `C30` and `C50`,
  become `C30_A` and `C50_A`; a key outside that set throws with the list of
  accepted keys (it read as 0). Sample options `tiltRowMrad`, `tiltColMrad`,
  `thickness` become `sampleTiltMrad` (`[row, col]`, no tilt when omitted)
  and `depthSpreadA`. The result's `rotationDeg` and `bfCount` become
  `scanDetectorRotationDeg` and `numBrightFieldPixels`.
- The remote server's HTTP parameters name their axis and unit (protocol
  version 2): `sx`, `sy` become `scan_row`, `scan_col`; `cx`, `cy` become
  `center_col_px`, `center_row_px`; `column_start`, `column_stop` become
  `col_start`, `col_stop`; `det_bin` becomes `detector_bin`; `inner`, `outer`
  become `inner_bright_field_fraction`, `outer_bright_field_fraction` on
  `/api/browse/realspace` and `inner_radius_px`, `outer_radius_px` on
  `/api/browse/realspace-shape`; the modes `CoMx`, `CoMy` become `CoMcol`,
  `CoMrow`. `BrowseService.plan(det_bin=)` becomes `detector_bin=`. The
  sessions payload gives `scan_sampling_A`, `detector_sampling_mrad` and
  `detector_sampling_inv_A` as `[row, col]` pairs in place of
  `k_pixel_size_mrad`.
- The portable SSB phase record (`ssb.json` beside `ssb.npy`) is
  `"schema": "quantem.ssb-phase/2"` with snake_case keys of the shared
  vocabulary: `format`/`schemaVersion` become `schema`, `sourceIdentity`
  `source_identity`, `rows`/`columns` `shape`, `phaseFile` `phase_file`,
  `phaseEncoding` `phase_encoding`, `phaseUnits` `phase_unit`, `phaseSHA256`
  `phase_sha256`, `runMetadata` `run_metadata`; in `calibration`
  `beamEnergyKeV` becomes `voltage_kV`, `semiangleMrad` `semiangle_mrad`, the
  two scan steps `scan_sampling_A`, the two detector steps
  `detector_sampling_mrad`, `centerRow`/`centerColumn`
  `bright_field_center_px`, `brightfieldRadiusPixels` `bright_field_radius_px`,
  `excludedDetectorPixels` `excluded_detector_pixels`; `c10Nanometers`,
  `c12Nanometers`, `phi12Radians` become `aberrations` with `C10_nm`, `C12_nm`,
  `phi12_deg`; `rotationDegrees` becomes `scan_detector_rotation_deg` with
  `detector_transposed`. Version-1 records are refused. The response of
  `/api/ssb/saved-results` is `"schema": "quantem.ssb-phase-results/2"` with
  `source_identity`. The exporter's flag `--expected-source-id` becomes
  `--expected-source-identity`.
- The stored fit block written by `SSB.reconstruct_series` also holds
  `bright_field_center_px` and `detector_sampling_mrad`.
- `SSBResult.scan_sampling_A` becomes `object_sampling_A`: the pixel size of
  the reconstructed image, always a `[row, col]` pair (it was the scan step
  divided by the upsampling factor under the scan step's name, a scalar when
  the two axes were equal). The scan step is `scan_sampling_A` only.
- `scan_sampling_A` and `detector_sampling_mrad` of `SSB`, `SSB.open`,
  `SSB.reconstruct_series` and `parallax.run`, and `scan_sampling_A` of
  `optics.ssb_upsampling_factor`, are `(row, col)` pairs; one number raises
  `ValueError`. The stored fit block holds the pair.
- Counts, factors and samplings name their unit: `det_sampling` becomes
  `detector_sampling_mrad` (`SSB`, `SSB.open`, `parallax.run`, the session
  attribute and the result signature); `upsample` becomes `upsampling_factor`
  (`SSB.reconstruct`, `SSB.show_trials`, `SSBResult`); `tilt_mrad` becomes
  `sample_tilt_mrad` (`SSB.preview`, `SSB`, `SSBResult`); `n_trials` becomes
  `num_trials` (`SSBResult`); the trial-count argument `trials` becomes
  `num_trials` (`SSB.find_aberrations`, `SSB.reconstruct_series`, the result
  signature and `SSBSeriesResult`); `det_shape` becomes `detector_shape`
  (`H5Writer`, detector sessions). Trial records hold `sample_tilt_mrad` as
  one `[row, col]` pair in place of `tilt_row_mrad` and `tilt_col_mrad`.
- An HDF5 master written by `io.save` stores the detector shape once, as the
  root attribute `detector_shape`, the one the master reader reads; the copy
  on `entry/data` (`det_shape`) is gone.
- A ShowPtycho calibration (`snapshots/cal.json`) read by MPS SSB is the file
  quantem.widget writes: `"schema": "quantem.ssb-browser-calibration/2"` with
  the keys `bright_field_rows`, `bright_field_cols`, `bright_field_center_px`,
  `bright_field_radius_px` and `detector_sampling_mrad` (they were `bf_rows`,
  `bf_cols`, `bf_center`, `bf_radius_px`, `det_sampling_mrad_px`), the scan
  crop `scan_region` as `[row_start, row_stop, col_start, col_stop]` and
  `scan_shape` as `[rows, cols]` (it was `scan_region.shape`). Another
  calibration is refused.
- A rotation fitted from the centre-of-mass curl is
  `scan_detector_rotation_mod180_deg`: `DPCResult.rotation_deg` and
  `ScreeningResult.rotation_deg` take that name, `DPCResult.use_transpose` and
  `ScreeningResult.transposed` become `detector_transposed`. The argument
  `rotation_angle_deg` of `dpc.run` and `screening.prepare` becomes
  `scan_detector_rotation_deg`, with a new `detector_transposed`; a result
  then holds the given angle in `scan_detector_rotation_deg`, wrapped into
  [0, 360): `dpc.run(scan_detector_rotation_deg=-20)` reports `340`, where
  `rotation_deg` held `-20`. `DPCResult.report()` names its flag column
  `Detector transposed` (it was `Detector-axis swap`) and adds the column
  `Known modulo 180`. Angle and flag of a result describe one geometry: with
  a given rotation `dpc.run` has no `scan_detector_rotation_mod180_deg`, and
  `screening.prepare` holds the given angle modulo 180 there beside the
  given flag, with the fit kept in `metadata["parameters"]`. A fitted
  `scan_detector_rotation_mod180_deg` lies in [0, 180]: an unrotated scan
  fits the end point of the search grid, float32 pi, which reads
  `180.000005`; 0 and 180 describe the same estimate. The fit and its phase
  are the ones returned before.
- A saved SSB result record and its signature carry
  `"schema": "quantem.ssb-result/7"`, a string; it was the number 6.
- `ScreeningResult.mean_dp` becomes `mean_diffraction_pattern`, as does its
  array in the screening cache; in `ScreeningResult.metadata` and the cache
  record, `exact_accumulation.mean_dp_divisor` becomes
  `mean_diffraction_pattern_divisor`.
- `scripts/benchmark_screening.py` writes its report with the result's
  names: `probe_center_row` and `probe_center_column` become
  `bright_field_center_px` (`[row, col]`), `probe_radius_pixels` becomes
  `bright_field_edge_radius_px`, `rotation_degrees` becomes
  `scan_detector_rotation_mod180_deg`, `transposed` becomes
  `detector_transposed`, and the product `mean_dp` becomes
  `mean_diffraction_pattern`. `scripts/benchmark_ssb_mps_scaling.py` writes
  `elapsed_s` (it was `elapsed_seconds`).
- The WebGPU detector engine names its iDPC geometry as the other engines
  do: the parameters `rotationDeg` and `useTranspose` of
  `DetectorCompute.maskedIDpc` and `maskedIDpcBuffer` become
  `scanDetectorRotationDeg` and `detectorTransposed`.
- The SSB parity harness (`tests/parity/ssb_parity_case.py`,
  `ssb_parity_gate.py`, `tests/metal/ssb_parity_check.swift`,
  `ssb_fit_trajectory.swift`) reads and writes the shared names. In
  `case.json`: `bf_center` and `brightfield_center` become
  `bright_field_center_px`, `bf_radius_px` and `brightfield_radius_px`
  become `bright_field_radius_px`, `rotation_angle_deg` becomes
  `scan_detector_rotation_deg`, `aberrations` becomes `aberrations_A_rad`
  (Angstrom and radians, the engines' units), `det_sampling_mrad` (one
  number) becomes `detector_sampling_mrad` (`[row, col]`), `det_sampling_rad`
  becomes `detector_sampling_rad`, and `scan_sampling_A` is a `[row, col]`
  pair. Its exported calibration is `quantem.ssb-browser-calibration/2`, with
  `bf_excluded_detector_pixels` as `excluded_detector_pixels`. In the gate's
  report, `num_bf` becomes `num_bright_field_pixels` and the aberration keys
  `C10`, `C12`, `phi12` become `C10_A`, `C12_A`, `phi12_rad`; the trajectory
  record's `bestC10Nanometers`, `bestC12Nanometers`, `c10Nanometers` and
  `c12Nanometers` become `bestC10Angstroms`, `bestC12Angstroms`,
  `c10Angstroms` and `c12Angstroms`. A case or calibration exported earlier
  is exported again.
- The screening cache is `<master stem>.screening-v4.npz` and carries
  `"schema": "quantem.screening-cache/4"` in place of `"version": 3`; a
  version-3 cache is rebuilt.
- A stored SSB fit is read only when it carries the current fit schema:
  `"schema": "quantem.ssb-fit/7"` in the `computed.ssb` block of
  `live/screen/<dataset>/config.json`, written by `SSB.reconstruct_series`,
  and `"schema": "quantem.ssb-result/7"` in a saved result record. Any other stored fit is treated as
  absent and fitted again on first use, including every fit stored by earlier
  versions; none of its values is converted. Until now a block was accepted
  whenever it declared nm, so `reconstruct_series(probe_reference_frame=...)`
  reconstructed with aberrations fitted under the earlier automatic detector
  sampling, which give a different image.
- Native Swift `.qem` readers accept scientific metadata schema 2 only and
  refuse schema 1 with a message that names it, as the Python reader does;
  `NativeQEMMetadataUnits.legacySchema` is removed, and the scaled-uint16 writer
  refuses caller metadata of schema 1. `NativeANSSnapshot.metadata` and
  `NativeQEMMetadataUnits.recordedMetadata` give the flat keys of specification
  0.0.4: `detector_sampling` with `detector_sampling_unit` becomes
  `detector_sampling_mrad` or `detector_sampling_inv_A`, and `semiangle_mrad`,
  `dwell_time_us` and `camera_length_mm` are added. The native writer
  (`saveSnapshot`, `MetalQEMExporter`) no longer copies calibration, source
  metadata or loader state into the retained `metadata` block (`scan_sampling_A`,
  `detector_sampling_inv_A`, `detector_sampling`, `detector_sampling_unit`,
  `voltage_kV`, `source_metadata`, `source_shape`, `axis_order`,
  `lossless_exact`, `pixel_origin`). The native reader drops from the
  retained block of every file what the Python reader drops: the 47 names of
  `NativeQEMMetadataUnits.notRetained` (the flat calibration keys, the
  unit-less detector pair, the copies of `scientific_metadata` groups and the
  loader's host and resident state), every key that starts with `entry/`,
  and a `hot_pixel_correction` record without the current schema.
- `NativeMicroscopeMetadata.beamEnergyKeV` becomes `voltageKV`: it holds the
  accelerating voltage in kV.
- The native SSB engine names the units it computes in. `c10Nanometers` and
  `c12Nanometers` of `MetalSSBAberrations` and `SSBOptimizationPoint` become
  `c10Angstroms` and `c12Angstroms` (the values were Angstrom),
  `MetalSSBHigherOrder.magnitudeNanometers` becomes `magnitudeAngstroms`, and the
  `rotationDegrees:` argument of `MetalSSBEngine.reconstruct`, `phaseVariance`
  and `optimize` becomes `scanDetectorRotationDegrees:`. `MetalSSBSavedRun`
  carries `"schema": "quantem.ssb-saved-run/2"` in place of `schemaVersion: 1`;
  its `rotationDegrees` and `optimizedRotationDegrees` become
  `scanDetectorRotationDegrees` and `optimizedScanDetectorRotationDegrees`, and
  an earlier saved run is refused with the reason.
- The native SSB phase pair (`SSBPhaseArtifact`, `savePair`,
  `MetalSSBSavedRun.savePhasePair`) reads and writes `quantem.ssb-phase/2`, the
  record of the Python reader; version-1 `live.ssb` pairs and single-file
  `.ssbresult` copies are refused. `SSBPhaseArtifact.schemaVersion` becomes
  `schema`, `rows` and `columns` become `shape`, `phaseUnits` becomes
  `phaseUnit`, `c10Nanometers`, `c12Nanometers` and `phi12Radians` become
  `aberrations` (`SSBPhaseArtifact.Aberrations` in nm and degrees), and
  `rotationDegrees` becomes `scanDetectorRotationDegrees` with
  `detectorTransposed`. In `MetalSSBCalibration`, `beamEnergyKeV` becomes
  `voltageKV`, `scanStepRowAngstroms` and `scanStepColumnAngstroms` become
  `scanSamplingRowAngstroms` and `scanSamplingColumnAngstroms`,
  `detectorStepRowMrad` and `detectorStepColumnMrad` become
  `detectorSamplingRowMrad` and `detectorSamplingColumnMrad`, and `centerRow`
  and `centerColumn` become `brightfieldCenterRowPixels` and
  `brightfieldCenterColumnPixels`; its JSON is the record's `calibration` block.

### Fixed

- `io.inspect` of a merged scaled `.qem` returned neither `maped_merge` nor
  `maped_summary`, which `io.load` of the same file returns. It returns both.

- A given rotation kept in [0, 360) (`SSB`, `SSB.set_rotation`, `SSBResult`,
  trial records, `ParallaxResult`, a rotation given to `dpc.run` or
  `screening.prepare`) could read `360.0`: a negative angle smaller than the
  rounding of 360 wraps to the period itself under the plain modulo. These
  angles are wrapped by one function, once, in double precision, and that
  case reads `0.0`; every other value is unchanged, bit for bit. The
  rotation fitted from the centre-of-mass curl is not wrapped.
- A `live/screen/<dataset>/config.json` that `SSB.reconstruct_series`
  created held only `computed.ssb`. The file is quantem.live's screening
  record, read there only with `"schema": "quantem.screen-acquisition/2"` at
  its top level, so the acquisition was left out of the Screen list. A record
  this package creates now carries that key
  (`ssb.persistence.SCREEN_ACQUISITION_SCHEMA`). In a record that exists
  nothing outside `computed.ssb` changes, and a record with another top-level
  schema, or none, raises `ValueError` naming the file, the schema found and
  the one written, before anything is written: a fit is not stored into a
  record of an earlier layout (quantem.live's `tools/rewrite_screen_config.py`
  rewrites such a record once).
- The native SSB phase pair stored the engine's Angstrom under
  `c10Nanometers` and `c12Nanometers`, ten times the value for a reader that
  took them as nm, and the azimuth in radians. The record now states
  `C10_nm`, `C12_nm` and `phi12_deg`, and the rotation as the physical angle
  in [0, 360).
- The SSB fit-trajectory report never matched a Python reference candidate: it
  looked for `C10_nm` in the recorded search parameters, which are `C10_A`,
  `C12_A` and `phi12_deg`. The reference script printed those two values as
  `None` for the same reason.
- `SSB.bright_field_center_px` and `SSBResult.bright_field_center_px` are in
  pixels of the full detector for every input. A session built from an
  encoded acquisition or `SSB.open` reported the centre inside its decoded
  bright-field crop.
- The remote catalogue listed every master that records its scan step as a
  `[row, col]` pair as uncalibrated: it called `float()` on the pair.
  `k_pixel_size_mrad` read a key no reader writes and was always null.
- The remote server ignored a query parameter it does not declare, so a
  client of protocol version 1 (`sx`, `sy`, `cx`, `cy`, `det_bin`, `inner`,
  `outer`) received HTTP 200 with the pattern of scan position (0, 0) or the
  default mask. Every route now refuses an undeclared parameter with HTTP
  400, naming the unknown parameters, the protocol version and the
  parameters the route takes.
- One master whose `scan_sampling_A`, `detector_sampling_mrad` or
  `detector_sampling_inv_A` attribute is not a pair of numbers (one number
  for both axes, as earlier versions wrote) made `/api/browse/sessions`
  answer HTTP 500 for the whole catalogue. That master is listed
  uncalibrated, and its entry states why in the new field `cal_reason`
  (null otherwise).
- A DPC or screening fit made with exchanged detector axes could only be
  handed to SSB as a bare angle, so the exchange was dropped without a
  message. The flag travels with the angle (`detector_transposed`) and SSB
  refuses True.
- The portable SSB phase record wrote the angle below 180 without the
  reversal flag as its rotation, 180 degrees off for a reversed scan, gave the
  detector step as semiangle over a radius other than the one the fit used,
  and copied aberrations from a fit block of any schema (Angstrom under nm
  keys in older blocks). It now exports only a current fit, with its physical
  rotation and the detector sampling the fit used.
- `SSB.open` on an exact bright-field column export (Apple backend) left the
  session's `detector_sampling_mrad` unset while the backend derived one, so
  `SSB.reconstruct_series` stored a fit block with
  `"detector_sampling_mrad": null` and exporting that fit raised `TypeError`.
  The session now holds the sampling the backend uses (the export's
  calibration, else the semi-angle over the export's disk-edge radius) and
  the fit block always holds the pair. A session built from a dense array
  without `detector_sampling_mrad` still leaves the measurement to its
  backend and reports None.
- A DM4 detector step read 1/nm when loaded and 1/angstrom after a `.qem`
  round trip, a factor of 10 with a different unit string. Both now read
  1/angstrom (`detector_sampling_inv_A`); 1/Å and 1/µm axes are read too.
- A caller's `dwell_time_us` and `camera_length_mm` are written to a new
  `.qem`. They were dropped.
- `dwell_time_us` of an HDF5 master is the exposure `count_time` when the
  master holds one, else the frame period `frame_time`, for every detector
  and for a time stored as float32, as the saved copy records it. The loader
  reported the frame period of every master, so the value of an ARINA master
  changed after a round trip (49.6 to 49.5 us).
- After a reciprocal-length calibration override the detector unit read
  `1/Å`; it is `1/angstrom` like every other route (now the key
  `detector_sampling_inv_A`).
- Loading a `.qem` replaced the header's `source_dtype`,
  `file_counts_exact` and `working_counts_exact` with what the load knew (the
  stored dtype, exact), so a copy of a file rewrote them: a uint32 source
  became `uint16`, and counts changed by a pixel correction or by zeroed flag
  markers became exact. The integer, float32 and CPU reference readers now
  return these three keys as the header states them and add none where it
  has none; a copy stores the same values as its original.
- A `.qem` made from an HDF5 master that `io.save` wrote stored
  `NX_class: "NXdata"`, `signal` and `data_file_frame_ranges` in its
  header's `metadata` block, keys the format does not list and no reader
  uses: the master reader copied these layout attributes of the `entry/data`
  group into the dataset metadata. They are no longer copied.
- The two records of a merged multi-tilt result (`quantem_maped_merge_v1`,
  `quantem_maped_summary_v1` in a scaled `.qem` header's `attributes`) came
  back differently by reader: parsed as `maped_merge` and `maped_summary`
  from an HDF5 master, as raw JSON text under `metadata["attributes"]` from
  a `.qem` on an Apple GPU, and not at all from a `.qem` on the CPU. Every
  reader now returns them parsed under `maped_merge` and `maped_summary`;
  `metadata["attributes"]` is no longer returned.
- Calibration overrides were stored twice in one file, in volts and metres in
  `metadata` and in kV and angstrom in `scientific_metadata`. They are stored
  once.

## rc16 - 2026-10-11

### Fixed

- Reading scans that a resident has not yet received, or has already released,
  raises instead of returning uninitialized memory (CUDA streamed and paired
  counts, Metal counts).
- A detector session over a CuPy array that changes in place returns current
  dark-field and centre-of-mass values: every query reads the counts. rc15
  reused per-pattern totals and the field cached at the first call. For an
  array that will not change, `detector.prepare(data, immutable=True)` keeps
  those totals. On 65536 frames of 192 x 192 uint16 on an idle GPU a DF image
  takes 3.4 ms (1.8 ms with `immutable=True`) and a centre of mass 3.1 ms
  (0.01 ms repeated with `immutable=True`).
- Decoding a batch right after freeing earlier decoded batches could return
  wrong frames in arrays still being computed from them: the CUDA upload stream
  now waits for the caller's queued work. `io.load` was not affected.
- The mean pattern and scan-ROI sums and means of float32 acquisitions are the
  float64 total rounded once on CUDA and Metal. They were a float32 running
  total, up to 8 units in the last place off on real data.
- The mean pattern and detector total raise after `release_rows_before` or on a
  partly loaded source instead of dividing the held rows by the full scan
  count. A scan-ROI reduction of a float acquisition raises when a selected
  scan was released.
- Dense detector reductions of signed integer data keep their sign.
- Metal LZ4 decoders bound their reads and writes; damaged blocks raise
  `ValueError`.
- A Metal acquisition dropped without `close()` returns its GPU memory. Metal
  detector sessions on integer frames no longer build a binned, clipped copy in
  the background.
- `io.save` to a narrower integer dtype refuses counts that do not fit. It used
  to wrap on CUDA and CPU and clamp on Metal.
- Schema-2 calibration overrides reopen with their exact recorded values
  (0.35 came back as 0.35000000000000003).
- CUDA decode batches above 4 GiB keep every frame in place. A truncated or
  corrupt LZ4 chunk raises `ValueError` on CUDA instead of decoding to garbage,
  and the CUDA LZ4 decoder bounds its reads as well as its writes: damaged
  bytes inside intact framing used to end the process's CUDA context. The
  frame reader refuses a chunk whose block lengths do not end at the chunk's
  last byte (a file still being written, a bad copy); the unchecked walk read
  past the chunk. The paired loader stops at a block length that points past
  its chunk.
- A failed CUDA HDF5 load frees the chunks it already encoded. Prepared `/dp`
  HDF5 stacks load on CUDA and Metal. uint32 HDF5 datasets with flagged-pixel
  markers load on Metal. `release_rows_before` frees rows of float
  acquisitions. Frame sums of uint16 Torch MPS tensors are exact.
- `parallax.run(fit_aberrations=True)` measures beam angles from the
  bright-field centroid with SSB's detector calibration. Earlier values were
  wrong for any disk not at the detector corner.
- CUDA SSB compiles its FFT kernels per GPU and computes on the acquisition's
  GPU. A fit on a second GPU previously returned wrong aberrations with loss 0.
- A CUDA reduced-BF preview context follows a `set_rotation` made after the
  context was created. On MPS, `set_rotation` with anisotropic `det_sampling`
  keeps every aperture pixel.
- `search_ranges` accepts `[low, high]` lists for `phi12_deg` on CUDA. `SSB`
  accepts NumPy scalar `scan_sampling_A` and `det_sampling`. The tilt fit no
  longer changes the process-wide warning filters.
- Browser: the product-first virtual image of uint16 HDF5 data summed only the
  low 8 bits of each count. Virtual images and centres of mass over masks of
  one to three pixels also counted detector pixel 0. Dense virtual images
  accumulate integer counts exactly and round once. SSB at a bright-field
  count below the loaded one paired resident G(q,k) chunks with the wrong
  geometry. Bitshuffle-LZ4 decoding of frames that are not a whole number of
  blocks (for example 96 x 96 uint16) read the final block with the wrong
  plane stride; frames whose pixel count is not a multiple of 8 are refused
  with a message. The `.qem` centre of mass rounds the exact moment quotient
  once.

### Changed

- `device.release_cached_memory(verbose=False)` returns the gigabytes this
  process handed back to the driver. It collects garbage first, then empties
  Torch's cache, CuPy's cached FFT plans and memory pool on every GPU the
  process has opened, CuPy's pinned pool, and on Apple silicon the Torch MPS
  and MLX caches. It opens no GPU the process has not used. `verbose=True`
  prints one line per GPU. rc15 returned `None` and emptied CuPy's pool on
  the current GPU only. With a 1 GiB tensor in a reference cycle and a 1 GiB
  CuPy or MLX array dropped, it returns 2 of 2 GiB to the driver on CUDA and
  on Metal.
- `parallax.run` takes `scan_sampling_A` plus `semiangle_mrad` or
  `det_sampling`. With `fit_aberrations=True` it returns C10 and C12 in nm,
  phi12 in radians (scan frame) and `ParallaxResult.rotation_angle_deg`, in
  the convention `SSB` takes. Parallax coarse alignment bins the detector
  around the disk centre.
- CUDA `find_aberrations(tilt=True)` repeats bit for bit for a fixed seed, and
  its objective is about 5% faster.
- `movie.save_mp4(backend="auto")` prints one line when it uses the CPU
  writer. `gpu_id` is an explicit keyword, and other keyword options raise.
- `detector.fit_probe` accepts an acquisition and raises `ValueError` when the
  pattern has no disk. `block_stride` is refused for byte-rANS series.
  `device.release_cached_memory()` also empties the Torch caches.
  `io.__all__` lists `DataRepresentation` and `PairedLoader`. Array loads
  record `detector_mask_policy` and `flagged_markers_stored_as_zero`.
  `load_timings` keeps the encode, index and ready times only.
- `SSBSeriesResult.metadata()` returns a DataFrame. MPS BF-column exports
  declare their columns in the calibration.
- The CUDA LZ4 decode kernel takes about 11% longer than in rc15 with its
  write and read bounds (0.407 ms to 0.451 ms for 4096 frames of 192 x 192
  uint16 on an RTX PRO 6000); a whole decode, which the host copy dominates,
  is unchanged within run-to-run noise.
- Browser: the centre-of-mass shader divides at one call site, a restructuring
  intended for Direct3D FXC; values are unchanged on NVIDIA and the FXC
  compile has not been checked. SSB builds resident G(q,k) one chunk at a time
  (build peak 252 MB to 177 MB on the 1793-column gate; setup 742 ms to 471 ms
  on a 256 x 256 scan). The product-first virtual image indexes masked blocks
  in place (ADF on 80000 frames about 500 ms to about 200 ms).
- Packaging: `jinja2` is no longer a dependency. `movie` installs
  PyNvVideoCodec on Linux and Windows. `[dev]` installs
  `opencv-python-headless` and no longer installs `websocket-client`. The
  source distribution includes `scripts/`.
- Documentation: install pages no longer pin rc8; the SSB API states that CUDA
  refuses scans larger than 1024 x 1024; native Swift/Metal SSB is documented
  for 128, 256 and 512 scans; dated records are listed in the navigation.
- Encoded sources keep their exact detector total while loading, so the mean
  pattern no longer decodes the acquisition (about 110 ms to 0.08 ms for
  262144 patterns). The encoded centre of mass decodes each stream once (about
  115 ms to 50 ms). Encoded scan-ROI sums, means and maxima decode only the
  selected scans.
- `io.load` reads the next block while the GPU decodes and encodes (530 ms to
  about 380 ms for 262144 patterns). A `.qem` save writes its body once (a
  2.46 GB save took 5.3 to 6.5 s at best and now takes 1.7 to 2.3 s).
- SSB session build transforms spectra in place and decodes the bright-field
  crop directly. `parallax.run` takes 214 ms instead of 330 ms on a 512 x 512
  scan.
- Detector sessions over bounded views of dense and paired acquisitions reduce
  on the acquisition's own backend; reductions by bounded reads are removed
  (viewer start on a paired 512 x 512 scan: 65 s to 0.14 s).
- Metal scan-ROI sums, means and maxima of encoded counts are computed on the
  device. HDF5 stacks stored as one gzip chunk per frame are inflated on up to
  8 threads.

### Removed

- `detector.virtual`; `io.load(scan_order=...)` and list-valued precision
  `scan_region`; `io.save(compression_level=...)` and the non-LZ4 Arina
  codecs; joint series of dense CuPy arrays; `device.resolve("webgpu")` and
  `Dataset4dstemGPU.residency`; reading version-1 precision exports; the
  `QUANTEM_GPU_HDF5_TRUST_FRAME_SOURCE_CACHE` switch.
- `parallax.run(voltage_kV=...)`, `ParallaxResult.to_ssb_aberrations()` and
  the `rotation_angle_deg()` method. `SSB(backend="webgpu")`.
  `SSBSeriesResult.show()`: quantem.gpu no longer imports quantem.widget.
- Browser: the SSB compressed-HDF5 source path, the rANS export-folder
  loaders, the scan quarter-turn kernel, `webgpu/index.ts`, `dense.ts`, the
  `loadLocalH5Master` alias, the frame-index manifest, DPC magnitude,
  experimental bslz4 decoders and upload routes, and SSB debug globals.
- The benchmark registry (`benchmarks/`), the profile matrix, the
  backend-status generator, the documentation fingerprint manifest and the
  pages generated from them. Measured numbers remain in the documentation as
  dated tables. `scripts/benchmark_webgpu_h5_browser.py`,
  `benchmark_webgpu_h5_product_browser.py`, `webgpu_cdp.py`,
  `benchmark_detector_products.py` and `audit_android_vulkan_fixture.py`.

### Added

- `plot` extra (Matplotlib, IPython, QuantEM) for `SSBResult.show`,
  `SSB.show_trials` and the rotation histograms.
- `scripts/benchmark_ssb.py` and `tests/benchmarks/data_paths.py` time the hot
  paths and print output digests. The rc15 to rc16 comparison is recorded in
  `docs/maintainer/2026-10-11-lz4-read-bounds-and-chunk-walk.md`.
- Headed browser gates for dense HDF5 decode, product-first and resident sums
  against raw counts, SSB bright-field subsets, and the SSB build peak.
- `io.save` returns a result that records the save time, file size and
  uncompressed size of a `.qem` save and displays as a table in a notebook.
- `detector.prepare(data, immutable=True)` for dense CuPy arrays that will not
  change.
- `tests/benchmarks/viewer_stages.py` times load, mean, viewer start-up and one
  interaction on CUDA or MPS. The speed round is recorded in
  `docs/maintainer/2026-10-11-speed-round.md`.

## rc15 - 2026-10-09

- WebGPU loads a served `.qem` through uncached HTTP ranges (`qemHttpFiles`)
  and reads every payload byte once: four 64 MiB chunks are authenticated in
  flight, their read buffers are recycled, and the stream bytes are staged in
  mapped GPU groups of at most 256 MiB that the decoder binds directly. A
  512 x 512 x 192 x 192 uint16 file of 1.19 GB (1.85 GB) is resident in 2.7 s
  (4.1 to 4.2 s) on an RTX PRO 6000 in Chrome, from 4.7 to 5.5 s (7.1 to
  9.5 s) when the whole file was fetched first; every count is unchanged. A
  chunk larger than one group is split on the GPU, and the per-block upload of
  other sources is capped at 256 MiB per group (a 4.29 GB mapping failed on
  Apple Metal).
- `RansResidentSeries` loads a `.qem` series as one resident set per
  acquisition, three loads in flight, after every header has been checked for
  one geometry, dtype and validity mask; a mismatched series is rejected before
  any payload is read. Masked sums batch across separately loaded sets.
  Cancelling or failing a download or series load releases its GPU buffers.
  The series status names each file once (`2/7 t1.qem: Verifying .qem 5/18`),
  and a one-file load adds no `1/1` prefix.
- Resident rANS sets lend their exact uint32 detector sums to a display as
  borrowed views (`imageViewsU32`), and refuse an acquisition whose image does
  not start on the device's storage offset alignment. The view type
  `Uint32ImageView` and `validateUint32ImageView` live in
  `detector/webgpu/borrowed-image.ts`, beside the resident sets that produce
  them.
- quantem.gpu no longer ships a browser display:
  `src/quantem/gpu/display/webgpu` (colormap engine, display statistics, FFT,
  filters and geometry, about 8,300 lines) is deleted, and `webgpu/index.ts`
  no longer exports `GPUColormapEngine` or `createGPUColormapEngine`.
  quantem.widget owns browser display. `webgpu/sources.json` lists no display
  files. The Metal display
  (`display/metal/display.metal`), `display/colormaps.json` and the Python
  `quantem.gpu.display` LUT functions remain.
- `io.save(..., dtype="float16")` raises a `ValueError` that names
  `dtype="scaled_uint16"`: `io.load` refused the float16 export it wrote.
  Saved version-1 single-scale scaled uint16 files still reopen.
- The Android Vulkan sources (`native/vulkan`, about 15,200 lines) are deleted;
  no consumer or CI job built them. The backend registries, `backend_status.py`
  and the documentation list five backends (CPU reference, CUDA, Python MPS,
  native Swift/Metal, WebGPU).
- `quantem.gpu.io.__all__` lists `Dataset4dstemGPU`, the type `load` returns.
  The classifiers declare Python 3.14.
- Removed `PairedFeed`, the double-buffered block iterator over paired-count
  sources; nothing outside its own test used it. `PairedCounts.decode_blocks`
  reads the same 512-scan blocks. Also removed: `SSBSeriesResult.alignment`,
  the WebGPU exports `resetGPUDevice`, `ssbHasLocalSource`,
  `show4DSTEMLocalH5Files` (use `collectShow4DSTEMLocalH5Files`) and
  `WEBGPU_FFT_CONFIGS`, and unreferenced Swift helpers, scripts and harnesses.
- CI type-checks every WebGPU source listed in `webgpu/sources.json`, not only
  the entry point, with quantem.widget's compiler options (`npm run typecheck`,
  including `noUnusedParameters`). The install notes give the Intel Mac and
  Windows on ARM limits of the extras.

## rc14 - 2026-10-08

Includes everything since rc5; rc6 to rc13 were published without rolling this file.

- Exact detector products (`masked_sum_exact`, `reduce_frames_exact`,
  `reduce_frames_max`) on CuPy float arrays raise `TypeError`, as on Torch and
  NumPy; they returned truncated integers (0 for 0.75) before. The float ANS
  centre of mass is the absolute detector centre like every other source, 0 for
  an empty frame, so one empty frame no longer makes the DPC field NaN.
- CUDA SSB refuses scans larger than 1024 positions per side instead of
  center-cropping them, and prints one line when it pads a smaller or
  non-square scan with its mean pattern.
- WebGPU scan-ROI patterns read float32 data as float32 and sum counts exactly
  in 64 bits; a 32-bit sum wrapped above 2^32. `applySlots` reads back adopted
  slots and slots with spare RGBA capacity.
- The CPU HDF5 reference raises when the pixel mask shape differs from the
  frames instead of skipping the mask. The CPU movie writer uses the CUDA and
  MPS grid, so its frames no longer clip the last column of a scaled grid.
- `wait=False` on `frame` and `masked_sum` completes the result on sources that
  do not queue queries; it raised `TypeError` on dense CUDA series before.

- `[cuda]` installs CuPy for CUDA 13 (`cupy-cuda13x[ctk]>=14.0`) with PyTorch 2.11
  or newer, so CuPy and PyPI torch share one CUDA 13 toolkit with uv and pip alike
  (uv picked a CuPy without NVRTC before). CUDA needs an NVIDIA driver 580 or newer.
- MPS opens Arina `uint32` masters with flagged pixels, as CUDA does: the default
  median correction replaces the 0xFFFFFFFF flagged pixels on the GPU before every
  count is checked to fit `uint16`. It raised `NotImplementedError` before.
- PyTorch 2.3 or newer is a dependency: `import quantem.gpu` loads the readers
  that return PyTorch tensors.
- Restructure the package into layers (`device`, `formats`, `resident`, `io`,
  `detector`, `geometry`, `dpc`, `parallax`, `screening`, `ssb`, `remote`), with
  CUDA, MPS and WebGPU code in `cuda/`, `mps/` and `webgpu/` beside the science it
  implements. No `backends/`, `compute/` or underscore modules remain; private
  import paths changed without aliases, public spellings (`io.load`, `io.save`,
  `SSB`, `detector.*`, `dpc.*`, `parallax.run`, `screening.prepare`) did not. The
  native Swift and Android Vulkan sources moved to `native/swift` and
  `native/vulkan`; the Android app and Direct3D folders are removed.
- `io.load` keeps acquisitions ANS encoded on CUDA and MPS; dense data is the
  explicit CPU reference (`backend="cpu", representation="dense"`). Removed: the
  Python packed family (QGPUH5 and QGIX readers, packed residents, `uint4`),
  block-indexed count-rANS, the prepared 66-acquisition series and `source112`,
  `MultiChunkedFrames`, the `scan_indices`, `random_positions`, `drift`,
  `detector_bin`, `output` and `devices` load options, and dtype casts other than
  `scaled_uint16`. Re-export old packed files from their original acquisitions
  as `.qem`. `io.inspect` reports a master whose frame count is not a square scan
  as not ready and asks for `scan_shape`.
- The remote service keeps browsing and saved SSB results; the remote SSB and
  MAPED services, `serve-ssb-mps`, `prepare-browse` and compact residency are
  removed. `SSB` rejects raw residents and never releases data it borrowed;
  screening, centre of mass, DPC and series detector queries run on MPS encoded
  data; parallax and scan rotation take the encoded acquisition.

- Fit the bright-field disk automatically once per encoded acquisition for
  `detector.bf(data)`, `detector.adf(data)` and `detector.df(data)`. Reuse the
  geometry across calls; optional center/radius overrides affect only that call.
  Mutable arrays are refitted so measurement edits cannot leave stale geometry.

- Rename `detector.mean_dp(data)` to `detector.mean(data)` and
  `detector.auto_probe(mean_dp)` to `detector.fit_probe(mean_dp)`, preserving
  the disk estimator. Update callers together; the previous names are removed.
  BF/ADF/DF now use the shared session reduction for ANS sources. Reuse fitted
  geometry across detector calls to avoid repeated mean-pattern reductions.

- Decode selected detector streams directly for CUDA streamed integer ANS
  indexing, preserving exact values while avoiding full-pattern expansion
  for detector-pixel scan images and detector crops.

- Index loaded 4D acquisitions directly with integers, slices and ellipsis to
  obtain GPU tensors, e.g. `data[10, 12]`. Integer indexing now selects scan
  rows rather than tuple fields; access storage and metadata with `.data` and
  `.metadata`. Strides decode their bounding region; advanced indexing is
  unsupported.

- Remove the `ssb.compute` compatibility imports. Remove obsolete test-path
  aliases; test commands now use current paths or suite names.

- SSB C10/C12 values and search ranges now use nanometers at the public API.
  Earlier releases passed angstrom values under nm labels: divide manually
  retained C10/C12 numbers and search bounds by 10 when migrating. For example,
  an old input of 100 now becomes 10 nm. Angles remain radians. Saved fits must
  declare nm and use the current schema; rerun old fits rather than silently
  converting their units.
- CUDA SSB previews support 1x, 2x, 3x, 4x and 8x output sampling, including
  tilt/depth correction with C10/C12. Fitting and diagnostic loss remain on
  the native scan grid. Higher-order aberrations and upsampled MPS/WebGPU
  previews are not supported. Finer sampling does not guarantee finer resolution.
- SSB.open follows the ANS-only GPU acquisition policy. Re-export older
  prepared-packed files from their original acquisitions as .qem; there is no
  packed override. Remove the unreachable packed-loading branch and update
  loading tests to use the supported source formats.
- Fix optional master-path handling in the Swift real-acquisition test so the
  Metal test target compiles.

- Share exact SSB phase and calibration as versioned JSON/NumPy pairs, with
  Python and native Swift readers, source-content matching, checksum validation
  and non-overwriting publication. No reconstruction buffers are exported.

- Recognize EMPAD-G1/G2 float32 exports and supported EMD 1 datacubes with
  versioned metadata provenance; optionally apply a confirmed mean-dark reference
  on Metal without changing original packed measurements.
- Prepare native SSB directly from packed detector columns, report preparation
  progress, preserve calibration and manual higher-order controls in saved runs,
  and use the measured blocked four-row FFT schedule for full-aperture fitting.
  Unqualified persistent-scheduler experiments remain outside production code.
- Native Metal SSB accepts uint8, uint16, and uint32 BF columns without integer
  narrowing, and saves complete versioned reconstructions with calibration,
  optimization history, source identity and image checksums for local reopening.
- Share percentile selection, display-limit conversion, and reusable-buffer
  Metal range/histogram encoding through `MetalImageRuntime`. Mixed integer
  and float batches use one caller-owned command buffer; synchronous helpers
  use the same implementation. Colormap tables remain in `MetalDisplayKernels`.
  Float statistics now expose raw `valueRange` rather than ordered reduction bits.
- Open original little-endian uint32 bitshuffle/LZ4 Arina acquisitions into the
  same lossless packed Metal residents as uint8/uint16, decoding all 32 planes
  in bounded windows with exact UInt64 detector sums and fused uint32 DPC
  accumulation. Wider compressed reads and read-ahead are registered as the
  20260909 uint32 experiments (raw evidence in the private evidence archive);
  big-endian input is rejected explicitly.
- Import validated NXem companion metadata (`_em_metadata.h5`) for Arina
  masters: regular-scan sampling with explicit length units, beam energy,
  convergence angle, camera length and reciprocal sampling, bound to the
  companion's identity for catalog invalidation. Mismatched or unitless
  metadata is reported, not guessed.
- Add `NativeScientificExport` for writing full-resolution scalar planes and a
  UTF-8 JSON metadata record into one new HDF5 file, and `countSummary()` that
  reduces the exact per-scan totals on Metal into one UInt64 without rereading
  the packed 4D payload.
- Match the CPU histogram reference to the Metal bins (`floor(fraction * 256)`,
  maximum in the last bin, constant images in the center bin) and use the
  shader's signed log1p mapping for logarithmic thresholds. A nonempty UInt32
  range of only `UInt32.max` is no longer reported as empty.

- `session.masked_sum(..., block_stride=k)` on a paired native series sums every
  k-th 512-scan block of the mask (every k-th scan row of a 512-wide raster) and
  leaves the other rows of `out` untouched, at about 1/k of the device time. The
  rows written are exact; consecutive queries at one stride build on each other
  incrementally and a change of stride starts from a full plan. A viewer uses it
  to keep every tile moving with a fast detector drag and follows it with one
  exact batch when the pointer pauses.
- The paired residual decoder refills its bit reservoir once per three coded
  pairs from a prefetched window instead of checking before every symbol, and
  the packed warp reduction biases products as it multiplies: 28 percent fewer
  instructions and 26 to 28 percent less device time per detector update on
  recorded centre drags, byte-identical outputs (`docs/performance/data/paired-decoder-2026-09-09.json`).
  Incremental masks whose change touches no index leaf skip the index pass and
  copy the previous sums.
- Add the opt-in paired-count tANS resident layout (`quantem.gpu.resident.cuda.paired`):
  `PairedCounts` codes complete 512-scan blocks with 32 Poisson pair models and an
  adaptive polar interaction index, `detector.prepare` selects the paired query
  kernels automatically, and `PairedCounts.save`/`load` reopen the exact resident
  arrays without decoding. `io.load(..., representation="paired")` streams complete
  uint16 acquisitions (one path or a list) through a direct-I/O loader whose
  shard reads run ahead across files, and reopens saved paired resident forms
  from their `QGPUPAIR` magic. `PairedCounts.decode_blocks` and `PairedFeed`
  hand whole 512-scan count blocks (optionally as float32 amplitudes) to
  reconstruction consumers from a prefetch stream; `io.inspect` reports saved
  paired forms and `PairedLoader.stream` yields public results per acquisition
  for applications; native streamed queries accept `wait=False` with
  `session.finish()` so a viewer keeps the device busy while it plans the next
  mask. The default byte-rANS layout and every existing load path are unchanged.
- Add experimental native EMPAD XML/RAW loading into lossless float32-bit
  packed Metal residents, with full-source parity, compensated BF/ABF/ADF,
  CoM and mean diffraction. Cooperative packing and reductions reuse bounded
  staging; optional source-checksum metadata never replaces full source reads.
  On an Apple M5 (24 GB), overlapped first-use hashing reduced native
  full-resident presentation from 2.11-2.39 s to 1.53-1.61 s for a public
  4.36 GB acquisition, at unchanged 4.36 GB residency. This is not a cold-I/O,
  subsecond-first-open or universal 120 Hz claim. Reproduction and limitations
  are retained as experiment 20260908-empad-first-open in the private evidence
  archive.

- Load original Arina HDF5 acquisitions (bitshuffle uint8/uint16, no crop or
  bin) directly into exact lossless block-packed Metal residents on Apple GPUs,
  with every count round-trip verified and about 2.0-2.5 GB resident per full
  512x512x192x192 uint16 acquisition. Exact detector updates use bounded
  8x8 region sums within the resident's own memory, and multi-resident detector
  updates run as one Metal submission. Packing plans (disposable per-source
  layout metadata) are written off the load thread, so a first open no longer
  pays the write on the load path. Registered the Apple M5 load-capacity
  measurement: a full load is GPU-saturated (two concurrent loads gain only
  24% throughput), so subsecond loading on that device needs a faster decode
  kernel rather than more overlap.

- Route sealed CUDA packed loading and SSB through `io.load`, add source-preserving
  browse-registry preparation, and expose authenticated prepared CoM without
  dense detector expansion or new end-to-end timing claims.
- Retain dense loading alongside direct packed readers, consolidate native
  Metal/Vulkan and WebGPU implementations, and document per-representation
  operation limits. Add packed inspection, raw-reconstruction admission,
  exactness reporting, and cross-language buffer-lifetime regressions without
  changing benchmark claims or silently transcoding between representations.
- Add one backend-neutral 4D-STEM representation API across Python,
  Swift/Metal, WebGPU, and remote receipts. `io.load()` now returns
  `Dataset4dstemGPU`, reports `lossless_packed` or `dense` separately from dtype
  and residency, auto-detects prepared Lossless Pack Format sources, and uses
  the descriptive `detector_bin` spelling while retaining compatibility aliases.
- Add a loopback-only CUDA browse service for native applications, with exact
  virtual-detector and selected-diffraction transport, acquisition monitoring,
  crop/bin admission, automatic whole-dataset placement across multiple GPUs,
  and bounded per-device resident caches. The service has no `quantem.live`
  dependency and ships through the `[cuda,remote]` install extras. A single
  `quantem-gpu-remote` Conda environment is provided for workstation setup,
  and memory admission preserves native `uint32` capacity when an Arina source
  cannot be narrowed losslessly.
- Add drift-aware sparse ptychography batches to `quantem.gpu.io.load()`: a
  shared random position set can be paired with one dense integer or
  fractional `(row, col)` drift field per source, matching the scan grid.
  Raw diffraction patterns remain unchanged and corrected float32 probe
  positions are recorded in `metadata["drift_batch"]`.
- Add `load(..., output="torch")` for direct Torch tensors on CUDA, MPS, and
  CPU, including recursive conversion of multi-dataset results. Add the
  notebook-friendly `quantem.gpu.device.profile()` diagnostic for reporting
  host, Python environment, Torch version, and the resolved compute backend
  without manual printing.
- Organize public compute around strict scientific domains: `io`, `detector`,
  `dpc`, `parallax`, `screening`, `device`, and `SSB`. CUDA, MPS, and WebGPU
  implementations now live below each domain's `compute` or `backends`
  directory; deleted flat APIs and the top-level WebGPU package are not kept as
  compatibility aliases.
- Add native four-byte unsigned (`dtype='uint32'` / `u32`) load and virtual-image
  support across CUDA, MPS, and WebGPU, and add CUDA packed `dtype='u4'`
  output for true 4-bit counts (`0..15`). CUDA and MPS selected sums use wider
  internal accumulation before float32 display output; MPS and WebGPU load paths
  preserve native `uint32` unless the caller explicitly requests `uint8` browse
  clipping. WebGPU product-first low8 sidecars now reject `uint32` sources
  instead of silently dropping high bits. Public `dtype='u4'` now means packed
  two-counts-per-byte storage with exact range audit and CUDA BF/DF/CoM kernels;
  it no longer aliases NumPy's four-byte `<u4` storage token.
- Add MPS cache-miss generation for `screening.prepare()`. CUDA still
  uses the RawKernel reduction path; MPS now streams raw HDF5 row chunks through
  chunk-backed Metal BF/DF/CoM reducers, records timing/memory metadata, and
  matched CUDA on local anonymized real-data agreement: mean DP/BF/DF exact,
  CoM max abs error `7.63e-6`, and matched rotation/radius.
- Add WebGPU GPU-resident DPC row/col and iDPC reducers to the canonical
  domain-owned Show4DSTEM WebGPU engine. The browser path now computes CoM,
  global CoM mean, centered DPC components, and fixed-rotation iDPC in WGSL,
  with direct browser agreement against NumPy/CUDA references and a local
  anonymized full-512 no-bin real-data NVIDIA WebGPU stress run. The latest headed signoff
  reports DPC row/col max abs error `7.63e-6`, iDPC mean abs error `4.70e-6`
  (`3.05e-5` max, float32 FFT tolerance), GPU-resident display medians of
  about `14.9/13.2/13.2 ms`, and full recompute medians of
  `13.7/19.3/22.7 ms` for DPC row/DPC col/iDPC on an RTX PRO 6000
  Blackwell WebGPU adapter after batching the browser FFT passes into one
  command submission. The browser benchmark harness now has a
  `--require-local-profile` guard so local-file timing runs cannot silently
  accept the URL/fetch fallback path.
- Sign off the Show4DSTEM WebGPU local-H5 detector-bin path for explicit
  `detBin=2/4/8` on full `512x512x192x192` and true crop-256 real evidence.
  The WGSL load path now zeroes raw bad detector pixels before binning and
  keeps the binned output free of raw-detector bad-pixel indices. Headed Chrome
  on an RTX PRO 6000 Blackwell WebGPU adapter matched corrected-frame integer
  checksums exactly against the zero-bad-before-bin reference, with full-load
  count-audited low8 page profiles `1.199/1.212/1.106 s` and crop-256
  20-repeat medians `0.774/0.755/0.733 s` with p95
  `0.798/0.813/0.775 s`; native non-low8 `uint16` `detBin=2` was also exact
  at `2.651 s`.
- Sign off WebGPU product-first BF selected-block loading on true
  real-acquisition `1024x1024x192x192` evidence with BF radius `30`. Headed
  Chrome on an RTX PRO 6000 Blackwell WebGPU adapter matched an independent
  Python reference exactly (`max_abs=0`, `mean_abs=0`, mismatches `0`) with
  4-run median wall `4.92 s`, page/profile `4.85 s`, product stage `1.56 s`,
  selected compressed payload `6.88 GB`, and `4.19 MB` output. This is a
  product-first signoff, not full-stack no-bin browser browse/load signoff.
- Harden the WebGPU selected-block staging uploader so reused staging buffers
  are not remapped until the previous submitted copy work has completed. The
  browser product benchmark now validates fixture existence, mounted file
  count, required reference arrays, and product-debug hooks before reporting
  timings.
- Refresh the MPS SSB performance status on a 24 GB Apple M5 reference laptop:
  radius-30
  `512x512` object steering is real-time (`10.86 ms` mean), radius-30 exact
  phase/loss is reviewable but not CUDA-like (`76.28 ms` mean), and full-active
  `512x512` exact phase/loss remains slow (`528.90 ms` mean). The docs now keep
  BF policy, object-wave steering, and exact phase/loss timing separate.
- Record true real-acquisition `1024x1024x192x192` CUDA and MPS HDF5
  load/decode signoffs: no hidden bin/crop, `uint16` output, selected
  corrected frames bit-exact against direct HDF5, `77.31 GB` resident,
  `4.704 s` wall on CUDA and `4.617 s` wall through the chunk-backed MPS path.
- Keep Show4DSTEM browser VI/DPC ownership beside the detector and DPC domains: detector
  and scan mask builders now live with the canonical WebGPU compute source, and
  the widget source-contract tests verify the frontend does not reintroduce
  local BF/DF/DPC mask helper implementations.
- Add a CUDA RawKernel virtual-image backend for resident CuPy uint8/uint16
  4D-STEM data, wire `compute_backend(cupy_array)` to it, and add exact parity
  tests against the old CuPy selected-pixel reduction. Add a
  `virtual_image_kernel_support()` probe plus a maintainer checklist covering
  CUDA, MPS, and WebGPU browser paths, including the future
  `1024x1024x192x192 uint8` target. The CUDA path now uses warp-shuffle
  selected-pixel reducers, a custom total-count reducer, fused dense
  `total - complement` output, and per-viewer detector-index caching. On a
  local anonymized full 512x512x192x192 real-data benchmark, median BF/ADF/DF drag
  latency improved from 4.96/16.16/62.64 ms on the old widget Torch path to
  1.35/3.86/1.84 ms with bit-exact output. On a local seven-tilt detector-bin2
  benchmark, per-panel BF/ADF/DF medians are now 0.54/1.35/0.53 ms with
  max absolute error 0.
- Add a CUDA RawKernel CoM/DPC reducer for resident CuPy uint8/uint16 data.
  The fused kernel accumulates total intensity, detector-row moment, and
  detector-column moment in one detector pass and caches the full-detector CoM
  field per backend. On a local anonymized full 512x512x192x192 uint16 benchmark, DPC
  CoM improved from 200.42 ms to 12.39 ms with max absolute error 0; on a
  local seven-panel detector-bin2 benchmark, first full-grid DPC improved
  from 373.14 ms to 24.63 ms with max absolute error 0, and repeated DPC reads
  use the backend cache.
- Clarify the cross-backend CoM/DPC product-kernel tracker: MPS uses raw Metal
  `com_u8`/`com_u16`, while WebGPU already has WGSL masked CoM source under
  the DPC domain but still needs the same GPU-resident buffer/cache
  parity path as virtual-image dragging.
- Add canonical reusable WebGPU/TypeScript browser compute beside each
  scientific domain. The Show4DSTEM and ShowPtycho browser engines are shipped
  as package data for widget builds.
- Fix MPS SSB fixed-aberration loss reporting for cached 512x512 geometry. The
  cached path now treats the 512 column-kernel sum-of-squares as a scalar, so
  real-data MPS fixed phase/loss and sparse optimizer parity pass against the
  CUDA reference artifacts.
- Add MPS Metal uint8 virtual-image kernels and route
  `load(..., backend="mps", dtype="u8")` through chunk-backed Metal IO, so
  Show4DSTEM browse loads do not materialize a giant Torch-MPS tensor.
- Add the MPS dense-mask `total - complement` cache path so dark-field style
  Show4DSTEM drags use sparse complement reads on Metal, matching the CUDA
  kernel strategy.
- Make CUDA SSB batch variance deterministic for sparse 256/512/1024 row
  transforms, clarify the ShowPtycho UI handoff, and document that WebGPU/WGSL
  runs in the browser while reusable source lives beside its scientific domain.

## rc5 - 2026-07-14

- Add the first documentation site with install/backend tutorials, simplify the
  tutorial language around BF/DF/ADF/DPC, add movie rendering docs, and add a
  backend coverage matrix for CUDA, MPS, CPU, and remaining migration work.
- Add an Apple Metal/MPS MP4 rendering backend so `save_mp4(...,
  backend="auto")` tries CUDA/NVENC, then MPS/Metal, then CPU/ffmpeg.

## rc4 - 2026-07-14

- Correct installed package version reporting so `quantem.gpu.__version__`
  matches the `quantem.gpu` distribution version from TestPyPI installs.

## rc3 - 2026-07-14

- Add `load(path, scan_region=(row_start, row_stop, col_start, col_stop))` as
  the crop-first HDF5 API.
- Move MPS crop-first sparse HDF5 decode and the lazy multi-dataset MPS loader
  into `quantem.gpu.io`, leaving `quantem.widget.multidataset_mps` as a
  compatibility re-export.
- Add real-data CUDA/MPS parity tests for crop-first HDF5 IO and MPS SSB sparse
  optimizer objective checks on a full 512x512 acquisition.
- Match MPS SSB fixed-preview phase output to CUDA's mean-of-per-BF-phase
  contract, tighten real-data phase parity thresholds, and add a fused
  MLX/Metal correction kernel that reduces MPS sparse objective timing from
  about 26 ms/candidate to about 7 ms/candidate on a 24 GB Apple M5 reference
  laptop.

## rc2 - 2026-07-14

- Publish the first `quantem.gpu` release candidate to TestPyPI as the
  multi-backend accelerated STEM package for QuantEM (`cuda`, `mps`, `cpu`),
  with a Quick Start README showing install, device reporting, HDF5 crop load,
  virtual detector products, and widget migration usage.
- Move the HDF5 GPU IO/decompression hot path into `quantem.gpu.io`, including
  CUDA bitshuffle/LZ4 chunk decode, pinned-buffer master loading, scan-region
  crop loading, MPS Metal bitshuffle/LZ4 chunk IO, and CPU reference decode for
  parity.
- Add device policy helpers (`device_report`, `select_device`) and import-light
  lazy exports so `import quantem.gpu` works without CUDA/CuPy installed.
- Move BF/DF/ADF, mean diffraction pattern, masked-sum, virtual image, CoM/DPC,
  and iDPC compute paths into `quantem.gpu`, with parity tests against the
  legacy widget/live paths.
- Move SSB compute APIs from `quantem.live` into `quantem.gpu.ssb`, including
  CUDA reference parity, MPS/MLX preview and C10/C12/phi12 free-fit paths, and
  real-data parity/speed checks used during migration.
- Move MPS chunk-backed product compute and movie export helpers into
  `quantem.gpu`, leaving widget responsible for frontend display/export
  orchestration.
- Wire the `quantem.widget` migration branch to depend on
  `quantem.gpu>=0.0.1rc2`, so widget HDF5 loading and accelerated products can
  call the new package without changing public widget APIs.
- Add release automation for `gpu-v*` tags, TestPyPI trusted publishing through
  `release.yml`, MIT license packaging, and an NVIDIA nvCOMP CUDA LZ4
  BSD-3-Clause third-party notice.
