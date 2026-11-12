# Native Direct Ptycho: complete-workflow optimization study

This local development study targets loading through a displayed, checked
fifth-order reconstruction. It has not established a universal few-second
workflow. Earlier preparation gains are recorded separately in the
[preparation study](2026-11-05-direct-ptycho-preparation.md).

## Workload and evidence

The physical device is an Apple M5 Max MacBook Pro with 128 GB unified memory,
running macOS 26.4. Three acquisitions retain all 512 × 512 scan positions and
192 × 192 detector pixels, original counts and calibration, all active BF
evidence, 200 initial trials, exact-derivative Newton fitting and fifth-order
refinement with the existing half-band consistency check. No tolerances were
relaxed. Regularized Phase and Complex retain their existing CPU float64
linear solve; this is a native Swift/Metal pipeline, not an all-GPU claim.

Sources start at backend `a08cad242afe25e956ba0a1447865b1d1c4dce31`
and application `a4968550db09c0a0c35be87fef8df0f3f6b1e243`, with local
changes. Exact source manifests, binary hashes, commands, control snapshots,
images and raw receipts belong to private experiment
`direct-ptycho-eight-hour-20261106`. These results do not describe a released
package or an installed colleague build.

## Final complete-workflow observations

The final local artifact was also exercised with original HDF5 inputs, from
launch through a confirmed exact Phase drawable after fifth-order validation.
Each acquisition had three fresh-application observations, with operating-system
caches uncontrolled. All three reconstruction methods matched their retained
image hashes exactly.

| Acquisition | Launch to checked Phase, median | Observed range | Run to checked Phase, median |
|---|---:|---:|---:|
| A | 10.30 s | 10.11–10.40 s | 8.31 s |
| B | 13.75 s | 9.34–14.30 s | 11.79 s |
| C | 11.90 s | 10.71–12.19 s | 10.01 s |

Adjacent original/final/original journeys gave final-to-control ratios of
1.025, 0.997 and 0.992 for A, B and C respectively, using the mean of the two
surrounding controls. These small samples establish no consistent speedup.
The complete-workflow target remains unmet. A nominal thermal state and AC
power were recorded, but they do not prove constant clock rates or an idle
desktop.

The all-method journeys had final process lifetime peak footprints of 18.93,
28.30 and 19.08 GiB, compared with 19.15, 28.59 and 19.23 GiB for the neighboring
original controls. These include CPU and Metal/driver accounting; they are not
GPU-only allocation peaks and do not qualify a physical 16 GB device.

## Final QEM and saved-result observations

Three fresh-app QEM journeys per acquisition used the same final artifact,
full counts, recorded calibration, 200 trials and fifth-order validation.
The endpoint was a positive native Phase drawable; OS file caches were not
controlled. All Phase hashes matched the original HDF5 workflows.

| Acquisition | Launch to checked Phase, median | Observed range | Run to checked Phase, median |
|---|---:|---:|---:|
| A | 6.50 s | 6.49–6.57 s | 5.07 s |
| B | 8.71 s | 8.69–8.86 s | 7.24 s |

These are format/workflow observations, not an interleaved original-versus-final
QEM comparison or evidence of a code-change speedup. The few-second target is
still unmet.

A later adjacent HDF5/QEM/HDF5 check on the same final app measured
7.13/6.50/7.08 s for A and 9.53/9.24/9.86 s for B. QEM launch/setup to
preparation was 1.50–1.55 s, versus 2.09–2.24 s for HDF5. Fitting times
were similar within these adjacent observations. The later HDF5 runs were
also faster than the earlier series, so its larger historical gap cannot be
attributed to QEM alone. The desktop/process state was not isolated enough
to assign the runtime variation to a particular cause.

A two-acquisition QEM batch completed in 13.13 s, preserving both retained
Phase hashes and original saved-result bytes. Opening a saved result took
0.61 s to a positive drawable; restarting the app and restoring it took
1.01 s. Neither operation started another fit. These are single observations
of scientific-result reuse, not new-data reconstruction timings.

## Native stage observations

The following stage observations were recorded earlier in the study and are
not the final-artifact observations above. Fresh application caches were used.
Operating-system file caches were not
evicted, so these are not cold-disk measurements. The normal desktop remained
active. Native production-controller actions drove visible windows; positive
Metal drawable receipts establish the stated presentation boundaries. This is
not a physical mouse or keyboard acceptance test.

| Acquisition | Input | Launch to preparation start | Preparation start to initial checked SSB | Preparation start to checked fifth-order SSB | Subsequent exact Phase calculation |
|---|---|---:|---:|---:|---:|
| A | QEM | 1.46 s | 2.43 s | 4.67 s | 0.24 s |
| B | QEM | 1.55 s | 3.89 s | 7.05 s | 0.45 s |
| C | HDF5 | 1.90 s | 4.08 s | 7.66 s | 0.24 s |

These are individual stage traces, not warm medians or matched speedups. Phase
was selected after the checked SSB image. Its calculation time cannot be
substituted for an independently observed launch-to-Phase presentation time.
Fitting and validation account for most remaining latency. Removing a
consistency check would change the scientific workflow, not optimize it.

## Retained changes and limits

- Fourier preparation workspaces shrink after preparation and grow again when
  required. A matched full-size fit saves approximately 136 MiB while retaining
  exact outputs; no general latency improvement is claimed.
- Progress publication is bounded to 30 Hz in the application. First, stage
  and completion updates remain immediate; scientific computation and
  cancellation checks are preserved.
- Rotation and band-mask changes invalidate retained derivative values while
  preserving reusable workspaces. Reusing values from a different model would
  be incorrect.
- Large irregular scan axes use the qualified Bluestein preparation described
  in the [kernel registry](../../kernels.md). The existing 2e-6 error bound is
  retained. A 1,020-case sweep passed, but some irregular sizes are slower;
  this is an accuracy fix.
- An indexed Fourier-evidence view remains opt-in. It saves approximately
  2.9–5.1 GiB during fitting but has mixed latency, including a small regression
  on one acquisition. It is not the faster default.

Alternative candidate threadgroup sizes, loop unrolling, inactive-phasor
branches, flat preview accumulation and bounded CPU solves were compared with
adjacent controls and rejected where they provided no consistent gain. Exact
scientific output alone is not sufficient to accept a performance change.
Runtime drift was visible; one unusually fast repetition was not promoted.

## Count preservation and interaction

All three original acquisitions passed independent SHA-256 comparisons of
every decoded resident count: 9,663,676,416 canonical little-endian uint32
values each. Saved QEM controls and a newly exported QEM also passed against
their original HDF5 sources. BF/ABF/ADF were exact, CoM/DPC differences were
zero and iDPC retained its existing tolerance. Earlier sampled-DP checks are
not described as complete count verification. The exhaustive audit takes
about 134 seconds per acquisition and is outside loading benchmarks.

The new QEM's native export/quit/reopen journey preserved calibration and
microscope values. Final-artifact export and reopening times were 1.29 s and 0.81 s;
those are separate operations, not reconstruction time. Physical encoded
allocation sizes can differ while decoded counts remain identical.

Native Phase/Complex slider journeys recorded approximately 45–58 displayed
frames per second in most tested cases. The larger 4× Complex reconstruction
of acquisition B reached about 24 frames per second; its exact release image
took 0.46–0.57 s. Preview-window construction is additional work. A small
kernel's throughput does not establish a 60 fps complete viewer.

The real rectangular 90 × 68 scan and a finite region of a real EMPAD XML/RAW
acquisition also completed fifth-order reconstruction and saved-result
reopening. Explicit, documented calibration was used where the source lacked
it. Nonfinite source values and half-band disagreement remained visible;
neither was hidden to obtain a successful screenshot.

Injected critical-memory-pressure recovery preserved displayed results and
calibration. It does not qualify this workload on a physical 16 GB computer.
Release packaging, notarization, physical interaction gates and other devices
are outside this study.

## Reduced-budget failure control

A small synthetic Python/MPS case with known C10 = -8 nm illustrates why this
study preserves 200 trials. Starting from the wrong 180-degree branch, mapping
the opposite branch after only 40 primary trials returns C10 = +16.797 nm and
agreement 0.2918. The independent 40-trial-per-branch control returns -7.966 nm
and 0.9815. With 200 trials, mapped and independent fits recover -7.967 and
-7.965 nm, within the original defocus tolerance. The failed reduced-budget
result is retained in the private evidence; no numerical tolerance was relaxed.
These are low-order synthetic diagnostics, not full native-workflow timings or
proof of unique specimen tilt and depth.

## Follow-up boundary

The primary candidate evaluator already batches candidate dispatches into one
command buffer and waits once. Its final band reduction intentionally uses
CPU Double arithmetic. Moving that reduction to a float32 GPU tree would
change precision and summation order, so it is not an equivalent optimization.

The half-band checks mutate model weights and invalidate retained derivative
values before and after each mask. They cannot safely run concurrently on the
same mutable model. A future separate-state evaluator must demonstrate bounded
workspace lifetimes, identical numerical outputs and actual full-workflow
speedup; it is not implemented or qualified by this study.
