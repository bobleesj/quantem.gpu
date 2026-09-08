# Incremental EMPAD detector integration

The full public 256×256 scan now updates wide ABF/ADF images faster by
integrating only pixels entering or leaving the mask. **120 Hz everywhere is
not achieved.** The final two-acquisition native test measured:

| Gesture | Full-integration baseline | Incremental final |
| --- | ---: | ---: |
| Wide ABF move | 40.2 Hz | 70.3 Hz |
| Wide ABF resize | 39.3 Hz | 106.9 Hz |
| Wide ADF move | 30.1 Hz | 48.6 Hz |
| Wide ADF resize | 30.2 Hz | 73.4 Hz |

These are observed native presentations, not an identical-power statistical
benchmark. Isolated incremental runs reached roughly 53–56 Hz for wide ADF
movement. Selected DP remained approximately 120 Hz. Exact intervals, failed
118 Hz gates and build identities are retained in [result.json](result.json).

## Accepted implementation

The backend keeps a compensated high/low float pair per scan position:
524,288 bytes for this source, not another 4D volume. It reuses only a completed
command's sum. Incomplete commands, large mask changes and periodic rebasing
trigger a complete integration. Nonfinite previous sums recover by integrating
the current mask. Allocation-budget checks can use the stateless full-sum path
instead. Default execution uses 128 threads per scan position; experiments can
select the full-sum control with `QGPU_EMPAD_INCREMENTAL=0`.

All 1,073,741,824 original samples remain bit-exact. Product references keep
1e-6 relative and absolute tolerances. Fourteen synthetic cases and 96 real
aperture changes pass, including nonfinite removal and signed cancellation.
Eight sequence steps check every scan position; the other 88 check 128
distributed positions against independently summed original float64 samples.

## Rejected variants and regression checks

- 32 threads were slower than 128 in the native A/B/A sequence.
- 256 threads did not consistently improve the AC-powered comparison.
- A compensated five-stage SIMD tree passed parity but did not materially
  improve center movement. Its implementation was removed; its source patch,
  measurements and hashes remain in the local experiment archive.

The final ARINA single-source native run passed 119.3–120 Hz with 8.33 ms p95
intervals. Two seven-source tests completed 30 navigation commands exactly
once, with seven residents totaling 14.75 GB and selected DP at 120 Hz.
Wide comparison phases varied from 80.2 to 113.6 Hz; earlier runs varied from
61 to 116.5 Hz. That is not a controlled zero-regression proof or a 120 Hz pass
for seven-way wide detectors. No ARINA kernel was changed by this experiment.

The application still requires its experimental EMPAD feature flag. Installed
releases, dependency pins and repository remotes were not changed.
