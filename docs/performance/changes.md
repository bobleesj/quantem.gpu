# Revision and change ledger

This ledger separates documentation revisions from the implementation revision
that produced a benchmark. A documentation edit never makes an older timing a
measurement of the current code.

- **Ledger reviewed:** 2026-08-20
- **Integration base:** `334b7b5`
- **Current clean local stack:** `d65911a`
- **Current measured checkout:** `8c47a466` (source tree `c3094dcf`)
- **Documentation branch:** `mps-subsecond-pipeline`

The current follow-up adds exact prepared QH5 binning (`e0e92b4`), optimized
word-major detector binning (`ff3c7fd`), and exact resident summaries
(`d65911a`) after the native Swift/Metal SSB product (`e1da9bc`). It does not
relabel an older CUDA, Python MPS, WebGPU, source-load, or application
measurement.

## Latest documentation changes

| Commit | Change | Performance-number effect |
|---|---|---|
| `d65911a` | Added provenance-bound exact resident summaries and overflow-safe `uint64` detector moments | adds separate one-time summary-build and prepared-reopen rows; does not replace first compressed-source load or resident-compute rows |
| `e1da9bc` | Added `MetalSSBKernels`, exact 512×512 reconstruction/loss, deterministic 200-trial TPE plus Nelder–Mead fitting, focused tests, and a standalone benchmark | adds separate native Swift/Metal SSB rows; does not replace differently configured CUDA, Python MPS, or WebGPU rows |
| Current profiling-registry follow-up | Added the 35-cell platform/module schedule, human run index, manifest validator, CI gate, and continuous-profiling guide | none; the existing timings and scientific acceptance states are unchanged |
| `e052dfb` | Combined current platform profiling evidence with the three parity-qualified scientific fixes | existing measured baselines remain tied to their recorded revisions |
| `146238a` | Clarified that the SSB PyTorch functions are executable teaching references, not production kernels | none |
| `d83c8a1` | Rewrote SSB examples with explicit named layouts and ordinary PyTorch FFT calls | none |
| `7182ce4` | Made the SSB reference progressive and executable from top to bottom | none |
| `a319507` | Added equation-paired PyTorch SSB reference functions | none |
| `5a897d9` | Added the implementation and benchmark dashboard | no measurement changed; existing evidence was indexed |
| `f16ad05` | Completed operation contracts, source maps, and parity gates | none |
| `6502ed0` | Turned API pages into typed integration contracts | none |
| `de5050c` | Separated kernel implementations from remote compute | none |
| `c7cfccd` | Standardized public `(row, column)` notation | none |
| `a25ab84` | Reorganized the site around scientific operations and runtimes | none |

Local documentation commits are recorded as branch-relative identifiers until
they are published. Do not construct public GitHub commit links for an
unpublished revision.

## Latest benchmark-evidence changes

This table records promotion decisions, not another copy of the measurements.
The current values and distributions live only in
[Verified benchmark results](results.md).

| Evidence change | Current disposition | Why |
|---|---|---|
| `20260819-air-exact-resident-summary` | Promoted as separate native Swift/Metal prepared-product evidence | Seven fresh-process reopens reproduce nine same-device products byte-for-byte on Phil and the physical 8 GB M2 Air. Summary creation, prepared reopen, resident load, compressed-source load, and GUI paint remain distinct boundaries. |
| `20260819-native-metal-ssb` | Promoted as a separate native Swift/Metal SSB row | It uses its own full-BF fixture and optimizer implementation; warm prepared compute is not raw-source load, application wall time, or physical 8 GB signoff. |
| `PLATFORM-PROFILE-2026-08-19` | Current cross-platform profile | It supplies atomic timing, memory, dtype/bin, device, date, and parity fields; fixtures C and D remain explicitly separate. |
| Exact streamed screening follow-up | Promoted | It derives masks from the complete detector sum and transparently reruns BF/DF when the provisional mask differs. The first-chunk-mask candidate failed full-scan parity and remains rejected. |
| Deterministic CUDA SSB calibration follow-up | Promoted | Three seeded fits reproduce fitted parameters, phase, object, and loss. The earlier atomic-objective run produced two fitted minima and remains rejected despite being faster. |
| Current MPS prepared companion | Rejected | Stored columns do not match the declared detector-bin coordinate grid; only the raw detector-bin-2 reconstruction retains current numerical evidence. |
| Current WebGPU profile | Promoted for the measured load and resident-compute boundaries | UI paint, physical 8 GB signoff, per-pixel CoM/DPC/iDPC error arrays, and calibration remain explicit gaps. |
| `THREE-HOST-512-U16-2026-08-19` | Superseded for current headline timing | It remains useful same-fixture history, but newer comparable profile rows own current values. |
| Detector-bin-4 products | Mixed | Integer products and CoM pass their stated gates; cross-backend iDPC remains blocked and cannot inherit an integer-parity check mark. |
| Native-detector MPS CoM/iDPC | Blocked as native-resolution evidence | The public interaction sidecar uses detector bin 2; direct full-resolution CoM passes, while iDPC remains blocked. |
| `M2-AIR-BIN4-E2E` | Separate application evidence | It proves a physical 8 GB detector-bin-4 application path, not a no-bin library load or native SSB memory gate. |
| `CUDA-STOCHASTIC-IO` | Historical, labeled first-process | The storage-cache eviction procedure was not retained, so it cannot be called cold. |
| `WEBGPU-VISIBLE-512` | Historical single visible run | No timing distribution was retained, so it is not a median. |
| Other July rows with missing host identity | Historical diagnostic | Their numerical evidence can guide optimization, but incomplete hardware provenance prevents current release signoff. |

Accepted and rejected performance experiments remain in the
[optimization ledger](../maintainer/backend-optimization-matrix.md); older SSB
layout experiments remain in the
[SSB performance history](../maintainer/ssb-performance.md).

## Adding a new revision

1. Add the complete benchmark row with date, exact measured revision, physical
   device/runtime, source shape/dtype, cache state, crop/bin/load plan, timing
   boundary, memory, calibration, and parity artifact.
2. Explain whether it replaces a truly comparable row or remains separate.
3. Update the dashboard only after the detailed row is complete.
4. Update the evidence manifest when a fingerprinted evidence page changes.
5. Keep rejected or incomparable experiments discoverable rather than deleting
   them.
