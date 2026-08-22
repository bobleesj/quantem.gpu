# Revision and change ledger

This ledger separates documentation revisions from the implementation revision
that produced a benchmark. A documentation edit never makes an older timing a
measurement of the current code.

- **Ledger reviewed:** 2026-08-22
- **Integration base:** origin/main `75be74e`
- **Current clean benchmark source checkpoints:** canonical Python/CPU metadata
  and parity `68dbe3a`; canonical native Swift/Metal load `5106ca4`
- **Historical measured checkouts:** consolidated Python MPS `0bc9378`;
  native controlled-source `c0ea444`; retained CUDA and WebGPU rows remain
  bound to the exact revisions shown in the results ledger
- **Documentation branch:** `benchmark-platform-computer-matrix`

The native stack extends exact prepared QH5 binning (`e0e92b4`), optimized
word-major detector binning (`ff3c7fd`), and exact resident summaries
(`d65911a`) with bounded streaming, private residency, and controlled macOS
source-page measurement through `c0ea444`. It does not relabel an older CUDA,
Python MPS, WebGPU, source-load, or application measurement.

## Latest documentation changes

| Commit | Change | Performance-number effect |
|---|---|---|
| `68dbe3a` | Normalized source and working detector metadata, then reran canonical CPU/Python MPS bins 1/2/4/8 | replaces the `0bc9378` headline with exact full-volume current rows: MPS p50 0.406624/0.477740/0.370645/0.340210 s; cache/source-page states remain explicit and none is arbitrary-source cold |
| `5106ca4` | Added the canonical cross-layout logical-pixel contract used by native Swift/Metal evidence | adds current physical M5 24 GB bin2/bin4 distributions and one pressure-gated full-native bin1 smoke; the bin1 smoke is partial because it caused 758,448,128 B of swapouts |
| `0bc9378` | Consolidated exact MPS bin1/2/4/8 kernels, overflow and partial-tail guards, resident reuse, and exception-safe cleanup | adds final-head post-warmup fresh-destination p50 0.414824/0.457153/0.382109/0.356258 s for bins 1/2/4/8; controlled cold and application E2E remain open |
| `e4a35f9` | Adopted the accepted exact WebGPU integer CoM/DPC/iDPC source without importing obsolete campaign harnesses | no new timing claim; preserves the existing physical-Apple numerical evidence at the consolidated source boundary |
| `f0f39c9` | Added scratch-free exact full-`uint16` decode, fused exact detector-sum kernels for bins 2/4/8, bin-2 specialization, lazy LZ4 scratch, and source-shard-aligned pipelining | historical pre-consolidation MPS rows remain retained; they no longer headline current source |
| Current MPS lifecycle-audit follow-up | Separates resident payload, driver allocation sampled after load and release, process RSS, pressure, and swap | replaces the pressure-contaminated 2026-08-19 MPS load headline with revision `be035c4` fresh-process and explicit-release warm-process rows; no loader arithmetic changed |
| `c0ea444` | Aligned the raw benchmark state with explicit macOS `F_NOCACHE` control and sealed seven replacement runs | promotes a controlled uncached-source-page full-native package row; its distribution replaces schema v7 only because the older raw state was contradictory, not because the IO path changed |
| `00ba8bd` | Reused the immutable source identity while preserving controlled source reads | kept the same exact source/load boundary; the later `c0ea444` run owns durable timing because its state label is internally consistent |
| `3bb3845` | Added exact full-native detector-bin-1 loading into private Metal residency | established the 18 GiB exact resident path; later controlled-source runs own the current distribution |
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
| `20260822-mps-load-sub500ms` | Promoted for current Python MPS exact warm-process and independent-process loading | Four-bin full-output parity is byte exact. ABBA measurements preserve process/source state and explicit release. Bins 2/4/8 meet strict p50 <=0.5 s; full native bin1 is 0.523 s. Source pages remain warm and uncontrolled, so the result is not cold storage or application E2E. |
| `20260822-mps-load-memory-audit` | Promoted for Python MPS process-state and memory reporting | Seven fresh processes per detector bin and seven explicit-release warm-process repetitions preserve exact selected-frame hashes, record logical payload plus driver allocation sampled after load and release, process RSS, pressure, and swap, and pass independent exact detector-bin/product parity. Source pages were warm and uncontrolled, so neither protocol is cold-storage evidence. |
| `20260819-platform-profile-mps` load headline | Superseded, preserved | Its repeated-load harness cleared the Torch cache but did not call `MPSChunked4DSTEM.free()` on direct PyObjC Metal outputs. Full bin1 accumulated roughly 18 GiB per repetition, free memory fell from 86% to 41%, and p50 rose to 2.273 s. The artifact remains useful as pressure-failure evidence, not current loader timing. |
| `0822-97-fnocache-provenance-v8` | Promoted as controlled native package evidence | Seven fresh processes use a new index root, fresh private destination, and explicit `F_NOCACHE` source descriptors; volume and product hashes are exact. It is an audited-source package boundary, not arbitrary-source cold or application E2E, and the detailed result remains above the requested target. |
| `0821-96-bin1-controlled-cold` schema v7 | Superseded, preserved | Its command applied `F_NOCACHE` but the raw state token said source pages were unspecified. The timing and raw trials remain retained; schema v8 replaces it only for state-consistent reporting. |
| `20260819-air-exact-resident-summary` | Promoted as separate native Swift/Metal prepared-product evidence | Seven fresh-process reopens reproduce nine same-device products byte-for-byte on MacBook Pro (M5 Max, 128 GB) and MacBook Air (M2, 8 GB). Summary creation, prepared reopen, resident load, compressed-source load, and GUI paint remain distinct boundaries. |
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
