# Revision and change ledger

This ledger separates documentation revisions from the implementation revision
that produced a benchmark. A documentation edit never makes an older timing a
measurement of the current code.

- **Ledger reviewed:** 2026-08-19
- **Integration base:** `334b7b5`
- **Current combined local stack before this follow-up:** `e052dfb`
- **Current measured checkout:** `8c47a466` (source tree `c3094dcf`)
- **Documentation branch:** `platform-parity-profile-integration`

The current profiling-registry follow-up changes documentation, machine-readable
run policy, and CI validation. It does not change production Python, CUDA,
Metal, Swift, or WebGPU kernels or relabel an older measurement.

## Latest documentation changes

| Commit | Change | Performance-number effect |
|---|---|---|
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

| Evidence | Latest retained interpretation | Why it changed or remained separate |
|---|---|---|
| `PLATFORM-PROFILE-2026-08-19` | Added current CUDA, Python MPS, native Swift/Metal, WebGPU, and CPU-reference rows with atomic p50/p95/max, memory, dtype/bin, and parity fields | replaces older overview values only when the operation, cache boundary, source shape/dtype, and load plan are equivalent; fixtures C and D remain explicitly separate |
| Exact streamed screening follow-up | MPS full-scan mean DP and CoM are byte-exact; BF/DF are value-exact across public `float32` and reference `uint64`; build `6.711 s`, validated warm reopen `20.803 ms` | replaces the faster `3.451 s` first-chunk-mask candidate because that candidate failed full-scan parity |
| Deterministic CUDA SSB calibration follow-up | 200 seeded TPE trials plus Nelder–Mead are byte-deterministic across three full fits; p50 `11.168 s` | replaces the `8.096 s` atomic-objective headline because identical seeds produced two fitted minima; the slower deterministic result is the accepted claim |
| Current MPS prepared companion | rejected; raw detector-bin-2 phase/loss instead passes CUDA at `1.2815e-6` rad and `7.45e-9` loss error | the stored companion columns do not match the declared detector-bin coordinate grid, so its faster reconstruct timings are not scientific evidence |
| Current WebGPU profile | full-scan load bins 1/2/4/8 and resident detector/SSB compute are measured on Chrome 151 with Apple M5 Max Metal-3 | supersedes comparable July overview timings; UI paint, physical 8 GB signoff, per-pixel CoM/DPC/iDPC error arrays, and calibration remain explicit gaps |
| `THREE-HOST-512-U16-2026-08-19` | Added one exact-revision, byte-identical-fixture matrix for CUDA GPU 0 and Python MPS detector bins 1/2/4/8, plus release Native Swift/Metal diagnostics | replaces stale representative overview rows only where the full-scan source/dtype/bin/cache boundary is comparable; differently configured July and physical M2 Air rows remain historical or separate |
| Current detector-bin-4 products | Integer mean/total/BF/DF and Apple-to-Apple CoM/iDPC are byte-exact; CUDA CoM passes `1e-5`; CUDA-versus-MPS iDPC is blocked at `2.84e-5` maximum error | performance is no longer allowed to inherit a check mark from integer parity when the phase product fails |
| Native-detector MPS CoM/iDPC | public automatic detector-bin-2 interaction sidecar is blocked as native-resolution evidence; direct full-resolution Metal CoM passes in `83.3 ms`, while iDPC remains blocked | the newer exact diagnostic distinguishes the full-resolution kernel from the faster changed-sampling sidecar |
| `M2-AIR-BIN4-E2E` | Added as physical 8 GB M2 Air first-process application evidence at measured revision `2c047160`; integration revision `e662d7fe` | newer Apple application evidence, but detector bin 4 makes it incomparable with no-bin library rows |
| `CUDA-STOCHASTIC-IO` | labeled first-process, not cold | the storage-cache eviction procedure was not retained |
| `WEBGPU-VISIBLE-512` | labeled a single visible run | no distribution was recorded, so it is not a median |
| July SSB CUDA and MPS rows | timing unchanged; device restored as RTX PRO 6000 Blackwell and Apple M5 `Mac17,2` respectively | the same-revision SSB performance record and frozen MPS fixture retain the hardware identity |
| Other July MPS/CUDA rows with missing host identity | retained as historical diagnostics | numerical evidence remains useful, but incomplete hardware provenance prevents release signoff |

The authoritative row-level values remain in
[Verified benchmark results](results.md). Accepted and rejected performance
experiments remain in the
[optimization ledger](../maintainer/backend-optimization-matrix.md).

## Adding a new revision

1. Add the complete benchmark row with date, exact measured revision, physical
   device/runtime, source shape/dtype, cache state, crop/bin/load plan, timing
   boundary, memory, calibration, and parity artifact.
2. Explain whether it replaces a truly comparable row or remains separate.
3. Update the dashboard only after the detailed row is complete.
4. Update the evidence manifest when a fingerprinted evidence page changes.
5. Keep rejected or incomparable experiments discoverable rather than deleting
   them.
