# Revision and change ledger

This ledger separates documentation revisions from the implementation revision
that produced a benchmark. A documentation edit never makes an older timing a
measurement of the current code.

**Ledger reviewed:** 2026-08-19  
**Implementation baseline for this documentation branch:** `origin/main` at
`4f89e08`  
**Documentation branch:** `developer-centered-documentation`

The documentation commits after `4f89e08` change navigation, API explanation,
notation, equations, and evidence presentation. They do not change production
Python, CUDA, Metal, Swift, or WebGPU kernels.

## Latest documentation changes

| Commit | Change | Performance-number effect |
|---|---|---|
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
| `M2-AIR-BIN4-E2E` | Added as physical 8 GB M2 Air first-process application evidence at measured revision `2c047160`; integration revision `e662d7fe` | newer Apple application evidence, but detector bin 4 makes it incomparable with no-bin library rows |
| `CUDA-STOCHASTIC-IO` | labeled first-process, not cold | the storage-cache eviction procedure was not retained |
| `WEBGPU-VISIBLE-512` | labeled a single visible run | no distribution was recorded, so it is not a median |
| July MPS/CUDA/SSB rows with missing host identity | retained as historical diagnostics | numerical evidence remains useful, but incomplete hardware provenance prevents release signoff |

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
