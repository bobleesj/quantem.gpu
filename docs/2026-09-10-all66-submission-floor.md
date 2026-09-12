# All-66 Metal submission floor

Experimental measurement, not a native-app speedup or release.
Base revision: abff4e0; exact v22/v23 source snapshots are sealed by the linked manifests.

Same 66 full-resolution entropy acquisitions, 512×512×192×192 uint16 each,
no crop or binning, unchanged encoded source and exact tile index.

## Findings

Counterbalanced diagnostic protocol, 18 measurements per mode, four-second idle:

| Mode | Median wall ms | p95/max ms |
|---|---:|---:|
| Resource declaration, initialization/index, tiny probe; no entropy decode | 505.6 | 544.9 |
| Selected dense compressed-word scan; no ANS/sparse decoding | 518.0 | 549.2 |
| Full exact query | 515.3 | 557.2 |

This supports a large cost outside entropy arithmetic. It does not measure
physical cache hit rates or establish an additive decomposition of overlapping
driver and GPU stages. The diagnostic modes never publish scientific images.

A separate fresh-process scientific A/B/A tested one serial compute pass with
the same 17 grouped64-record grids, source/destination resources declared once:

| Arm | Median wall ms | p95/max ms |
|---|---:|---:|
| A1 normal 17 encoders/commands | 415.5 | 456.8 |
| B one compute pass | 588.7 | 728.1 |
| A2 normal 17 encoders/commands | 422.4 | 468.7 |

The candidate is 39–42% slower and stays disabled. It is distinct from the
previously rejected one-command/17-encoder experiment.

All 54 scientific queries in A/B/A match all 66 complete reference image hashes.
Frozen Linux BF/ABF/ADF and six DP fingerprints also pass. Allocated Metal bytes
remain 86,860,316,672; swap delta is zero; teardown returns to 720,896 Metal bytes.
Release build: 242 tests executed, 27 explicit opt-in skips, zero failures.

The first diagnostic run had an incorrect source-only/total-residency assertion,
was stopped, and remains preserved. The corrected diagnostic run passed.
No source precision, memory representation, loading or app policy was relaxed.

## Open gates

- 300 ms exact-return p95 remains unmet; native publication was not retested.
- Prepared entropy with unspecified source-page state is not cold HDF5 loading.
- Background desktop activity and inherited nice5 were preserved.
- A future diagnostic can omit encoded-source declarations only in a no-read,
  no-publication probe while retaining all 66 allocations. It has not been run.
  It may distinguish source-resource residency/validation from dispatch/index cost.
- Hardware GPU L2/L3 counters remain unavailable; no cache-hit claim is made.

## Registry

- [Preserved failed v21](../experiments/20260910-all66-submission-floor/manifest.json)
- [Successful diagnostic v22](../experiments/20260910-all66-submission-floor-r2/manifest.json)
- [Rejected scientific candidate v23](../experiments/20260910-all66-single-compute-pass/manifest.json)
