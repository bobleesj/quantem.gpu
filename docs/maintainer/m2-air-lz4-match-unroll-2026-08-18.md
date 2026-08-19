# M2 Air LZ4 match-copy unroll

This follow-up is based on the frozen `aac92d42` M2 Air IO handoff. It changes
only the scalar low-plane LZ4 decoder in `qh5idx.metal`: dependency-safe
four-word `packed_uchar4` match copies replace scalar loop iterations for
offset-1 and offset-2 patterns and for offsets of at least 16 bytes. Offsets
4 through 15 keep the original sequential loop. Load geometry, crop, scan and
detector binning, dtype, count audit, products, and provenance are unchanged.

The physical test used the same `495f1d6d...` Live4DSTEM executable for both
arms; only the bundled `qh5idx.metal` changed from `45c0eba8...` to
`13fea6a4...`. On the 8 GB `Mac14,2` M2 Air, seven process-isolated alternating
Fixture A/Fixture B loads were measured for frozen A, candidate, and frozen B.

| Fixture | Frozen pooled Metal p50/p95/max | Candidate Metal p50/p95/max | Frozen pooled wall p50/p95/max | Candidate wall p50/p95/max |
|---|---:|---:|---:|---:|
| Fixture A | 1.715/1.739/1.739 s | 1.615/1.629/1.629 s | 2.086/2.355/2.355 s | 1.985/1.989/1.989 s |
| Fixture B | 1.719/1.731/1.731 s | 1.618/1.624/1.624 s | 2.114/2.266/2.266 s | 2.043/2.148/2.148 s |

Both fixtures retained the full `512x512` scan, no crop, scan bin 1, explicit
detector sum bin `192x192 -> 48x48`, and uint16 resident data. Selected
diffraction hashes remained `255b94c5a4b37122` and `cc1b9e849138c351`.
Frozen/candidate/frozen BF, ABF, ADF, CoM row/column, DPC row/column, and iDPC
exports were byte-identical. Peak process footprint was about 1.43 GB, swap
delta was zero, system memory remained 77% free after the campaign, and macOS
reported no thermal or performance warning.

AGX sampling on candidate Fixture B recorded 100% device-utilization p50/p95/max,
with 71.8% of samples at or above 90%. The remaining approximately 1.62-second
Metal interval is therefore the next optimization target; this commit does not
claim the requested one-second wall target.

Verification before commit:

- QuantEM.GPU Swift suite: 60 executed, 4 opt-in real-QH5 skips, 0 failures.
- Real-QH5 CPU/Metal focused suite: 4/4 passed twice.
- Physical Air A/B/A: 42 source loads plus six full product exports.
- Rejected separately: aligned 32/64-bit copies, mask-word loads, concurrent
  mapping, disabled read-ahead, and unroll depth 8.

Adoption boundary: Live4DSTEM should consume this as a new pin only after its
current `aac92d42` regression completes. Do not rewrite the frozen handoff or
enable benchmark-gated paths implicitly.
