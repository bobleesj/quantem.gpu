# QuantEM.GPU overnight evidence index

Generated 2026-08-22 from clean coordinator revision
`23d25619cfe22d5e89761fda2d2796a7c82ba090`. This is a source-only audit. It
did not run a GPU, reopen a fixture, copy an evidence bundle, update the public
documentation, or qualify a consumer.

## Question

Which Swift/Metal, Python MPS, WebGPU, selective-loading, Rodman, and MJGOAT
artifacts from the overnight campaign support a bounded claim, which refute an
experiment, and which remain pending?

## Decision

**Mixed.** Thirteen artifacts support their stated bounded claims, four
record refuted experiments, and two remain pending. Thirteen atomic measurements
are admissible under their original boundaries. None is arbitrary-source cold
HDF5 or Live4DSTEM wall-to-wall evidence, so the campaign does **not** establish
a cold-load target on Phil, Steve Kerr, or Rodman.

## What we learned

- Native Swift/Metal full-native detector-bin-1 reached `0.313870 s` p50 for a
  prepared-index, private-resident load on Phil. Full private readback and
  catalog time are outside that boundary.
- The controlled Swift/Metal package boundary on Phil reached `0.524590 s` p50
  across seven processes. Its `1.572781 s` p95/max outlier is retained.
- Current accepted Python MPS warm/uncontrolled-page p50 values are `0.522754 s`,
  `0.498708 s`, `0.383772 s`, and `0.356969 s` for detector bins 1, 2, 4, and 8.
- Rodman admitted the full `512x512x192x192 uint16` volume once at `3.827780 s`.
  One sample is not a percentile distribution, and swap increased materially.
- The historical WebGPU full packed-uint16 candidate is refuted. Resident mode
  0 bypassed its integer CoM branch, and iDPC exceeded the frozen tolerance.
  A later paired-u32 implementation has only a six-case hardware smoke, not a
  full-fixture parity or performance qualification.
- Selective loading is source-qualified for CUDA and Python MPS semantics, but
  native Swift/Metal lacks bulk indexed selection and WebGPU still reads whole
  intersecting files. All physical selective-I/O benchmark cells remain open.
- MJGOAT CUDA exact screening is now source-qualified at clean implementation
  `5ee2016e` and evidence commit `47bb6e42`. The best like-for-like accepted
  warm-source A-B-B-A row is `1.208146 s` p50, while the later final-source
  distribution is `1.546580 s` p50 in a CPU-contended window. Neither is cold.
- Physical WebGPU SSB completed on Phil, but phase parity is refuted: maximum
  wrapped error was about `0.0597773 rad` against a frozen `0.0002 rad` limit.
  Its `329 ms` first-use and `17.9 ms` repeated p50 timings are diagnostic only.

## Main caveat

“Prepared,” “warm,” “controlled uncached,” “resident compute,” and “cold
arbitrary source” are different experiments. The rows below preserve their
original cache state and timed boundary. They must not be collapsed into one
leaderboard.

## Immutable fixture used by accepted timing rows

| Fixture | SHA-256 | Scan | Detector | Source dtype | Compression | Shards |
|---|---|---:|---:|---|---|---:|
| `real-512x512x192x192-u16-bslz4-27shard-master-fixture-c` | `c9c0d968fae70b8911ae925d676b90007886970fa99fe296a47cfe07844bbfe9` | `512x512` | `192x192` | `uint16` | bitshuffle-LZ4 | 27 |
| `real-512-native-detector` | `4802ec16ba241fef439e9dcb1c28e94f9cf9d95f773df9c5c8c3b5f7ed8192c4` | `512x512` | `192x192` | `uint16` | bitshuffle-LZ4 | — |

Private filesystem paths are intentionally absent. Every accepted timing row
in the machine index binds to one listed identifier and hash.

## Accepted load and product measurements

| Platform | Computer | Area | Scan | Source detector | Detector bin | Working detector | Working dtype | Cache state | Samples | p50 (s) | p95 (s) | Max (s) | Single wall (s) |
|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| Python MPS | Phil | loading | `512x512` | `192x192` | 1 | `192x192` | `uint16` | warm OS pages; uncontrolled | 6 | 0.522754 | 0.540934 | 0.544737 | — |
| Python MPS | Phil | loading | `512x512` | `192x192` | 2 | `96x96` | `uint16` | warm OS pages; uncontrolled | 14 | 0.498708 | 0.503702 | 0.505150 | — |
| Python MPS | Phil | loading | `512x512` | `192x192` | 4 | `48x48` | `uint16` | prepared plans; warm or uncontrolled OS pages | 6 | 0.383772 | 0.390006 | 0.391839 | — |
| Python MPS | Phil | loading | `512x512` | `192x192` | 8 | `24x24` | `uint16` | warm OS pages; uncontrolled | 6 | 0.356969 | 0.359302 | 0.359820 | — |
| Native Swift/Metal | Phil | loading and products | `512x512` | `192x192` | 1 | `192x192` | `uint16` | controlled uncached; already audited source | 7 | 0.524590 | 1.572781 | 1.572781 | — |
| Native Swift/Metal | Phil | loading and products | `512x512` | `192x192` | 1 | `192x192` | `uint16` | prepared indexes; source pages unspecified | 6 | 0.313870 | 0.318865 | 0.318865 | — |
| Native Swift/Metal | Rodman | loading and products | `512x512` | `192x192` | 1 | `192x192` | `uint16` | prepared indexes; source pages unspecified | 1 | — | — | — | 3.827780 |
| Native Swift/Metal | Rodman | loading and products | `512x512` | `192x192` | 2 | `96x96` | `uint16` | prepared indexes; destination reused | 6 | 0.683492 | 0.690893 | 0.690893 | — |
| Native Swift/Metal | Rodman | loading and products | `512x512` | `192x192` | 4 | `48x48` | `uint16` | prepared indexes; destination reused | 12 | 0.631540 | 0.645521 | 0.645521 | — |

## Accepted CUDA screening measurements

| Candidate or distribution | Computer | Scan | Detector | Cache state | Samples | Baseline p50 (s) | Candidate p50 (s) | Candidate p95 (s) | Candidate max (s) | Exact-complete p50 (s) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| exact mask guard | MJGOAT | `512x512` | `192x192` | warm source pages; unspecified | 3 per arm | 2.506634 | 1.663968 | 1.718105 | 1.724120 | 1.551314 |
| contiguous read plan | MJGOAT | `512x512` | `192x192` | warm source pages; unspecified | 3 per arm | 1.317555 | 1.208146 | 1.241817 | 1.245558 | 1.104858 |
| final source distribution | MJGOAT | `512x512` | `192x192` | warm source pages; no prepared index; CPU contention retained | 3 | — | 1.546580 | 1.552670 | 1.553347 | 1.429524 |

All rows use source `uint16`, scan bin 1, detector bin 1, no crop, exact
`uint64` screening accumulation, and byte-exact retained public arrays. They
time `quantem.gpu.screening.prepare` through exact completion. First usable is
exact complete; there is no preview result. The three rows represent different
experiment phases, so only each recorded baseline/candidate arm is a valid
before/after comparison.

The compact source index reduced source-index p50 from `0.236894 s` to
`0.012524 s`. Its prepared package p50 was `1.393478 s`, p95 `1.559568 s`,
and max `1.578022 s` over three samples. Lookup is accepted, but that end-to-end
distribution remains provisional because the retained outlier is not discarded.
The clean source/evidence commits, manifest, migration note, and ledger are
locally resolved and hash-verified. Raw CUDA trial bundles remain external
`local-evidence://` references with sealed manifest hashes; they were not copied
into this Phil evidence-index worktree.

## Peak-memory definitions and measurements

- **Logical resident bytes** are the scientifically addressable working tensor,
  not the process RSS.
- **Accelerator peak bytes** are the sampled Metal allocation or working-set
  peak retained by the source artifact.
- **Process peak RSS** is resident process memory sampled by the operating
  system. Private GPU allocations can therefore be much larger than RSS.
- **Process peak footprint** is the operating-system physical-footprint metric
  when the artifact recorded it. It is not interchangeable with RSS.

| Platform | Computer | Detector bin | Logical resident (B) | Accelerator peak (B) | Process peak RSS (B) | Process peak footprint (B) |
|---|---|---:|---:|---:|---:|---:|
| Python MPS | Phil | 1 | 19,327,352,832 | 19,801,456,640 | 739,966,976 | — |
| Python MPS | Phil | 2 | 4,831,838,208 | 6,107,774,976 | 617,414,656 | — |
| Python MPS | Phil | 4 | 1,207,959,552 | 2,483,896,320 | — | — |
| Python MPS | Phil | 8 | 301,989,888 | 1,577,926,656 | 613,924,864 | — |
| Native Swift/Metal | Phil | 1 | 19,327,352,832 | 19,940,737,024 | 1,064,058,880 | — |
| Native Swift/Metal | Phil | 1 | 19,327,352,832 | 19,940,737,024 | 685,670,400 | — |
| Native Swift/Metal | Rodman | 1 | 19,327,352,832 | 19,940,737,024 | 669,876,224 | 20,033,504,288 |
| Native Swift/Metal | Rodman | 2 | 4,831,838,208 | 5,445,222,400 | 694,779,904 | 5,563,728,952 |
| Native Swift/Metal | Rodman | 4 | 1,207,959,552 | 1,821,343,744 | 702,283,776 | — |

The CUDA final-source distribution streamed the `19,327,352,832` decoded bytes
instead of retaining a full logical tensor. Median per-trial host peak RSS was
`2,286,784,512 B`; process GPU allocated peak was `2,238,152,192 B`; process
GPU reserved peak was `5,050,266,112 B`; and total-card peak was
`6,532,890,624 B`, including a `1,472,462,848 B` coexisting idle-service
baseline. Thus the process reserve stayed below the six-GiB gate, while the
separately reported total-card peak must not be mislabeled process memory.

The two Phil detector-bin-1 rows have different timing boundaries and process
measurements; their equal logical/Metal allocation does not make them duplicate
benchmarks. The missing Python MPS detector-bin-4 RSS is an unresolved evidence
cell, not zero.

## Accepted resident DPC/iDPC measurement

| Platform | Computer | Input | Samples | Wall p50 (ms) | Wall p95 (ms) | Wall max (ms) | Peak Metal (B) | Peak RSS (B) | Peak footprint (B) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Native Swift/Metal | Rodman | resident `512x512 float32` CoM row/column maps | 15 | 1.979 | 3.640 | 3.640 | 9,895,936 | 36,618,240 | 69,141,104 |

This boundary starts after loading and CoM reduction. Rotation row/column maximum
absolute error is `1.1920928955078125e-7`. iDPC maximum absolute error is
`9.5367431640625e-6`, p99 is `4.76837158203125e-6`, and normalized maximum
error is `1.6974102170991063e-7` against float64 NumPy followed by float32
publication.

## Accepted correctness and contract artifacts without a promoted speed row

| Family | Artifact | Accepted scope | Evidence revision | Consumer eligible |
|---|---|---|---|---|
| Python MPS | exact-bin overflow | fail-closed full/selective/sidecar unsigned-sum range checks | `72df01c8f0a1a2f4b68164f4e27ff89a6ff39371` | yes |
| Python MPS | partial bitshuffle tail | exact native `uint16`/`uint32` source tails for bins 1/2/4/8 | `83a61411eb45b2f8549f00dfa178cc9de19bf23e` | yes |
| Cross-backend I/O | selective-load audit | source contract and portable focused parity only | `58577c81390419d6237465092d50a8fb80dbe36d` | no |
| MJGOAT CUDA | owner guard | fail-closed shell ownership decision only | `05d6bbf7ca367f47bb65a88c87e3ad568563d646` | no |

Python MPS native uint8-source bitshuffle decode is **not** proven. The accepted
`uint8` option is an explicit clipped output produced from a native `uint16` or
`uint32` source; resident uint8 kernels and `output_dtype='u8'` do not establish
native uint8 HDF5 source support.

## Refuted artifacts

| Family | Artifact | First divergence or measured result | Status |
|---|---|---|---|
| Native Swift/Metal | private preparation / two-stage pipeline | exact-load p50 regressed `0.322143 s -> 0.361828 s`; package p50 regressed `0.529075 s -> 0.565636 s` | reverted |
| WebGPU | historical packed-uint16 iDPC candidate | zero-rotation 3,746 violations; optimized rotation 2,587 violations at unchanged `rtol=atol=1e-5` | refuted |
| WebGPU | full-active-BF SSB | maximum wrapped phase error `0.0597773 rad` against `0.0002 rad`; loss absolute error `1.0248e-8` passed | refuted |
| MJGOAT CUDA | service routing | current service cannot promise request-level GPU0-only execution while excluding GPU1 | refuted before request |

The WebGPU rejected bundle's load and product timings are deliberately absent
from the accepted measurement table. Separate FFT/Poisson evaluation against
the same hardware DPC had no violations, placing the first divergence before
centering: packed-uint16 resident mode 0 bypassed a uint8-only integer branch.

The refuted WebGPU SSB run used a prepared exact detector-bin-2 BF-column
companion on Phil's physical Apple Metal-3 adapter. First use was `329 ms` wall
and `33.2 ms` GPU; five repeated exact phase/loss calls were `17.9/18.0/18.0 ms`
wall p50/p95/max and `16.6/16.8/16.8 ms` GPU. Resident G(q,k) was
`649,498,624 B`, browser-tree peak RSS was `1,748,828,160 B`, and WebGPU total
device memory was not exposed. These values remain diagnostic because phase
parity failed. Scan sizes 128 and 512 were exercised; declared sizes 256 and
1024 remain untested in this lane. The scientific source identity remains
`uint16`; the companion's `uint8` storage is an exact lossless count encoding,
not evidence for native uint8 HDF5-source decode.

## Pending artifacts

| Family | Artifact | Verified so far | Missing gate |
|---|---|---|---|
| Python MPS SSB | fixture recovery | exact detector-bin-2 companion and explicitly sampled bounded pass | clean source; frozen randomized-pair engine; 200-trial plus Nelder-Mead gate |
| WebGPU | paired-u32 CoM | six row/column bitwise hardware-smoke cases, including saturation | full real-fixture DPC/iDPC parity and performance |
| Rodman Swift/Metal | bin-2 barrier consolidation | source tests only | uncontended physical smoke and A-B-B-A |

The WebGPU iDPC artifact family is classified refuted overall because its only
full real-fixture run failed. The later smoke is retained inside that family as
pending work; it is not a parity fix or performance claim.

## Unresolved metric cells

1. Cold arbitrary-source HDF5 discovery, index, load, first usable, exact
   completion, and peak memory on Phil.
2. Cold arbitrary-source HDF5 discovery, index, load, first usable, exact
   completion, and peak memory on Rodman.
3. Every current-revision Steve Kerr 8 GB metric.
4. Live4DSTEM wall-to-wall load, prepared reopen, A-B-A switching, and headed
   interaction evidence.
5. Python MPS detector-bin-4 atomic accepted-arm process RSS.
6. WebGPU full-fixture paired-u32 DPC/iDPC parity and performance.
7. WebGPU true range-read selective I/O with ordered and duplicate selectors.
8. Native Swift/Metal bulk indexed selective loading.
9. CUDA arbitrary-source controlled-cold and Live4DSTEM application evidence.
10. Public CUDA result fields for the exact internal ABF, ADF, and total maps.
11. MPS SSB clean 200-trial calibration under the frozen engine.
12. Rodman detector-bin-2 candidate device qualification.
13. A repeated Rodman full-native distribution without paging contamination.

## Review surfaces

- Machine-readable index: [`evidence-index.json`](evidence-index.json)
- Source-only validator: [`validate_evidence_index.py`](validate_evidence_index.py)
- Experiment manifest: [`manifest.json`](manifest.json)

The machine index is authoritative for exact artifact paths, hashes, source and
evidence revisions, dirty-state fingerprints, cache labels, sample counts,
geometry, dtype, bin/crop, parity claims, memory fields, and limitations. This
draft intentionally does not update or replace the public benchmark registry.
