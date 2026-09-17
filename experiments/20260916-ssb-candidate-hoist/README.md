# Can the per-evaluation SSB loss pipeline be hoisted across candidates?

**Verdict: the per-evaluation pipeline is fully candidate-dependent.** No buffer that the
pipeline recomputes per evaluation is byte-identical across two aberration triples, so there is
no work to hoist and the bit-exact saving from hoisting is **0 bytes per evaluation**.

## Why this was worth testing

The 8937-term fit costs 228 evaluations and ~62 s. If any per-evaluation buffer were a function
of the dataset alone, producing it once per session would be the only bit-exact 2x left. The
question had to be answered on the real path, not from the shape of the code.

## Method

`probe-digest.patch` adds an env-gated device digest kernel plus accessors to
`MetalSSBEngine`, and one digest per buffer write in the fused 512 loss path
(`MetalSSBKernels.swift:1560`). `probe_hoist.swift` then loads the real ARINA 512x512x192x192
master (8937 active bright-field terms, 11.37 GB resident), prepares the full `G` cache once,
and evaluates the loss with two triples **on the same engine**: the pinned optimum
`(6.603671591975075, 0.09848762314231685, 1.1341200767509283)` and `(5.0, 0.5, 0.0)`.

The digest is a per-thread FNV-1a over 16 consecutive `uint32` reduced with a commutative
low/high pair, so it is deterministic under any thread scheduling. It was validated three ways:
against a host implementation of the same fold (known answer and one-word mutation), against an
all-zero region, and by repeating a whole process.

## Buffer table (run 1; run 2 reproduces every value bit for bit)

| buffer | bytes/eval | identical across candidates | verdict |
| --- | --- | --- | --- |
| `G` Hermitian cache, 9 chunks | 9,407,729,664 | yes | candidate-independent, already produced by `prepare()` |
| `bf_geometry` (`ssb.metal:479`) | 142,992 | yes | memoised on rotation (`MetalSSBKernels.swift:1334`) |
| `twiddle`, `q_row`, `q_col` | 8,192 | yes | geometry constants |
| `chi_trig` (`ssb.metal:901`) | 2,097,152 | no | function of `c10, c12, phi12` |
| `cross_trig` (`ssb.metal:1003`) | 73,211,904 | no | function of `c10, c12, phi12` |
| column-pass output (`ssb.metal:1129`, buffer 5) | 9,407,729,664 | no, 1118/1118 batches | every load is scaled by the candidate tables |
| Nyquist correction (`ssb.metal:1200`) | 73,269,248 | no, 1068/1118 batches | 50 batches are an all-zero correction in both runs |
| `phase_sum`, `phase_sum_sq` (`ssb.metal:1280`) | 2,097,152 | no | moments over the candidate phase |

`column-pass output` is the intermediate that the row pass reads back at
`ssb.metal:1293`; the 512 path fuses that read into the moment kernel, so the row-pass
product and the moments accumulator are the same buffer.

## Traffic split and the size of the prize that is not there

Per evaluation the pipeline moves 28,524,539,904 B: 9,407,729,664 B candidate-independent
(33.0%) and 19,116,810,240 B candidate-dependent (67.0%). The candidate-independent share is a
*read* of a cache that `prepare()` already builds once per session (2.16 s measured), so it
cannot be removed by hoisting: it is the transform's input. Had it been free, the bound would
have been 9.408 GB x 228 / 131 GB/s = **16.4 s of the 62 s fit**; the measured saving is 0.

The earlier traffic closure is consistent with this and rules out hidden per-evaluation cache
work: 267 ms/evaluation matches three 9.408 GB passes at 105-138 GB/s. A per-evaluation transpose
or rewrite of the 9.4 GB cache would add 75-150 ms per evaluation and is not present.

## Limits

- The streamed tail path (`ssb.metal:928`, `:963`, used only when the cache budget is smaller
  than the bright-field count) is candidate-dependent by the same construction - it applies the
  same correction tables and a per-candidate inverse transform - but was **not** measured here;
  the fit runs the fully cached path.
- The probe's digest dispatches add ~9.5 GB of device reads per evaluation, so its wall seconds
  are not a timing result. Use the objective-profile experiment's 267 ms/evaluation for timing.
