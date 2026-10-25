# Classify fine-grid SSB mirror reuse once per frequency

The common 3× and 4× views of a full 512 × 512 × 192 × 192 acquisition
retain the same fifth-order optics, tilt, depth, complete bright-field support,
precision and preprocessing. The paired CUDA correction previously classified
native alias boundary columns inside each bright-field iteration. Classify each
output frequency before that loop and reuse the boolean. Independent boundary
corrections and every nonzero arithmetic operation remain unchanged.

This is restricted to the existing 512-square real-FFT higher-order fine-grid
path. Native reconstruction, fitting, Newton refinement, loss diagnostics,
MPS and browser rendering are unchanged. There is no new cache, workspace,
public API or kernel family.

## Matched timing

RTX PRO 6000 Blackwell Max-Q Workstation Edition, CUDA, one prepared session.
Ten rounds changed C10 and rotated variant order; medians exclude two warmups.
Device synchronization surrounds reconstruction. Transfer and browser work are
excluded. The reference includes the previously accepted zero-contribution skip.

| Upsampling | Reference | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| 3× | 48.10 ms | 45.42 ms | 5.6% |
| 4× | 52.50 ms | 49.83 ms | 5.1% |

This is a further backend improvement, not a measured notebook frame rate or
60 fps result. The correction still dominates reconstruction. The separate
native-grid loss diagnostic on slider release remains unchanged.

## Validation and rejected candidates

All twenty matched outputs are array-identical. Eighty full-data cases over
two acquisitions, factors 1, 2, 3, 4 and 8, varied rotation, defocus,
astigmatism, tilt and depths from zero to 60 nm are also array-identical.

Skipping phase evaluations for zero aperture weights was faster but changed
phase by up to 1.98e-6; it is rejected under the exact-output contract. Skipping
inactive-block preparation was exact but produced little gain. Shared coordinate
staging was exact and slightly faster in combination, but added shared-memory
traffic and about 0.5 ms improvement; the smaller classification change is the
accepted implementation. Raw timings and all candidates are retained privately.

Reference kernel SHA-256:
`6f5621e3db38aa5e75f300e89ae5b19c1124609417a960924410bd352e8478c9`.
Candidate kernel SHA-256:
`026f16815ffe7e8312a42437863e939fcfd45546b469019e210591fce9d97822`.
Private evidence: `20261023-ssb-fine-grid-classification`.

Browser automation remained disabled in the host application. Slider gestures,
FFT display and actual presentation rate are not verified by these measurements.

## Installed notebook validation

The installed candidate passed all 31 CUDA wave-average/thick-sample tests
in 127.27 seconds. Five repository privacy checks passed. The maintained
advanced CUDA tutorial completed from a fresh kernel, including search,
rotation comparison and ShowPtycho construction.

Twelve changed-C10 previews per factor then matched all 24 pre-change output
hashes exactly. Warm median reconstruction plus GPU-to-host copy was 45.44 ms
at 3× and 49.47 ms at 4×; the first observation is excluded. These independent
installed-run timings exclude network and browser work. Use the alternating
matched table above to quantify the optimization's speedup.
