# Selective exact prefix-overlay planner screen

Status: **refuted for the ≥20% held-out residual-reduction hypothesis**. This
is a CPU-only planner experiment. It did not compile or run a Metal kernel,
allocate GPU memory, decode count data, or measure response time.

## Question

Can a bounded per-leaf exact prefix basis, with no more than 75 extra prefix
fields per source, be selected from other BF/ABF/ADF transitions and reduce
residual pixels by at least 20% on the held-out large ADF center move
`(5,8) → (12,20)`?

## Protocol

- Uses the current `PairedRuntimeTANSPolarPlan` CPU planner at `leafPixels=16`,
  `layout=radial1`, and joint planning disabled.
- Reconstructs the same 20 named detector-mask geometries in the seven-source
  ANS series benchmark. It verifies those names and all seven source output
  hashes per mask against the frozen 20-mask reference artifact.
- Replays 17 exact mask transitions: 16 training transitions across BF, ABF,
  and ADF, and the ADF center-8-to-center-20 transition held out from leaf
  selection.
- A selected 16-pixel leaf may add cumulative exact sums for its first 4, 8,
  and 12 pixels in radial1 order. The planner chooses three segment
  corrections and converts them into coefficients over those cumulative
  prefixes. Uncovered signed coefficients remain exact direct residuals.
- Ranks leaves by the sum of positive reductions in the existing proxy
  `selected fields + 4 × residual pixels`, then evaluates budgets from 0 to 25
  leaves. It also compares all-modality training with ADF-only training.
- An explicitly labeled target-only selection is reported only as an oracle
  diagnostic; it leaks the held-out target and is not a train/test result.
- Applies the same four source-invalid detector pixels found in all seven
  masters: indices `5319, 15050, 21710, 29965`. Their detector-mask fingerprint
  and source audit are retained in the existing
  [radial1 validity screen](../20260913-apple-m5-ans-radial1-leafwidth-cpu-screen/results/screen.json).

## Result

All 20 reference mask names and 140 full-scan source/mask output-hash records
matched. The shared four-pixel validity mask was applied before planning. All
17 transitions reconstructed exactly for both the existing plan and the
prefix overlay, and synthetic full-`uint16` scalar products matched for seven
source seeds. The output-hash records validate the frozen source/mask set; this
CPU experiment does not decode source count data or regenerate those hashes.

| Selection | Stored prefix fields/source | Active extra queries/source on held-out ADF | Held-out residual pixels/source | Residual reduction | Proxy cost/source |
|---|---:|---:|---:|---:|---:|
| Existing planner | 0 | 0 | 1,067 | — | 4,639 |
| 16-transition training | 75 | 7 | 1,045 | 22 (2.1%) | 4,558 |
| ADF-only training | 75 | 23 | 1,011 | 56 (5.2%) | 4,438 |
| Target-only oracle (leaks holdout) | 75 | 26 | 921 | 146 (13.7%) | 4,081 |

The 25-leaf cap means at most 75 retained prefix sums per source, or 525 across
seven sources. These are field counts, **not measured bytes**. The oracle's
13.7% residual reduction is an upper-bound diagnostic for this particular
prefix basis, not a generalizable result. Even the oracle does not approach a
4× reduction; the training-selected result is substantially smaller. The
≥20% held-out reduction hypothesis is therefore refuted, and no Metal follow-up
is justified from this screen alone.

## Reproduce

From the repository root:

```sh
sh experiments/20260913-apple-m5-ans-selective-prefix-overlay/run.sh
```

The script compiles the production CPU planner plus the isolated experiment
driver and writes `results/prefix-overlay.json`. It uses no GPU device.
