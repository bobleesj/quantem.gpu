# Exact ANS polar leaf-width screen

## Question

On the same 18 BF/ABF/ADF mask transitions, does changing the exact polar
planner leaf width from 64 to 32 or 16 reduce its aggregate planner proxy and
the large ADF center-8-to-20 proxy without worsening the worst transition?

## Protocol

- 192×192 detector and the exact 18 transitions copied from
  `20260913-apple-m5-ans-layout-census/main.swift`.
- Compare `leafPixels` 16, 32, and 64 while holding `layoutKind: "polar"`
  fixed, matching the production runtime's default layout.
- Explicit all-valid geometry (`validPixels` is 36,864 ones); no source data
  or bad-pixel masks are loaded.
- Every `PairedRuntimeTANSPolarPlan.reconstructedDelta()` must exactly equal
  the full signed mask delta before its record is accepted.
- Record selected field count, residual pixel count, and
  `estimatedCost = fields + 4 × residuals` per case, with aggregate and worst
  cases by width. The proxy is not a latency model or timing result.
- CPU-only; no GPU or Metal benchmark. Consider a Metal A/B only if a changed
  width lowers aggregate and ADF center-8-to-20 estimated cost and does not
  increase the worst transition cost versus the current 64-pixel default.

## Run

```sh
swiftc experiments/20260913-apple-m5-ans-leaf-width-screen/source/PairedRuntimeTANSPolarPlan.swift \
  experiments/20260913-apple-m5-ans-leaf-width-screen/main.swift \
  -o /tmp/ans-leaf-width-screen
env -u QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN \
  -u QGPU_PAIRED_RUNTIME_JOINT_PLAN \
  /tmp/ans-leaf-width-screen \
  experiments/20260913-apple-m5-ans-leaf-width-screen/results/leaf-width-screen.json
```

The source snapshot pins the planner implementation used by the screen. No
wall-speed claim is made from this CPU census.

## Result

All 54 plans reconstructed the complete signed delta exactly. The summary
below reports selected fields, residual pixels, and the planner's estimated
cost (`fields + 4 × residuals`):

| Leaf pixels | Total fields | Total residuals | Total estimated cost | Worst case | ADF center-8-to-20 |
| ---: | ---: | ---: | ---: | --- | --- |
| 16 | 973 | 12,394 | 50,549 | ADF center-8: 147 / 1,501 / 6,151 | 218 / 1,335 / 5,558 |
| 32 | 697 | 12,836 | 52,041 | ADF center-8: 103 / 1,553 / 6,315 | 168 / 1,443 / 5,940 |
| 64 | 452 | 13,664 | 55,108 | ADF center-8-to-20: 88 / 1,783 / 7,220 | 88 / 1,783 / 7,220 |

For the last two columns, values are `fields / residuals / estimated cost`.
Against leaf64, leaf16 reduces aggregate proxy cost 8.27%, the target ADF cost
23.02%, and worst-case cost 14.81%. Leaf32 reduces those costs 5.57%, 17.73%,
and 12.53%, respectively. Both widths pass the CPU screening gate for a real
Metal A/B; leaf64 remains the current default/control. Leaf16 has the lower
proxy of the two candidates, but this planner census does not establish GPU
speed or memory behavior.

The full per-case field/residual/cost census is recorded in
[`results/leaf-width-screen.json`](results/leaf-width-screen.json). Each case
also records whether the planner selected the index; cases with `used_index`
false used its direct fallback. No wall-speed result was measured.

| Transition | Leaf16 fields / residuals / cost | Leaf32 fields / residuals / cost | Leaf64 fields / residuals / cost |
| --- | ---: | ---: | ---: |
| bf-center-1 | 0 / 186 / 744 | 0 / 186 / 744 | 0 / 186 / 744 |
| bf-center-8 | 37 / 420 / 1,717 | 42 / 446 / 1,826 | 28 / 466 / 1,892 |
| bf-center-8-to-20 | 90 / 562 / 2,338 | 71 / 614 / 2,527 | 40 / 726 / 2,944 |
| bf-radius-1 | 0 / 296 / 1,184 | 0 / 296 / 1,184 | 0 / 296 / 1,184 |
| bf-radius-8 | 22 / 418 / 1,694 | 13 / 424 / 1,709 | 12 / 424 / 1,708 |
| bf-radius-20 | 35 / 356 / 1,459 | 17 / 360 / 1,457 | 20 / 376 / 1,524 |
| abf-center-1 | 0 / 352 / 1,408 | 0 / 352 / 1,408 | 0 / 352 / 1,408 |
| abf-center-8 | 77 / 1,306 / 5,301 | 58 / 1,334 / 5,394 | 42 / 1,376 / 5,546 |
| abf-center-8-to-20 | 175 / 1,018 / 4,247 | 129 / 1,140 / 4,689 | 74 / 1,286 / 5,218 |
| abf-radius-1 | 0 / 420 / 1,680 | 0 / 420 / 1,680 | 0 / 420 / 1,680 |
| abf-radius-8 | 35 / 1,090 / 4,395 | 24 / 1,090 / 4,384 | 24 / 1,110 / 4,464 |
| abf-radius-20 | 49 / 776 / 3,153 | 32 / 780 / 3,152 | 24 / 812 / 3,272 |
| adf-center-1 | 0 / 568 / 2,272 | 0 / 568 / 2,272 | 0 / 568 / 2,272 |
| adf-center-8 | 147 / 1,501 / 6,151 | 103 / 1,553 / 6,315 | 74 / 1,631 / 6,598 |
| adf-center-8-to-20 | 218 / 1,335 / 5,558 | 168 / 1,443 / 5,940 | 88 / 1,783 / 7,220 |
| adf-radius-1 | 0 / 616 / 2,464 | 0 / 616 / 2,464 | 0 / 616 / 2,464 |
| adf-radius-8 | 39 / 638 / 2,591 | 17 / 654 / 2,633 | 12 / 666 / 2,676 |
| adf-radius-20 | 49 / 536 / 2,193 | 23 / 560 / 2,263 | 14 / 570 / 2,294 |
