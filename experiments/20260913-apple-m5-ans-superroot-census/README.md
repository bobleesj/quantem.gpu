# Exact ANS polar superroot census

Status: refuted for promotion to GPU. This was a CPU-only geometry screen; no
production source, application build, GPU, microscope counts, or actual
per-source validity mask was involved.

## Question

Does a mathematically exact third polar hierarchy level (superroot → root →
leaf → pixel), grouped by 4 or 16 roots, reduce aggregate and target large-ADF
planner proxy work relative to the existing two-level leaf16/radial1 plan?

## Scope and parity gate

- 192×192 detector, existing radial1 ordering, 16 detector pixels per leaf,
  and 16 leaves per root.
- Same 18 BF/ABF/ADF center and radius transitions as
  `20260913-apple-m5-ans-layout-census/main.swift`, including ADF center-8→20.
- Geometry masks are explicitly all-valid. This is not a claim about the
  seven source files' actual validity masks.
- The candidate greedily selects modal coefficients at leaf, root, then
  superroot levels. Pixel residuals preserve every disagreement. Reconstruction
  must equal each complete signed mask delta for all 18 transitions and all
  three plans (54 exact checks).
- Planner proxy remains `nonzero selected fields + 4 × nonzero residual pixels`;
  it is an operation-count proxy, not a GPU latency model.
- Promote only to a GPU experiment if the candidate improves aggregate proxy
  and ADF center-8→20 without worsening the worst-case proxy. No speedup is
  claimed by this census.

## Run

```sh
swiftc src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSPolarPlan.swift \
  experiments/20260913-apple-m5-ans-superroot-census/main.swift \
  -o /tmp/ans-superroot-census
/tmp/ans-superroot-census \
  experiments/20260913-apple-m5-ans-superroot-census/results/census.json
```

## Result

All 54 complete signed-delta reconstructions passed. The all-valid mask proxy
was:

| Plan | Selected fields, total | Residual pixels, total | Proxy cost, total | Worst transition proxy |
| --- | ---: | ---: | ---: | ---: |
| Existing two-level leaf16/radial1 | 1,794 | 6,808 | 29,026 | 4,639 |
| Three-level, 4 roots/superroot | 1,705 | 6,808 | 28,937 | 4,639 |
| Three-level, 16 roots/superroot | 1,722 | 6,808 | 28,954 | 4,639 |

The 4-root and 16-root layouts reduced aggregate proxy cost by only 0.31% and
0.25%, respectively. Neither changed the target ADF center-8→20 transition:
all plans had 371 fields, 1,067 residual pixels, and proxy cost 4,639. The
candidate only re-expresses the same leaf-level approximation at higher levels;
it does not remove any target residual decoding. The CPU promotion gate
therefore fails, so no GPU experiment is recommended from this result.

The current radial1 leaf16 permutation is 147,456 bytes. Computing
`superroot = root / groupRoots` arithmetically needs no added persistent index
array. Even a conservative explicit root→superroot map replicated for seven
sources would add at most 4,032 bytes. Fully materializing every extra
superroot ID/coefficient query record for seven sources adds at most 2,016
bytes (4-root grouping) or 504 bytes (16-root grouping) per update. These are
layout-based upper bounds, not GPU allocations measured on hardware.

No source validity, microscopy counts, GPU timing, GPU-memory use, or FPS was
measured. This is a planner proxy census only.
