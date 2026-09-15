# Exact geometry census for radial-half ANS index layout

Status: refuted for the stated aggregate-and-worst-case hypothesis. This was a
CPU-only geometry census; no GPU run was made.

## Question

Does the already-supported static `radialhalf` detector grouping reduce exact
field-plus-residual work for realistic BF/ABF/ADF center and radius changes,
relative to the current `radial1` layout, without trading a target-specific
win for worse neighboring interactions?

## Scope and protocol

- The detector grid is 192×192; leaf width is 16, matching the full seven-source
  experiment configuration.
- Compare only existing static layouts `radial1` and `radialhalf`.
- Census includes 18 exact signed-mask transitions spanning small/large center
  changes and radius changes in BF, ABF, and ADF, including ADF center-8→20.
- Use all-valid geometry for this first CPU screen. Reconstruct every planned
  signed delta exactly and retain per-transition selected-field count,
  residual count, and the existing estimated-cost proxy. This is not a GPU
  timing, packed-size, or seven-source memory result.
- Run a full seven-source A/B/A Metal measurement only if `radialhalf` improves
  aggregate and worst-case planner work without worsening the large ADF target.

## Run

```sh
swiftc src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSPolarPlan.swift \
  experiments/20260913-apple-m5-ans-layout-census/main.swift \
  -o /tmp/ans-layout-census
/tmp/ans-layout-census \
  experiments/20260913-apple-m5-ans-layout-census/results/layout-census.json
```

The executable writes a JSON result. The plan uses only exact ±1/0 mask
coefficients, does not access microscope pixel counts, and allocates no GPU
buffers.

## Result

All 36 plans (18 transitions × two layouts) reconstructed their signed deltas
exactly. Across the 18 geometries, radialhalf increased total proxy cost from
29,026 to 33,730 (+16.2%) and total residual count from 6,808 to 7,990. For the
target ADF center-8→20 move, residuals increased from 1,067 to 1,857 and proxy
cost from 4,639 to 7,785. Radius-only changes often improved, but center shifts
regressed; the static candidate therefore does not generalize to the target
interaction. No GPU timing or memory measurement was warranted.
