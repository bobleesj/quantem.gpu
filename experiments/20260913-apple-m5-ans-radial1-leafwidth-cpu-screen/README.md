# Radial1 leaf-width screen with the seven-source ADF validity mask

Status: leaf32 rejected before GPU timing. This CPU-only screen compares the
current scan512 path's radial1/leaf16 plan with radial1/leaf32 using the exact
ADF masks and the real bad-pixel list from all seven source masters.

## Inputs and method

- The current scan512 runner sets `QGPU_PAIRED_RUNTIME_POLAR_LAYOUT=radial1`
  and `QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS=16`; leaf16 is the production
  comparison point for the 59.44 ms best-path baseline.
- Native HDF5 catalog-only inspection found the same four bad-pixel indices in
  all seven masters: `5319, 15050, 21710, 29965`. The detector-mask SHA-256 is
  `33f5b1988e3f4360e9578a8855884e5bdcb2cbc2b6b25c0c830d65bec6c69b47`.
- Apply that validity mask to both masks before forming signed deltas. Compare
  radial1 leaf widths 16 and 32 on 26 exact ADF transitions: the stored
  center-8→center-20 jump, center/radius changes, and 20 adjacent one-column
  moves. Each plan's complete reconstructed signed delta must equal the
  validity-filtered input delta.
- Cost is the existing proxy `selected field count + 4 × residual pixel count`.
  This is a CPU planning screen, not a GPU timing or performance model.

## Result

All 52 plans reconstructed exactly. Leaf32's proxy was higher on all 26
transitions, so it does not warrant a seven-source GPU A/B/A.

| Geometry | Leaf16: fields / residuals / cost | Leaf32: fields / residuals / cost |
|---|---:|---:|
| ADF center-8→20 | 371 / 1,067 / 4,639 | 185 / 1,953 / 7,997 |
| All 26 ADF transitions | 1,079 / 11,143 / 45,651 | 518 / 13,237 / 53,466 |

For center-8→20, leaf32 raises the proxy by 72.4% and residual pixels by
83.0%. Across all transitions, its aggregate proxy is 17.1% higher. The lower
field count does not offset the larger residual set. The target changed-pixel
count is 6,269, matching the seven-source benchmark after validity filtering.

## Reproduction

The planner screen uses the current `PairedRuntimeTANSPolarPlan.swift` source:

```sh
swiftc src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSPolarPlan.swift \
  experiments/20260913-apple-m5-ans-radial1-leafwidth-cpu-screen/main.swift \
  -o /tmp/ans-radial1-leafwidth-cpu-screen
env -u QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN \
  -u QGPU_PAIRED_RUNTIME_JOINT_PLAN \
  /tmp/ans-radial1-leafwidth-cpu-screen \
  experiments/20260913-apple-m5-ans-radial1-leafwidth-cpu-screen/results/screen.json
```

The retained per-transition records are in `results/screen.json`. No Metal
device was used and no source count data were decoded.
