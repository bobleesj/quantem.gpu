# Frozen MPS fixtures on Apple's virtual GPU

The frozen values are unchanged. Physical Apple GPUs still require bit identity
for these fixtures. Count decoding, mean diffraction, and integer reductions
retain their original exact assertions on every host.

The GitHub macOS runner reports `Apple Paravirtual device`. Run 35543593812
(both attempts, revision 79f2246) and run 35544175130 (revision 042a3cb)
reproduced the same six sets of floating discrepancies. The physical Apple M5
passed the original exact checks. The filtering, FFT, and window implementation
had not changed since ca9e2a5; the paired-ANS mean changes do not call them.
This isolates the discrepancy to the host-dependent floating implementation,
not the new count reducer. It does not identify an individual driver instruction.

| Frozen operation | Shape | Observed maximum absolute difference | Virtual-host absolute allowance |
| --- | --- | --- | --- |
| Gradient, sigma 1 | 512 × 512 | 1.9073486e-6 | 2e-6 |
| Gradient, sigma 2 | 512 × 512 | 2.3841858e-7 | 2.5e-7 |
| Gradient, sigma 4 | 512 × 512 | 5.9604645e-8 | 6e-8 |
| FFT | 520 × 520 | 5.197525e-4 | 6e-4 |
| Cosine window, edge 16 | 192 × 192 | 2.9802322e-8 | 6e-8 |
| Cosine window, edge 96 | 192 × 192 | 5.9604645e-8 | 6e-8 |

The FFT discrepancy is not one ULP: its maximum observed relative difference
was 2.9723285e-4. Its band is specific to this frozen, unnormalized 520-square
transform and must not be reused as a general scientific tolerance. Absolute
bands avoid a near-zero relative-error loophole in the windows and gradients.

The test records every accepted host difference and still fails on nonfinite
values, larger differences, other operations, other sizes, or other devices.
Policy-scope checks run in both XCTest and the standalone checker. Do not
recapture these fixtures or extend a band merely to make a later failure green.

Reproduce physical-host checks with
`bash scripts/check_metal_scientific_numerics.sh`; the `Metal package` CI job
runs the same frozen checks under XCTest on the virtual GPU.
