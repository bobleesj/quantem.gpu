# Synthetic QEM 0.0.1 conformance references

MIT licensed, like this repository. No private or real acquisition data.
Frozen on 2026-09-16; generated from the explicit formulas in
`scripts/build_qem_conformance.py`. Do not regenerate to hide a test failure.

- `u8-interval-boundary`: 537 scans in one chunk; crosses the 512-scan interval.
- `u16-multiple-chunks`: 537 scans in two chunks, 9x9 detector edge tiles,
  zero, constant, sparse, literal and entropy-coded count columns.
- `float32-special-bits`: two chunks containing positive/negative zero,
  infinities, a specific NaN payload, positive/negative fractions and a subnormal.

Each `.npy` is the original measurement oracle; compare float arrays by uint32
bit patterns, not float equality. `manifest.json` freezes file and count hashes
and public metadata. These correctness fixtures are not compression benchmarks.
`invalid-metadata.json` defines portable mutations which readers must reject.
Older schema-1 references remain unchanged in `../qem-v1`.
