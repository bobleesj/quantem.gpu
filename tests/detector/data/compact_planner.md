# Compact detector planner fixture

`compact_planner.npz` contains only the seven host planner metadata arrays from
the validated `compact-prepared-series-v1` checkpoint. No measured intensity
arrays, compressed source payloads, or device buffers are included.

- Source: `global-state.npz` from the retained prepared-series checkpoint.
- Source SHA-256: `c499fa252b142a1a9d000c873646e6d3f43b9cd05dbcc7e801db08f29d83b99f`
- Fixture SHA-256: `d6d7a1dc2928fb605ffa5627b38da4d79734a409fec48a64d0e4adb253bf2925`
- Fixture size: 83,661 bytes.
- Array selection: `valid`, `leaf_of`, `column_cost`, `tile_cost`, `parents`,
  `omitted`, and `stored_positions`, with the source `planner__` prefix removed.

The companion planner C++ source was copied byte for byte from experiment
`0906-167-native-detector-planner/source/native_planner167.cpp`.
Its SHA-256 is `990d3755f591bba413e2eb29df5e2d904e83a3d146f85c061489f7e208365320`.

The tests compare independently vectorized NumPy planning and direct integer
mask sums for translated and fractional annuli, exact boundaries, empty and
full masks, and large forward/reverse jumps. They require a C++17 compiler but
do not require a CUDA device. Validation ran with `CUDA_VISIBLE_DEVICES=''`.
The private planner has no CuPy imports; the package's existing top-level SSB
import does import CuPy even for this CPU-only test.

Integer coefficients and reconstructed masks/counts are checked exactly.
Diagnostic planner cost permits an absolute `1e-12` rounding difference because
NumPy uses a pairwise reduction while the original C++ accumulates serially.
The measured maximum was `4.547473508864641e-13` across 36 plans; repeated
runs produced the same result. No scientific tolerance or source code changed.
