# Metal detector kernels

`reductions.msl` contains detector-reduction Metal source. The adjacent
[`kernels.py`](../kernels.py) reads, compiles, and dispatches it.

- `reductions.msl` — masked_sum, detector_sum, prefix-sum, bin2.
- New detector kernels belong here; other scientific domains keep their own
  implementation directories.

Kept as `.msl` files (not Python strings) for syntax highlighting, real compiler
errors, and isolated Git history. Public detector APIs remain in the detector
domain; consumers do not need to import this resource path.
