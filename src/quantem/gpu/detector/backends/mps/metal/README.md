# Metal detector kernels

`reductions.msl` contains detector-reduction Metal source. The adjacent
[`kernels.py`](../kernels.py) reads, compiles, and dispatches it.

- [reductions.msl](reductions.msl) implements dense masked/detector sums,
  prefix and row-span sums, explicit detector binning, mean diffraction,
  radial accumulation, and CoM. Exact integer and floating-output entry points
  have separate contracts; they are not interchangeable precision modes.
- Packed and ANS count owners dispatch through their own IO-resident kernels,
  not by expanding into this dense array path. See the
  [representation contract](../../../../../../../docs/api/representations.md).
- New detector kernels belong here; other scientific domains keep their own
  implementation directories.

Kept as `.msl` files (not Python strings) for syntax highlighting, real compiler
errors, and isolated Git history. Public detector APIs remain in the detector
domain; consumers do not need to import this resource path.
