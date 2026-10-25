# Native detector median contract

The native encoded source and corrected resident view use `native_median3x3`:
valid neighbors inside a 3 × 3 detector window, the floored mean of the two
middle counts for an even neighborhood, and zero if no valid neighbor exists.
This preserves the frozen native counts.

Since 2026-10-25 the original-HDF5 Metal loader (`hotPixelPolicy: .median`)
and `NativeHotPixelCorrection.median3x3` round that mean half to even, as
Python `fill_flagged` does; see
[2026-10-25-original-load-median-fill.md](2026-10-25-original-load-median-fill.md).

The Python resident correction uses its separately validated planned-neighbor
policy: expand the window when needed and round half to even. These policies
are different; neither caller may silently select the other's kernel.

The native encoded source and corrected resident view previously compiled the
Python `hot_median` kernel while binding its older buffer interface. The kernel
now expects neighbor offsets and indices, so those bindings were invalid. This
caused the frozen count test to fail before the SSB integration as well as after
it. Rebinding alone would also change eight of the 595 frozen count values by
one, because of rounding.

The repair gives the native policy its own resource and entry point. It changes
neither the Python kernels nor any frozen fixture or numerical tolerance.

## Verification

`bash scripts/check_metal_scientific_numerics.sh` passes on physical Apple
hardware, including exact frozen counts and means, filtering, FFT, correlation,
calibrated reductions and indexed file loading. The frozen check now exercises
both append-time and read-time median correction, verifies that the raw source
is unchanged, and checks that releasing a corrected view preserves its
caller-owned source.

The same fixture is exercised by the macOS CI job. Physical and CI-device
results are separate checks.
