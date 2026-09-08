# Complete original EMPAD rereading

The final development build presented the full 4.36 GB acquisition in **0.961,
0.896 and 0.989 s** across an open, forced reload and folder round trip. Every
run reread and losslessly packed all 65,536 frames. These are valid-checksum-cache
hits with unspecified OS page-cache state, not cold reads or retained 2D images.

With checksum reuse disabled, the same native workflow took **2.19–2.27 s**.
SHA-256 calculation alone took 1.31–1.33 s. The general first-open one-second
target therefore remains unmet. An intermediate candidate startup also took
1.036 s despite a metadata hit; subsecond behavior is not guaranteed.

## Changes

- `readv` scatters each original detector frame directly into bounded Metal
  staging storage, avoiding another RAW buffer allocation and copy. Only the
  format's footer is discarded, never physical detector pixels.
- A small checksum record binds the completed logical hash to the file path,
  device, inode, size, mtime and ctime. Corrupt or changed records are rehashed.
- The app uses `~/Library/Application Support/Live4DSTEM/Indexes/EMPAD/`.
  These records contain identity metadata, not measurement arrays; the existing
  **Privacy & Local Data** cache-clearing action removes them safely.

Both cache-miss and cache-hit paths passed bitwise comparison of all
1,073,741,824 detector samples. BF/ABF/ADF, total, mean DP and CoM pass the
unchanged 1e-6 float64-reference tolerances. Fourteen synthetic tests include
source mutation, cache corruption, source-path protection, cancellation,
packing widths and bounded memory. Native viewer-buffer parity, failed-file
recovery, FFT restoration and forced reload passed.

[result.json](result.json) retains source/build identities, exact measurements,
native control results and limitations. The installed application was not
changed. Wide EMPAD detector interaction still fails the 120 Hz target; see
the [aperture experiment](../20260908-empad-incremental-apertures/manifest.json).
