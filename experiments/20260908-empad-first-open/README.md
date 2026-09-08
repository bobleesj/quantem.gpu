# First-open EMPAD loading

First-open native presentation improved from **2.11-2.39 s to 1.53-1.61 s**
on the same original 4.36 GB acquisition. The one-second target remains unmet.
These are full resident-backed presentations, with checksum reuse disabled.
Source-page state was not controlled; this is not a cold-storage claim.

## Change

SHA-256 previously blocked the producer after every read. An ordered serial
digest queue now hashes the existing input buffer concurrently with Metal
packing and the next bounded read. A second hash cannot be submitted before
the first finishes, and successful publication and cancellation both drain the
queue. Source validation and the original SHA-256 identity remain mandatory.

Two 16 MiB first-use windows replace one 32 MiB serial window. The intermediate
512-frame candidate was slightly faster but increased transient memory by about
32 MiB. The selected 256-frame version restores comparable peak memory while
keeping most of the speedup. Cached-hash loads retain their existing 512-frame
path and still reread every source frame.

This is an IO/checksum scheduling improvement, not a new Metal shader or a
change to the scientific packing format. EMPAD is original float32 RAW, not
ARINA bitshuffle/LZ4. No binning, clipping, physical detector cropping, float
quantization, image-cache substitution, or dense second volume is used.

| Measurement | Serial control | Selected bounded pipeline |
| --- | --- | --- |
| Backend full resident | 1.83-1.98 s | 1.34-1.42 s in window probe |
| Native full presentation | 2.11-2.39 s | 1.53-1.61 s in final app |
| Packed resident | 4,358,391,856 bytes | Identical |
| Observed process peak footprint | Approximately 4.43 GB | Approximately 4.42-4.43 GB |

The remaining SHA-256 work is approximately 1.31 s and is now almost entirely
on the critical path. CommonCrypto did not materially outperform CryptoKit.
The initial library probe lacked a per-window autorelease pool; the corrected
bounded probe peaked at 42 MB RSS. Initial corrected-probe samples overlapped a
short fixture test and are not used as a precise library speed comparison.
These measurements establish neither a theoretical hardware minimum nor
universal performance on other acquisitions.

## Verification

- 14 source/packing/cancellation/cache tests passed without skips.
- All 1,073,741,824 physical detector values matched original float32 bits.
  BF/ABF/ADF, total, mean, and DPC products passed unchanged float64-reference
  tolerances. The 32.801 s audit is excluded from loading time.
- Actual native surfaces passed parity. BF/ABF/ADF, contrast, colormaps, light
  and dark appearance, FFT, failed-file retention, forced reload, ARINA folder
  replacement, and return were exercised through the native controller hooks.
  This is not a physical pointer/Finder test or a 120 FPS certification.
- A new active-load test found an existing duplicate-open guard discarding
  returns to the retained folder. The app now ignores duplicate requests only
  when idle. Three active replacement cycles passed with stable allocation, and
  the sequence is included in `Tests/NativeUI/drive_empad.py`.
- 27 native harness unit tests passed. The final test app exited normally and
  zero target processes remained.

One initial serial control failed after the detector unexpectedly became
Custom. Its cause is not established and the failed log is retained. Repeated
controls and the candidate passed. Two ARINA first-presentation records are
missing in the final extended journey; readiness is verified, but those timing
values are not reported. Seven-tilt/FPS acceptance was not rerun here.

## Reproduction and evidence

`compare.py` runs serial/pipeline/serial with checksum reuse disabled. Pass
`--windows` for the 512/256/512 staging comparison. `hash_probe.swift` checks
full-original-file SHA-256 library parity. `cancel_native.py` reproduces the
active-loading return-to-folder sequence with a small and large original.

`result.json` records samples, identities, memory, failures, and remaining gates.
Raw logs, screenshots, and source snapshots are preserved in the local
`2026-09-08 EMPAD First Open` validation archive. The final app hash is
`60314a8e4290c4ca0da940d96f4f606b8badf476bca560b258fde945bf5ec120`.
The installed app was not replaced and nothing was committed or published.
