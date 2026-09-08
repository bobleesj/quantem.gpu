# EMPAD native frame budget

Apple M5, 24 GB, 120 Hz; original float32 acquisitions; no cropping, binning,
clipping or conversion to integer counts. Live4DSTEM native hooks drive the
visible window. No physical mouse/Finder claim or cold-I/O claim.

## Controlled topology experiment

The first candidate gives each scan position four 32-lane SIMD groups instead
of one. It changes work distribution, not resident storage. The same executable
was used for the one/four/one-group sequence, selected by
`QGPU_EMPAD_DETECTOR_GROUPS=1` or `4`.

| Native presentation, updates/s | One group A | Four groups B | One group A2 |
|---|---:|---:|---:|
| Unique-pixel selected DP | 119.7 | 119.7 | 119.7 |
| Wide ABF center | 118.5 | 120.0 | 119.5 |
| Wide ABF resize | 114.5 | 119.5 | 114.5 |
| Wide ADF center | 80.5 | 119.0 | 82.5 |
| Wide ADF resize | 81.8 | 119.5 | 81.2 |

Original full AutoDisk scan `(64,64,128,128)`. Packed resident remains
236,838,288 bytes; sampled final Metal allocation is 260,521,984 bytes in all
three arms. Full original opens to first presentation: 0.929/0.967/0.942 s.
Background public-data downloading was active; these are not isolated-storage
benchmarks. No speedup in loading is claimed from the topology change.

Separate diagnostic runs measured detector GPU p95 10.723 → 5.977 ms. These
component timings are not end-to-end FPS. The panel cannot display beyond
120 Hz. Occasional 16.67–25 ms presented intervals remain in the native traces.

The prior 108.6 selected-DP updates/s was measured with a sinusoidal trajectory
over a 64×64 grid. Its 360 inputs contain only 326 consecutive distinct rounded
positions. A separately labelled unique-pixel-per-tick trajectory measures
119.7 presented updates/s; the test was strengthened, not the DP kernel sped up.

An initial control attempt timed out at BF after receiving unplanned custom
detector updates; cause was not established. It is retained as a failed run,
not counted as acceptance. The repeated full journeys and native control/
FFT/mixed-folder test passed.

## Larger original source and numerical regression

[Adan Mireles, Zenodo 17246822](https://zenodo.org/records/17246822), CC BY 4.0:
experimental MoS₂–MoSe₂ EMPAD, full `(256,256,128,128)` scan. Original RAW is
4,362,076,160 bytes. Published MD5 `c50f643c2cc87360bfdc746afd026cce` verified;
SHA-256 `5f5bbae2295aea62a1d6bd74c6611ae7c913d473c3ae83435fc85eda87abd325`.

The first full audit preserved all 1,073,741,824 original sample words exactly,
but 8/65,536 ABF sums failed the frozen `rtol=1e-6, atol=1e-6`. Signed values
nearly cancel; dropping each lane's low-order compensation before combining
partials loses absolute accuracy. The failed run remains evidence.

The revised kernel carries a high/low pair through both reduction levels using
Neumaier compensation, compiled with fast math disabled. It adds only 32 bytes
of transient threadgroup scratch, not a persistent resident cache. A small
signed-cancellation regression supplements the original-file audit. The
topology table above precedes this numerical fix; final native measurements
must be recorded separately and cannot inherit its speed claim.

## Final corrected-kernel native results

| Full original scan | Selected DP | Wide ABF | Wide ADF | First visible open |
|---|---:|---:|---:|---:|
| 64×64 | 120.0/s | 119–120/s | 118–119.5/s | 0.949 s |
| 256×256 alone | 120.0/s | 23.4–23.9/s | 18.7/s | 13.3 s |

The small case passes an explicit **118 steady updates/s minimum and p95 gap
≤8.475 ms**, not a claim of zero missed refreshes: its last wide-ADF center
trace has one 33.33 ms maximum gap. Counts, startup delay, per-input-second rate
and interval distributions remain in `result.json`. The larger speed gate fails;
functional success alone cannot produce a speed pass. `drive_folder.py` now
exits nonzero when an explicitly requested presentation gate fails.

All 1,073,741,824 MoS₂–MoSe₂ sample words are bit-exact. Corrected BF/ABF/ADF,
total, mean and CoM pass the original tolerances. The actual viewer's three
aperture buffers have maximum relative errors below 5.96e-8. Light/dark,
contrast, colormaps, FFT, failed-file retention, forced reload and return from
an ARINA folder passed. The larger resident is **4,358,391,856 bytes**; noisy
signed float32 data does not share the compact integer dataset's compression.

A mixed-size folder exposed an additional UI regression: rounding on the
64×64 reference grid skipped positions in the active 256×256 scan. Keeping
fractional shared coordinates and rounding only on the active scan restores
29.9 →119.7 selected-DP presentations/s. The final native viewer DP matches
the exact requested original pixel `(row=1,column=103)` bit-for-bit; a CPU
test compiles the production statements and checks 1,024 mixed-grid mappings.
Both distinct original acquisitions load and switch; three navigation
requests each apply once. Retained switches are not original reconstruction
timings. Mixed-folder wide apertures remain only15.7–20updates/s.

Eight backend synthetic tests and 27 native-harness tests pass. Installed
v0.0.9 is unchanged. The final app executable hash is
`78520d25c67365a9e6b2994b288ce10986d66ce7f2b7327bd41fbabad6f28c30`;
bundled EMPAD Metal source hash is
`cab19cdba3e4df52b624fd5bf0d1720c5cc1b9f7939e18d5501b5cb6604e2334`.

## Reproduction and boundaries

- Backend: build `tests/hardware/metal/swift_original_packing`; set
  `EMPAD_SOURCE_PARITY_EXE`, run `test_empad_source.py` and
  `test_empad_public.py`. For the larger source set
  `EMPAD_PUBLIC_CASE=mos2-mose2` and `EMPAD_PUBLIC_RAW` to the original RAW.
- The parity executable's `all` selection streams one decoded DP at a time to
  disk, avoiding a second full dense 4D host array during the billion-word audit.
- App: enable `LIVE4DSTEM_ENABLE_EXPERIMENTAL_EMPAD=1`, then run
  `Tests/NativeUI/drive_folder.py --capture --unique-scan` and
  `drive_empad.py --dataset mos2-mose2`. The latter compares actual viewer
  buffers with independent float64 sums, then exercises controls and recovery.
- Run `--diagnostics` with `LIVE4DSTEM_EMPAD_PROFILE=1` separately from clean
  presentation measurements. Record both executable and bundled Metal source
  hashes: Metal source is compiled at runtime and may change without changing
  the executable's hash.

Experimental development build only; no installed app, release archive,
notarization, dependency pin, public repository or push changed here. Average-DP
UI and the remaining EMPAD release gates are not certified by this experiment.
