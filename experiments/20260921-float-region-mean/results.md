# Float32 ANS region means

## Scope and result

Apple M5, 24 GB; full 256×256×128×128 float32 EMPAD acquisition from the
public Zenodo 15987625 fixture set. No crop, bin, precision reduction, CPU
reduction, or expanded resident cube. Resident storage remains 4,337,173,504
bytes. This result concerns float32 Metal region means, not all formats or CUDA.

The previous implementation decoded all scan chunks for each region. The new
kernel skips nonintersecting chunks and gathers independent literal/constant
lanes directly. Entropy lanes stream through registers. Ordered chunk passes
preserve the reference's division and compensated accumulation order.

| Moving 43-pixel selection, 24 samples | Before median | After median | After p95 |
| --- | ---: | ---: | ---: |
| Circle | 225.523 ms | 2.484 ms | 2.581 ms |
| Rectangle | 225.607 ms | 2.735 ms | 3.232 ms |

These are compute-call wall times, **not FPS**. All 25 frozen pre-change GPU
output hashes matched exactly. A second real 128×128×128×128 float32 acquisition
was saved and reopened as .qem: all 26 region cases matched the old GPU path.
Signed, nonfinite, subnormal, constant, sparse, and entropy-lane synthetic
fixtures passed 14 cases each with 64-frame RAW and 512-frame .qem windows.
Generated .qem files were removed by the managed test runner.

## Native application verification

Live4DSTEM base 547a02e with local mean-frame state isolation, direct Metal
publication, and content-generation presentation tracing. Signed development
app executable SHA-256:
`ed91ca60125896314b1b1db83ad0a198abe5083405b378172daa30376f6ee9b6`.
This candidate is **not notarized**.

Opened the real RAW file through the native Open File panel. Drove Circle and
Rectangle movement, corner resizing, Point switching, and keyboard movement.
Displayed mean bounds matched the selected bounds after settling. Original
data and export locations were not modified. Microphone stayed off.

Short native pointer gestures are not a sustained 120 Hz input source. An
explicit test-menu trajectory therefore supplied 360 controller inputs over
three seconds, using the production controller, GPU calculation, histogram,
Metal renderer, and native window. This supplements, not replaces, pointer
hit-testing. Only positive drawable-presented timestamps were counted, with
one timestamp per content generation and duplicate display times removed.

| Sustained trajectory | Unique displayed results | Rate | Frame interval median / p95 | Maximum gap |
| --- | ---: | ---: | ---: | ---: |
| Circle, 1,457 points | 345 | 116.94/s | 8.333 / 8.333 ms | 33.333 ms |
| Rectangle, 1,849 points | 344 | 116.60/s | 8.333 / 8.333 ms | 41.666 ms |
| Circle repeat | 340 | 116.90/s | 8.333 / 8.333 ms | 41.666 ms |

Request-to-visible median was 31.48, 26.74, and 27.89 ms respectively. Display
buffering latency is not the same as update throughput. These are near-120 Hz
results, **not a locked 120 FPS guarantee** for arbitrary selection sizes.
The window was active, unoccluded, on a 120 Hz-capable screen, with low-power
mode off and nominal thermal state. The test used one verified app process.
After measurement, additional operator interaction was detected; automation
stopped and the app was left available to the user. Later input is not part of
the reported trajectory results.

## Rejected attempts and remaining limits

- Fast-math compilation changed floating-point hashes. Rejected; the selected
  mean uses a separately cached strict Metal library, leaving other codecs'
  compilation policy unchanged.
- Kernel-only acceleration did not establish native presentation performance.
  The first pointer run showed 28–31 updates/s. Separating mean state from the
  root controller and publishing directly to the Metal canvas removed avoidable
  SwiftUI publication work; sustained cadence was then measured independently.
- A first synthetic fixture omitted the EMPAD record tail and correctly failed
  reader validation. The fixture, not the reader, was corrected.
- Point/mode switching and 15 cancellation/generation guard checks passed.
  An attempted native in-flight cancellation occurred after the short trajectory
  had already finished; it is not claimed as in-flight cancellation evidence.
- Larger circles, background-corrected runtime means, and other devices require
  separate performance qualification. No universal 120 FPS claim is made.

## Reproduction

Build `EMPADRegionMeanParity` in `tests/hardware/metal/swift_original_packing`.
Run it on an original RAW/XML or .qem file with `EMPAD_MEAN_VERIFY=1` to compare
every detector value's bit pattern against the retained GPU control. Optional
`EMPAD_MEAN_QEM` must point to disposable managed scratch storage. Run
`tests/hardware/metal/test_empad_region_mean.py` with `EMPAD_REGION_MEAN_EXE`
pointing to the executable. The control environment flag is measurement-only;
it is not a user-facing dense residency mode.

For native cadence, build the app against this local backend, enable its
existing UI test hooks and frame-pacing recorder, open the original acquisition,
choose Circle or Rectangle and its size, then choose **UI Test → Run Mean Region
Drag**. Report actual content presentation timestamps, not timer ticks.
