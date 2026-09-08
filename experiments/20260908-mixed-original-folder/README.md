# Mixed original-file native validation

**Pass for the tested ARINA and experimental EMPAD workflow. NumPy is not
supported by the native app and was not included as a successful load.**

App `f8b06ac09b64e254666da4a3e67c5392302b967b`, backend
`12608ce28c6382f1413b169de747f4b4177b1202`, Apple M5 with 24 GB memory.
Executable SHA-256:
`cd219e2be0537f119a99fe2f1594c69d9e02113eed0270ef114ba6a6e692ce34`.
No app or production backend code changed. No package was replaced or released.

## Coverage

- One nested folder contains seven distinct full original HDF5 acquisitions
  and three verified original EMPAD acquisitions (PdPt, SnSe, MoS2-MoSe2).
  Hardlinks preserve original filenames, shard references, and bytes without
  another full data copy.
- Two complete forward/back traversals visited all ten, followed by rapid
  switching. All 42 navigation requests applied exactly once with no retries.
- Three additional EMPAD journeys checked actual selected-DP bits and complete
  BF/ABF/ADF images against independent float64 reductions at `rtol=atol=1e-6`.
  Maximum reported relative discrepancy was below `6e-8`.
- EMPAD journeys exercised contrast, colormaps, light/dark appearance, FFT,
  malformed-file retention without a stuck loading banner, forced reload,
  replacement by the seven-file integer folder, in-flight replacement, and
  return with the FFT fingerprint restored. Their six navigation requests
  applied once without retries.
- The repeated mixed-folder journey then compared all seven integer sources,
  exercising large ABF/ADF center and radius drags and selected-DP dragging.
- Each harness owned one native app process. Runs exited normally; no target
  process remained after teardown. No source crop, binning, or clipping.
- Screenshots were captured from the native app and inspected. The initial
  folder screenshot shows seven ARINA and three EMPAD entries, visible
  diffraction/virtual images, logarithmic DP, linear image, and FFT hidden.

These are real native windows driven by existing controller hooks, **not**
physical pointer/Finder-picker or fresh-distribution acceptance. The EMPAD
development gate was enabled. No blanket claim that all UI behavior is covered.

## Descriptive results, not an isolated speed comparison

The original NumPy transfer wrote to the same disk during these tests. OS file
pages were uncontrolled; new hardlinked paths initially lacked matching packing
metadata. No cold-I/O claim or causal performance improvement is justified.

- First traversal: seven integer loads reached resident-backed presentation
  in 2.43-2.68 seconds; the large float acquisitions in 1.77 and 2.19 seconds.
- Evicted integer return visits performed real reconstruction and presented in
  1.03-1.18 seconds. Retained switches presented in 0.034-0.056 seconds; those
  are **not** full reread timings.
- The first traversal's sampled peak allocation was 13,338,853,376 bytes,
  below its 17,162,698,752-byte budget. This is observed Metal allocation,
  not total process peak RAM or proof of no swap.
- Seven-way comparison retained approximately 14.75 GB of packed data;
  final Metal allocation was 14,891,384,832 bytes. The 1.49-second comparison
  preparation reused retained residents and is not seven-source loading time.
- Single-view integer DP: 119.7 distinct presentations/second steady; large
  ABF/ADF: 119-120. Seven-way selected DP: 120 steady, with 62.55 ms initial
  presentation delay. All-seven ADF center/resize: 73.9/93.6 updates/second;
  ABF center/resize: 91.5/117.0. Sustained 120 everywhere **did not pass**.
- First-presented markers collected after teardown are retained. A missing
  marker at an earlier polling checkpoint is not zero latency. Individual
  EMPAD recovery journeys do not contain a timing marker for every action;
  numerical/control success does not fill those gaps.

An initial EMPAD recovery attempt passed a single integer file where the driver
requires a multi-file folder. It failed that assertion, not the scientific
checks; its failure receipt is retained. The corrected invocation reran the
same checks without changing assertions and passed.

`result.json` contains redacted metrics, evidence fingerprints and original
HDF5 hashes. `summarize.py` generates it from the retained local native logs.
Full logs, screenshots, and private acquisition paths remain in the local
validation folder, not this repository.
