# Live BF and ADF on a growing CUDA scan

With the detector aperture held fixed, sum only the newly encoded positions
and keep the values already displayed. A changed aperture still re-evaluates
the available scan.

The matched comparison replayed one recorded 512 × 512 scan with
192 × 192 uint16 diffraction patterns and the acquisition's detector mask
on an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (driver
580.173.02). Each implementation queried the same encoded allocation every
4,096 positions. The BF mask was a centered 22-pixel disk and the ADF mask
was a centered 30–80-pixel annulus. Query order alternated between
implementations. The reference was revision `36f22e7864918cd0326ac63884856cd018e2018c`.
The candidate ran the `preview.py` from `65ffd80776d64c30cfdd241d618d83b669bb846f`
on that same reference runtime.

| Mask | Reference median / p95 | Candidate median / p95 | Final 16 updates, median before → after |
| --- | ---: | ---: | ---: |
| BF | 1.83 / 2.85 ms | 1.17 / 2.09 ms | 2.01 → 1.15 ms |
| ADF | 1.67 / 2.23 ms | 1.00 / 1.04 ms | 2.11 → 1.02 ms |

Wall time covers the synchronous GPU-backed query through the returned host
image. The table uses 63 observations per mask and version and excludes the
first query. Detector sums agreed exactly at every update, including the
first. The CUDA live-encoder contract passed six tests, including a changed
aperture and skipped chunk arrivals. This comparison excludes ingest time,
notebook transport and browser paint. It does not establish microscope
display latency or frame rate.

Private evidence ID `codex-live-detectors-1117`.
