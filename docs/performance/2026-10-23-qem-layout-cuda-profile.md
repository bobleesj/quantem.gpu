# QEM index size and CUDA loading, 2026-10-23

Historical six-acquisition evidence. The [current five-acquisition selection](2026-10-23-default-five-io.md) excludes 96 × 96 detector inputs from future runs and active examples; the completed measurements below are retained unchanged.

Removing the spatial index does not fix acquisition-06's expansion. Its ANS
payload alone is 1.645 GB, already larger than the entire 1.367 GB HDF5 input.
The 2048² CUDA load is predominantly waiting for storage reads in the observed
profiles; a separate pinned transfer control reaches 28.76 GB/s.

Private experiment: `20261023-qem-layout-cuda-profile`. This follow-up retains
the same scientific source fingerprint as the earlier six-acquisition suite,
`2873ac0ffac23c49c89a7d5311ffbcc8836ead6af640acf6df10f5a292290df8`,
on base revision `e8750f3e365208421675f47bbdc6314b135f7146`. Only benchmark
accounting and profiling instrumentation changed. CUDA ran on the same physical
RTX PRO 6000 Blackwell Workstation Edition, with a PCIe generation-5, eight-lane
link, Python 3.14.6, CuPy 14.2.0 and CUDA runtime 12.9.

## Byte accounting

The existing converter recreated all six acquisitions, with exhaustive source
verification: 217,432,719,360 values compared and zero differences. No counts,
flagged pixels, dimensions or precision changed. Acquisition-06 again used the
explicit disposable benchmark retry after the production converter declined a
larger copy. Complete byte counts match the original CUDA-produced files.

The authenticated QEM directory accounts for ANS payload, stream offsets,
models, spatial words, spatial starts, spatial bit widths, padding and envelope
bytes. GB is decimal. The final column subtracts the three spatial arrays from
the complete file; it is not a supported file or a measured load without indexes.

| Acquisition | HDF5 (GB) | ANS payload (GB) | Spatial index (GB) | Full QEM (GB) | QEM minus spatial index (GB) |
|---|---:|---:|---:|---:|---:|
| 03 | 0.717 | 0.434 | 0.032 | 0.489 | 0.457 |
| 06 | 1.367 | 1.645 | 0.047 | 1.715 | 1.669 |
| 01 | 3.040 | 1.828 | 0.132 | 2.054 | 1.923 |
| 04 | 3.374 | 2.107 | 0.153 | 2.355 | 2.202 |
| 08 | 13.476 | 8.428 | 0.607 | 9.413 | 8.806 |
| 10 | 54.595 | 34.414 | 2.409 | 38.333 | 35.924 |

For acquisition-06, spatial indexes are only 2.71% of QEM. Removing them leaves
a file-size estimate 22.08% above HDF5. Removing every other structure as well
would leave the ANS payload, still 20.35% above HDF5. This establishes that the
current ANS count representation is the principal cause of expansion; the
contributions of model selection and the input's statistics have not been
isolated. HDF5 already uses bitshuffle plus LZ4.

For acquisition-10, spatial indexes occupy 6.28% of QEM. Their removal alone
would leave 35.924 GB to store, authenticate and load, with functionality and
index reconstruction costs requiring a separate implementation and benchmark.

## Loading times

These are the retained matched suite's repeated medians, not the new profiled
loads. Each cell is HDF5 / QEM in seconds. Geometry, first observations and
verification boundaries remain in the [complete comparison](2026-10-23-default-six-io.md).

| Acquisition | CUDA | MPS | Swift/Metal |
|---|---:|---:|---:|
| 03 | 0.21 / 0.08 | 0.24 / 0.03 | 0.38 / 0.07 |
| 06 | 0.24 / 0.22 | 0.36 / 0.11 | Unsupported / 0.22 |
| 01 | 0.44 / 0.26 | 0.77 / 0.12 | 1.41 / 0.27 |
| 04 | 0.55 / 0.29 | 0.78 / 0.14 | 1.53 / 0.29 |
| 08 | 1.48 / 1.12 | 3.68 / 0.52 | 6.32 / 1.09 |
| 10 | 11.96 / 10.47 | 12.11 / 2.20 | 24.28 / 4.54 |

The statistic is the median of observations two and three. Loads stop at
encoded GPU residency. OS caches were not controlled. Native acquisition-06
loads an independently verified MPS-produced QEM. Apple and CUDA observations
use different computers and storage; these are not isolated GPU comparisons.

## CUDA contention and profile

The ten-second pre-run observation showed no sustained device compute
activity. The cooperative GPU lock was acquired for both the conversion suite
and the follow-up profile. A resident scientific service remained present, but
no active competing scientific computation was identified in the sampled
process telemetry. Desktop graphics activity was present: individual display
processes briefly reached 14% SM utilization during the suite. This was a
shared desktop, not an exclusively idle device.

An existing rsync checksum verification was also active, with one observed
two-second interval reading 475 MB. Its source and destination were two other
physical disks, not the benchmark NVMe. It could still affect shared CPU or
page-cache resources; its contribution was not isolated. No other process was
stopped. This does not establish that all background activity was harmless.

New ordinary QEM reloads for the 2048² acquisition took 17.27, 10.61 and
18.41 seconds. These variable observations are retained separately from the
earlier 10.47-second repeated median. Two subsequent interval-only profiles
of the same verified 38.333 GB file gave:

| Profile | Load wall time (s) | Process storage reads (GB) | At least one read call active (s) | At least one SHA-256 call active (s) |
|---|---:|---:|---:|---:|
| First | 24.12 | 38.138 | 23.90 | 7.50 |
| Second | 12.47 | 37.933 | 12.34 | 6.81 |

Read and checksum intervals overlap across four workers; the columns must not
be added. Read calls include waiting and copying into pinned buffers. The
Linux process storage-read counter increased by 99.0% of the file size on the
second profile, despite preceding loads. These observations support storage
reads and the read/hash pipeline as the dominant observed bottleneck, rather
than another scientific job saturating the GPU. They do not isolate media
latency from scheduling or explain every page-cache eviction.

A separate bounded control copied 2 GiB through a reused 64 MiB pinned host
buffer and device buffer. Three CUDA-event measurements were 28.764, 28.762
and 28.748 GB/s, a median of 28.762 GB/s. This demonstrates substantially more
transfer capacity than the approximately 3.07 GB/s complete second profiled
load. It is a transfer control, not a 38 GB acquisition benchmark. Host upload
enqueue timings are not presented as GPU transfer durations.

QEM loading does not re-encode or expand all counts, so low compute-engine
utilization alone does not demonstrate inefficient GPU execution. The HDF5
CUDA path already encodes this acquisition in about 1.20 seconds in the
earlier measurements. QEM removes that work but must still read, checksum and
transfer tens of gigabytes. Storage savings are about 30%, not an order of
magnitude, and cannot by themselves imply a large loading speedup.

## Reproduction and artifacts

`scripts/benchmark_qem.py` now includes integer layout accounting by default.
Add `--profile-reload` for one additional instrumented Python load after normal
reload trials. The profile records selected host call intervals and process
IO counters without weakening checksum or source verification.

The private archive retains source and runner identities, all six exact byte
ledgers, full-count results, process/device telemetry, storage-device checks,
the interval-only profile, transfer controls, and static IEEE-style PNG/PDF/SVG
tables. Initial combined cProfile function statistics were excluded because
their cross-thread accounting was inconsistent; conclusions use the separate
interval-only follow-up. Existing timing records were not replaced. No codec
optimization, index-free reader or release qualification is claimed.
