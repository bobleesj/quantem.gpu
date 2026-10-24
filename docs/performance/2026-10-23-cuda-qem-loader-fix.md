# CUDA QEM loading: cached-block scheduling, 2026-10-23

The integer CUDA QEM reader now keeps independent file handles open, reads and
checks up to sixteen 64 MiB blocks concurrently, prioritizes blocks already in
the Linux file cache, and uploads each verified block as soon as it is ready.
The previous reader used four slots and consumed futures in file order.

For the 2048² scan with 192² detector pixels, controlled empty-file-cache loads
improved from **10.61 s to 6.59 s**. With the tail half cached, they improved
from **10.44 s to 4.78 s**. The ordinary five-input comparison improved from
**10.49 s to 5.10 s** on the largest acquisition, with changing cache residency
explicitly recorded. These are complete authenticated encoded-resident loads;
the dense tensor is not expanded during loading.

## Setup and timing boundary

Physical CUDA: RTX PRO 6000 Blackwell Workstation Edition, 96 GB; Core Ultra 7
265K; 125 GiB host RAM; one WD_BLACK SN850X 8 TB SSD, ext4, PCIe 4.0 ×4.
The host and GPU were shared with desktop applications. GPU monitoring recorded
desktop bursts, but no sustained competing scientific compute during the
controlled loads. This is not exclusive-host qualification.

Private experiment: `20261023-cuda-qem-loader-fix`. Base revision:
`e8750f3e365208421675f47bbdc6314b135f7146`, with the recorded working-tree snapshot.
Five-input measurement source fingerprint:
`6f22b5a3795fd34e30f8f837d4e97145ea2023a0455559e787b435882d1cf568`.
Candidate loader SHA-256:
`f26e63c920c5532e9b2c81ee75f7706b23c6c8e0c113ec76bf78002ff9347646`.
Frozen original loader SHA-256:
`ca2971acd642f687d4609bb07ddc027e65fc8c3ac1264a77a474bf0c290996cb`.

Each measured call uses the public `io.load` API and returns only after GPU
synchronization and complete header/body authentication. Full original-count
comparison is separate from load timing. Source data, dtype, detector validity,
encoded arrays, indexes and tolerances are unchanged. No checksum was disabled.
The five native acquisitions exclude both 96 × 96 detector inputs.

The source fingerprint includes test files. The later addition of an indexed
virtual-detector assertion changes the hardware-gate fingerprint to
`ad80a4d5d0b58638dcdf533bdfc83eeedd1fc3348ad2f95f38fd3320c1f8f6bd`;
the measured loader bytes above are unchanged.

## Controlled cache results

Two observations per arm, with arm order reversed on the second repetition.
Only the task-owned QEM file received `POSIX_FADV_DONTNEED`; no global cache
flush, operating-system tuning or process termination was performed. Cache
residency was measured using `mincore` before timing.

| Initial file cache | Original, 4 readers (s) | New, 16 readers (s) | Original physical reads | New physical reads |
|---|---:|---:|---:|---:|
| Empty, measured 0% | 10.6105 / 10.6098 | 6.6340 / 6.5435 | 38.333 / 38.333 GB | 38.333 / 38.333 GB |
| Tail half, measured 49.25–50% | 10.5290 / 10.3536 | 4.8220 / 4.7351 | 37.979 / 38.333 GB | 19.166 / 19.166 GB |

A sequential traversal of the uncached prefix can evict a cached tail before
reaching it. Scheduling cached blocks first avoids that observed reread.
Increasing to eight workers alone reached about 6.6 s in the exploratory
uncontrolled-cache sequence; the sixteen-reader controlled implementation
reaches about 5.8 GB/s when every byte comes from storage.

The cache inspection maps no readable data pages and does not fault in or pin
the full file. Its result is a best-effort scheduling hint, never a correctness
condition. Unsupported or restricted inspection retains sequential scheduling.
Pinned staging is bounded at 1 GiB, versus 256 MiB previously. Destination
allocation remains the exact encoded body size. Upload-completion events guard
each staging-slot reuse; every block is copied to its original arena offset.

## Five-input comparison

For each acquisition: three original loads, three candidate loads, then three
original loads again. First observations are separate; repeated means median of
observations 2–3, not a cold-storage claim. Each arm uses the same QEM bytes.

| Acquisition | Scan × detector | Original A1 first / repeated (s) | Candidate first / repeated (s) | Original A2 first / repeated (s) |
|---|---|---:|---:|---:|
| acquisition-03 | 256² × 192² | 0.074 / 0.068 | 0.054 / 0.055 | 0.079 / 0.071 |
| acquisition-01 | 512² × 192² | 0.279 / 0.259 | 0.169 / 0.170 | 0.251 / 0.256 |
| acquisition-04 | 512² × 192² | 0.290 / 0.279 | 0.188 / 0.189 | 0.273 / 0.279 |
| acquisition-08 | 1024² × 192² | 2.013 / 1.070 | 0.675 / 0.682 | 1.052 / 1.047 |
| acquisition-10 | 2048² × 192² | 10.495 / 10.487 | 4.964 / 5.104 | 6.574 / 8.437 |

The four smaller candidate inputs were fully cached. The largest candidate
started with only 35.9–36.3% of its 38.333 GB cached and read 24.48–24.88 GB
from storage. Original A1 started with 0.2–42.5% cached but still read about
38.25 GB each time. A2 benefited from changed cache state and varied from
6.57 to 9.88 s. The controlled table above isolates the improvement more
cleanly than this ordinary repetition sequence.

Historical MPS and Swift/Metal measurements remain in the
[five-input comparison](2026-10-23-default-five-io.md). They ran on different
storage and an Apple M5 Max with unified memory. Their repeated-cache states
were uncontrolled. GPU compute capability alone does not rank these loading
workflows. The separate [copy-only control](2026-10-23-cuda-qem-cache-recheck.md)
takes 1.33 s to move this byte count across the measured CUDA host link,
excluding storage reads, allocation and authentication.

## Why the historical 66-acquisition load was faster

Recovered original experiment records confirm the remembered result. The
prepared archive held **98,732,699,648 bytes** for 66 complete
512 × 512 × 192 × 192 acquisitions. The single-SSD median was **17.706 s**
(three observations; 5.576 GB/s). A fresh-process two-SSD load took **8.041 s**
(12.278 GB/s), followed by **1.149 s** of independent scientific-product
validation. The historical summary also records a 7.97 s observation.

Those records are `0906-261-query-ready-oneSSD-load-control` and
`0907-11-cold-prepared-load`. The recovered cold-loader source is preserved
at archived revision `e8e1155419ab430bbebb5abb6509f31612a4acf1`. This comparison
uses the original JSON and source, not a remembered estimate.

That prepared tANS archive is a different representation from the current
integer QEM container. Its load used two aligned packs, four pinned slots,
`O_DIRECT`, and direct copies into final resident owners. The cold-loader
source explicitly performs **no per-record hashing during load**. It validates
24 complete all-acquisition scientific-output arrays afterward. That is useful
product parity, but is not exhaustive comparison of all original detector
counts and does not have the current QEM load's authentication boundary.
The historical one-SSD control also used an existing context and owners;
its boundary differs from the fresh-process two-SSD case.

The present acquisition contains 38.333 GB of QEM data and as many scan points
as sixteen 512² acquisitions. Different acquisition data and codecs prevent
attributing compressed size differences to the codec alone. Dataset count or
scan dimensions alone do not specify bytes read. At 5.8 GB/s, the corrected
single-SSD reader already matches the historical single-SSD bandwidth while
retaining every block hash.

A new matched diagnostic read alternating blocks from identical temporary QEM
copies on two physically distinct SSDs. Every block hash and the public load's
header authentication remained enabled; all destination offsets were unchanged.
Both copies' file pages were evicted before each observation, with measured
0% residency. Arm order was reversed for the second repetition.

| Storage for the same 38.333 GB QEM | Observation 1 (s) | Observation 2 (s) |
|---|---:|---:|
| Single WD SSD | 6.6875 | 6.6273 |
| Second NVMe SSD | 5.9827 | 6.3011 |
| Both SSDs, alternating blocks | 4.6491 | 4.6433 |

Both NVMe links negotiated PCIe 4.0 ×4. Two-copy physical reads totaled
38.69–38.70 GB, including readahead. This diagnostic demonstrates the additional
storage-bandwidth benefit with full block authentication. It is **not** a new
public multi-disk QEM API, a striped QEM format, or a measurement of the original
66-acquisition archive. Its timing is kept separate from the five-input public
loader table and from full original-count verification.
The diagnostic subsequently compared all 154,618,822,656 original values
exactly on CUDA with zero differences. Both task-owned file copies were removed
after archiving the evidence; source acquisitions were untouched.

## Verification and limits

All **215,016,800,256 original values** across the five acquisitions were
compared exactly on CUDA, including flagged detector values: **zero
differences**. No crop, binning, precision reduction or masking substitution.

The targeted physical CUDA checks executed **23 tests successfully**; two
MPS-only collection cases were explicitly skipped and are not CUDA evidence.
The checks cover reordered and single-block loading, tail blocks, staging reuse,
exact decoded values and virtual-detector sums, corruption at the first/middle/
last block, release after failure and successful retry.
The active CUDA source installation was then updated from an unchanged original
loader, and all seven focused restore tests passed from that actual import path.
Existing processes were not restarted; a fresh process is required to import
the changed module.

The broader `scripts/check_ans_io.py --backend cuda` gate recorded **145 passed,
5 failed FP8 product cases, and 23 blocked real-data cases** because the required
Zenodo collections were not supplied. It is therefore not an overall acceptance
pass or release qualification. The native integer and typed round-trip cells
passed. Raw evidence and temporary-export lifecycle stay in the private archive.
