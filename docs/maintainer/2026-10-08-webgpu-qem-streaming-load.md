# WebGPU .qem loading: streamed, authenticated once, staged on the GPU - 2026-10-08

## Question

The browser loader (`RansResidentSet.loadQemFile`) read every payload byte twice:
once to verify the 64 MiB SHA-256 chunks, then again block by block to upload the
encoded streams. A served viewer also had to fetch each file whole before
admission could start. Can a served `.qem` stream straight into admission, be
read once, and land in GPU groups small enough for every browser to map, with
every count unchanged?

## Change

- `qemHttpFiles`: HEAD plus ranged GETs with `cache: "no-store"`; one ranged
  response per file streams the authentication chunks, filled in place by a
  BYOB reader when the body is a byte stream, copied from a default reader
  otherwise. Short and long responses are rejected.
- Admission keeps four reads and digests in flight. A chunk's 64 MiB read
  buffer returns to a pool of at most four only after its checksum matched and
  its bytes were copied; the pool empties when the stream ends or fails.
- With a device, each authenticated chunk's stream bytes are copied into mapped
  GPU groups of at most 256 MiB (browsers refuse much larger mapped
  allocations; a 4.29 GB mapping failed on Apple Metal WebGPU). Offset and model
  tables are copied from the same authenticated bytes. The decoder binds the
  staged groups with each block's byte offset (pad bits 28-29) instead of
  reading the payload again. A chunk larger than one group is split across
  groups on a word boundary, and each block a split cuts is copied whole on the
  GPU from the authenticated groups, so nothing is read after authentication.
  The per-block upload path of other sources is capped at 256 MiB per group too.
- Series compatibility is decided from the authenticated headers before any
  payload is read or staged. `RansResidentSeries` loads one resident set per
  acquisition, three loads in flight (the next one to publish and two ahead),
  after checking every header, and publishes them in file order;
  compare-grid batches group computes by set and submit once.

## Setup

- Linux, RTX PRO 6000 Blackwell GPU 0; Chrome 147 headed, Vulkan forced to the
  NVIDIA driver, adapter `nvidia`/`blackwell`, subgroup size 32 (asserted before
  any timing). Device requested with the adapter's maximum buffer limits, as the
  widget does (maxBufferSize 4294967292, maxStorageBufferBindingSize 2147483644).
- Files: two private MAPED tilt acquisitions, 512 x 512 scan, 192 x 192 detector,
  uint16, 16 chunks of 16384 scans each. File A is 1.19 GB (chunk payloads up
  to 70 MiB), file B 1.85 GB (up to 114 MiB). Served from the local disk (page
  cache warm) by a Range-capable Python server on localhost.
- Before: `dev` 3880d9a, which admits only a `File`, so the served file is
  fetched whole as a Blob first. After: this change, both from the same Blob
  and streamed with `qemHttpFiles`. Three interleaved runs each.
- Exactness: three complete patterns per file against the original detector
  file with its declared 3 x 3 median correction (CPU, 110592 values), and full
  512 x 512 BF and ADF images against an independent CUDA decode of the same
  `.qem` (524288 values), plus SHA-256 of all products across all runs.

## Results

Seconds from nothing to a resident, checkpointed set (fetch + admission and
decoder build), per run:

| File | Loader | Fetch whole file | Admission + decoder | Total | Payload groups |
|---|---|---|---|---|---|
| A 1.19 GB | dev, Blob | 0.86, 0.89, 0.84 | 4.62, 4.29, 3.82 | 5.48, 5.18, 4.67 | 1 |
| A 1.19 GB | port, Blob | 0.82, 1.34, 0.78 | 2.61, 2.46, 2.41 | 3.43, 3.81, 3.19 | 4 |
| A 1.19 GB | port, streamed | - | 2.65, 2.68, 2.65 | 2.65, 2.68, 2.66 | 4 |
| B 1.85 GB | dev, Blob | 2.73, 1.38, 1.38 | 6.77, 5.72, 5.68 | 9.51, 7.11, 7.06 | 1 |
| B 1.85 GB | port, Blob | 2.60, 1.37, 1.46 | 3.76, 3.71, 3.75 | 6.37, 5.08, 5.21 | 8 |
| B 1.85 GB | port, streamed | - | 4.09, 4.09, 4.18 | 4.09, 4.09, 4.18 | 8 |

The checkpoint build is unchanged (0.26 to 0.62 s). Every product hash is
identical across the 15 runs of each file (the table above plus the runs
below); every loader shows 0 differing values against both references.

Split path. These chunks never exceed a group under real limits, so the same
streamed loads were repeated through a device wrapper that lowers the group
limit to 32 MiB and counts the bytes of every buffer the load creates. Each
chunk is then split across groups and every block a split cuts is copied once
on the GPU. The peak equals the resident size after loading:

| File | Groups | Admission + decoder | GPU peak |
|---|---|---|---|
| A, real limits | 4 | 2.71, 2.74, 2.77 | 1.586 GB |
| A, 32 MiB groups | 34 | 2.67, 2.59, 2.60 | 1.621 GB (+34.7 MB of copied blocks) |
| B, real limits | 8 | 3.93, 4.16, 4.11 | 2.361 GB |
| B, 32 MiB groups | 66 | 4.04, 4.04, 4.03 | 2.516 GB (+155 MB of copied blocks) |

## Conclusion

Reading the payload once cuts admission plus decoder build from 4.3 to 2.5 s
(file A, medians) and from 5.7 to 3.7 s (file B). Streaming removes the
whole-file fetch, so a served file is ready in 2.7 s and 4.1 s instead of 5.2 s
and 7.1 s. Counts are bit-identical. The split path costs no measurable time and
one extra copy of each cut block in GPU memory. Blob fetch times vary from 0.8
to 2.7 s between runs in one page; the streamed path does not hold the file in
the page.

## Not taken

- Snapshot decode-kernel changes from the same sprint (byte-wide column
  metadata, checkpoint shortcuts for zero, constant, raw and sparse columns,
  sparse-event integration, a direct slot lookup table) have no recorded GPU
  parity and change WGSL that must also compile on Windows FXC; they are left
  for a separate decision.
- The opt-in stored spatial index and the large-mask rebase that depends on it.
- Per-stage `performance.measure` timings and page globals: debug
  instrumentation only.
- An earlier attempt on file B lost its page during the first baseline (dev)
  evaluation; it did not recur in the three recorded runs.
