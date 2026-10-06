# CUDA memory pool retention after ANS loading and MAPED merging - 2026-10-04

## Question

After MAPED merges seven ANS tilts on CUDA and closes them (`release_tilts=True`),
the process keeps about 8 GiB more than the 5.93 GiB merge, and `free_all_blocks()`
cannot return it. Right after loading, CuPy's default pool holds 10.55 GiB for
7.76 GiB of encoded tilts. Why, and can released residents return their memory
without changing a decoded value, slowing loading or the merge, or touching the
Apple MPS path? Can the complete MAPED run stay under 16 GB?

## Setup

- One NVIDIA RTX PRO 6000 Blackwell (96 GB), shared with other jobs. Python 3.14,
  CuPy 14.2.0, PyTorch 2.13.0.
- quantem.gpu `19c2b0a0` (base `fbfadf03`); MAPED `fac7b8ef` (merge regions of
  up to 4096 scan positions, 62 regions with `crop=True`).
- Seven native Arina tilts, 512 x 512 scan, 192 x 192 detector, uint32 files
  whose counts fit uint16. `io.load(io.discover(folder))` builds seven runtime
  ANS residents (7.76 GiB). `MAPED.merge` stores the merge with
  `io.load(MergedRegions, dtype="scaled_uint16")` (5.93 GiB, 62 regions).
- Process memory: `nvidia-smi --query-compute-apps` for the process, polled every
  50 ms (peaks shorter than that can be missed). CuPy: `MemoryPool.used_bytes()`
  and `total_bytes()`. Torch: `max_memory_allocated()` and `max_memory_reserved()`.
- Bit-exactness: SHA-256 over every array kept by every tilt and by the merged
  result, before and after, on the same MAPED commit.
- Timings: interleaved before/after processes; minima and medians from runs
  where the GPU showed 0 % utilization from other jobs.

## Cause

Both residents are `StreamedCounts` (`src/quantem/gpu/_compact/streamed.py`):
the tilts through `io/_streamed.py:175` (`source.append(raw)`), the merge
through `io/backends/cuda/precision.py:200` (`encode_ans`). Before this change,
`StreamedCounts.append` allocated the kept arrays from CuPy's default pool right
next to the scratch of the same chunk (line numbers before the change):

| Line | Array | Lifetime | Size per 16384-scan tilt chunk |
|---|---|---|---|
| 167 | `scratch` (encode) | scratch | 1.21 GB |
| 168 | `sizes`, `states` | scratch | 4.7 MB each |
| 169 | `models` | kept | 1.2 MB |
| 183 | `offsets` | kept | 4.7 MB |
| 186 | `payload` | kept | about 70 MB |
| 231, 237, 240 | `widths`, `starts`, `words` (`_index`) | kept | small |
| 116 | `valid` | kept | 36 KB |

The decoded counts (`raw`, 1.21 GB) and the decompressor buffers live in the same
pool. CuPy splits a request out of any free cached block and can only return a
block when every piece of it is free. A 70 MB payload or a 36 KB `valid` array
carved out of a freed 1.2 GB scratch block therefore keeps the whole block
reserved for the life of the resident; after `close()`, the released pieces
merge back but the block stays pinned by pieces of the next tilt or of the merge.
`load_h5_ans` freed the pool only at the start of each load, never at the end.

## Fix

1. `_compact/streamed.py`: new `retain(arrays)` copies the six kept arrays of a
   chunk into one exact allocation outside the pool
   (`cp.cuda.using_allocator()`), packed at 256-byte offsets and returned as
   views. The allocation is freed with its last view, so `close()` returns it
   at once. `append` and `index_encoded` keep `retain(...)`; `valid` and the
   per-device entropy tables are also allocated outside the pool. All
   intermediates stay pooled scratch.
2. `io/_streamed.py`: the CUDA H5 load returns its pooled scratch when it ends
   (an `ExitStack` callback that runs after the reader and corrector close).
3. `io/_precision.py`: a CUDA precision load returns cached blocks before
   converting and its own scratch after it.
4. `io/_array_resident.py`: the float32 path, which drops the count index after
   `append`, calls `retain` on the three code arrays so the dropped index bytes
   are not kept inside the shared allocation.

Decoded values and encoded bytes are unchanged; no MPS code path changed.

## Results

Memory (GiB, `nvidia-smi` for the process, same MAPED commit):

| Step | Before | After |
|---|---|---|
| After loading 7 tilts | 11.10 | 8.42 |
| Peak while loading | 11.10 | 12.03 to 12.08 |
| After loading, pool freed | 9.97 | 8.42 |
| CuPy pool after loading, used / held | 7.76 / 10.55 | 0.00 / 0.00 |
| Merge peak, tilts kept (cold / warm) | 16.94 / 17.04 | 17.08 / 17.19 |
| Route 2 peak, fresh process (3 runs) | 16.71 to 16.73 | 16.98 to 17.19 |
| After route 2 (tilts closed) | 15.08 | 6.77 |
| Plus pool freed, tilts dropped, both caches | 14.66 | 6.75 |
| CuPy pool after release, used / held | 5.93 / 13.87 | 0.00 / 0.00 |

Time (seconds):

| Step | Before | After |
|---|---|---|
| Load 7 tilts, warm page cache, quiet GPU, minimum | 3.49 | 3.56 |
| Load 7 tilts, same, typical | 3.49 to 3.58 | 3.56 to 3.63 |
| Encode and index inside the load | 0.66 to 0.68 | 0.66 to 0.67 |
| Warm merge, 10 runs, median | 2.05 | 2.09 |
| Route 2 in a fresh process (includes compiling), median of 3 | 3.83 | 3.81 |

Bit-exactness: the SHA-256 of all seven tilts' kept arrays and of the merged
result's kept arrays are identical before and after; merged RMSE 0.005432755805088518
in both.

What sets the merge peak after the fix (route 2, 62 regions):

| Part | GiB |
|---|---|
| Seven tilts plus the complete merge | 13.69 |
| Torch reserved by the merge kernel (allocated peak 1.60) | 1.90 |
| CuPy conversion scratch, pool high-water | 0.80 |
| CUDA context after compiling | about 0.7 |
| Rounding of exact allocations (112 tilt chunks, 62 merge chunks) | about 0.2 |

MAPED region size, measured with the fix only (route 2 in a fresh process,
one run each, includes compiling):

| Region frames | Regions | Time (s) | Peak (GiB) | After (GiB) | Torch reserved (GiB) | Merged RMSE |
|---|---|---|---|---|---|---|
| 4096 (MAPED default) | 62 | 3.84 | 16.98 | 6.77 | 1.90 | 0.0054328 |
| 2048 | 124 | 4.09 | 16.46 | 6.86 | 1.07 | 0.0053741 |
| 1024 | 248 | 6.35 | 15.51 | 7.00 | 0.68 | 0.0053380 |

Allocation granularity measured on this GPU (200 exact allocations each):

| Requested | Device memory per allocation |
|---|---|
| 0.6 MB | 0.69 MB |
| 1.2 MB | 2.09 MB |
| 4.7 MB | 6.29 MB |
| 47 MB | 48.2 MB |
| 70 MB | 71.3 MB |

## Conclusion

Released tilts now return their memory: after route 2 the process holds the
merge plus the context (6.77 GiB) instead of 15.08 GiB, and after loading it
holds the tilts (8.42 GiB) instead of 11.10 GiB. Output is bit-identical. Cost:
about 0.06 s on a 3.5 s load and about 0.04 s on a 2.05 s warm merge (one exact
allocation and copy per chunk, plus scratch allocated again for each tilt now
that it is returned).

The run peak does not drop; it rises by 0.1 to 0.5 GiB. Before, the merge
scratch and payloads were carved out of the 2.8 GiB of load scratch the pool
could never return; now that scratch is returned and the merge allocates its
own. The peak is 13.69 GiB of live data plus the merge working set in two
allocators (torch 1.90, CuPy 0.80) plus the context. Staying under 16 GB needs
a smaller MAPED merge working set: 1024-frame regions reach 15.51 GiB (merge
about 1.65 times slower, RMSE changes because each region keeps its own
calibration), 2048-frame regions 16.46 GiB.

The load peak rises from 11.10 to 12.03 GiB: each tilt's 16384-scan staging
(decoded counts 1.21 GB, encode scratch 1.21 GB, decompressor buffers) now sits
beside the tilts already loaded instead of being partly reused for their
payloads. It stays below the merge peak.

## Rejected ideas

- **One exact allocation per array** (`using_allocator()` around each kept
  `cp.empty`): every allocation above about 1 MiB is rounded up to whole 2 MiB
  pages. With six arrays per chunk this cost 0.35 GiB after loading (8.77 GiB)
  and the warm merge peaked at 18.04 GiB. One allocation per chunk fixes it.
- **A dedicated `cp.cuda.MemoryPool()` per resident**, as
  `backends/cuda/packed.py` does: a fresh pool still requests each allocation
  from the device exactly (same rounding), destroying the pool object frees
  memory that views may still use, and release must also free the pool.
- **Only freeing the pool at the end of each load**: kept arrays are still
  carved out of scratch blocks, so `close()` still cannot return them.
- **Keeping cached scratch for the merge to reuse** (no free before converting):
  the 1.13 GiB block left by the detector summaries stays beside the merge;
  warm merge peak 17.41 to 17.61 GiB.
- **Writing payload and index straight into the exact allocation** instead of
  copying: needs every size before the compact and pack kernels, so `_index`
  would be split in two; the copy costs about 0.05 ms per chunk on a quiet GPU.
- **`cudaMemcpyAsync` instead of an elementwise copy**: equal on a quiet GPU
  (0.225 and 0.227 ms for 70 MB including the allocation), slower under load.
- **Encoding each merge region in smaller chunks** to shrink the encode scratch:
  more chunks means more rounding, and it changes the saved chunk layout.

## Not covered

- Exact allocations bypass CuPy's free-and-retry on out-of-memory; on a nearly
  full GPU a kept array can fail where the pool would have freed cached blocks
  first.
- Other CUDA residents still allocate from the default pool:
  `CudaANSResidentCounts` and `CudaPackedResidentCounts` (`backends/cuda/_ans.py`,
  used by saved `.qem` reopening and by float16 or global-scale precision), and
  the arena in `io/_streamed_file.py`.
- The MPS path was not changed and was not rerun here.
