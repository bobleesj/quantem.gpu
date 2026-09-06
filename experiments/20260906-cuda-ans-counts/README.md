# Bounded CUDA count-ANS extraction

The narrow implementation is committed as `6c6e64f5634708e2b2fef4d58e084cccbfe6c602`,
after the independent fixture commit `db552c4`, followed by integer-index fix
`ff4e86dd197c74ef79889534cdc246332e4c76a9`. It depends on the coordinator-owned
`quantem.gpu.io._ans_contract` validation module. This is a private accelerator
primitive, not a separate loader, disk format, publication, or performance claim.

## Preserved source and scientific contract

The byte-normalized recurrence derives from the preserved
`experiments/0905-detector-rans/detector_rans.cu` and the later series runtime.
Those dirty research sources were read-only throughout. Their snapshots remain
under `outputs/representation-integration-20260906/preservation/cuda-research`.
No historical source, service, benchmark, or failed experiment was removed.
Table-ANS moving-detector research remains unintegrated: its retained code has
hardcoded geometry, runtime patching, and external acquisition dependencies.

One stream is one detector column in a bounded scan block. Tail blocks retain
all remaining positions. Arrays are payload `uint8`, offsets `uint64`, model
selectors/context offsets `uint32`, symbols/cumulative/frequencies `uint16`, and
literal flags `uint8`. Literal columns retain little-endian `uint16` values,
including 65535. Probability scale is 1 through 15; normalization lower bound is
2^23. Initial state, frequency partitions, source range, exact terminal state,
and exact consumed bytes are checked before admission.

Logical dtype remains explicitly `uint8` or `uint16`. A declaration of `uint8`
requires every decoded value to be at most 255. There is no mask, crop, bin,
rescale, float conversion, or silent truncation in this layer.

## Private integration seam

`CudaANSResidentCounts` accepts the canonical arrays and metadata dimensions.
The coordinator owns authenticated file parsing, calibration, mask semantics,
public load/save/conversion APIs, and source provenance.

- `decode_block_device(block_index)` returns a caller-owned native-count block.
- `gather_diffraction_device(scan_positions)` preserves requested order and
  duplicates; `extract_diffraction_device(row, column)` reads one native DP.
- `detector_sum_device(binary_mask)` returns exact scan-shaped `uint64` counts.
- `to_packed()` returns an independent `CudaPackedResidentCounts` with the same
  methods, not a dense staging tensor. The original source stays usable until
  the caller explicitly calls `release()`.
- `resident_bytes` and `nbytes` count physical owned arrays; `logical_nbytes`
  reports the dense equivalent. Neither is process RSS nor allocator reserve.

ANS admission traverses all streams without a dense output allocation. A later
random DP decodes only each requested detector stream's prefix within its block,
not all preceding scan blocks. Duplicate requests currently repeat that work.
No constant-time ANS access or real-time timing is claimed.

## Exact direct ANS-to-packed conversion

The private packed layout consists of `uint32` words, `uint64` word offsets, and
`uint8` bit widths from 0 through 16. Streams are word-aligned; a zero-width stream
owns no payload words. Width 16 retains the complete native `uint16` range.

The first pass decodes streams to a bitwise-OR width and word count. A small
prefix sum allocates disjoint word spans. The second pass decodes directly into
those spans. No full decoded tensor is created. Direct DP access then extracts
bits without entropy prefix decoding. Binary-mask sums fuse extraction with
exact integer accumulation.

`conversion_owned_buffer_peak_bytes` includes retained ANS arrays, new packed
arrays, and named index/error scratch. It is a logical live-buffer count, not a
sampled driver, allocator-reserve, RSS, or whole-card peak. Framework scratch and
context overhead still need physical measurement. The conversion checks
addressability and available device bytes before its payload allocation; the
original remains intact if conversion fails.

## Executed checks

| Check | Computer/runtime | Result | Boundary |
|---|---|---|---|
| Focused IO contracts plus new hardware cases | MacBook Pro, Apple M5 Max; Python 3.13 | 159 passed, 11 skipped | 10 explicit CUDA ownership skips; one existing opt-in skip |
| Actual integer kernel serial harness | Apple clang 21.0.0 | 6 passed | CPU-only recurrence/bit-layout adjudication, not GPU scheduling or memory behavior |
| Nine kernel compilation | Linux, NVRTC 13.1.115, compute_120 | Passed, empty compiler log | Compiler only; no CUDA context or launch |

The fixture is a native `(3, 5, 2, 3)` array with four-position blocks and a
three-position tail. Independent encoder/reference logic checks ordinary counts,
constant contexts and rare maximum literals. The serial harness compiles the
actual `.cu` source, replacing only launch/intrinsic/atomic scheduling with
serial host equivalents. It checks exact DPs, duplicate order, mask products,
malformed states, declared-dtype overflow, ANS-to-packed conversion, and bit
widths 0/1/7/16 including word crossings. It does not establish GPU performance,
concurrent atomic correctness, stream lifetime, physical memory, or real-data
qualification.

Review found a signed/unsigned NumPy addition could promote flattened scan
indices to floating point. `ff4e86d` normalizes both coordinates to uint64 first;
the added no-GPU test preserves ordered duplicate indices above 2^53 exactly.
The earlier 158-pass log remains retained rather than overwritten.

The copied coordinator validation dependency remains deliberately untracked in
this subtask worktree. Its patch SHA is recorded in the manifest, and the exact
runtime source archive and dependency patch are preserved under the raw log
directory's `runtime-ff4e86d/`. This subtask does not claim a standalone clean
source checkout before the coordinator integrates its shared dependency.

```bash
PYTHONPATH=src python -m pytest tests/contracts/io \
  tests/hardware/cuda/test_cuda_ans_counts.py -q

# Only after obtaining an uncontended owned CUDA window:
QUANTEM_CUDA_ANS_TEST=1 CUDA_VISIBLE_DEVICES=GPU_UUID PYTHONPATH=src \
  python -m pytest tests/hardware/cuda/test_cuda_ans_counts.py -q
```

Raw local logs are in `outputs/representation-integration-20260906/cuda-ans-counts/`.
The Linux compiler source and runner are also retained at
`local-evidence://representation-integration-20260906/cuda-ans-counts/`.
Final inspection found service PID 2303 on both devices and Sunshine PID 19980
on device 0. They were not stopped, changed, or used for these checks. No task GPU
process, loop, tunnel, or server was started.

## Remaining gates

Physical CUDA execution, complete real-data parity, repeated cold/warm/prepared
timing, peak driver/allocator/host memory, full API dispatch, source calibration,
end-to-end application interaction, and throughput remain unqualified. No
benchmark registry was relabeled. SSB and derived Fourier storage are out of
scope for this change.
