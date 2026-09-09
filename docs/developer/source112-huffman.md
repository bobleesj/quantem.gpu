# Explicit Source112 Huffman64 representation

The browser Source112 loader accepts an explicit backend representation option:

```typescript
const source = await Source112ResidentSet.loadFiles(
  device, grantedFiles, status, signal, { representation: 'huffman64' },
);
```

Omitting the option preserves the existing tANS path. This is an internal backend
selection for fresh stock-widget qualification, not a new scientist-facing codec
control or a 120-update/s claim. Both paths retain all66 native uint16 acquisitions,
original source hashes, detector validity policy and exact integer products.

The selected path authenticates original source records and derives deterministic
canonical Huffman codebooks from the already authenticated ANS decoding metadata.
No additional source, codebook download or CPU native-count decoding is required.
The tree uses stable lexicographic symbol-list ties and canonical length/symbol
ordering, matching the independently validated Python reference.

Preparation happens before the load promise resolves. The loader skips the old
9.443GB restart cache and initial product bindings. It repacks and checks each
payload group while its original bytes remain alive, then builds checked hot64
and cold128 checkpoints after all groups are replaced. It then converts only the
checkpoint metadata to hot32/cold128: hot cursors use 11-bit fields, cold state/cursor
triples occupy two words, and stream starts use 16-bit block-relative offsets plus
one absolute base per 32 streams. Each group is built and independently verified
before its old cache is retired. All hot native values,
retained dense words and sparse component words must pass complete coverage gates.
The final sum and pattern/gather pipelines are adopted together. Every compute
stage retains eight storage bindings and one uniform; dense sums use the validated
WG128 bank-padded implementation with 16 columns per workgroup. No intermediate
cache generation is exposed to scientific calls.

The source owns its buffers and preparation state directly. No instance method
replacement, external owner cast, browser global, manual pause or GC hook is used.
Abort, corruption, allocation failure or device loss releases the incomplete
source. Legacy tANS cache mutation on a Huffman source requires a reload. A failed
error-scope pop cannot consume a caller-owned scope.

`loadProfile` includes the selected representation, preparation time, complete
native/retained-word counts, cache bytes, `restartCacheLayout` and logical peak.
For this path the final layout is `huffman64-compact`; `denseRestartStatus()`
reports 64-value hot segments and actual compact byte counts. Preparation time
includes both metadata stages. The peak is a monotonic
lifetime high-water mark for owned buffers plus representation-preparation scratch,
including later lazy display images and aligned display uniforms. It excludes
renderer-owned resources and temporary scientific readbacks. Logical admission uses
the previous tANS cache allocation budget for payload conversion, then bounds
compact-cache conversion to 2 GiB above the intermediate Huffman64 allocation.
The 64-value layout uses fewer checkpoints than the earlier 32-value layout.
This reserves more space for browser display surfaces while retaining the complete
encoded source and exact detector results. Physical memory and drag throughput
require fresh device qualification; logical allocation success alone is insufficient. These logical bounds
do not establish physical VRAM availability or guarantee immediate driver retirement. Actual device OOM scopes remain authoritative. `readyMs` includes
source loading and representation preparation; widget first-presentation timing
must additionally include its first scientific product and draw.

CPU validation covers deterministic books, malformed metadata, preparation
blocking, abort/failure cleanup, scope ownership, unchanged default loading and
explicit-load readiness ordering. Production shader bodies match the private
GPU-proved migration/decoder implementation. A fresh stock load without diagnostic
retirement hooks, original-reference scientific parity and sustained live trials
are still required before changing the default.

### Authenticated local record prefetch

Local loading reads and authenticates up to four queued records in parallel
while the current record uploads, with a one GiB logical host reservation. GPU admission remains ordered. Two 16 MiB mapped staging buffers alternate
CPU copies and GPU buffer copies, followed by one completion fence per record.
The ring owns 32 MiB and is disposed, with all map promises drained, before
Huffman payload or checkpoint preparation allocates additional buffers. Every record still passes
its preserved SHA-256 digest before any of its bytes reach the GPU. Pending
Blob reads/digests cannot be cancelled; abort and failure therefore drain their
handled result before owner cleanup completes.

The fixed 1 GiB host reservation counts the current raw record and each queued
record's raw bytes plus a possible WebCrypto input snapshot. A digest completion
releases its snapshot reservation; releasing a yielded record releases its raw
reservation. Reads start only when both the four-record queue limit and the byte
budget permit. Individual records remain limited to 256 MiB. Oversized records
reject before disk reads. This measures application reservations, not browser
internal memory or driver upload staging. Prefetch restarts at each payload group
boundary; all queued operations drain on exit, including consumer failure.

Load profiles distinguish `payloadReadMs` (sum of disk read durations),
`payloadHashMs` (sum of digest durations), `payloadReadWaitMs` (time admission
waited for authenticated records), `payloadStageMs` (ordered upload/fence time),
`payloadLoadMs` (whole payload loading interval), and `metadataAndSetupMs`
(metadata admission and pipeline setup). `peakHostPrefetchBytes` records the
maximum reservation. Overlapping timings must not be summed as elapsed time;
`readyMs` still measures the complete load, including representation preparation.
These fields enable a fresh physical-device measurement; they do not establish
any loading-speed target.

### Mapped ingress profile

`payloadHostCopyMs` measures copies into mapped staging; `payloadCopySubmitMs`
measures command encoding/submission. `payloadMapPendingMs` sums asynchronous
mapping durations, while `payloadMapBlockedMs` measures actual waits for reusable
slots. `payloadFenceWaitMs` measures the record completion fences.
`payloadStageMs` remains the elapsed complete-record ingress duration. Mapping
intervals overlap; these counters must not be added together as wall time.

`payloadStagingBytes` is the current logically owned ring allocation, zero before
checkpoint preparation. `peakPayloadStagingBytes` is 32 MiB. The lifetime
`peakResidentBytes` includes this staging allocation while it exists. These are
owned GPU buffer bytes, not browser RSS, CPU copy snapshots or immediate physical
VRAM retirement. No adapter, extra scientific binding or scientist-facing option
is introduced by the ingress path.
