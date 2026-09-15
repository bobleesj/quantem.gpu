# Resident-only interaction index

## Question and implementation

Can a loaded Normal ANS resident acquire a fine detector index without reopening
the source file, re-encoding the measurement payload, or recomputing its existing
images and DPC statistics?

The experimental SPI `prepareResidentDetectorIndex(maximumAdditionalBytes:)`
builds a `radial1fine4` index directly from retained Metal buffers. It keeps Normal
encoding and its public interaction mode unchanged. It is not the existing Fast
profile: Fast also changes stream order/offset layout and substitutes compact
events in eligible streams. No UI path calls the new SPI yet.

The existing operation lock serializes preparation with reads. Index publication
is transactional; failure/cancellation retains the old resident. Repeated calls
are idempotent. `releaseResidentDetectorIndex()` frees the optional index without
releasing counts. No process-global options are mutated.

## One real seven-source trial

Apple M5, 24 GB; seven full uint16 512x512x192x192 acquisitions. Allocation budget
17,162,698,752 bytes. Initial loading is outside the index preparation timer.

| Measurement | Result |
|---|---:|
| Resident-only index preparation, seven sources | 7.741 s |
| Normal residency before | 7.085 GB |
| Indexed residency after | 10.938 GB |
| Extra retained index memory | 3.853 GB |
| Full-map equality checks, add/remove | 70/70 |
| Selected raw-DP equality checks | 7/7 |
| HDF5 rereads during upgrade | 0 |

The probe checks five masks per source (BF, wide ABF, large ADF and two center
displacements), before index preparation, after preparation, and after removal.
It also tests early cancellation, repeated preparation, unchanged original load
metrics, and restoration of resident bytes after removal.

This is parity against the same resident before modification, not an independent
original-HDF5 count oracle. The separate ANS/bit-packed mismatch remains unresolved.
No 120 FPS, end-to-end UI latency, true peak system-memory, or 16 GB fit claim.

## Why the current Fast toggle is slower

The native app currently rebuilds each resident from the original HDF5 and
prepares new display products. A prior seven-source Fast return took 26.31 s:
recorded fused decode/encoding-size work totaled 12.24 s; compact stream writing
5.21 s; metadata prefixes 0.22 s; consolidation 0.42 s. Remaining wall time includes
index construction, pipeline setup and frontend work, not a separately isolated
index measurement.

The 7.74 s prototype and 26.31 s full Fast conversion do different work. Do not call
this a full-Fast speedup. It demonstrates that source rereading is avoidable for
index preparation. To replace the existing toggle, next compare interaction rates
and exact outputs with full Fast, then implement bounded resident-to-resident event
transcoding only if the compact events are necessary. Retain existing image and
calibration objects across such a transition. Do not expose a third UI mode solely
to hide an unqualified performance change.

## Reproduce

Run `bash experiments/20260914-resident-index-only/build.sh`, then invoke
`.build/release/resident-index-probe INPUT_FOLDER INDEX_DIRECTORY` on the seven-source
fixture. Preserve stdout JSONL and stderr. Exact source/executable hashes and raw
artifact identity are recorded in `manifest.json`. No application release or push.
