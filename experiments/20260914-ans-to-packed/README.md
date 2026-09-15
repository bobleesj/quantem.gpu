# Direct resident ANS to bit-packed conversion

Backend prototype, 2026-09-14. Not enabled in the viewer.

## Result

Seven already-loaded, distinct full uint16 acquisitions convert to the existing
bit-plane resident **in 6.58 s wall time**, without rereading HDF5. This fails the
1–2 s seven-source target. Each source took 0.85–1.17 s in that run.

Apple M5, 24 GB, macOS 26.6.2; each input is 512×512×192×192. No binning,
cropping, clipping, or new hot-pixel correction is performed by the converter.
It preserves the counts and correction policy of the existing ANS resident.

| Candidate/run | Scope | Conversion | GPU measure | GPU write | Parity |
|---|---|---:|---:|---:|---|
| Dynamic bit-plane array | First acquisition | 1.355 s | 0.309 s | 0.794 s | Pass |
| Fixed registers for widths ≤2 | First acquisition | 1.062 s | 0.313 s | 0.617 s | Pass |
| Fixed vectors for remaining widths | Seven, checks between conversions | 6.476 s summed | Per-source recorded | Per-source recorded | Pass |
| Same kernels, contiguous conversion retry | Seven, checks afterward | **6.578 s wall** | 2.345 s summed | 3.251 s summed | Pass |

These are exploratory single runs, not repeated-trial speedup certification.
The final run includes per-source acceptance DP read, release, and receipt
printing in its wall timer; reference preparation and final full-map comparisons
are outside that timer. Original loading is separate: approximately 16.0 s for
seven warm-source ANS loads in the final run. No cold-I/O claim.

Normal ANS resident buffers total 7.085 GB. Final packed resident buffers total
12.046 GB (decimal). The largest **post-conversion sample** of
`device.currentAllocatedSize` is 13.029 GB while the last old source still exists;
after its release, 12.061 GB. This is not continuous process/driver peak memory,
and does not qualify a 16 GB machine.

## Implementation and ownership

The experimental `PairedRuntimeTANSPrototype` SPI exposes:

```swift
let packed = try ans.makePackedResident(
  maximumAdditionalBytes: availableBytes,
  shouldCancel: { cancellationRequested })
// Accept/publish the replacement before explicitly releasing the old source.
ans.releaseResidentStorage()
```

Call on the interaction worker, not the main thread. The call is synchronous
and serializes against queries on that source. Cancellation/failure retains the
original. The caller owns deciding when it is safe to retire the old source.

1. Decode 512-count ANS streams on Metal to measure block-32 widths and sums.
2. Scan the small per-pixel length index on the host to allocate exact output.
3. Decode again directly to compact bit planes; counts stay in registers.
4. Reuse existing exact DPC summaries and the existing packed interaction factory.

The converter stages no dense 4D cube and builds no extra Fast ANS detector
index. Shards contain 4096 scans. It supports the deployed paired resident's
uint8/uint16 count types, not general arbitrary ANS formats or uint32 residents.
It does not mark a real acquisition as exhaustively round-trip verified.

CUDA and Python/MPS already have `to_packed()` using measure/allocate/write.
This follows that ownership pattern but targets the native viewer's existing
blockwise bit-plane ABI, rather than substituting a different packed format.
No packed-to-ANS reverse entry point was found in the inspected CUDA files;
the reverse native conversion is not implemented here.

## Correctness and failure checks

- Each seven-source run: 21 complete detector maps (three annular geometries)
  and 77 complete diffraction patterns match the original ANS source exactly.
  DP samples cover first/last scan, tile, packet, and shard boundaries.
- Independent synthetic original-count oracle: 262,144 values per offset
  format, 524,288 total, including 32767/32768/65535 and detector permutation.
  Coverage includes zero, constant, raw uint16, sparse events, compact events,
  and encoder-selected entropy models 82, 84–93. This does not cover all 32
  entropy models or every possible malformed stream.
- Early cancellation, cancellation after work starts, zero-memory-budget
  rejection, and retained-source access after failure pass.
- Unknown/interleaved mode 96 fails closed in the synthetic converter test.
- Release build, synthetic test, and profile registry validation pass. The
  first build failed on a missed call-site update when making the calibration
  helper static; that call site was corrected before any successful run.

Real-data reference is the existing ANS decoder, not an exhaustive independent
original-HDF5 count oracle. No native UI/FPS test of these newly converted
residents has run. Existing cross-storage UI map-hash discrepancies from the
earlier comparison remain a separate unresolved gate.

## Reproduce

```sh
bash experiments/20260914-ans-to-packed/build.sh
.build/arm64-apple-macosx/release/ans-to-packed-synthetic
.build/arm64-apple-macosx/release/ans-to-packed-probe "$FIXTURE_FOLDER" "$INDEX_FOLDER" 7
python3 scripts/check_profile_registry.py
```

Raw receipts are retained under `local-evidence://sep14-ans-to-packed/` and
hashed in the manifest. The evidence snapshot records the dirty source state;
the base commit alone is not sufficient to reproduce it.

## Decision

Keep this as an opt-in backend prototype. Do not remove Fast ANS or change the
viewer default yet. The forward conversion is promising but seven-source speed,
reverse conversion, transactional UI replacement, progress, and post-conversion
interaction still need qualification. Retaining both representations could
make return instant, but costs approximately their combined residency and is
not a free or low-memory solution.
