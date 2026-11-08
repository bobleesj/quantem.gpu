---
name: h5-to-qem
description: Convert 4D-STEM acquisitions (ARINA *_master.h5 sessions) into lossless, ANS-encoded .qem copies on the GPU with `quantem-gpu convert --verify --source-hashes`, to reduce storage and keep one checksummed research format. Use after copying a new microscope session, or when asked to "convert to qem", "h5 to qem", "compress the data", "reduce storage", "archive the session", or whether QEM saves space.
---

# h5-to-qem: HDF5 to lossless .qem on the GPU

A `.qem` copy holds the detector counts decoded on the GPU, stored in the narrowest unsigned type that holds every value (the source dtype is recorded) and ANS entropy-coded, with checksums, calibration and the embedded master. `io.load` and QuantEM.live read `.qem` directly. `restore_master` rebuilds the original HDF5 layout. Format details: [README, "Why QEM?"](../../README.md#why-qem-one-research-format-across-detector-vendors).

## When it pays off

Measured on ARINA acquisitions, 512x512 scan, 192x192 detector, uint16, bitshuffle+LZ4 HDF5:

| Data | HDF5 | .qem | Ratio | Why |
|---|---|---|---|---|
| Atomic-resolution 4D-STEM, 30 mrad, ~2.7 counts/px, 59% zeros (45 acquisitions) | 149.8 GB | 111.1 GB | 1.35x | ANS codes value frequencies; LZ4 does not |
| Nanobeam scans, 0.85 mrad, ~0.001 counts/px, 99.9% zeros (49 acquisitions) | 9.95 GB | 8.24 GB | 1.21x | LZ4 already shrinks near-empty frames; per-frame overhead remains |

Normal-dose data saves about a quarter. Near-empty nanobeam data saves less; convert it for one format and checksums.

## Steps

1. Record calibration first (a session `dataset.yaml`, or `--voltage-kv`, `--semiangle-mrad`, `--scan-sampling-a`). Missing values are reported, never guessed.
2. Estimate without writing:
   ```bash
   quantem-gpu convert <session_folder> --dry
   ```
3. Convert with full verification, with CPU threads capped:
   ```bash
   CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
     quantem-gpu convert <session_folder> --verify --source-hashes
   ```
   - A folder converts every `*_master.h5` below it; each `.qem` lands beside its master, or under `--out DIR` mirroring the layout.
   - `--verify` compares every decoded value with the source and prints `verified: N values identical`.
   - `--source-hashes` records the SHA-256 of every original file.
   - Keep the default `--hot-pixel-correction none` for archives: every stored value is retained.
   - About 8 s per 3.3 GB acquisition on one recent NVIDIA workstation GPU.
4. Check: one `.qem` per master, every log line says `values identical`, and `quantem-gpu validate <file.qem>` passes (no GPU needed). Report total HDF5 and `.qem` bytes.
5. Keep the HDF5 originals as the acquisition archive on separate storage. Remove them from working storage only on an explicit request; conversion never deletes sources.

## Notes

- Keep different experiment types in separate folders so each converts and screens with its own calibration.
- `convert` never modifies source files and never overwrites an existing `.qem`.
