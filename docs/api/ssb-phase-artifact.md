# Portable SSB results: JSON and NumPy

Save a result as two ordinary files beside the original measurements:

```text
data/
  sample_master.h5
  sample_data_000001.h5
  live/screen/sample_master/
    ssb.json
    ssb.npy
```

The `.npy` contains the numerical phase, not a colored preview. JSON is readable
metadata, with no base64 arrays. Keep both files together when sharing. No raw
data, complex object, or large Fourier evidence is duplicated. A 512-square
phase is 1,048,704 bytes as NumPy v1, plus its small JSON record.

## Schema 2 contract

- `schema`: `quantem.ssb-phase/2`. Records of version 1 (`format: live.ssb`,
  `schemaVersion: 1`, camelCase keys) are refused; export them again.
- `phase_file`: companion `.npy` basename in the same directory; no traversal.
- `shape`: `[rows, cols]` integers, 1 through 4096 inclusive.
- NumPy v1, C-order, little-endian float32 (`<f4`); finite values only.
- `phase_encoding`: `float32-le-row-major`; `phase_unit`: `rad`.
- `phase_sha256`: SHA-256 of raw row-major phase bytes, excluding the NumPy
  header. No uint16 conversion, clipping, normalization or quantization.
- `source_identity`: lowercase SHA-256 from the existing
  `live4dstem.dataset/v0.1` master/ordered-member byte identity.
- `calibration`: `voltage_kV`, `semiangle_mrad`, `scan_sampling_A` and
  `detector_sampling_mrad` (`[row, col]`), `bright_field_center_px`
  (`[row, col]`, full detector); optional `bright_field_radius_px` and
  `excluded_detector_pixels`. The detector sampling is the one the fit used.
- `aberrations`: `C10_nm`, `C12_nm`, `phi12_deg` (scan frame);
  `scan_detector_rotation_deg` in [0, 360) and `detector_transposed`: saved
  physical settings, independent of display contrast and colormap.
- `provenance`: nonempty string-valued object identifying producer, phase
  variant, recorded revision and calibration/identity limitations.
- `run_metadata`: readable JSON object preserving the original scientific run,
  including higher-order settings when recorded.

The exporter reads only a stored fit block of the current fit schema
(`computed.ssb` with `"schema": "quantem.ssb-fit/7"`); another block is
refused with the reason.

Publish NumPy first and JSON last. Never overwrite a different result. Repeating
identical exports is safe. Readers validate shape, dtype, finite values, units,
calibration, companion location and checksum. Unknown versions are rejected.
JSON is bounded to 32 MiB. The native explorer currently accepts 512 x 512 phase
images, although the interchange reader supports larger shapes.

## Shared implementation

Python `quantem.gpu.formats.ssb_phase` owns `acquisition_identity`, `export_result`
and `read_result`. The source-identity exporter currently supports ARINA masters
with external data members. It exports original `ssb_phase.npy` with matching
`computed.ssb`; it never substitutes a calibrated, picked, locked or denoised
image with different settings.

```python
from quantem.gpu.formats.ssb_phase import export_result, read_result

manifest = export_result(run_folder, master, expected_identity=verified_digest)
record, phase = read_result(manifest, expected_identity=verified_digest)
```

The default output is `<source folder>/live/screen/<source stem>/ssb.json`
with `ssb.npy` beside it. The source stem retains the complete filename except
its final extension. Different fits use numbered sibling folders (`-02`, `-03`);
identical exports reuse the original. The application publisher places BF, DF,
mean DP and CoM images in that same folder. This low-level exporter writes only
the verified phase and metadata. Older arbitrary JSON/NumPy pairs remain readable.

Swift `MetalSSBKernels.SSBPhaseArtifact.load` reads `.json` pairs of this
schema and `savePair(to:)` writes them, with the same keys and checks as the
Python reader. `MetalSSBSavedRun.savePhasePair(to:phase:)` exports native saved
runs without reconstruction buffers: the native engine works in Angstrom and
radians (`MetalSSBAberrations.c10Angstroms`, `c12Angstroms`, `phi12Radians`), and
the record states them once, as `C10_nm`, `C12_nm` and `phi12_deg`, with the
rotation in [0, 360) and `detector_transposed` false. Swift does not invoke
Python. Single-file `.ssbresult` copies and version-1 pairs are refused; export
them again.

Identity excludes external paths and filenames: unchanged data moved or renamed
still matches. Rewriting the H5 master or re-encoding shards changes identity.
Initial verification reads all source bytes; it is not a subsecond operation
or part of pointer interaction. A digest recorded at export does not prove the
historical provenance of an older run.

Storage discovery, bookmarks and defaults belong to applications. Reading exact
saved phase does not establish cross-solver reconstruction parity after changing
aberrations. Tests cover signed float bits, wrong source, invalid schema/units/
shape, missing and corrupt companions, and native/Python pair interoperability.
