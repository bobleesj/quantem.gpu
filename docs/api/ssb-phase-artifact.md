# Portable SSB results: JSON and NumPy

Save a result as two ordinary files beside the original measurements:

```text
data/
  sample_master.h5
  sample_data_000001.h5
  live/ssb/
    sample-01.json
    sample-01.npy
```

The `.npy` contains the numerical phase, not a colored preview. JSON is readable
metadata, with no base64 arrays. Keep both files together when sharing. No raw
data, complex object, or large Fourier evidence is duplicated. A 512-square
phase is 1,048,704 bytes as NumPy v1, plus its small JSON record.

## Version 1 contract

- `format`: `live.ssb`; `schemaVersion`: integer `1`.
- `phaseFile`: companion `.npy` basename in the same directory; no traversal.
- `rows`, `columns`: integer dimensions, 1 through 4096 inclusive.
- NumPy v1, C-order, little-endian float32 (`<f4`); finite values only.
- `phaseEncoding`: `float32-le-row-major`; `phaseUnits`: `rad`.
- `phaseSHA256`: SHA-256 of raw row-major phase bytes, excluding the NumPy
  header. No uint16 conversion, clipping, normalization or quantization.
- `sourceIdentity`: lowercase SHA-256 from the existing
  `live4dstem.dataset/v0.1` master/ordered-member byte identity.
- `calibration`: `beamEnergyKeV`, `semiangleMrad`, `scanStepRowAngstroms`,
  `scanStepColumnAngstroms`, `detectorStepRowMrad`, `detectorStepColumnMrad`,
  `centerRow`, `centerColumn`; optional `brightfieldRadiusPixels` and
  `excludedDetectorPixels`, using `MetalSSBCalibration` conventions.
- `c10Nanometers`, `c12Nanometers`, `phi12Radians`, `rotationDegrees`: saved
  physical settings, independent of display contrast and colormap.
- `provenance`: nonempty string-valued object identifying producer, phase
  variant, recorded revision and calibration/identity limitations.
- `runMetadata`: readable JSON object preserving the original scientific run,
  including higher-order settings when recorded.

Publish NumPy first and JSON last. Never overwrite a different result. Repeating
identical exports is safe. Readers validate shape, dtype, finite values, units,
calibration, companion location and checksum. Unknown versions are rejected.
JSON is bounded to 32 MiB. The native explorer currently accepts 512 x 512 phase
images, although the interchange reader supports larger shapes.

## Shared implementation

Python `quantem.gpu.io.ssb_result` owns `acquisition_identity`, `export_result`
and `read_result`. The source-identity exporter currently supports ARINA masters
with external data members. It exports original `ssb_phase.npy` with matching
`computed.ssb`; it never substitutes a calibrated, picked, locked or denoised
image with different settings.

```python
from quantem.gpu.io.ssb_result import export_result, read_result

manifest = export_result(run_folder, master, expected_identity=verified_digest)
record, phase = read_result(manifest, expected_identity=verified_digest)
```

The default output is `<source folder>/live/screen/<source stem>/ssb.json`
with `ssb.npy` beside it. The source stem retains the complete filename except
its final extension. Different fits use numbered sibling folders (`-02`, `-03`);
identical exports reuse the original. The application publisher places BF, DF,
mean DP and CoM images in that same folder. This low-level exporter writes only
the verified phase and metadata. Older arbitrary JSON/NumPy pairs remain readable.

Swift `MetalSSBKernels.SSBPhaseArtifact.load` recognizes `.json` pairs;
`savePair(to:)` writes them. `MetalSSBSavedRun.savePhasePair(to:phase:)` exports
native saved runs without reconstruction buffers. Swift does not invoke Python.
Legacy `.ssbresult` remains import compatibility to preserve saved work, not
the preferred output format.

Identity excludes external paths and filenames: unchanged data moved or renamed
still matches. Rewriting the H5 master or re-encoding shards changes identity.
Initial verification reads all source bytes; it is not a subsecond operation
or part of pointer interaction. A digest recorded at export does not prove the
historical provenance of an older run.

Storage discovery, bookmarks and defaults belong to applications. Reading exact
saved phase does not establish cross-solver reconstruction parity after changing
aberrations. Tests cover signed float bits, wrong source, invalid schema/units/
shape, missing and corrupt companions, and native/Python pair interoperability.
