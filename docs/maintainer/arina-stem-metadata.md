# Paired ARINA microscope metadata

## NXem companion files

For `acquisition_master.h5`, the Python loader also recognizes
`acquisition_em_metadata.h5`. Keep the master, all detector shards and this
companion together. Opening the master is sufficient; no separate metadata
argument is required.

The NXem reader checks the regular raster against the acquisition frame count
and any explicit scan dimensions before applying calibration. It reads scalar
metadata only. Scan pixel sizes are converted to Å, angular detector pixel
sizes to mrad and accelerating voltage to kV. Each quantity needs explicit,
recognized units. Row and column sampling are retained separately.

Zero or missing convergence angles remain unavailable during header inspection.
`screening.prepare` can estimate the missing semiangle from the full mean
diffraction disk when angular detector sampling is recorded. It uses the
half-plateau disk area and the angular area of one detector pixel, preserving
the estimate's provenance in `semiangle_mrad_source`. This does not independently
validate the detector calibration. An explicit recorded angle is retained.
A zero scan-controller
rotation does not establish a scan-to-detector rotation. Missing or mismatched
metadata must not become a guessed calibration. The paired file participates
in the source signature, so changing it invalidates cached metadata.

`tests/contracts/io/test_nxem_metadata.py` covers loading a nonsquare raster,
anisotropic units, saved metadata, companion changes and rejected calibration.

## Legacy STEM Metadata companion files

Opening `acquisition_master.h5` also reads `acquisition.h5`, when that
exactly paired file contains a `STEM Metadata` group. The Python loader and
native Swift catalog read its small attributes without decoding virtual images.
Keep the master, detector shards and microscope file together.

| Source attribute | Interpretation |
| --- | --- |
| `NumPixelsY`, `NumPixelsX` | Scan rows and columns, checked against detector frame count and any explicit raster |
| `Pixel Size` | Nominal scan step in nm, converted to Å |
| `IsPixelSizeCalibrated` | Only `1` permits promotion to calibrated scan sampling |
| `High Tension` | Microscope voltage in V |
| `Camera Length` | Nominal camera length in m |
| `Rotation` | Retained scan-controller setting; not used as scan-detector rotation |

These are the legacy STEM Metadata schema conventions. Other HDF5 layouts
require their own unit-bearing reader. The `Pixel Size` interpretation follows
the [RosettaSciIO ARINA reader](https://github.com/hyperspy/rosettasciio/blob/main/rsciio/arina/_api.py).

## What becomes automatic

`io.load(...)` returns a `Dataset4dstemGPU` with microscope voltage, camera
length, raw companion attributes and the paired filename in `metadata`.
Calibrated scan steps populate `sampling[:2]` in Å. A nominal step instead
appears in `metadata["scan_sampling_nominal_A"]`; the scan sampling remains
unspecified. The native inspector displays the nominal value and its warning.

Detector sensor pitch is not scan sampling. Camera length alone is not enough
to derive a calibrated angular or reciprocal detector step. No convergence
angle, detector sampling or scan-detector rotation is invented by this reader.

The microscope voltage takes precedence over a conflicting detector
`photon_energy` setting. Both source values and a warning are retained.
Unreadable or mismatched companion files cannot supply calibration. Native
catalog cache keys and Python source signatures include the paired file, so
updating or removing it invalidates earlier metadata evidence.

Saved `.qem` copies retain the original attributes, nominal step, calibration
flag and warning in `source_metadata`, as well as valid microscope quantities.

## Checks

- Python: `tests/contracts/io/test_stem_metadata.py` covers inspection, loading,
  saved metadata, nominal calibration and mismatched companions.
- Native: `tests/hardware/metal/test_catalog_calibration.py` covers the same
  source interpretation, cache refresh, removal and existing NXem behavior.
- Real-device acceptance separately checks original HDF5 GPU loading and the
  native inspector. Metadata tests alone do not qualify detector decoding.
