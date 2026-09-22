# QuantEM data (.qem), container version 1

## QEM specification 0.0.2

Changes from 0.0.1. There is no compatibility period: 0.0.2 readers refuse the
0.0.1 forms named here, and the file must be exported again from its source.

- Calibration quantities are named by array axis:
  `scan_controller/regular_scan/pixel_size_row` and `pixel_size_column`,
  `imaging_system/reciprocal_pixel_size_row` and `reciprocal_pixel_size_column`.
  The 0.0.1 names ending in `_y` and `_x` are refused in `electron_microscope`
  and in `calibration_overrides`. Source files keep their own x and y names
  inside `source_metadata`; a reader maps source y to row and source x to column.
- `processing` is required. It lists every operation between the source
  measurements and the stored measurements. Each record has an `operation` name
  and a boolean `changes_measurements`.
- Two operations are defined in addition to `lossless_storage`:
  `exact_integer_narrowing` (`changes_measurements: false`, with `source_dtype`
  and `stored_dtype`) and `flagged_pixel_replacement`
  (`changes_measurements: true`, with `method` and `pixel_count`).

### Validate a file

```bash
quantem-gpu validate acquisition.qem another.qem
```

The command needs no GPU. For each file it checks the envelope, the header and
every body checksum, the codec layout, and the metadata rules of this
specification, then prints one JSON report. The report lists `processing` and,
separately, `measurements_changed_by`, so a reader sees at once whether the
stored counts differ from the source. A file whose retained reader record shows
replaced detector pixels or a narrower stored dtype, without the matching
`processing` record, fails validation. The exit status is nonzero when any file
fails. `tests/data/qem-v2/invalid-metadata.json` lists header mutations that
every conforming reader must refuse; the Python and Swift readers share them.

## Microscopy-friendly metadata (since 0.0.1)

New writers use `quantem.scientific-metadata/2`. The binary container and
measurement codecs remain unchanged. Readers accept metadata schemas 1 and 2 subject to the 0.0.2 rules above;
older schema-1-only readers must reject schema 2 rather than reinterpret units.
Existing files are not rewritten. Previously distributed schema-1 consumers
need an updated backend before opening schema-2 files.

Schema 2 stores normalized quantities and calibration overrides in these units:

| Quantity | JSON unit | Typical display |
| --- | --- | --- |
| Scan-axis sampling | `angstrom` | Å |
| Reciprocal-length detector sampling | `1/angstrom` | Å⁻¹ |
| Angular detector sampling and convergence semi-angle | `mrad` | mrad |
| Accelerating voltage | `kV` | kV |
| Beam energy, when explicitly supplied | `keV` | keV |
| Scan dwell time | `us` | µs |
| Camera length | `mm` | mm |

For example, `{"value": 0.42, "unit": "angstrom"}` means 0.42 Å per scan
step. Applications may display another convenient unit, but must read the unit,
not infer it from a field name. Exposure, dwell and frame interval are not
synonyms; angular and reciprocal-length sampling are not interchangeable.
Original `source_metadata`, evidence and provenance are preserved. Codec-private
restoration metadata and internal calculation APIs keep their existing units;
the public scientific metadata is the authoritative human-readable unit layer.
Measurement bytes remain exact. Calibration conversion uses floating-point
arithmetic without display-style rounding; physical equivalence is tested to a
relative tolerance of `1e-14`. For identified ARINA sources, `count_time` takes
precedence over `frame_time` when recording the scan timing. The original
fields remain available separately; generic NXmx photon energy is not inferred
to be an electron accelerating voltage.

Open an acquisition, preserve its scientific measurements and calibration, save
one portable file, then restore the same measurements without the source folder.
The extension names quantitative electron microscopy, not dimensionality or a
compression algorithm. This is a project format, not a claim of an industry
standard or formal NeXus compliance.

See the [field-by-field metadata map](qem-metadata-mapping.md) and the
[portable references and validation workflow](qem-interoperability.md).

### Calibration ownership and independent readers

For schema 2, `scientific_metadata` is the sole calibration authority. Recorded
axis `sampling` owns the scan/detector step; duplicate microscope step fields
must agree within `1e-14` relative tolerance. Explicit `calibration_overrides`
take precedence for calculations, never for rewriting the recorded values.
Private restoration fields are derived conveniences, not a second authority.
Missing public calibration means unknown; a reader must not revive a stale
private value. Schema 1 retains its historical restoration contract.
Reciprocal lengths use cycles per distance, not radians per distance: there is
no implicit factor of `2π`. Angular sampling in mrad is a distinct quantity.
Known calibrated quantities require nonempty provenance. Source coverage is
explicit; unknown vendor fields may be retained but must not be guessed.

The reference CPU path is explicit (`backend="cpu"`), not a silent replacement
for an accelerated backend. It preserves exact measurements and is intended
for portability, small examples and independent verification, not speed claims.

## Scientific contract

No implicit crop, binning, masking, background subtraction, integer narrowing or
float quantization is permitted. Integer counts and floating-point bit patterns
must round-trip exactly. Two operations are permitted only when declared in
`processing`. A writer may store uint32 source counts as uint16 after checking
every stored value, refusing the file when any count exceeds 65535
(`exact_integer_narrowing`). Masked pixels are not an exception to this range
check. The record requires verified exact-count evidence from the reader, not
just a difference between source and stored dtypes. A writer may replace detector
pixels flagged in the source pixel mask, for example with the median of their valid neighbors
(`flagged_pixel_replacement`); all other pixels stay exact, and the record
states that measurements changed. A detector pixel marked invalid in the header
`valid` mask is excluded from analysis, but its stored source value must still
round-trip exactly unless replacement is explicitly declared in `processing`.
The mask never permits an undeclared change to stored values.
Axes are ordered, named and sized; row precedes column.
Each calibrated axis has an explicit value, unit and provenance. An absent
quantity is unknown, not zero. Detector sensor pitch is not specimen scan step.

The metadata vocabulary follows the microscope companion files used with ARINA:
`electron_source/accelerating_voltage`,
`illumination_system/semi_convergence_angle`,
`scan_controller/regular_scan/pixel_size_row` and `pixel_size_column`,
`scan_controller/regular_scan/dwell_time`, and `imaging_system/camera_length`,
all below `electron_microscope`. Source x maps to column and y to row.
Other detectors map to these scientific quantities without changing their
recorded detector identity. Angular and reciprocal-length sampling are distinct.

A converter that walks the complete source master declares
`source_metadata_coverage: "exhaustive"`. It stores every field under its HDF5
path and every attribute as `path@attribute`, so a unit sits beside its value
(`.../count_time` and `.../count_time@units`). Arrays longer than 64 values are
named with their dtype and shape, and travel complete in
`metadata.source_master_file`, the master file itself as zlib then base64.
`metadata.source_files` records the name, size and SHA-256 of every source file.

Keep interpreted quantities separate from `source_metadata`, which retains the
original reader-provided fields and units. `source_metadata_coverage` must say
whether this is a reader-retained subset or an exhaustive archive. A bounded
display collector must never claim complete original-file preservation.
Recorded calibration and explicit overrides are separate; overrides carry
provenance. Compression alone never claims that a correction was applied.

## Binary envelope

All integer fields are unsigned little-endian. The first 56 bytes contain:

| Offset | Bytes | Meaning |
| --- | ---: | --- |
| 0 | 8 | ASCII `QEMDATA1` |
| 8 | 8 | UTF-8 JSON header length |
| 16 | 8 | Body start, exactly 56 + header length |
| 24 | 32 | SHA-256 of the JSON header |

The JSON header is at most 16 MiB. It declares `container = "quantem.qem"`,
`container_version = 1`, `codec`, and `scientific_metadata`. Codec versions are
independent of the container version. Unknown versions or codecs fail with an
actionable error; they must never be guessed from the filename. Body checksums
cover consecutive 64 MiB blocks including alignment padding. Complete byte
coverage and bounds must be validated before GPU consumption.

The first implemented codec is `runtime-column-rans-spatial-v2`, retaining the
existing exact uint8/uint16 stream arrays and spatial indexes. Its existing
`profile`, `version`, `interval`, `shape`, `dtype`, `valid`, `chunks`, `bytes`,
`sha256`, and `metadata` fields retain their meanings. The scientific metadata
schema is dimension-independent; this first codec remains 4D-only. Naming the
container generically does not imply that every dimensionality or dtype already
has an implemented decoder.

## Migration and implementation status

The retired `.ans` containers (`QGPUSTRM` and `QGANS`) are no longer supported.
Open the original acquisition and save a new `.qem` copy; renaming a legacy file
does not migrate its header, and the readers reject those magics instead of
guessing a codec. Publishing a destination is atomic and must not overwrite an
existing file.

Native Swift and Python share the envelope and stream codec. Hardware parity,
source-format qualification, floating-point codecs and actual application UI
coverage must be reported separately from metadata-schema support. WebGPU and
arbitrary 3D/5D codecs are not implied by this contract.

### Native EMPAD floating-point codec

`float32-bit-lanes-rans-v1` preserves float32 bit patterns in two encoded uint16
lanes per detector pixel, including signed zero, NaN payloads, and infinities.
Detector geometry is stored explicitly, not inferred as 128×128. The former
`empad-xor-row-packed-v1` acquisition codec is retired: re-export original data.
See the [normative codec specification](qem-codecs.md).

`empad` stores the original format identity, reader-retained microscope metadata,
scan calibration and supplier correction statement. An explicitly selected dark
reference saves its detector-shaped mean float32 plane and calibration identity in the
checksummed header. The original sample values remain encoded unchanged; the
restored plane is subtracted once for display and products. Changing this saved
recipe currently requires reopening the original and exporting another copy.
Encoded EMPAD2 sensor words still require matching gain/dark calibration and are
not accepted as float32 merely because an XML label says float32.

### Supported entry points

- Python: `io.save("copy.qem", encoded_resident, backend="mps")`, then
  `io.load("copy.qem", backend="mps")`. The shared integer codec also has a CUDA
  reader, but CUDA execution must be qualified independently. Explicit
  `backend="cpu"` reads both integer and EMPAD float32 codecs as original dense
  measurements, and writes supported NumPy arrays. Python GPU float32 ANS-QEM
  reading is implemented; physical backend qualification remains separate. See the [Python workflow](qem-python.md) and
  [normative codec specification](qem-codecs.md).
  Normalized quantities are available as `acquisition.metadata["scientific_metadata"]`.
- Native Swift: `NativeNPYSource(url:)` followed by
  `MetalRuntimeANSResidentSource.load(array:device:)` and `saveSnapshot(to:)`
  converts C-order little-endian `uint8`/`uint16` NumPy `.npy` arrays. The four
  axes must be `(scan_row, scan_column, detector_row, detector_column)`.
  `NativeCountArray` lets DM4 and NumPy reuse the same bounded Metal encoder.
  The NumPy header is retained, but microscope sampling is unknown, not guessed.
  Floating-point, signed, Fortran-order, zipped and non-4D arrays are rejected
  with corrective guidance. This native entry point does not imply automatic
  `.npy` dispatch in the Python API.
  For native float32 NumPy and `.qem`, use `NativeEMPADSource.open` followed by
  `MetalEMPADResidentSource.load` and `saveQEM(to:)`. Read actual dimensions from
  `source.detectorShape`; do not use the static EMPAD sensor dimensions for an
  arbitrary array. Application dispatch and rendering need their own integration tests.
  Run `bash scripts/check_npy_qem_roundtrip.sh counts.npy` on Metal to verify
  every diffraction pattern and three detector masks against the input counts,
  plus metadata preservation. Add `--reject=unsupported.npy` for negative cases.
- Shared Metal conversion: `MetalQEMExporter.save(_:to:device:)` accepts
  validated count-array, indexed HDF5, or EMPAD readers.
  It owns bounded conversion and atomic output; the calling app owns scheduling,
  progress presentation, and explicit dark-reference/already-corrected choices.
  It has no dependency on app preferences, dialogs, or a particular viewer.
- Scaled/quantized products, uint32 counts and other arbitrary dense arrays do not yet
  have QEM export codecs. Unsupported exports fail rather than writing another
  container under a `.qem` filename.

`scientific_metadata.calibration_overrides` stores explicit user edits separately
from recorded quantities and axis sampling. Each entry uses the same microscope
path with `value`, `unit`, `provenance: "user_override"`, and nonempty `evidence`.
In schema 2, scan sampling uses `angstrom`; beam voltage uses `kV`; dwell time
uses `us`; camera length uses `mm`; semi-angle uses `mrad`. Detector row/column
sampling uses a shared unit of `mrad` or `1/angstrom`. Both members of each sampling pair are
required. Invalid or incomplete calibration is rejected, not guessed.

The native `calibrationOverrides` API retains its existing calculation units
(m, V, s, m, mrad, and detector mrad/1/nm/1/Å). Conversion happens only at the
scientific-metadata boundary. Schema-1 files retain their original unit meanings.
Native exporters accept `calibrationOverrides`. Omitting it preserves existing
saved edits; supplying a complete dictionary replaces them in the new copy;
supplying `[:]` clears them. `NativeQEMCalibration` validates and reads these
quantities. Native readers expose the serialized dictionary under
`qem_calibration_overrides`; applications apply it ahead of recorded calibration.
Only `.qem` carries calibration overrides; no other destination is accepted.
Python integer readers expose effective scan sampling, detector sampling and
beam voltage in acquisition metadata while retaining the complete scientific
metadata, including original values and every override. CUDA execution still
requires independent hardware qualification.

Reader-retained source metadata is preserved, not a full archive of every HDF5
object or vendor binary tag. Do not describe this as complete original-file
preservation. This populates the existing optional field in container version 1;
the binary envelope and codec are unchanged.

## Verification gates

- Original -> saved -> reopened: exact shape, dtype, bad-pixel mask and retained
  source metadata; calibrated axes and microscope quantities unchanged.
- Real independent source frames and detector reductions, not merely two reads
  through the same decoder. Distinguish sampled checks from exhaustive checks.
- Moved-file reopen without relying on original filesystem paths.
- Explicit rejection of truncated files, unknown codecs and altered checksums.
- Performance measured separately: save, checksum/read, resident restoration,
  first native presentation and interaction; include device and cache state.

Run `bash scripts/check_qem_roundtrip.sh original new-copy.qem` for a release-mode
native check. The test compares seven selected DP frames, calibration and all
reader-retained fields. Integer BF/ADF/DF images are compared in full before and
after saving, with sampled independent host reductions; EMPAD reductions are
checked against host float64 sums. A generated fixture also checks saved dark
subtraction, negative values and rejection of corrupted headers/bodies.
`scripts/check_qem_collection.sh` visits supported acquisition entry points in a
local testing collection, removes only its own temporary copies, and reports
rejections separately in its output. It never changes the original acquisitions.

Run `bash scripts/check_qem_calibration_roundtrip.sh counts-uint16.npy new-copy.qem`
to compare every original DP count, assert microscopy-friendly saved units,
restore overrides without local preferences, and check resident-save preservation
and an explicitly uncalibrated copy of the original. The command also creates
`new-copy.preserved.qem` and `new-copy.reset.qem`; all destinations must be new.
