# QEM metadata mapping and completeness

This is the implemented container-v1, scientific-metadata-schema-2 contract,
documented as QEM specification 0.0.3, not a claim that every vendor
field is understood. The [format specification](qem-format.md) owns the binary
envelope. The format is independent of the compression algorithm and app version.

## Status vocabulary

- **Preserved:** reader-retained source fields are carried without replacing them.
- **Normalized:** a known quantity is mapped into a common path with explicit units.
- **Missing:** no valid recorded quantity exists; omit it, never invent a zero.
- **Unsupported:** the current reader cannot interpret or export that representation.

Normalization and preservation are complementary. `source_metadata_coverage` is
`reader-retained`, not `exhaustive`. Keep original acquisitions for archival use.

## Field-by-field map

Paths below are relative to `scientific_metadata`. Microscope paths are relative
to its `electron_microscope` object. Source `y` means row and `x` means column.

| Quantity | Common destination | Input recognized by native reader | Saved unit/status |
| --- | --- | --- | --- |
| Scan/detector shape | `axes[].name`, `axes[].size` | Validated source dimensions; DM4 axes reversed into scan row/column, detector row/column | Preserved, four named axes |
| Count representation | header `dtype`, `shape` | Native uint8/uint16; native EMPAD float32 codec | Preserved; no implicit narrowing |
| Scan row/column step | `axes[0:2].sampling`, `scan_controller/regular_scan/pixel_size_row` and `pixel_size_column` | ARINA/NCEM scan calibration; DM4 `ImageData.Calibrations.Dimension.*.Scale/Units`; EMPAD reader calibration | Normalized to `angstrom`; unknown omitted |
| Detector row/column step | `axes[2:4].sampling` | Reader detector calibration; DM4 reciprocal-nm axes | Normalized to `mrad` or `1/angstrom`; angular and reciprocal-length units remain distinct |
| Beam voltage | `electron_source/accelerating_voltage` | NCEM same path (V/kV); ARINA `entry/instrument/detector/incident_energy` (eV/keV); DM4 `ImageTags.Microscope Info.Voltage` (V) | Normalized to `kV`; existing electron-energy conversion policy unchanged |
| Convergence semi-angle | `illumination_system/semi_convergence_angle` | NCEM same path with rad/mrad | Normalized to mrad; not inferred from detector pitch |
| Dwell time | `scan_controller/regular_scan/dwell_time` | NCEM same path with s/ms/us/µs/μs | Normalized to `us` |
| Camera length | `imaging_system/camera_length` | NCEM same path with m/cm/mm | Normalized to `mm` |
| Angular detector sampling | `imaging_system/reciprocal_pixel_size_row` and `reciprocal_pixel_size_column` | NCEM `reciprocal_pixel_size_y` (row) and `reciprocal_pixel_size_x` (column) with rad/mrad | Normalized to mrad; named by array axis, not a conversion to reciprocal length |
| Detector identity | `source_metadata.camera_model`, `camera_id` | DM4 `ImageTags.Acquisition.Device.Source Model/Source ID`; equivalent retained camera fields | Preserved; never replace K3 identity with ARINA |
| Acquisition processing | `source_metadata.acquisition_processing` | DM4 `ImageTags.Acquisition.Parameters.High Level.Processing` | Preserved text, not assumed to mean background-corrected |
| Acquisition date | header `metadata.acquisition_date` | DM4 `ImageTags.SI.Acquisition.Date`; reader-provided date | Preserved when present; not a required normalized microscope field |
| Vendor fields | `source_metadata` | Fields retained by the source reader; DM4 names prefixed `dm4.` | Preserved subset, not every vendor object |
| Original XML/JSON documents | `source_documents[]` | Native EMPAD companion XML and explicitly attached UTF-8 XML/JSON | Complete text, filename, media type and SHA-256; separate from interpreted quantities |
| User calibration | `calibration_overrides` | Explicit scan/detector sampling, voltage, semi-angle, dwell, camera-length edits | Normalized values, units, `user_override` provenance and evidence; original fields stay separate |
| Lossless storage history | `processing` | Exporter operation | `lossless_storage`, `changes_measurements=false` |
| Dark/background recipe | header `empad` | Explicit native EMPAD dark reference and supplier correction evidence | Saved recipe/plane/identity; packed sample remains unchanged; subtraction applied once |
| Bad-pixel validity | header `valid` for integer codec | Reader detector-validity mask | Preserved independently of raw counts |
| Specimen (0.0.3) | `sample` (id, name, geometry, growth_direction, orientation_relationship) | Session `dataset.yaml` `specimen:` at conversion | Declared; provenance `dataset.yaml` with its sha256 as evidence; never derived |
| Specimen components | `sample/components/<label>` (role, chemical_formula, zone_axis, cif) | `specimen.components.<label>`; the CIF file as a JSON document in `source_documents[]` | Declared; lattice and space group read from the CIF, not stored |
| Thickness estimates | `sample/components/<label>/thickness_estimates[]` | `files.<n>.thickness.<label>[]` (`value_nm` becomes `value` in `angstrom`) | Declared estimates with method and region; automatic readings stay in result files until a person records them |
| Components in view | `sample/components_in_view` | `files.<n>.components_in_view` | Written only when declared |

Only finite positive calibration quantities with recognized units are promoted.
Missing metadata is not fabricated. A supported field may still be missing in a
particular acquisition. Known metadata in Python and Swift uses the same unit
contract, but runtime/source-reader qualification remains separate.

## Source-specific boundaries

| Source | What is supported | What must not be claimed |
| --- | --- | --- |
| K3 DM4 | Native uint8/uint16 4D counts, calibrated reciprocal-nm detector axes, retained selected-image tags | Arbitrary DM4 images/dtypes, every vendor object, inferred scan calibration from sensor pitch |
| ARINA/NCEM HDF5 | Reader-qualified counts and companion microscope vocabulary | Archiving every HDF5 object, every acquisition variant, or every private tag |
| EMPAD | Native float32 codec; explicit Python CPU reference encode/decode and float-export XML/RAW import | Treating encoded EMPAD2 words as corrected float32; Python GPU QEM float decoding remains unsupported |
| NumPy | Native little-endian C-order 4D uint8/uint16, retained NumPy header | Microscope metadata that was not supplied; signed/float/Fortran-order export through this count codec |
| Derived/scaled results | Separate result contracts | Arbitrary scaled uint16, uint32, 3D/5D or reconstruction export as QEM is not yet qualified |

SSB result images/calibration remain separate JSON/NumPy result artifacts; they
are not silently embedded as an acquisition's recorded microscope calibration.

## Original documents and reviewed edits

`source_documents` preserves the original UTF-8 text, including unknown fields,
comments and whitespace. Each entry contains `filename`, `mediaType`, `content`
and `sha256`; the digest covers the UTF-8 bytes of `content`. Readers validate
the digest before accepting a document. The current limits are 16 documents and
4 MiB total document content. XML document types and external entities are not
accepted. JSON must contain a top-level object.

Three distinct records must not be conflated:

- `source_documents`: original document text, available after the source file is
  removed or the acquisition moves to another machine.
- `electron_microscope` and `axes`: supported, normalized recorded quantities.
- `calibration_overrides`: explicit reviewed edits, without replacing the original
  recorded quantities or document text.

The native `NativeMetadataImport.read` API previews recognized EMPAD XML or
scientific-metadata JSON quantities before a client applies them. Unknown vendor
XML/JSON is preserved without guessed units. Saving an attachment alone must not
change calibration. Contents can include names and paths; inspect them before
sharing a QEM file. Document preservation does not imply exhaustive DM4/HDF5 tag
capture, nor does it establish CUDA support for the float32 measurement codec.

## Source map

- Swift: `NativeDM4Source`, `NativeMicroscopeMetadata`, `NativeQEMMetadata`,
  `NativeQEMCalibration`, `NativeQEMFile`, `MetalQEMExporter` under
  `src/quantem/gpu/swift/Sources/`.
- Python: `io/_qem_metadata.py`, `io/_streamed_file.py` and source-specific readers.
- The [reference and validation workflow](qem-interoperability.md) distinguishes
  byte integrity, field preservation, exact decoded counts and hardware coverage.
