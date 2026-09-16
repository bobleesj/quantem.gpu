# Native acquisition formats and metadata

For portable copies, use the versioned [QuantEM data (.qem) contract](qem-format.md).
It reuses the microscope vocabulary documented below, preserves reader-retained
source fields, and distinguishes scientific metadata from the payload codec.
Native `.qem` exports support integer K3/ARINA and calibrated EMPAD float32;
the codec table and explicit limitations are documented in that contract.

`Native4DSTEMIO` owns source identification, layout validation, calibration and
microscope metadata parsing for native clients. Applications should reuse these
readers, not duplicate HDF5 paths or infer physical units from filenames.

The pipeline is **inspect source and companions → validate shape, dtype and
units → read bounded original frames → GPU packed residency → scientific
products**. Header and metadata parsing run on the host; that is not CPU
decompression of the scientific volume. This page describes native Swift/Metal
support, not a promise that Python, CUDA, WebGPU or iOS supports the same files.

## Layout protocol v1.1

This section is the **native acquisition layout protocol, version 1.1**: a written
contract for interpreting existing vendor files. Revision 1.1 adds K3 DM4 and
compressed acquisition snapshots to version 1 without changing existing reader
contracts. It does not introduce a new
on-disk file format or claim an official vendor version. `R` means required
for the shown layout; `O` means optional. Optional physical quantities need
valid units and values before being promoted to calibration.

### ARINA master and NCEM/NXem companion

```text
acquisition_master.h5                         R: detector acquisition
└── entry
    ├── data
    │   └── <link-name>                       R: detector stack/link(s)
    │       → shard.h5:<recorded target>         all referenced shards required
    └── instrument/detector                  O: geometry/beam metadata
        ├── detector_distance                   length
        ├── y_pixel_size                        detector row pitch, not scan step
        ├── x_pixel_size                        detector column pitch
        └── incident_energy                     eV or keV

acquisition_em_metadata.h5                    O: NCEM/NXem companion
└── electron_microscope
    ├── scan_controller
    │   ├── scan_type                           "regular"
    │   └── regular_scan
    │       ├── n_pixels_y                      scan rows
    │       ├── n_pixels_x                      scan columns
    │       ├── n_frames                        1 for this scan-calibration path
    │       ├── pixel_size_y + @units           specimen row step
    │       ├── pixel_size_x + @units           specimen column step
    │       └── dwell_time + @units             O
    ├── electron_source/accelerating_voltage     O, with units
    ├── illumination_system/semi_convergence_angle  O, with units
    └── imaging_system
        ├── camera_length                       O, with units
        ├── reciprocal_pixel_size_y             O, angular units
        └── reciprocal_pixel_size_x             O, angular units
```

The stack/link names and targets above are symbolic, not required literal
filenames. The original packed-loader contract defines accepted rank, dtype,
compression and block geometry. The scan block must establish matching
dimensions before companion metadata is imported. An absent companion does
not prevent detector loading; it leaves microscope quantities unavailable.

### EMPAD-G1 XML and RAW

```text
acquisition.xml
└── <root>
    ├── raw_file @filename                   R: referenced RAW
    ├── pix_y / pix_x                        scan rows / columns
    ├── scan_parameters @mode="acquire"      alternative recorded scan shape
    │   └── scan_resolution_y / scan_resolution_x
    ├── timestamp @isoformat                 O: acquisition date text
    ├── exposure_time                        O: milliseconds
    └── iom_measurements                     O: fields in XML mapping below

referenced.raw                               R
└── scanRows × scanColumns records
    └── 16,384 float32 detector words + 256 footer words per record
```

Shape can also be supplied by the supported `scan_xN_yN.raw` convention or
explicit caller input. Conflicting sources fail; file length is not used to
guess a square scan. Footer words are not scientific detector pixels.

### EMPAD-G2 float32 XML and RAW

The XML dtype alone is insufficient evidence of a processed export. Some raw
acquisition software writes `float32` for encoded analog/counter/gain words.
The native reader conservatively rejects raw-offset metadata accompanied by a
persistent bit-30 marker in three sampled frames. This guard is not an exhaustive
encoding detector and never selects a gain calibration automatically. Encoded
G2 needs matching sensor calibration and even/odd dark handling; mean-dark
subtraction of IEEE-754 reinterpretations is scientifically invalid.

```text
acquisition.xml
└── <root>
    ├── sensor
    │   ├── type                             R: EMPAD2
    │   └── shape                            R: 128,128
    ├── scan
    │   ├── type                             R: scan
    │   ├── shape                            R: positive raster dimensions
    │   └── exposure_time                    O: seconds
    ├── rawfile
    │   ├── filename                         R: referenced RAW
    │   └── datatype                         R: float32
    └── iom_measurements                     O: fields in XML mapping below

referenced.raw                               R
└── scanRows × scanColumns × 128 × 128 float32 values, no G1 footer
```

This is a float export contract, not decoding of native encoded integer
EMPAD2 acquisition words. Exported raster dimensions and exact byte length
govern loading; an acquisition frame counter containing flyback frames does
not add extra raster positions.

### EMD 1 HDF5 4D datacube

```text
acquisition.h5 (or .emd)
├── @authoring_program = "emdfile"             R
├── @version_major = 1                        R
└── datacube_root                             R: hard link
    ├── datacube                             R: hard link
    │   ├── data                             R: hard-linked contiguous F32LE
    │   │                                      [scan0, scan1, 128, 128]
    │   ├── dim0 … dim3                      not interpreted for calibration
    │   └── metadatabundle/SoM2k              O
    │       ├── high tension                    V
    │       └── camera length                   m
    └── metadatabundle/calibration            O
        ├── R_pixel_size / R_pixel_units        isotropic scan step; A or Å
        ├── Q_pixel_size / Q_pixel_units        isotropic angular step; mrad
        └── convergence_semiangle_mrad          mrad
```

Minor-version attributes are not currently selection gates. The shape must be
rank four with positive scan dimensions and 128×128 detector dimensions. The
dataset must have valid in-file storage bounds and exactly the expected byte
count; external storage, soft/external links and compressed/chunked layouts
are rejected. A Velox scalar-image EMD does not satisfy this 4D contract.

### K3 DigitalMicrograph DM4

K3 is a detector identity, DM4 is its acquisition container, and `.qem` is a
saved compressed representation. These are separate properties. A `.dm4`
extension alone is not evidence of K3. `NativeDM4Source` validates the DM4 v4
header and selects the unique rank-four image; a survey image is not selected.
`sourceFormat` becomes `K3 DM4` only when the selected image's recorded
`Source Model`, after trimming and case normalization, equals `K3`. Other or
missing models remain `DigitalMicrograph DM4`.

```text
acquisition.dm4
└── ImageList/<selected image>
    ├── ImageData
    │   ├── DataType                         R: 6 (uint8) or 10 (uint16)
    │   ├── Dimensions/<axis>                R: four positive dimensions
    │   ├── Data                             R: complete little-endian counts
    │   └── Calibrations/Dimension/<axis>
    │       ├── Scale                        R: recorded axis sampling
    │       └── Units                        R: detector axes "1/nm"
    └── ImageTags
        ├── Acquisition/Device/Source Model  O: "K3" identifies this detector
        ├── Acquisition/Device/Source ID     O: recorded device identifier
        ├── Acquisition/Parameters/High Level/Processing  O: vendor description
        ├── Microscope Info/Voltage          O: positive accelerating voltage, V
        └── SI/Acquisition/Date               O: original date string
```

DM4 dimensions and calibrations are reversed together into public
`(scan_row, scan_col, detector_row, detector_col)` order. There is no spatial
transpose, crop, binning, count rescaling or automatic median correction.
Native loading rejects ambiguous four-dimensional images, unsupported dtypes,
endianness, missing reciprocal-axis calibration and incomplete payloads.

| Recorded quantity | Native metadata or dataset field | Units / missing behavior |
| --- | --- | --- |
| Device model | `camera_model`, derived `sourceFormat` | Recorded text; absent model does not imply K3 |
| Device identifier | `camera_id` | Original text; optional |
| Acquisition processing | `acquisition_processing` | Original text, e.g. `Gain Normalized`; not evidence of background subtraction |
| Reader interpretation | `sourceFormatVersion` | `digitalmicrograph/native-counts-v1`; not a vendor software version |
| Scan axis scales | `sourceScanCalibration` | Row/column Å per pixel: nm × 10, µm/um × 10,000, A/Å unchanged; unsupported units leave scan calibration absent |
| Detector axis scales | `kPixelSizeRow`, `kPixelSizeCol`, `kPixelUnit` | Recorded 1/nm divided by 10 into 1/angstrom |
| Voltage | `electron_microscope/electron_source/accelerating_voltage` and `@units` | Positive V in originals; snapshots may normalize to kV |
| Acquisition date | `acquisitionDate` | Original text, not filesystem modification time |
| Source file size | `sourceBytes` | Total bytes of the currently opened DM4 or compressed file, not bytes per scalar; original logical volume size follows shape and dtype |
| Selected-image metadata | `dm4.<relative tag path>` | Scalar/string values; unrecognized typed tags retain descriptors and base64 bytes |

Native detector products use uint32 accumulation on this path. A uint16 camera
geometry whose possible detector sum exceeds UInt32.max is rejected rather than
overflowing; Python MPS provides a separate uint64 product path. Successfully
opening a K3 file does not imply SSB supports its scan size: native SSB currently
accepts 128×128, 256×256 and 512×512, not arbitrary 100×100 or 210×210 scans.

### Saved compressed acquisitions (`.qem`)

Identify the saved copy by `QEMDATA1` magic, container `quantem.qem`, and
`codec=runtime-column-rans-spatial-v2`, not extension alone. The retired `.ans`
containers are rejected; open the original acquisition instead.

```text
acquisition.qem
├── 56-byte prefix: QEMDATA1 magic, JSON byte count, body offset, JSON SHA-256
├── UTF-8 JSON header
│   ├── container, container_version, codec, version, interval=512, shape, dtype
│   ├── valid, chunks (encoded-array offsets and lengths)
│   ├── bytes, sha256 (64 MiB body-block checksums)
│   ├── scientific_metadata (axes, acquisition metadata, overrides)
│   └── metadata
│       ├── scan_sampling_A, detector_sampling_inv_A, voltage_kV
│       ├── acquisition_date
│       └── source_metadata: retained source tags and native format identity
└── body: encoded detector streams and reusable spatial indexes
```

CUDA/Python writers also store normalized `camera_model`, `camera_id`,
`acquisition_processing` and `source_kind` directly in `metadata`. Native
readers accept that placement as well as the native `source_metadata` placement;
recorded DM4 device tags remain a fallback. An original K3 acquisition therefore
retains `sourceFormat=K3 DM4` after reopening, while `sourceKind=ans-snapshot`
and `storageSchema=runtime-column-rans-spatial-v2` identify its current storage.
Unknown non-DM4 sources are never relabeled K3 by this reader.

Metadata inspection reads only the bounded header; resident loading verifies
all compressed body checksums before GPU use, restores existing encoded streams
and indexes, and does not re-encode or require the original DM4. Native saving
preserves source metadata and original counts, refuses overwrite, and publishes
atomically. No conversion of existing files is required for this protocol
revision. See [K3 opening, saving and verification](../integrations/k3-dm4-qem.md)
for APIs and timing boundaries.

### Conformance and evolution

1. Identify from validated contents, never from a filename alone.
2. Validate required fields, axes, dtype, links and byte counts before loading.
3. Preserve original numerical values and source identity through residency.
4. Return absent optional calibration as absent; separate user assumptions.
5. Retain the reader identifier and calibration evidence with derived results.
6. New interpretation rules need documented schema evolution and independent
   fixtures. Do not reinterpret an existing fixture merely to pass a test.

This protocol documents current coverage, including gaps. It does not promise
exhaustive raw metadata export or introduce a new Swift `Protocol` abstraction.

## Supported layouts

HDF5 is a container, not a detector type. An EMD file can contain a 4D datacube,
a scalar image, or an unsupported layout. Inspect contents before selecting a
reader. Public array coordinates are `(scanRow, scanColumn, detectorRow,
detectorColumn)`; preserve recorded EMD axis order without implicit transpose.

| Source | Identification and scope | Metadata source |
| --- | --- | --- |
| ARINA HDF5 | Validated master and linked detector stacks; native packed loading supports uint8, uint16 and uint32 under its encoding constraints | Master/catalog fields |
| ARINA HDF5 + NXem | Same acquisition with a readable, scan-shape-matched `_em_metadata.h5` companion | Companion microscope metadata |
| EMPAD-G1 XML/RAW | Recorded scan shape; each frame has 128×128 float32 values and a 256-word footer | G1 XML |
| EMPAD-G2 processed XML/RAW | Processed float32 export with `sensor/type=EMPAD2`, 128×128 sensor and raster shape; no G1 footer; encoded acquisition words are unsupported | G2 XML |
| EMD 1 HDF5 datacube | `authoring_program=emdfile`, major version 1, `/datacube_root/datacube/data`, contiguous little-endian float32, shape `(scan0, scan1, 128, 128)` | EMD calibration and supported SoM2k fields |
| Velox EMD scalar image | Separate catalog image/calibration path | Supported Velox metadata; not proof of a 4D acquisition |
| K3 DM4 | Validated DM4 v4, unique native-count 4D image, recorded camera model K3 | Selected-image DM4 calibration and device tags |
| DigitalMicrograph DM4, other/unknown camera | Same native-count reader without assuming K3 | Recorded calibration; camera identity optional |
| Saved copy (`.qem`) | QEMDATA1 container `quantem.qem` and runtime-column-rans-spatial-v2, uint8/uint16 | Checksummed header, scientific metadata schema and calibration overrides |

The two ARINA labels distinguish metadata availability; they are **not official
ARINA v1/v2 file-format versions**. See [original HDF5 packed loading](original-hdf5-metal-packing.md)
and [the native load contract](native_4dstem_io.md) for encoding/masking limits.

XML and its named RAW need not share a basename. `NativeEMPADSource.open`
resolves a unique matching sibling XML when passed the RAW. Conflicting shape,
incompatible byte length, or changed source files fail rather than silently
loading a partial volume. The float reader preserves all recorded float32 bits,
including signed and fractional signal; no background subtraction, gain
correction, clipping, cropping or binning is applied.

Encoded integer EMPAD2 words, arbitrary HDF5 groups, compressed/chunked EMD
datacubes, external EMD links and float64 EMD are **not supported by this float
reader**. Standalone auxiliary offset files are not automatically acquisitions.

## Reader schema and provenance

`NativeEMPADSource.formatIdentifier` reports the parsing contract:

- `empad-g1-float32-xml/v1`
- `empad-g2-float32-xml/v1`
- `emd1-contiguous-float32/v1`

`formatName` is the display label. These identifiers are independent of package
version, acquisition-software version and GPU packing schema. They do not
rewrite or version the original files. ARINA currently uses catalog
`metadata["sourceFormat"]` labels rather than these versioned identifiers.

Retain the source identity, original shape/dtype, reader identifier and
calibration evidence with derived results. EMD's `sourceDataset` and
`sourceAxisOrder` describe the interpreted dataset. The `SoM2k` metadata namespace
and `emdfile` authoring attribute establish the supported export layout; they
do not establish detector generation or a complete acquisition-tool version.
A unified acquisition-software/version property is not currently exposed.

## ARINA detector metadata and the NCEM/NXem variant

The supported NCEM/NXem variant uses **ARINA HDF5 + NXem microscope
metadata**. Call this the NCEM/NXem variant in documentation; the runtime label
remains `ARINA HDF5 + NXem metadata`. Detection is based on the companion's
contents and scan dimensions, not the institution or filename alone. A generic
NXem companion is not proof that a file originated at NCEM.

For `acquisition_master.h5`, the reader checks
`acquisition_em_metadata.h5`. This adds microscope/scan information that a
detector master alone may not contain. The supported fields are:

| Field | HDF5 path | Interpretation |
| --- | --- | --- |
| Scan type | `/electron_microscope/scan_controller/scan_type` | Must be `regular` for this calibration path |
| Scan rows | `/electron_microscope/scan_controller/regular_scan/n_pixels_y` | Positive integer; y maps to row |
| Scan columns | `/electron_microscope/scan_controller/regular_scan/n_pixels_x` | Positive integer; x maps to column |
| Scan frames | `/electron_microscope/scan_controller/regular_scan/n_frames` | Must be 1 for this calibration path |
| Row step | `/electron_microscope/scan_controller/regular_scan/pixel_size_y` | Explicit length unit; output nm/pixel |
| Column step | `/electron_microscope/scan_controller/regular_scan/pixel_size_x` | Explicit length unit; output nm/pixel |
| Beam energy, convergence, dwell, camera length, angular sampling | See the common microscope table below | Unit-validated optional quantities |

Accepted scan-length units are m, mm, um/µm and nm. The detector master also
supports scan-step alternatives, searched in this order for each axis:
`/entry/instrument/scan/{y,x}_pixel_size`,
`/entry/instrument/scan/step_{y,x}`,
`/entry/measurement/scan_step_{y,x}`, and
`/entry/scan/{y,x}_pixel_size`. Both axes require explicit supported units.

Detector geometry is distinct from scan calibration:
`/entry/instrument/detector/detector_distance` and
`/entry/instrument/detector/{y,x}_pixel_size` yield angular sampling using
`atan(sensorPitch / detectorDistance)`, converted to mrad. This legacy geometry
reader defaults missing length units to meters; it must not be mistaken for
specimen scan-step metadata. Explicit NXem angular sampling is interpreted by
the typed microscope reader below.

Calibration preference is source HDF5 scan sampling, then a shape-matched NXem
companion, then the catalog's supported unambiguous Velox sibling calibration.
Retain `spatial_calibration_source`, `microscope_metadata_source`, and any
`microscope_metadata_warning` with results. FOV is derived, not a separately
measured quantity: scan rows/columns multiplied by their respective steps.

### Retained metadata is not a complete file dump

The ARINA HDF5 display-metadata collector retains a bounded dictionary of up to
**100 entries per inspected file**. It skips `/entry/data`, detector pixel-mask
values and unsupported/large values; dataset units are appended to values and
other attributes use `path@attribute` keys. This preserves additional descriptive
fields without treating them as calibrated scientific quantities. It does not
guarantee retention of every field in a large file.

EMPAD XML uses an explicit field allowlist, and EMD uses the specific paths
documented below; neither currently exposes an exhaustive raw metadata tree.
Therefore distinguish **documented interpreted metadata**, **retained raw
metadata**, and **all metadata physically present in the file**. Complete
metadata export is not currently provided by these readers.

## Common microscope quantities

`NativeMicroscopeMetadata(metadata:)` interprets normalized paths below
`electron_microscope/`. Unknown units, nonfinite or nonpositive values produce
`nil`, not guessed calibration. A separate `@units` entry takes precedence over
an inline unit. XML/EMD readers normalize their known fields to this contract.

| Normalized path | Accepted units | Typed output |
| --- | --- | --- |
| `electron_source/accelerating_voltage` | V, kV | `beamEnergyKeV` |
| `illumination_system/semi_convergence_angle` | rad, mrad | `semiConvergenceAngleMrad` |
| `scan_controller/regular_scan/dwell_time` | s, ms, us, µs, μs | `dwellTimeMicroseconds` |
| `imaging_system/camera_length` | m, cm, mm | `cameraLengthMillimeters` |
| `imaging_system/reciprocal_pixel_size_y` | rad, mrad | `angularRowMrad` per pixel |
| `imaging_system/reciprocal_pixel_size_x` | rad, mrad | `angularColumnMrad` per pixel |

Beam energy also accepts `entry/instrument/detector/incident_energy` in eV/keV.
The NXem companion must match scan dimensions; unreadable or mismatched
companions generate a metadata warning rather than contributing calibration.

### EMPAD XML field mapping

| Quantity | G1 XML path and convention | G2 XML path and convention |
| --- | --- | --- |
| Voltage | `iom_measurements/high_voltage`, V | `iom_measurements/ColumnSourceHighVoltage`, V |
| Camera length | `iom_measurements/nominal_camera_length`, m | `iom_measurements/ColumnOpticsGetCameraLengthNominalCameraLength`, m |
| Exposure | `exposure_time`, ms | `scan/exposure_time`, s |
| Detector sampling | `iom_measurements/calibrated_pixelsize` multiplied by 1e9 for the G1 inverse-nm convention | `iom_measurements/calibrated_diffraction_angle`, rad/pixel, applied to both axes |

G2 `calibrated_pixelsize` is deliberately not interpreted as reciprocal-length
sampling. For G1, equal positive `full_scan_field_of_view/x` and `/y` values in
meters, with a positive `/scale_factor`, yield isotropic scan sampling from
FOV divided by that factor and the maximum scan dimension. Alternatively,
`iom_measurements/optics.get_full_scan_field_of_view` supplies a legacy JSON
row/column FOV pair in meters, divided by the respective scan dimensions.
Both are under `iom_measurements/`. Unsupported/partial calibration stays absent.

### EMD field mapping

Under `/datacube_root/metadatabundle/calibration/`, the reader uses:

- `R_pixel_size` with `R_pixel_units=A` or `Å`: isotropic scan step in angstroms.
- `Q_pixel_size` with `Q_pixel_units=mrad`: angular detector step on both axes.
- `convergence_semiangle_mrad`: convergence semi-angle in mrad.

Under `/datacube_root/datacube/metadatabundle/SoM2k/`, `high tension` is in volts
and `camera length` in meters. The reader does not currently derive independent
anisotropic sampling from EMD dimension arrays or import arbitrary microscope
fields. It does not provide EMD acquisition time or dwell time in this schema.

Scan calibration is returned separately as `scanCalibration`, with source
provenance and evidence. FOV is derived from scan dimensions and step. Unknown
signal units must not be labeled electrons; background-corrected EMPAD float
values are not automatically calibrated electron counts or dose.

## Native client integration

### Documented background subtraction

`NativeEMPADSource.backgroundSubtractionEvidence` reports an explicit supplier
README declaration without changing any measurement. Version 1 recognizes
unindented relative file/directory headings followed by an indented sentence stating
`already background subtracted` (hyphenated spelling also accepted). Headings
must resolve to the opened RAW/XML/HDF5 file or a containing directory under
that README. Negated, uncertain or conflicting statements remain unknown.
This bounded reader is not a general natural-language interpreter and never
infers correction from a folder name, file extension, microscope or detector.

Clients should display `Yes · supplier README`, retain the evidence filename
and sentence, and prevent a second mean-dark subtraction for these sources.
This declaration establishes reported background subtraction only: it is not
proof of gain calibration or electron units. Unknown sources keep the existing
explicit correction workflow. Encoded EMPAD2 words are still rejected even if
a README calls them corrected; source validation cannot be bypassed by prose.
README edits invalidate an in-progress source snapshot. Small README metadata
reads take place during inspection, not during interactive detector reductions.

```swift
import Foundation
import Native4DSTEMIO

let source = try NativeEMPADSource.open(URL(fileURLWithPath: inputPath))
let microscope = NativeMicroscopeMetadata(metadata: source.microscopeMetadata)
let readerSchema = source.formatIdentifier
let scanCalibration = source.scanCalibration
// Pass the validated source to the shared packed-float Metal loading path.
// Preserve optional values and provenance when building the client model.
```

This inspection does not load a complete resident. Loading, source identity
audits and publication of a usable GPU resident have separate completion
boundaries. Applications own folder selection, auxiliary-file policy, scheduling,
cache admission, notes, user overrides and UI. Keep assumed values separate
from recorded metadata; a client default such as 30 mrad is not a source fact.

## Source map and verification

Sources live in `src/quantem/gpu/swift/Sources/`:
`Native4DSTEMIO/NativeEMPADSource.swift`, `Native4DSTEMCatalogBuilder.swift`,
`NativeMicroscopeMetadata.swift`, and `CNativeHDF5/CNativeHDF5.c`.

`tests/hardware/metal/test_empad_source.py` covers float bit parity, layouts,
metadata, source changes, cancellation and memory budgets.
`check_empad_acquisition.py` independently compares real-source DP samples and
full-scan BF/ABF/ADF/total products. Catalog calibration has its own
`test_catalog_calibration.py` fixtures. Native UI and release qualification are
separate consumer gates. Layout support alone does not establish cold-I/O time,
peak memory, frame rate or support on another runtime.
