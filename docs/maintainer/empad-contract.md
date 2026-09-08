# EMPAD source contract v1 (implementation in progress)

Open a conventional EMPAD XML/RAW acquisition and recover each recorded
diffraction pattern in `(scan_row, scan_col, detector_row, detector_col)` order.
The native format entry point is `NativeEMPADSource.open(_:scanShape:)`.

## Scientific meaning

- Each record contains 130 × 128 little-endian float32 words. The first
  128 × 128 are detector measurements; the remaining 256 words are a footer,
  not two additional detector rows. All physical detector pixels are retained.
- Keep IEEE-754 sample bits unchanged, including signed/fractional values,
  negative zero, NaN and infinity. Do not reinterpret measurements as integer
  electron counts, round to uint16, subtract a pedestal or clip negatives.
- Scan shape comes from XML, the explicit `scan_xN_yN.raw` convention, or an
  explicit caller `(row, col)` shape. File length alone never proves a square
  scan. Conflicting metadata and truncated/extra records fail closed.
- XML raw filenames resolve to the sibling basename. Do not load external
  entities or follow an acquisition computer's absolute filesystem path.
- Frame selection retains request order and duplicates. No transpose, binning,
  scan crop or detector mask is implicit. Physical calibration is unknown
  unless supplied in supported acquisition metadata or explicitly by the user.
  Importing sampling changes coordinate labels, never recorded measurements.

## Acquisition sampling and missing dimensions

`NativeEMPADSource` exposes optional `scanCalibration`,
`diffractionSamplingInverseNanometers`, and `acquisitionDate`. These reuse the
existing source-calibration model and do not add a second loading API.

- EMPAD 1.2 `full_scan_field_of_view` stores the same maximum-axis field in
  `x` and `y`, multiplied by `scale_factor`. The reader divides by that factor
  and the larger scan dimension, then converts meters to angstroms. Unequal
  fields, nonpositive factors and nonfinite values do not establish calibration.
- Legacy `optics.get_full_scan_field_of_view` provides the row/column fields
  in meters. Each is divided by its corresponding scan dimension.
- Legacy `calibrated_pixelsize` follows the EMPAD conversion to inverse
  nanometers (`value * 1e9`). Newer exports omit it; no camera-length estimate
  or assumed diffraction sampling substitutes for missing metadata.

These conventions follow the documented
[RosettaSciIO EMPAD reader](https://github.com/hyperspy/rosettasciio/blob/main/rsciio/empad/_api.py).
The implementation independently parses only the recognized metadata fields;
no external reader code or detector correction is included.

An otherwise unresolved shape raises `NativeEMPADSourceError.missingScanShape`
with the resolved RAW URL. Callers can request original dimensions and retry
the same entry point. This is distinct from malformed XML, conflicting
dimensions, or incomplete records. The app keys confirmed dimensions by that
RAW URL so XML and RAW aliases share one answer. The original files remain
unchanged. Tests cover rectangular sampling, unavailable calibration, exact
selected float bits, and native metadata entry followed by quit/relaunch.

This does not implement EMPAD-G2. Its original acquisition needs both a
matching background and sensor calibration, as specified by the
[EMPAD-G2 reader](https://github.com/sezelt/empad2). Neither an arbitrary gain
map nor the conventional float32 interpretation is a valid substitute.

## Scope and remaining gates

This describes conventional float32 EMPAD exports, not EMPAD-G2 uncalibrated
acquisition words, and not arbitrary RAW files. Import correctness alone does
not certify GPU residency or Live4DSTEM support. The current integer packed
resident and integer reductions cannot represent these measurements. A
lossless float representation and floating-point reductions are implemented in
`MetalEMPADResidentSource`, but consumer qualification is still required before
enabling the application route. No dense uint16 fallback is permitted.

The float resident encodes each 128-word detector row as an XOR base, common
leading/trailing bits and a bit-packed residual. It preserves all IEEE-754 bits
and supports direct Metal selected-DP extraction and mask integration. Storage
can exceed logical float32 size for incompressible rows; do not promise an
integer-data compression ratio. Loading uses bounded 64-frame windows, with
private packed payloads and descriptors. There is no complete dense volume or
saved 2D image standing in for a resident. Interaction methods encode into
caller-owned command buffers without CPU frame readback or synchronous waits.
The load path does synchronize bounded preparation/packing commands. The caller
must serialize ownership and wait for commands before publishing/reusing outputs.
Source device/inode, size, modification time and change time are checked before
and after bounded reads and before resident publication. A changed RAW or XML
invalidates the load, including same-size edits with restored modification time.
These checks detect filesystem changes; they are not a file lock.

`Metal4DSTEMResidentCapabilities.empad(_:)` exposes the actual float32 tensor,
packed bytes and available products. Selected DP numerics are exact float32
bits, not integer counts. Mean DP and CoM now have native float kernels;
CoM and float image buffers feed the shared DPC/iDPC and FFT kernels. The
backend advertises these compute prerequisites through `fullInteractiveResident`.
That flag is not a native application or release certificate: the experimental
consumer still has separate UI integration gates (including its average-DP
control). Capabilities cannot be obtained after release.
The logical SHA-256 covers little-endian detector words in scan order. Source
tensor identity hashes the UTF-8 domain `quantem.gpu.empad-tensor/v1\0float32-le\0`,
four little-endian UInt64 shape dimensions and the lowercase logical hash's
ASCII bytes. Footer bytes, names and XML formatting are not tensor identity;
this receipt is not the checksum of the original container file. Computing
the digest occurs during the bounded source reads, not during interaction.

Aperture integration uses compensated float32 summation with strict arithmetic.
Selected non-finite values propagate; unselected values do not participate.
Floating reductions are qualified against a float64 reference with `rtol=1e-6`,
`atol=1e-6`, not falsely described as exact integer sums.

Mean DP is the arithmetic mean of all scan positions, with the same float
tolerance. CoM is `sum(value * coordinate) / sum(value)` for zero-based detector
row and column coordinates. Signed weights participate without clipping. A zero
total or any non-finite detector measurement yields NaN CoM coordinates. These
derived operations do not change packed samples; their consumer integration
and live UI qualification remain separate gates.

Required evidence: independent raw-word parity, rectangular scan orientation,
XML and RAW opening, footer separation, retained negatives/fractions/non-finite
bits, selected DP parity, float BF/ABF/ADF and DPC parity, memory admission,
cancellation, and native mixed-format folder/drop tests. Version 0.0.9 remains
the installed qualified application until these and the archive gates pass.

## Initial physical-device evidence (2026-09-08)

On Apple M5 (24 GB), the native reader and packed Metal tests cover six
synthetic workflow cases: rectangular XML/RAW source loading, original bit
recovery with repeated/out-of-order selection, malformed/ambiguous file
rejection, and cross-window resident/float-aperture parity with a tiny-budget
rejection, source mutation rejection with restored modification time, and
cancellation before allocation or after the first 64-frame window followed by
a successful retry. XML checks reject conflicting or incomplete dimensions,
ambiguous source filenames and external entities. These fixtures include negatives, fractions, signed zero, a NaN
payload, infinities and a subnormal. They do not establish full-size performance.

The public AutoDisk Pd@Pt original RAW acquisition passed a separate full-data
audit: all 67,108,864 detector sample words from the 64 × 64 scan matched the
original file byte-for-byte after GPU packing and extraction. BF/ABF/ADF and
total-image sums matched the independent NumPy float64 reference within the
unchanged tolerance. The source SHA-256 matched its published Git LFS pointer:
`c431b6ae2a506b189ae86fbaf254d97de7c9258383c6dde96c7a871d1a9f7805`.

The initial uncompensated sum failed twice identically: BF maximum relative
error `1.61481776e-6`, with 93 of 4,096 values beyond the `1e-6` limit.
Compensated summation reduced maximum relative errors to `5.92664287e-8` (BF),
`5.92822801e-8` (ABF), `5.89445828e-8` (ADF), and `5.88597142e-8` (total).
No tolerance, sample or reference was changed. Packed payload and descriptors
occupied 236,838,288 bytes versus 268,435,456 logical float32 detector bytes.
This excludes display products, transient allocation and process RSS.

The existing Arina original-packing regression suite also passed all 38 cases
in 270.20 seconds. Those HDF5 acquisitions are synthetic fixtures, not a new
real-folder UI qualification or a loading benchmark.

Reproduce after building the `EMPADSourceParity` product in
`tests/hardware/metal/swift_original_packing`:

```sh
EMPAD_SOURCE_PARITY_EXE="$PWD/tests/hardware/metal/swift_original_packing/.build/release/EMPADSourceParity" \
  python3 tests/hardware/metal/test_empad_source.py -v
EMPAD_PUBLIC_RAW=/path/to/pdpt_x64_y64.raw \
EMPAD_SOURCE_PARITY_EXE="$PWD/tests/hardware/metal/swift_original_packing/.build/release/EMPADSourceParity" \
  python3 tests/hardware/metal/test_empad_public.py -v
```

The full audit intentionally reads every reconstructed DP back for independent
comparison; this is verification, not the interactive or loading timing path.
No FPS, cold-I/O, 512 × 512 scaling, application or release claim follows from
these tests. Python MPS, CUDA, WebGPU, Direct3D and Vulkan EMPAD loading remain
unsupported; the canonical backend matrices record the native path as partial.

## Native consumer and cooperative aperture experiment

The experimental Live4DSTEM route now discovers XML/RAW files, loads original
samples into this float resident, and uses the existing viewer for selected DP,
BF/ABF/ADF, derived products and FFT. It is gated separately from the installed
release. The input stage and float packing differ from ARINA's bitshuffle/LZ4
integer loader; the session, presentation and interaction workflow are shared.

See `experiments/20260908-empad-native-cooperative` for the controlled native
serial/cooperative/serial test. A 32-lane group cooperates on each aperture sum,
retaining compensated accumulation rather than assigning an entire pattern to
one thread. Complete sample parity, float64 product parity and actual native
presentations are distinct acceptance checks. No dense resident is added.

## Current native loading and interaction controls

The current reader scatters complete RAW records directly into bounded shared
Metal staging buffers; only the format's two footer rows are omitted. SIMD
row description and unique-writer packing preserve every physical float32 bit.
The optional `sourceHashCacheURL` argument stores a source-snapshot-bound
checksum record, never detector arrays. A matching record skips checksum
computation, not original reads, packing, or source-mutation checks. Report
`reusedSourceHash` separately from OS page-cache state and retained residents.

Completed aperture sums now retain a compensated high/low pair per scan
position. Changes integrate entering and leaving mask pixels. Incomplete prior
commands, large mask changes and periodic rebasing use a complete sum; removing
nonfinite pixels recomputes affected frames. This adds eight bytes per scan
position, not a second 4D representation, and respects the allocation budget.
The original full-sum control remains available for experiments with
`QGPU_EMPAD_INCREMENTAL=0`. Scientific tolerances are unchanged.

See the [loading experiment](../../experiments/20260908-empad-hash-cache/manifest.json)
and [aperture experiment](../../experiments/20260908-empad-incremental-apertures/manifest.json)
for native presentation evidence and unmet performance targets. Kernel speed
does not establish a 120 Hz UI result.

## Format references and public validation candidates

- [AutoDisk demonstration and original RAW](https://github.com/swang59/AutoDisk_Demo):
  S. Wang, T. Eldred, J. Smith and W. Gao, *AutoDisk: Automated Diffraction
  Processing and Strain Mapping in 4D-STEM*. Use the original repository's
  attribution and terms. The file was downloaded for local qualification;
  the scientific data are not bundled with the app or this repository.
- [Cornell PrScO3 data](https://data.paradim.org/doi/ssmm-2j11/): Z. Chen et al.,
  *Electron ptychography achieves atomic-resolution limits set by lattice
  vibrations*, Science 372 (2021), 826–831, DOI: 10.1126/science.abg2533.
  Downloaded MATLAB export SHA-256:
  `e5c470797e90a21381b9fdb0a353493f3ead96fcafdd4b9a163bb110059e472a`.
  Inspection found `dp` stored as float64 `(4096, 256, 256)` in MATLAB HDF5,
  whereas the acquisition note describes 128 × 128 detector data. No padding,
  orientation or conversion was guessed. It is not the original RAW validation
  fixture and was not fed into the float32 loader.

- [RosettaSciIO EMPAD format](https://hyperspy.org/rosettasciio/supported_formats/empad.html)
- [LiberTEM EMPAD reader](https://libertem.github.io/LiberTEM/_modules/libertem/io/dataset/empad.html)
- [Public 256 × 256 scan, 128 × 128 EMPAD detector](https://zenodo.org/records/17246822):
  `scan_x256_y256.raw`, published MD5 `c50f643c2cc87360bfdc746afd026cce`.
  Acquisition metadata and author attribution remain with that record. The
  complete 4,362,076,160-byte original was downloaded and checksum-verified.
  All 1,073,741,824 physical detector samples passed bitwise resident parity;
  BF/ABF/ADF, total, mean DP and CoM passed independent float64 references at
  the unchanged 1e-6 relative/absolute tolerance. Native UI performance is a
  separate gate: see `experiments/20260908-empad-original-load`.
- [Cornell-associated EMPAD-G2 work and dataset reference](https://www.paradim.org/highlights/MIP_120)
  describes a different acquisition generation; do not conflate its raw format
  with conventional EMPAD float32 exports.
