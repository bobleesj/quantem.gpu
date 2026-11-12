# Native EMPAD Direct Ptycho input contract

The native Direct Ptycho engine accepts signed float32 detector columns using
`MetalDirectPtychoCountType.float32`. Negative and fractional measurements remain
unchanged; no integer conversion, camera marker replacement, scan binning or
detector crop is applied. The same Fourier, fitting and reconstruction kernels
serve integer and float inputs. This is a development capability, not application
release qualification.

`MetalEMPADResidentSource.encodeDetectorColumns` returns detector-major,
row-major scan planes. It preserves requested ordering and duplicates, applies
only the resident's explicitly selected background correction, and uses bounded
ANS decode workspace. An explicitly padded output has zeros outside the source
scan; scientific consumers must decide whether padding is valid for their model.

The detector reduction used to build geometry is the signed sum in detector
coordinates. It must be finite and have positive integrated aperture intensity.
The bright-field estimator accepts float means without quantization. Invalid
measurements must not silently become zero counts. The explicit missing-scan
policy below records any working-data interpolation.

For XML that retains both search and acquisition grids, a search grid may be
selected only when its recorded dimensions match the RAW filename and the
complete file length matches that detector record layout. Otherwise loading
fails. Sampling remains unavailable in this ambiguous-mode case; a verified
calibration must be supplied separately. Original XML is retained as provenance.

Validation includes an analytic two-impulse Fourier reference at 128, 256 and
512 scan sizes, integer incident-dose regression, independent bright-field disk
selection, exact signed detector-column extraction, background correction,
duplicate selections, padding and metadata conflict checks. Real acquisitions,
rendered app behavior and speed require separate evidence.

## Isolated missing scan positions

Loading audits each detector pattern for non-finite float32 values, including
when reopening an original-word QEM archive. `resident.scanValidity` records
zero-based `(row, column)` positions. The original encoded detector words are
never patched: QEM export retains their NaN payloads and embeds the validity
record in microscope metadata.

Mean DP excludes invalid patterns and divides by the number of valid patterns.
CoM/DPC retain a NaN at a missing position; DPC centering uses finite vector
pairs, so a single missing pattern does not invalidate the whole image. The app
leaves iDPC unavailable for a scan with missing positions rather than silently
filling an integrated image.

Direct Ptycho can request `encodeDetectorColumns(..., interpolateMissing: true)`
after `requireInterpolatableScans(rows:columns:)`. Only isolated interior gaps
with four finite original neighbors qualify. Each detector pixel is replaced
in the transient working buffer by the arithmetic mean of the same pixel in
the up/down/left/right scan neighbors. This is scan-space interpolation, not
hot-pixel smoothing or recovered experimental counts. Adjacent gaps and boundary
gaps fail fitting unless the selected scan region excludes them.

Saved fits record the original invalid frame, its donor frames, correction
method `four-neighbor-dp-mean/v1`, calibration overrides and mean-DP policy.
The application shows an invalid-scan warning. Synthetic tests verify exact
preservation of valid float32 words, the neighbor-mean oracle, finite-pair DPC,
and QEM metadata and NaN-payload round trips. Real-data and native save/reopen
checks remain separate from these unit tests.
