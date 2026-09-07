# Direct packed uint16 experiment

Status: experimental; no release or performance qualification.

The scientific operation is unchanged: load all scan positions and native
detector pixels, preserve every unmasked integer, and evaluate exact detector
sums and selected diffraction patterns without expanding the full resident.
Only the authenticated, explicitly declared detector exclusions may be zeroed;
their constant raw values remain recoverable from source-bound metadata.

QGIX v3 header encoding 1 is unchanged (uint8, widths 0 through 8).
Experimental header encoding 2 uses the same checkpoint/nibble layout for
uint16 working counts: nibble values 0 through 14 are literal bit widths;
nibble 15 means width 16. Blocks requiring 15 bits are stored with 16 bits.
Checkpoints and payload offsets count actual stored words, not nibble codes.
This keeps full uint16 support without a wider per-block header. Older readers
must reject encoding 2, not interpret it as encoding 1.

Encoding 2 requires working_dtype uint16 and working_logical_sha256 of masked
little-endian uint16 counts in scan-major order. Prepared DPC uses the existing
uint16 moments/v2 contract. Encoding 1 retains its existing uint8 digest and
moments/v1 contract. Dtype and encoding must agree. Integer parity is exact.

Implementation scope: native Swift/Metal and independent Python reference.
Other backends remain unsupported for encoding 2 until independently tested.
Synthetic coverage must include 255, 256, 32767, 32768, and 65535, mixed block
widths, checkpoint boundaries, masks, and multiple resident updates. Real-data
conversion must verify original raw and working hashes before publication.
