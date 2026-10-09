# uint32 flagged-pixel markers and the gold_512 convert size - 2026-10-08

## Question

`quantem-gpu convert gold_004_master.h5` wrote nothing: the public gold_512
master (512 x 512 scan, 192 x 192 detector, uint32, 4 flagged pixels) failed
with "counts above 65535". After that was fixed, the dry run reported the copy
would be larger than the source (4.88 GB to 6.05 GB). Is that expected?

## Cause of the refusal

`convert` loads with `hot_pixel_correction="none"` so that every stored value is
kept. Arina writes 0xFFFFFFFF at every flagged pixel of a uint32 file, and the
CUDA (`io/encoded.py`) and Metal (`narrow_u32_u16` in `bslz4.msl`) narrowing
refused any value above 65535, flagged or not. The unmasked maximum is 174.

## Change

When uncorrected uint32 counts are stored as uint16, a value above 65535 at a
flagged pixel is stored as 0 and counted. Every valid count and every other
flagged value is the file's; a valid count above 65535 is still refused. The
pixel mask is saved with the copy and reads exclude flagged pixels, so no
measurement changes. Metadata records `flagged_markers_stored_as_zero`,
`detector_mask_policy="flagged-markers-stored-as-zero"` and
`file_counts_exact=False`; the scientific processing list declares
`flagged_marker_zeroing` (`changes_measurements=False`), and
`validate_declared_processing` refuses a copy that omits it. `convert`'s
verification checks the mapping value by value and that the count matches the
record.

## Size of the gold copy

Measured on GPU 0 (RTX PRO 6000 Blackwell) with 8192 sampled scans (every 32nd
block of 512 scans):

| Quantity | bits per value |
|---|---|
| HDF5 bitshuffle/LZ4 source (4.88 GB) | 4.04 |
| Encoded resident (6.05 GB): payload | 4.87 |
| Encoded resident: spatial index | 0.14 |
| Per-detector-pixel order-0 entropy | 2.58 |
| Entropy of one table for all pixels | 4.04 |

| Data property | Value |
|---|---|
| Zero fraction | 0.44 |
| Mean count | 16.3 |
| Values >= 32 | 21 % of all values |
| Detector pixels with more than 1 % of values >= 32 | 8168 of 36864 |

The integer codec (`resident/cuda/kernels/streamed.cu`) codes symbols 0 to 31
with one of 64 fixed rANS tables chosen per pixel and 512-scan interval; a
count of 32 or more is an escape symbol followed by two raw bytes. The gold
bright-field disk holds counts up to 233, so 21 % of values cost at least 16
bits: 3.4 bits per value from escapes alone. The copy is therefore larger than
the source, and `convert` keeps the HDF5 file. This is expected for dense,
high-count bright-field data; the size rule was not changed.

## Not changed

- The size rule (keep HDF5 when the copy is not smaller).
- The Metal kernel change is not run on Apple hardware in this session; run
  `QEM_TEST_BACKEND=mps pytest tests/hardware/test_uint32_flagged_master.py
  tests/hardware/test_qem_collection.py` on a Mac.
