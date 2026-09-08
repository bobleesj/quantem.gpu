# Full merged NumPy capacity audit

The existing lossless 128-word float XOR representation cannot fit this complete
merged dataset into a 24 GB Mac. This is a capacity result, not a statement
that every possible lossless compressor must fail.

| Measurement | Complete source |
| --- | ---: |
| Shape | 512 x 512 x 192 x 192 |
| Dtype | float32, little endian, C order |
| NPY file | 38,654,705,792 bytes |
| Audited words | 9,663,676,416 |
| Fractional words | 8,807,300,314 |
| Zero words | 856,375,257 |
| XOR payload | 36,429,611,552 bytes |
| Descriptors | 1,207,959,552 bytes |
| Packed total before allocation overhead | 37,637,571,104 bytes |

The SHA-256 from a separate complete source hash matches the audit's streamed
digest. Every value participates. Reads are bounded to 16 MiB; the audit does
not allocate a dense full volume. It was run on the source host using CPU/NumPy
to inspect capacity, not to benchmark native loading on the target Mac.

An initial strided sample projected 37.82 GB and about 75% fractional values.
That sample overrepresented edge positions. The complete audit supersedes it:
37.64 GB and about 91.14% fractional values. Do not quote sample statistics as
the whole acquisition's value distribution.

The payload calculation follows the current format: XOR each block's words
against its first word; remove common leading/trailing zero bits; retain the
residual width and a 16-byte descriptor. No IEEE-754 value conversion is used.
The native float kernels currently require 128x128 detector geometry, so this
audit measures the format extended over the linear word sequence, not an
already implemented native 192x192 loader. 36,864 pixels/frame is divisible
by the same 128-word block length.

Sequential loading or overlapping bounded staging can avoid a second dense
volume, but cannot remove the final packed payload requirement. Integer casting,
float16 conversion, cropping, binning, or calling a mapped file resident would
not satisfy this task. A different lossless representation must be demonstrated
to fit before promising complete residency, and the original 38.65 GB file
must still be read on each source reload unless a separately labeled exact
on-disk cache is adopted. There is no one-second or native UI claim here.

Reproduce: `python experiments/20260908-numpy-float-capacity/audit.py /path/to/merged.npy`.
The source stays private and is not bundled in the repository. Original transfer
and checksum receipts are kept with the target machine's local data folder.
