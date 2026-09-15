# Single-decode scratch experiment

The direct converter decodes each ANS stream twice: once for widths and once
for final bit-plane values. This candidate instead stores exact uint16 values
from the first pass in one reusable private 4096-scan window (301,989,888 bytes
per worker). The packer reads that window. No full dense acquisition is retained,
no original file is read, and the final packed bytes/counts are unchanged.

It is opt-in using `QGPU_PAIRED_CONVERSION_STAGING=1`. The ordinary converter
remains the lower-workspace two-decode implementation.

| Native Apple M5 trial | Workers | Seven-source forward wall |
|---|---:|---:|
| Previous two-decode comparison | 2 | 5.959 s |
| Single decode, pixel-order scratch | 2 | 5.345 s |
| Single decode, stream-rank scratch | 2 | 5.363 s |

The two staged runs pass 21 complete detector maps and 77 full diffraction
patterns each, before proceeding through a verified reverse conversion. Both
end at 12.046 GB packed residency. Maximum sampled device allocation was
13.925/13.939 GB during forward conversion, not a process-memory peak.

Staging gives a modest candidate gain; stream-rank ordering did not establish
an additional benefit. These are exploratory runs, not a new qualified default.
The 1–2 s seven-source target remains unmet. The additional scratch cannot be
ignored when admitting concurrent conversions.

An independent synthetic oracle checks all 262,144 input counts for each of
four combinations of staging on/off and compact/plain offsets (1,048,576
comparisons). It also checks the reverse packed reader, high uint16 values,
detector permutation, and rejection of unsupported modes. Coverage is bounded,
not an exhaustive real-data whole-volume oracle or every malformed stream.

Run using the round-trip harness in `20260914-packed-to-ans`; raw JSON receipts
and their hashes are registered in the manifests under
`local-evidence://sep14-ans-conversion-followup/`.
