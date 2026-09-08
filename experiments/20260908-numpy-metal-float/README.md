# Native Metal float bit-packing on the full merged NumPy source

**The kernel works exactly; the full packed representation does not fit 24 GB.**
Both complete runs preserved all 9,663,676,416 source float32 words bit-for-bit.
No CPU compression, CPU decompression, CPU prefix scan, conversion, clipping,
binning, or cropping was used.

This is one `(512, 512, 192, 192)` merged acquisition, not several datasets.
The original was transferred to the local unified testing folder and its entire
SHA-256 matched `24d7ec1fe83ec12faddf026a92b8f97f003202dc68a326664ec35f3a0d8c7057`.
The audit validates its exact header/length and checks that file identity and
modification time remain unchanged. It does not recompute a CPU source hash
inside the timing interval.

## Actual compute path

1. Read original file bytes directly into a bounded shared Metal input buffer.
2. Run the unchanged production `empad_describe_simd` kernel. Blocks contain
   128 consecutive words; native 192-column detector data is not resized.
3. Compute all payload offsets with three experiment-only GPU prefix kernels.
   The production loader's CPU descriptor loop is not used in this audit.
4. Run the unchanged production `empad_pack_simd` kernel into private storage.
5. Use the production `empad_word` decoder on the GPU for every original word
   and compare its raw UInt32 bits against the input. Read only the scalar
   mismatch counter and total packed byte count on the CPU.

The CPU handles filesystem IO, command submission, error checks and reporting.
It performs no scientific value conversion or codec work. The decoder is a
bit reinterpretation, not a numeric cast from float to integer.

The audit processes all source values in 64-frame windows and releases/reuses
each window. **It does not keep the complete acquisition resident**, publish a
preview, or launch a NumPy dataset in the app. The tiny audit footprint must not
be presented as the memory needed by the full dataset.

## Measured results on Apple M5

| Quantity | First | Repeat |
|---|---:|---:|
| Original float32 words verified | 9,663,676,416 | 9,663,676,416 |
| Packed payload bytes | 36,429,611,552 | 36,429,611,552 |
| Descriptor bytes | 1,207,959,552 | 1,207,959,552 |
| Required packed bytes before alignment | 37,637,571,104 | 37,637,571,104 |
| GPU describe + offsets + pack | 1.7246 s | 1.6200 s |
| GPU unpack + exact comparison | 0.8467 s | 0.8188 s |
| Original byte-read time | 4.2502 s | 4.2295 s |
| Complete bounded audit wall time | 8.4357 s | 8.2800 s |
| Observed audit Metal allocation | 19,628,032 bytes | 19,628,032 bytes |

These are synchronized command-buffer GPU timestamps and audit timings,
**not** resident-ready application load times. Shader compilation precedes the
timer. Source-page state was uncontrolled, so neither run demonstrates cold
SSD throughput. No performance improvement over another implementation is
claimed by these repeated correctness runs.

The actual packed payload averages **30.16 bits per value**, plus approximately
one bit per value for block descriptors. This kernel therefore finds little
removable redundancy in these particular float bit patterns. That differs
from low-valued integer counts that leave most upper bits unused. It is not a
proof that every other lossless float codec must have the same capacity.

## Provenance and reproduction

Production backend at `ab6352da6cf898479e5cdae22a842324db520c4e`; production
shader unchanged. The new experiment sources are retained beside this README.

SHA-256 fingerprints:

- Production `empad_float.metal`: `4b13ca7fedb79131f48300966b6765e18d44fc7b4e0fbb8da130866130c27c5b`
- `audit.swift`: `0a061996115e17a1bd8a3bbd712cefe220c2b27eca28a6fd6f78cc4854920760`
- `audit.metal`: `5dffabe7a325ec3c038d40f570c9c359d61d32344a597e718a436a1c795e7776`
- Executable: `f93bc37c2831f77412b4911cee2e82997368ba3bca9268f959efe33b9089d56f`

From the repository root, with `SOURCE` set to the verified original file:

```bash
swiftc -O experiments/20260908-numpy-metal-float/audit.swift -o /tmp/float-audit
/tmp/float-audit "$SOURCE" \
  src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/empad_float.metal \
  experiments/20260908-numpy-metal-float/audit.metal
```

NumPy parsing here deliberately accepts only the exact authenticated fixture;
it is not a general-purpose NumPy reader. No production API, installed app,
release or dependency pin was changed. A smaller exact GPU representation is
still needed before this entire source can fit the requested native resident
workflow. Do not substitute CPU archive compression or disk-backed previews.
