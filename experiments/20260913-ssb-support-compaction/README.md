# Fourier support compaction feasibility

## Result

For the measured full 8,937-BF geometry, a fixed-rotation interval layout
requires 7,042,386,864 bytes including metadata versus 9,407,729,664 bytes
for the complex64 Hermitian cache: 25.14% smaller. This is an exact count of
the proposed layout's bytes over all BF geometry, not an observed full-cache
Metal allocation. Production remains unchanged.

| Proposed storage | Total bytes | Reduction |
| --- | ---: | ---: |
| Existing dense half-spectrum | 9,407,729,664 | baseline |
| Fixed rotation, bitmask plus offsets | 7,277,215,544 | 22.65% |
| Fixed rotation, row intervals plus offsets | 7,042,386,864 | 25.14% |
| Any rotation, bitmask plus offsets | 9,326,656,304 | 0.86% |
| Any rotation, row intervals plus offsets | 9,069,270,704 | 3.60% |

This is not reversible compression of every Fourier value. It omits values
whose *corrected contribution* is identically zero for a declared geometry.
The raw acquisition is unchanged, and all selected BF terms are retained.
Changing geometry may require regenerating omitted Fourier evidence.

## Layout and contract

Each BF has 512 row descriptors, each with a UInt32 payload offset and a
packed UInt16 start/count pair. Interior retained coefficients occupy one
contiguous interval per row. Filling gaps between supported intervals is
conservative and costs a little storage but simplifies direct addressing.
Columns 0 and 256 are stored separately. Rows 0 and 256 are fully retained.
Metadata costs 36,605,952 bytes across all BF terms and is included above.
Complex64 payload values are copied without quantization.

The support test retains either shifted aperture disk, with a conservative
1e-4 inverse-Angstrom boundary margin. It uses the exact exported native
geometry for the established 30 mrad, 300 keV, 0.264 Angstrom scan-step case.
Fixed rotation is 158.88268568029937 degrees. The arbitrary-rotation envelope
retains frequencies within the aperture radius plus each BF radius.
All DC and Nyquist boundaries are preserved to avoid hiding the known
phase-objective endpoint issue.

Lower-order aberration changes do not change this aperture support. Scan
rotation, beam center, aperture, sampling, and BF selection changes invalidate
the fixed geometry contract. As a negative control, adding 20 degrees requires
4,120-16,962 new half-plane coefficients in the basic fixed-angle masks of the
eight sampled BF terms. A compact cache must never silently reuse an invalid
mask; rebuild or use a valid broader cache. Higher-order UI combinations and
retargeting are not integrated/tested in the production engine here.

## Evidence

- Original-file master hash matches the preceding full-aperture benchmark.
- Native Metal decoded eight real uint32 BF columns, each retaining the full
  512×512 scan. No CPU source decompression was substituted.
- Geometry census covers all 8,937 BF terms.
- NumPy complex64 payload round trips preserve every stored bit. The full
  complex correction reference, not the faulty projected-loss kernel, sees
  maximum corrected-spectrum error 0.0 in all 32 BF/layout cases.
- Three C10 settings were checked; rotation-safe layouts were checked at
  0, the reference angle, reference+20 degrees and 270 degrees. The inverse
  transform phase arrays match exactly in those cases. This is sampled
  BF reference evidence, not full 8,937-term objective parity.
- A standalone Metal row-interval lookup reproduced expected sample buffers
  bit-for-bit over 21 launches. The eight-BF compact buffers occupy 6,145,384
  bytes versus 8,421,376 bytes dense. Twenty warm lookup calls measured
  GPU p50/p95 0.1211/0.1815 ms. These small-buffer timings do not predict
  full-cache performance or SSB objective speed.
- Lookup deliberately uses integer bit copies, so it introduces no floating
  point rounding. No full-cache allocation, fused objective integration,
  optimizer timing, or UI performance claim is made.

Scripts: `probe.py` and `metal-check.swift`. Native extraction is opt-in via
`SSB_PROBE_EXPORT_BF=1` in the existing scheduling probe. Evidence is retained
under `local-evidence://ssb-support-compaction-20260913/`.

## Next gate

Implement direct interval fetch in the SSB consumer without expanding the
entire cache. Fix the separately established Nyquist objective discrepancy,
then run full-BF independent parity, cache invalidation, memory-peak and
adjacent performance tests. The conservative any-rotation layout saves too
little here to promise a major benefit. No branch, commit, push, or release
was created by this experiment.
