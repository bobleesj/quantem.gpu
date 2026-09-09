# Exact count-ANS codec and retained experiment formats

This is an experimental opt-in API. Start with
[experimental resident ANS](experimental-resident-ans.md) for current ownership,
Show4DSTEM usage, supported formats, and qualification limits.

The canonical count-ANS v1 representation preserves every native uint8/uint16
count, including hardware sentinels. It uses independent detector-column byte
rANS streams within scan blocks, a model selector for every stream, and an
explicit uint16 literal profile for incompressible columns. Dimensions are
`(scan_row, scan_column, detector_row, detector_column)`.

## Package ownership

`io.save(path, counts, format="quantem", compression="ans", backend="cpu")` is the explicit
reference encoder. It accepts four-dimensional NumPy arrays or memory maps,
uses bounded scan blocks, records source geometry and typed section checksums,
and publishes a new file atomically without overwriting an existing file.
No quantization, crop, binning or invalid-pixel replacement occurs. This is a
correctness reference, not an accelerated full-acquisition encoder.

```python
from quantem.gpu import io

saved = io.save("counts.ans", counts, format="quantem", compression="ans", backend="cpu")
```

The private `io._ans.ANSFile` reader validates the self-contained container and
returns one backend-neutral array contract through `runtime_arguments()`.
`io.backends.cuda._ans.CudaANSResidentCounts` owns the CUDA decoder. It copies
encoded arrays into owned device buffers and checks every stream's terminal
state before publication. Its block decode, point-pattern gather and binary
mask sum retain integer exactness; mask sums are uint64. No full dense source
is constructed. A direct packed conversion is also retained and tested.

The container contract is shared with the existing CUDA and MPS implementations.
This integration adds the WebGPU adapters and retains those native backends;
it does not constitute a new Metal hardware qualification.

The public loader supports exact CUDA residency and explicitly requested CPU
reference materialization. The returned resident owner keeps encoded buffers
until `release()`; its operations return caller-owned device results. No dense
acquisition is allocated when `representation="ans"` is selected:

```python
from quantem.gpu import io

source = io.load(saved.path, backend="cuda", representation="ans", device=0).data
try:
    pattern = source.extract_diffraction_device(0, 0)
    image = source.detector_sum_device(binary_detector_mask)
finally:
    source.release()

reference_counts = io.load(saved.path, backend="cpu", representation="dense").data
```

Show4DSTEM can export canonical files with
`export_show4dstem_rans_viewer([saved.path], "ans-viewer")` for browser WebGPU.
Its direct CUDA resident-owner factory is a separate protocol; this does not
claim that `Show4DSTEM(io.load(...))` accepts every canonical CUDA owner.
Existing HDF5 loading defaults are unchanged.

Section checksums detect corruption. Supply an independently retained
`expected_sha256` to `ANSFile` when source authentication is required. A valid
container checksum alone does not establish acquisition identity.

## Compatibility is explicit

| Retained format | Relationship to count-ANS | Implemented migration |
| --- | --- | --- |
| Seven-tilt detector-conditioned byte rANS | Same recurrence and literal profile; block-local offsets and per-model/per-column tables differ | `_ans_legacy._legacy_rans_arguments(record)` maps offsets and selectors into the canonical decoder without decoding or copying payload bytes |
| `quantem.gpu.count-ans.v1` | Canonical self-contained format | Reference encoder, container reader, CUDA decoder |
| `quantem-resident-source112-index180-v1` | `source112-tans1024-pair-v1` source plus `position9-flag1-count8-rank256-v1` sparse index | `_source112_archive._convert_source112_acquisition` decodes bounded source254 archive windows and streams complete acquisitions through the canonical writer |

The legacy adapter verifies the payload digest in the retained build record,
checks exact offset partitioning and model structure, and returns a read-only
payload memory map. Its caller must close that map after the runtime owns the
bytes. Legacy model archives lack independently recorded per-table digests;
compare decoded products and patterns against frozen acquisition references.

The retained tANS transition table is not a byte-rANS probability model, and
its sparse index is not an entropy stream. Archive admission identifies the
versioned storage layout independently of its provenance prefix; record hashes,
codec identifiers, native geometry and component bounds remain mandatory.
The migration reader uses the package's bounded source112 decoder and never
imports a native application's process or device state.

`_source112_archive._Source112Archive` reads the four native source components from
the immutable source254 archive. It verifies record and global-table hashes,
the dense/sparse partition, seek coverage and literal ownership of hardware
counts. It ignores interaction indexes. `_convert_source112_acquisition` iterates
all 262,144 native scan positions in 512-pattern decoding windows and passes
256-pattern blocks to the same canonical writer. Only one source chunk and one
decoded window are needed at a time; no full dense acquisition is constructed.
This explicit archival conversion requires host staging for the CPU reference
encoder. It is not the interactive viewer or a zero-upload display path.

```python
from quantem.gpu.io._source112_archive import _convert_source112_acquisition

# Explicit migration utility; original archive stays immutable.
_convert_source112_acquisition(
    source254_archive, "acquisition-0.ans", acquisition=0, device=0,
)
```

The converter retains source identity and full native geometry in the new
container. Output publication is atomic and refuses existing destinations.
There is no implicit crop, binning, clipping or invalid-pixel replacement. All
66 full-acquisition reencodes have not been run in this validation session;
the preserved-archive gate below checks bounded windows against original HDF5
counts and a complete canonical round trip of a labeled validation window.
No Tier D deletion is implied.

## Verification

`tests/contracts/io/test_ans_codec.py` tests native uint8/uint16 CPU round trips, CUDA
round trips, tail blocks, sentinel counts, requested-pattern order and
duplicates, binary detector sums, and direct packed conversion. All comparisons
use integer equality. GPU tests are explicit opt-ins:

```bash
CUDA_VISIBLE_DEVICES=0 QUANTEM_CUDA_ANS_TEST=1 PYTHONPATH=src \
  python -m pytest -q tests/contracts/io/test_ans_codec.py
```

The real-data test additionally reads `QUANTEM_LEGACY_RANS_BUILD` (the retained
seven-tilt `build-result.json`) and `QUANTEM_LEGACY_RANS_PRODUCTS` (the directory
containing `tilt-1/products` through `tilt-7/products`). It checks four frozen
mask products and the raw selected pattern for each acquisition using the
canonical CUDA decoder. These evidence paths are never inferred from another
process or its GPU state.

Validated on 2026-09-07 with CUDA-visible GPU0, NVIDIA RTX PRO 6000
Blackwell Workstation Edition: all 5 codec tests passed, including 28 frozen
products and 7 raw patterns, in 30.96 seconds. The save/import regression
selection passed 11 tests with 3 opt-in GPU/evidence skips. A wheel build
passed and contains the CUDA source required for runtime compilation. These
are correctness and packaging results, not latency benchmarks.

## Retained source112 migration validation

`tests/contracts/io/test_source112_ans_migration.py` compares 512 complete patterns from
source chunks 0, 523 and 1055 (local starts 0, 12288 and 15872) against original
HDF5 acquisitions. Every one of the 56,623,104 native uint16 counts matched.
The first 18,874,368-count window also passed source112 decode, canonical
streaming encode, public CUDA resident load and full count reconstruction.
The validation window is labeled as such; it is not a reduced replacement for
the full-acquisition converter.

```bash
CUDA_VISIBLE_DEVICES=0 QUANTEM_CUDA_ANS_TEST=1 \
QUANTEM_SOURCE112_ARCHIVE=/path/to/preserved/source254/archive \
PYTHONPATH=src python -m pytest -q \
  tests/contracts/io/test_ans_codec.py tests/contracts/io/test_source112_ans_migration.py
```

On GPU0 Blackwell this selection passed 5 tests with 1 optional seven-tilt
skip in 6.55 seconds. Public `io.load` CPU/CUDA paths are included in these
workflow tests.

## Remaining unification work

Both retained experimental formats can now enter the canonical CUDA codec:
seven-tilt through table/offset admission, retained source112 through explicit bounded
migration. The browser loads retained manifests through
`detector/backends/webgpu/rans.ts` and canonical containers through the package
`count-ans.ts` adapter. Stock Show4DSTEM exports canonical containers using
that adapter. Direct CUDA widget-owner compatibility remains a separate gate. An accelerated canonical
encoder and complete 66-acquisition conversion campaign are not qualified.
The small validation window encoded to 5,098,522 bytes; its compression does
not establish that the complete series will fit in the same device memory as
the original source112 archive. Full-series capacity needs its own admission
check before replacing the resident production source.
Changing production sources requires integer parity first. No latency claim
follows from this codec correctness work.
