# QEM references and interoperability checks

A portable file needs more than a readable header. Test three independent layers:
file integrity, metadata meaning, and decoded measurements.

## Validate a saved file without a GPU

```sh
python -m quantem.gpu.io.qem_validation acquisition.qem
```

The command reads the checksummed header and every encoded body byte in bounded
64 MiB blocks. It returns JSON and exits nonzero for invalid files. It never
decodes a measurement or starts a GPU. A valid checksum is not proof that a writer
encoded the right scientific values. Checksums detect damage, not authorship.

`integrity=verified` covers envelope length, metadata version/axes/override
contract and checksums. The integer codec additionally runs the production
array-span validator (`codec_layout=verified`). EMPAD float files also receive
chunk/row-descriptor bounds checks with `codec_layout=verified`. Neither check
decodes measurements. Unknown codecs are rejected explicitly.
`decoded_parity=not_checked` is always reported by this command.

See the [metadata map](qem-metadata-mapping.md) for fields normalized by readers,
fields merely retained, and missing/unsupported quantities. The integrity tool
does not assert that every normalized value is physically correct or that the
original vendor metadata is complete.

## Shareable synthetic references

The MIT-licensed `tests/data/qem-v1/` bundle contains uint8 and uint16 arrays,
native-written `.qem` files, and a JSON manifest with file/count SHA-256 hashes
and expected scientific metadata. No private acquisition is included. Each
array is `(3, 5, 16, 16)` in `(scan_row, scan_column, detector_row, detector_column)`
order and includes zero and maximum count values. The uint8 reference has
unknown calibration; the uint16 reference carries explicit synthetic overrides.

Download the reference files and keep them together:

- {download}`uint8.qem <../../tests/data/qem-v1/uint8.qem>` and
  {download}`uint8.npy <../../tests/data/qem-v1/uint8.npy>`
- {download}`uint16.qem <../../tests/data/qem-v1/uint16.qem>` and
  {download}`uint16.npy <../../tests/data/qem-v1/uint16.npy>`
- {download}`manifest.json <../../tests/data/qem-v1/manifest.json>`

Copy the bundle anywhere; the original build path is not required. The generator
scrubs its local source locator before freezing the checksummed files. Do not
regenerate fixtures to hide a parity failure. Generate candidates in a new folder:

```sh
python scripts/build_qem_references.py /tmp/new-qem-reference-bundle
```

Generation requires native Metal. Small reference arrays are correctness tests,
not performance or real-detector-format qualification.

## Run the same tests on each backend

```sh
pytest -q tests/test_qem_validation.py
QEM_TEST_BACKEND=mps pytest -q tests/test_qem_interoperability.py
QEM_TEST_BACKEND=cuda pytest -q tests/test_qem_interoperability.py
bash scripts/check_qem_reference.sh tests/data/qem-v1/uint16.npy tests/data/qem-v1/uint16.qem
```

The explicit backend test compares every decoded count with the original NumPy
array, re-exports through the production Python writer, moves the saved copy,
then compares counts and the complete scientific metadata again, with schema-1
units converted to the specified schema-2 units on new export. The frozen
schema-1 files are never regenerated to make a test pass. No CPU fallback
is permitted. Without `QEM_TEST_BACKEND`, hardware tests skip; an explicitly
requested but unavailable backend fails. Supply its Python-written file to the
native command to test the reverse direction as well.

CPU integrity tests are portable to macOS, Linux and Windows; configuring CI for
those systems does not itself mean all have executed successfully. Native Metal,
Python-hosted Metal and CUDA execution must each have their own recorded result.
Windows Metal is not supported. Python GPU EMPAD decoding, WebGPU QEM decoding and
arbitrary 3D/5D codecs are not implied by these integer-reference tests.

## Portable schema-2 conformance bundle

The additional MIT-licensed `tests/data/qem-v2/` bundle is entirely synthetic:

- {download}`uint8 interval boundary <../../tests/data/qem-v2/u8-interval-boundary.qem>`
  and {download}`original counts <../../tests/data/qem-v2/u8-interval-boundary.npy>`;
- {download}`uint16 multiple chunks <../../tests/data/qem-v2/u16-multiple-chunks.qem>`
  and {download}`original counts <../../tests/data/qem-v2/u16-multiple-chunks.npy>`;
- {download}`float32 special bits <../../tests/data/qem-v2/float32-special-bits.qem>`
  and {download}`original bits <../../tests/data/qem-v2/float32-special-bits.npy>`;
- {download}`frozen manifest <../../tests/data/qem-v2/manifest.json>` and
  {download}`invalid metadata mutations <../../tests/data/qem-v2/invalid-metadata.json>`.

The integer examples cross 512 scans and include detector edge tiles and extreme
counts. Float examples retain signed zero, infinities, NaN payloads and subnormals.
Compare floats by their uint32 bits. The explicit CPU reference tests require no
GPU and compare every decoded measurement, not just the header:

```sh
pytest -q tests/test_qem_reference.py tests/test_qem_metadata.py tests/test_qem_validation.py
bash scripts/check_qem_float_reference.sh tests/data/qem-v2/float32-special-bits.npy tests/data/qem-v2/float32-special-bits.qem
```

CI is configured for the portable tests on Linux, macOS and Windows. A configured
job is not evidence that a run has completed. GPU checks still require real
hardware and explicit opt-in. Generate proposed new fixtures in a new directory
with `scripts/build_qem_conformance.py`; never silently replace frozen files.

For real source qualification, also run `scripts/check_qem_roundtrip.sh`,
`scripts/check_qem_collection.sh` and `scripts/check_qem_calibration_roundtrip.sh`.
These cover different source/correction contracts; a synthetic reference does
not replace them.
