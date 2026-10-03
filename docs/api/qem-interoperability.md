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

## Python portability and redistribution examples

These are developer conformance and provenance examples. For ordinary GPU use,
see [Save and share your data](qem-python.md).

### Small synthetic CPU reference

```python
import numpy as np
from quantem.gpu import io

# Synthetic counts: (scan row, scan column, detector row, detector column).
counts = np.random.default_rng(7).poisson(2, (8, 12, 16, 16)).astype(np.uint16)
io.save("example.qem", counts, backend="cpu", metadata={
    "scan_sampling_A": [0.4, 0.4],
    "voltage_kV": 300,
    "source_metadata": {"data_origin": "synthetic example"},
})

with io.load("example.qem", backend="cpu") as acquisition:
    np.testing.assert_array_equal(acquisition.data, counts)
    print(acquisition.metadata["scientific_metadata"])
```

An existing destination is never overwritten. CPU reference encoding is for
portability and verification, not a claim of GPU-like speed. It uses bounded
encoding chunks; dense CPU decoding still requires RAM for the decoded array.
For large acquisitions, retain the accelerated encoded path where supported.

### Inspect metadata without a GPU or full decode

```python
info = io.inspect("example.qem")
metadata = info.metadata["scientific_metadata"]
row = metadata["axes"][0]
print(row["name"], row["size"], row["sampling"])
# scan_row 8 {'value': 0.4, 'unit': 'angstrom', 'provenance': ...}
```

Read `value` and `unit` together. Do not assume a number is in meters because a
different library expects meters. Missing sampling is unknown, not one or zero.
`source_metadata` retains the reader-provided original tags. Coverage is stated
explicitly; retaining tags is not a promise to archive every proprietary object.

The machine-readable contract is
{download}`qem-metadata-schema-v2.json <../../src/quantem/gpu/io/qem-metadata-schema-v2.json>`.
JSON Schema checks structure; the package validator additionally checks physical
consistency, versioning, spans and checksums:

```python
from quantem.gpu.io.qem_validation import validate_qem
report = validate_qem("example.qem")
print(report["integrity"], report["codec_layout"])
```

For a reader in another language, the first 56 bytes locate and authenticate a
UTF-8 JSON header. Follow the [envelope](qem-format.md) and
[complete codec definition](qem-codecs.md); no original filesystem path is
needed to decode the saved measurements.

### Export measurements again

```python
with io.load("example.qem", backend="cpu") as acquisition:
    np.save("restored.npy", acquisition.data)
```

NumPy does not carry the full QEM scientific record. Export
`acquisition.metadata["scientific_metadata"]` as adjacent JSON if you need that
calibration when sharing `.npy`. Float background recipes remain separate from
the original decoded array; never apply one silently during a format conversion.

### Share public data, including Hugging Face

Start with the [synthetic notebook](../examples/qem_portable.ipynb). The
distributed conformance bundle is synthetic and MIT licensed. It contains no
private acquisitions, usernames, microscope-session paths or research notes.

A future gold-data example should name a specific public repository, immutable
revision and source license, verify download hashes, then publish new `.qem`
derivatives alongside the originals. Public download access alone is not
permission to redistribute. Review source tags for identifying information
before publishing. Record the conversion software revision, source hashes,
exactness checks, shapes, dtype, units and any explicitly applied corrections.
Nothing in this workflow uploads data automatically.

#### Public QuantEM examples

The intended public home is
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data).
At revision `00179851c0015612bfb6e6438e02387f5ffff0ae`, its dataset card declares
MIT licensing. A conversion must preserve the applicable attribution and check
any file-specific restrictions as well.

Start with `4dstem/gold_128_npy_bin8/data.npy` and its `meta.json`: the metadata
declares uint16 measurements with shape `(128, 128, 24, 24)`, scan sampling in
angstrom and detector sampling in mrad. Preserve the entire source JSON, not
only its normalized numbers. In particular, its scan calibration is explicitly
inferred from a sibling acquisition rather than measured per file. Preserve
that qualification, the binning/averaging history and each optics source.
QEM conversion preserves the input array exactly; it does not undo earlier
averaging or establish the accuracy of a supplied calibration.

Include the source repository, immutable revision, relative source filenames,
SHA-256 hashes, license and converter revision in retained source metadata.
Check every decoded measurement and the normalized calibration before publishing
an additional `.qem` file. Keep originals until that migration is separately
approved. The repository also contains HAADF images, 1D tutorial arrays and
other assets outside the current four-axis QEM codecs; do not reshape or cast
those merely to claim every file is supported. This inventory is not a completed
dataset migration.
