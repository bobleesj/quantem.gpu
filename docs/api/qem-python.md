# Read, convert and share QEM from Python

QEM 0.0.1 combines exact measurements with explicit scientific metadata. It is
an experimental open format. The container is not HDF5; ordinary HDF5 readers
cannot open it. QuantEM provides GPU readers and an explicit CPU reference path.
These examples require a development version containing that reference path,
not the older published app or an assumed PyPI version.

## Open and save on your GPU

```python
from quantem.gpu import detector, io

with io.load("acquisition.npy") as acquisition:
    pattern = detector.prepare(acquisition).frame(0)
    io.save("acquisition.qem", acquisition)
```

Omit backend, representation and compression options for normal use. The loader
selects an available CUDA or MPS device, ingests supported originals into ANS,
and decodes selected products on demand. Saving retains measurements and
reader-provided metadata. See [supported inputs](io.md) for exact dtype,
geometry and calibration limits. Python coverage does not certify native Swift.

## Small synthetic CPU reference

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

## Inspect metadata without a GPU or full decode

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

## Convert supported source files

```python
# NumPy .npy, K3/other qualified DM3/DM4, or an EMPAD float-export XML/RAW pair.
with io.load("acquisition.npy") as acquisition:
    io.save("acquisition.qem", acquisition)
```

| Input | Python route | Limits |
| --- | --- | --- |
| NumPy `.npy` | `io.load(path)`, then `io.save` | 4D counts or float32; wider input requires an exact audit |
| DM3/DM4 including K3 | `io.load(path)`, then save | Install `quantem.gpu[dm]`; one qualified 4D count or float32 image |
| EMPAD-G1 XML/RAW float export | `io.load("scan.xml")` | 130x128 float32 records; retains the 128x128 detector, not footer words |
| EMPAD-G2 declared float export | Same XML route | Explicit 128x128 float32 raster; raw encoded detector words and their calibration are not supported by this reference importer |
| ARINA/NCEM HDF5 | `io.load(path)`, then `io.save` | Qualified original and generic array layouts; not arbitrary HDF5 structures |
| Other vendors or array dtypes | Not automatically supported | Use a verified source reader, preserve its calibration, and pass supported NumPy measurements; do not relabel bytes |

For a headerless EMPAD-G1 `.raw`, explicitly supply `scan_shape=(rows, columns)`.
This selects the documented G1 float layout; file size alone is not a detector
format detector. No background or gain correction is silently inferred.
Source footer words are not detector measurements; original XML is retained.
The reference exporter rejects an explicitly already-background-corrected float
array until that correction state is qualified across readers. Keep that array
and its metadata, or export the original measurements. Decoding never performs
an additional background subtraction.

For already encoded GPU counts, saving copies the encoded bytes and indexes:

```python
with io.load("acquisition.dm4") as acquisition:
    io.save("acquisition.qem", acquisition)
```

Use the runtime your machine supports. CPU is explicit, never a fallback for a
failed GPU operation. Qualified float32 `.qem` files reopen directly on CUDA
and MPS, with the saved geometry and correction provenance. See the
[field-by-field mapping](qem-metadata-mapping.md) for calibration limits.

## Export measurements again

```python
with io.load("example.qem", backend="cpu") as acquisition:
    np.save("restored.npy", acquisition.data)
```

NumPy does not carry the full QEM scientific record. Export
`acquisition.metadata["scientific_metadata"]` as adjacent JSON if you need that
calibration when sharing `.npy`. Float background recipes remain separate from
the original decoded array; never apply one silently during a format conversion.

## Share public data, including Hugging Face

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

### Public QuantEM examples

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
