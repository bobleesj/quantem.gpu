# QEM v1 reference files

These small, synthetic files are distributed under the repository's MIT license.
They contain no experimental acquisition or personal metadata.

Each NumPy array has shape `(3, 5, 16, 16)`, ordered as scan row, scan column,
detector row, detector column. The corresponding `.qem` contains exactly the
same integer counts. The uint8 file deliberately has unknown calibration;
the uint16 file has explicit synthetic calibration overrides.

`manifest.json` records file hashes, original-count hashes, and expected
scientific metadata. Preserve these frozen references when testing new readers.
Do not regenerate them to hide a regression. Generation and validation commands
are documented in `docs/api/qem-interoperability.md` at the repository root.

Checksums establish file integrity, not decoded parity. Test every count against
the `.npy` file using the backend under qualification.
