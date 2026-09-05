# Deploy a Linux CUDA host

QuantEM.GPU Remote uses the same Python package and CUDA kernels as in-process
CUDA. The dedicated environment makes the service dependencies and executable
repeatable without creating a second source tree.

## Create the environment

From a QuantEM.GPU checkout:

```bash
conda env create -f environment-remote-cuda.yml
```

The environment name is `quantem-gpu-remote`. The filename retains
`remote-cuda` for compatibility with existing automation. The equivalent
editable developer installation is:

```bash
python -m pip install -e ".[cuda,remote]"
```

The `cuda` extra provides CuPy and CUDA bindings. The `remote` extra provides
FastAPI and Uvicorn. The installed CUDA runtime must be compatible with the
host driver.

## Verify source and device

```bash
conda run -n quantem-gpu-remote quantem-gpu --help
conda run -n quantem-gpu-remote python -c \
  "import quantem.gpu as qgpu; print(qgpu.__version__, qgpu.device.detect())"
```

For a recorded deployment, also capture the Git revision or wheel hash,
Python executable, package freeze, CUDA runtime and driver, visible GPU list,
and the configured data-root identity. A package version alone is not source
provenance.

Before binding a packaged Windows consumer, verify
`GET /api/browse/capabilities` reports the intended exact
`implementation_revision` and a `packaged_service` object matching
`quantem.gpu.packaged-browse-service/v2`. Its compatibility chain must be
`quantem-live-browse/3 -> live4dstem-standalone/3 -> quantem-gpu-browse/1`.
Do not point the Windows client directly at the raw Browse v1 port; the
loopback adapter is the versioned seam.

## Start the service

```bash
conda run -n quantem-gpu-remote \
  quantem-gpu serve /data/4dstem --gpus auto --port 8780 \
  --implementation-revision <exact-git-sha>
```

`--gpus auto` makes every visible CUDA device eligible. Use an explicit list
when the host is shared and QuantEM.GPU owns only selected devices. Dataset
placement and fit are explained in [GPU admission and residency](admission.md).

## Prepare a lossless-packed deployment

Keep the acquisition master as the client-visible identity. For an already
qualified packed file, one call verifies every stored byte against its recorded
SHA-256 and creates a new deployment directory containing the integrity
manifest and trusted server registry:

```python
from quantem.gpu.remote import prepare_browse_source

registry = prepare_browse_source(
    "/data/4dstem/session/sample_master.h5",
    "/data/prepared/sample.h5",
    "/data/deployments/sample-v1",
    expected_source_sha256=qualified_source_sha256,
)
```

The equivalent installed command is:

```bash
quantem-gpu prepare-browse /data/4dstem/session/sample_master.h5 \
  /data/prepared/sample.h5 /data/deployments/sample-v1 \
  --expected-source-sha256 <qualified-source-sha256>
```

Neither source is modified, copied, cropped, binned, or repacked. Existing
output directories are refused. Preparation checks raw reconstruction support
and master/source shape agreement; the independently qualified seal remains
the authority for the scientific source. Source-bound calibration and prepared
moments must already be present if the application requires them. Creating a
packed file from raw acquisition data is a separate, explicit
[producer workflow](../api/native_lossless_pack_v1_producer.md).

Preparation reads the complete packed source once. Do not count it as loading
or include it in a prepared-load benchmark. The resulting `sources.json` is
accepted by `--compact-sources`; no hand-written registry is required.

For administrators combining several qualified datasets, the registry is an
explicit JSON list. Older whole-file-only bindings remain supported:

```json
{
  "schema": "quantem.gpu.compact-browse-sources/v1",
  "sources": [
    {
      "master": "detector/session/sample_master.h5",
      "compact": "prepared/sample-lossless-pack-v1.h5",
      "expected_whole_file_sha256": "<lowercase SHA-256>"
    }
  ]
}
```

Then add `--compact-sources /path/to/compact-sources.json` to the same
`quantem-gpu serve` command. This is trusted deployment configuration, not a
client path. The service rejects missing files, duplicate master bindings,
masters outside the served data folder, missing whole-file seals, shape drift,
and compact requests that ask for crop or bin transformations.

Generated bindings additionally record `chunk_integrity_manifest` and
`expected_chunk_integrity_manifest_sha256`. The service authenticates that
manifest before trusting its complete list of byte-range hashes. It never
automatically trusts an adjacent sidecar. CUDA's LZ4 profile verifies these
ranges concurrently; the direct-bitpacked profile retains its whole-file and
payload verification path.

For an in-process workflow, the same explicit seal can be passed through the
canonical loader:

```python
from quantem.gpu import io

integrity = io.SourceIntegrity.from_file(
    "/data/deployments/sample-v1/source.integrity.json",
    expected_sha256=qualified_manifest_sha256,
)
with io.load("/data/prepared/sample.h5", source_integrity=integrity) as loaded:
    diffraction = loaded.data.extract_diffraction(0, 0)
```

The manifest digest is recorded in the trusted registry. When both
`source_integrity` and `expected_source_sha256` are supplied, their whole-file
identities must agree. Loading does not expose chunk scheduling or kernel
controls. A load with only `expected_source_sha256` remains supported.

The default loopback binding is intentional. Do not expose the HTTP listener
directly on a public interface; use the connection patterns on the next page.

## Deployment verification

A service deployment is ready only when:

1. device detection reports the intended CUDA devices;
2. the capabilities response reports the expected protocol and implementation
   revision;
3. a small source can be discovered, loaded, and reduced;
4. exact output shape, dtype, bin/crop plan, and checksum match the frozen
   in-process CUDA reference; and
5. an over-budget request fails without changing scientific parameters.

For a compact deployment, also require a matching whole-file seal, exact
selected-diffraction and detector-product parity, and a residency telemetry
receipt. A service launch or successful HTTP response alone is not CUDA
runtime qualification.
