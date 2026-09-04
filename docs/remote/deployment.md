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
`quantem.gpu.packaged-browse-service/v1`. Its compatibility chain must be
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

To keep existing clients on the same catalogued master identity while using an
immutable compact resident artifact, create a trusted server-side registry:

```json
{
  "schema": "quantem.gpu.compact-browse-sources/v1",
  "sources": [
    {
      "master": "detector/session/sample_master.h5",
      "compact": "prepared/sample-exact-qgix-v1.h5",
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
