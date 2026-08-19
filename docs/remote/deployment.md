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

## Start the service

```bash
conda run -n quantem-gpu-remote \
  quantem-gpu serve /data/4dstem --gpus auto --port 8780
```

`--gpus auto` makes every visible CUDA device eligible. Use an explicit list
when the host is shared and QuantEM.GPU owns only selected devices. Dataset
placement and fit are explained in [GPU admission and residency](admission.md).

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
