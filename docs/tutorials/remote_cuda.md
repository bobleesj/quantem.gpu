# Connect a native client to remote CUDA

Remote CUDA keeps raw 4D-STEM data, decoding, and detector products on the GPU
workstation. A native client receives exact scientific arrays and metadata over
an authenticated SSH tunnel; the workstation does not need Live4DSTEM,
`quantem.live`, `quantem.widget`, or a web frontend.

## Install the service environment

The recommended setup is the repository's self-contained Conda environment:

```bash
conda env create -f environment-remote-cuda.yml
```

The environment is named `quantem-gpu-remote`. The equivalent editable
developer installation is:

```bash
python -m pip install -e ".[cuda,remote]"
```

The CUDA runtime must match the workstation driver. Verify the environment
before connecting a client:

```bash
conda run -n quantem-gpu-remote quantem-gpu --help
conda run -n quantem-gpu-remote python -c \
  "import quantem.gpu as qgpu; print(qgpu.__version__, qgpu.device.detect())"
```

## Connect from Live4DSTEM

Select an existing SSH alias and the remote data folder, then choose
**Connect**. The app starts the service inside the configured environment and
creates the SSH tunnel. A normal user does not need to choose a port or GPU.

For manual diagnostics, run:

```bash
quantem-gpu serve /data/4dstem --gpus auto --port 8780
```

The service binds to `127.0.0.1`; SSH owns authentication and encryption. Do
not expose this HTTP service directly on a public interface.

## GPU placement and memory

The service reserves up to 80% of each configured CUDA GPU for exact resident
4D-STEM data. The remainder is left for CUDA contexts, decoder scratch, derived
products, and concurrent allocations.

One dataset remains on one GPU. Multiple GPUs increase the number of datasets
that can be cached concurrently; their memory is not combined to make one
dataset fit. Free memory is checked again before each load. If the requested
source cannot fit on one configured GPU, the service rejects the plan rather
than silently cropping, binning, changing dtype, or splitting the volume.

Catalog discovery is recursive. Dataset identity includes the complete path
relative to the selected data root, so identically named folders in different
projects remain distinct.

## Scientific and transport boundary

The remote service owns source discovery, readiness, exact resident caching,
virtual-detector products, and selected diffraction arrays. The native client
owns presentation and interaction. Raw detector data stays on the CUDA host
unless the requested API explicitly returns it.

Every response must preserve source identity, scan and detector shape, source
and output dtype, scan and detector regions, scan and detector bins, backend,
and device. Detector binning is explicit and count-preserving. A binned result
must never be presented as native resolution.
