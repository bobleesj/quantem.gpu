# Remote CUDA service

Remote CUDA keeps raw 4D-STEM data, decoding, and detector products on the GPU
workstation. A client receives exact scientific arrays and metadata over an
authenticated SSH tunnel. The service has no presentation dependency.

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

## Start and connect

Run the service on the CUDA host:

```bash
quantem-gpu serve /data/4dstem --gpus auto --port 8780
```

The service binds to `127.0.0.1`; SSH owns authentication and encryption. Do
not expose this HTTP service directly on a public interface. A client connects
through a local SSH port forward and uses the versioned scientific protocol.

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

Scan and detector coordinates follow `(row, column) ≡ (y, x)`. Admission
telemetry is an estimate that includes resident and peak capacity; the server's
final load response remains authoritative when concurrent occupancy changes.
