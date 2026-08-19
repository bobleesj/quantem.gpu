# Linux CUDA service

The Linux CUDA service keeps raw 4D-STEM data, decoding, and detector products
on a CUDA host. A client may run on that same host through loopback or on
another machine through an authenticated SSH tunnel. In both cases, the
service exposes the same CUDA scientific backend and has no presentation
dependency.

“Remote” describes where a client happens to run; it is not a separate backend
or numerical implementation. Use [CUDA](cuda.md) for in-process kernel and
memory details, and this page for service deployment, transport, admission,
and resident dataset lifecycle.

## Install the service environment

The recommended Linux service setup is the repository's self-contained Conda
environment. The existing filename retains `remote-cuda` for compatibility:

```bash
conda env create -f environment-remote-cuda.yml
```

The environment is named `quantem-gpu-remote`. The equivalent editable
developer installation is:

```bash
python -m pip install -e ".[cuda,remote]"
```

The CUDA runtime must match the host driver. Verify the environment
before connecting a client:

```bash
conda run -n quantem-gpu-remote quantem-gpu --help
conda run -n quantem-gpu-remote python -c \
  "import quantem.gpu as qgpu; print(qgpu.__version__, qgpu.device.detect())"
```

## Start and connect

Run the service on the Linux CUDA host:

```bash
quantem-gpu serve /data/4dstem --gpus auto --port 8780
```

The service binds to `127.0.0.1`. A client on the same host connects directly
through loopback. A client on another machine connects through a local SSH port
forward, with SSH providing authentication and encryption. Do not expose this
HTTP service directly on a public interface. Both access modes use the same
versioned scientific protocol.

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

The Linux CUDA service owns source discovery, readiness, exact resident caching,
virtual-detector products, and selected diffraction arrays. The native client
owns presentation and interaction. Raw detector data stays on the CUDA host
unless the requested API explicitly returns it.

Every response must preserve source identity, scan and detector shape, source
and output dtype, scan and detector regions, scan and detector bins, backend,
and device. Detector binning is explicit and count-preserving. A binned result
must never be presented as native resolution.

Scan and detector coordinates follow `(row, column) ≡ (r, c)`. Admission
telemetry is an estimate that includes resident and peak capacity; the server's
final load response remains authoritative when concurrent occupancy changes.
