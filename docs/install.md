# Install

## Current source

Python 3.11 or newer is required. Record `git rev-parse HEAD` with your results;
the version field alone does not identify a development checkout.

Start with Python, then install the backend for your computer. The same public
API provides loading, virtual detectors, and SSB; supported options differ by
backend. Use the current source for ANS loading and the latest SSB features:

(apple-silicon-mac-mps-metal)=
### Apple Silicon Mac — MPS/Metal

```bash
git clone https://github.com/bobleesj/quantem.gpu.git
cd quantem.gpu
python -m pip install -e ".[mps]"
```

(nvidia-gpu-cuda-linux)=
### NVIDIA GPU — CUDA (Linux)

```bash
git clone https://github.com/bobleesj/quantem.gpu.git
cd quantem.gpu
python -m pip install -e ".[cuda]"
```

The Mac SSB backend uses MLX and Metal. The NumPy-like indexing interface,
such as `data[10, 12]`, returns PyTorch tensors and needs PyTorch on either
platform. Install a PyTorch build with support for your selected GPU.
For the static plotting examples, also install QuantEM:

```bash
python -m pip install torch quantem
```

Check accelerator availability before running the indexing examples:

```python
import torch

print(torch.backends.mps.is_available())  # Apple Silicon
print(torch.cuda.is_available())          # NVIDIA CUDA
```

Add `dm` for DM3/DM4 input (`".[mps,dm]"` or `".[cuda,dm]"`).
Save `git rev-parse HEAD` with your results.
The source version field still reads rc8, but its features have advanced
beyond that published candidate.

## Verify the install

```python
import importlib.metadata as md
import quantem.gpu as qgpu

print(md.version("quantem.gpu"))
print(qgpu.__version__)
print(qgpu.device.detect())
```

The distribution version and `qgpu.__version__` should match.

Record the printed version, device, and `git rev-parse HEAD` with your results.
Continue with [From acquisition to images](python-workflow.md).

For a historical environment, see the [rc8 installation record](maintainer/install-rc8.md).
