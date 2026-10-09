# Install

## Current source

Python 3.11 or newer is required. Record `git rev-parse HEAD` with your results;
the version field alone does not identify a development checkout.

Start with Python, then install the backend for your computer. The same public
API provides loading, virtual detectors, and SSB; supported options differ by
backend. Use the current source for ANS loading and the latest SSB features:

```bash
git clone https://github.com/bobleesj/quantem.gpu.git
cd quantem.gpu
```

(cuda-install)=
### NVIDIA GPU (CUDA)

```bash
python -m pip install -e ".[cuda]"
```

`[cuda]` installs CuPy for CUDA 13 with the CUDA libraries it links against, and
PyTorch 2.11 or newer, whose PyPI wheels use the same CUDA 13 libraries. CUDA 13
needs an NVIDIA driver 580 or newer (`nvidia-smi` reports the driver version).
Tested on Linux; CUDA on Windows is untested.

(mps-install)=
### Apple silicon Mac (MPS and Metal)

```bash
python -m pip install -e ".[mps]"
```

`[mps]` installs PyObjC Metal and MLX on an Apple silicon Mac.

(cpu-install)=
### Without a GPU

```bash
python -m pip install -e ".[cpu]"
```

`[cpu]` adds nothing to the base install. Pass `device="cpu"` to load the
dense CPU reference, `io.load(path, device="cpu")`. Tested on Linux and
Windows.

An Intel Mac is not supported by quantem.gpu: it has no Intel Mac backend, and
the `[mps]` extra installs nothing there. Each GPU extra carries a platform
marker, so a wrong pick installs no GPU package. When a GPU is present but its
runtime is missing, quantem.gpu prints one line with the install command for
that GPU.

The Mac SSB backend uses MLX and Metal. The NumPy-like indexing interface,
such as `data[10, 12]`, returns PyTorch tensors, so quantem.gpu installs
PyTorch 2.3 or newer. To use a particular CUDA build of PyTorch, install it
before quantem.gpu. For the static plotting examples, also install QuantEM:

```bash
python -m pip install quantem
```

Check accelerator availability before running the indexing examples:

```python
import torch

print(torch.backends.mps.is_available())  # Apple Silicon
print(torch.cuda.is_available())          # NVIDIA CUDA
```

Add `dm` for DM3/DM4 input (`".[cuda,dm]"` or `".[mps,dm]"`) and `movie`
for GIF and MP4 export (`".[cuda,movie]"` or `".[mps,movie]"`).
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
