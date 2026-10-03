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

## Older release candidate

```{admonition} Pin the documented candidate
:class: important
QuantEM.GPU and this documentation are an evolving pre-release draft. As of
2026-08-19, the release baseline was the exact TestPyPI candidate
`quantem.gpu==0.0.1rc8`. The commands below reproduce that baseline rather
than all current source examples. TestPyPI may list a newer candidate;
candidates are not assumed to be
interchangeable. Keep the equality pin, and advance it only after installation,
compatibility, scientific parity, and performance checks are repeated.
```

Install that exact release candidate from TestPyPI:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc8"
```

For Apple Silicon MPS testing:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps]==0.0.1rc8"
```

For CUDA machines, install the CUDA extra in an environment that already has a
compatible CUDA runtime:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda]==0.0.1rc8"
```

For GIF/MP4 movie rendering, install the movie extra. Combine extras when
movie rendering should use a device-specific backend:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[movie]==0.0.1rc8"

python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps,movie]==0.0.1rc8"
```

For [QuantEM.GPU Remote](remote/index.md) development, combine the service and
CUDA extras:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda,remote]==0.0.1rc8"
```

## Verify the install

```python
import importlib.metadata as md
import quantem.gpu as qgpu

print(md.version("quantem.gpu"))
print(qgpu.__version__)
print(qgpu.device.detect())
```

The distribution version and `qgpu.__version__` should match.

For a reproducible test report, record both printed versions, the Python
executable, platform/device, and the exact command above. Do not describe an
unpinned `--pre` install as equivalent to the documented candidate. Benchmark
rows can name other exact Git revisions because they are frozen historical
evidence rather than statements about the current package pin.
