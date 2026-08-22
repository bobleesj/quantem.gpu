# Install

```{admonition} Pin the documented candidate
:class: important
QuantEM.GPU and this documentation are an evolving pre-release draft. As of
2026-08-19, the Python examples target the exact TestPyPI candidate
`quantem.gpu==0.0.1rc6`, which matches the version declared by this source
tree. TestPyPI may list a newer candidate; candidates are not assumed to be
interchangeable. Keep the equality pin, and advance it only after installation,
compatibility, scientific parity, and performance checks are repeated.
```

Install that exact release candidate from TestPyPI:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc6"
```

For CUDA machines, install the CUDA extra in an environment that already has a
compatible CUDA runtime:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda]==0.0.1rc6"
```

For Apple Silicon MPS testing:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps]==0.0.1rc6"
```

For GIF/MP4 movie rendering, install the movie extra. Combine extras when
movie rendering should use a device-specific backend:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[movie]==0.0.1rc6"

python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps,movie]==0.0.1rc6"
```

For [QuantEM.GPU Remote](remote/index.md) development, combine the service and
CUDA extras:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda,remote]==0.0.1rc6"
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
