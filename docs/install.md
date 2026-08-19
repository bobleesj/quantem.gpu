# Install

Install the release candidate from TestPyPI:

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

For remote CUDA service development, combine the service and CUDA extras:

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
