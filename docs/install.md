# Install

Install the release candidate from TestPyPI:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc5"
```

For CUDA machines, install the CUDA extra in an environment that already has a
compatible CUDA runtime:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda]==0.0.1rc5"
```

For Apple Silicon MPS testing:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps]==0.0.1rc5"
```

For GIF/MP4 movie rendering, install the movie extra. Combine extras when
movie rendering should use a device-specific backend:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[movie]==0.0.1rc5"

python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[mps,movie]==0.0.1rc5"
```

For widget display testing, install widget and allow it to resolve
`quantem.gpu[movie]>=0.0.1rc5`:

```bash
python -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://test.pypi.org/simple \
  quantem.widget
```

## Verify the install

```python
import importlib.metadata as md
import quantem.gpu as qgpu

print(md.version("quantem.gpu"))
print(qgpu.__version__)
print(qgpu.device_report())
```

The distribution version and `qgpu.__version__` should match.
