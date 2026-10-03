# Reproduce the rc8 installation

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

For [QuantEM.GPU Remote](../remote/index.md) development, combine the service and
CUDA extras:

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu[cuda,remote]==0.0.1rc8"
```


These commands reproduce a historical release. Use the [current installation](../install.md) for the Python workflows.
