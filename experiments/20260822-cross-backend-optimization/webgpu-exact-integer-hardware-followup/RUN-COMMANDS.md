# Reproduction commands

The preserved candidate worktree was not edited. From
`work/quantem-gpu-public-main`:

```bash
QGPU_WEBGPU_TMP="$(mktemp -d)"
git branch --show-current
git rev-parse HEAD
git diff --full-index | shasum -a 256
shasum -a 256 src/quantem/gpu/detector/compute/webgpu/binning.ts
PYTHONPATH=src pytest -q tests/test_webgpu_sources.py
npx --yes esbuild@0.28.2 \
  src/quantem/gpu/detector/compute/webgpu/binning.ts \
  --bundle --format=esm --platform=browser \
  --outfile="$QGPU_WEBGPU_TMP/binning.js"
```

The final retained hardware run served the copied, hashed evidence bundle from
this directory:

```bash
python3 -m http.server 8877 --bind 127.0.0.1
```

In a second terminal:

```bash
QGPU_WEBGPU_TMP="$(mktemp -d)"
'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --headless=new \
  --no-first-run \
  --no-default-browser-check \
  --disable-gpu-sandbox \
  --enable-unsafe-webgpu \
  --use-angle=metal \
  --remote-debugging-port=9337 \
  --remote-allow-origins=http://127.0.0.1:9337 \
  --user-data-dir="$QGPU_WEBGPU_TMP/chrome-profile-final" \
  http://127.0.0.1:8877/harness.html
```

After the page completed:

```bash
python3 inspect_page.py
```

The browser and server were then stopped with `Control-C`. The accepted
scientific statement is limited to direct synthetic WebGPU kernel parity. The
commands do not control source-page cache, perform HDF5 IO, run a real full
4D-STEM volume, measure process/device peak memory, or exercise Live4DSTEM.

The initial 7.8 ms three-probe run is retained in the parent JSON as a
summary-only observation. Its harness was overwritten before review, so it is
not independently reproducible from a retained source file and is not used as
the strengthened full-output result.
