#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/gpu_release_check.sh

Runs the local quantem.gpu release gates:
  - Python compile smoke
  - local wheel build
  - twine check
  - wheel-content check for backend Python modules and Metal shader assets
EOF
}

for arg in "$@"; do
  case "$arg" in
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

echo "== quantem.gpu release check =="
echo "repo: $(pwd)"
echo "branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "== Python compile smoke =="
python -m compileall -q src/quantem/gpu

echo "== local wheel build/content check =="
rm -rf dist/gpu-release-check
python -m build . --wheel --no-isolation --outdir dist/gpu-release-check
python -m twine check dist/gpu-release-check/*
python - <<'PY'
from pathlib import Path
import zipfile

wheels = sorted(Path("dist/gpu-release-check").glob("quantem_gpu-*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"expected one wheel, found {wheels}")
wheel = wheels[0]
required = {
    "quantem/gpu/__init__.py",
    "quantem/gpu/io/hdf5.py",
    "quantem/gpu/io/bitshuffle.py",
    "quantem/gpu/io/backends/mps.py",
    "quantem/gpu/io/backends/metal/bslz4.msl",
    "quantem/gpu/compute/backends.py",
    "quantem/gpu/compute/mps.py",
    "quantem/gpu/compute/metal/reductions.msl",
    "quantem/gpu/detector.py",
    "quantem/gpu/dpc.py",
    "quantem/gpu/ssb/api.py",
}
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
missing = sorted(required - names)
if missing:
    raise SystemExit(f"{wheel} missing required files: {missing}")
license_files = [
    name for name in names
    if name.endswith(".dist-info/licenses/LICENSE")
]
if len(license_files) != 1:
    raise SystemExit(f"{wheel} missing LICENSE in dist-info/licenses")
print(f"wheel ok: {wheel}")
PY

echo "ALL LOCAL GPU RELEASE GATES PASS"
