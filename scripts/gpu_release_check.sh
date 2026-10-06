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
    'quantem/gpu/__init__.py',
    'quantem/gpu/detector/cuda/dense.py',
    'quantem/gpu/resident/mps/virtual_image.py',
    'quantem/gpu/detector/cuda/kernels/virtual_image.cu',
    'quantem/gpu/detector/webgpu/backend.ts',
    'quantem/gpu/detector/webgpu/qem-source.ts',
    'quantem/gpu/detector/workflow.py',
    'quantem/gpu/device/select.py',
    'quantem/gpu/display/__init__.py',
    'quantem/gpu/display/webgpu/colormaps.ts',
    'quantem/gpu/display/webgpu/fft.ts',
    'quantem/gpu/display/webgpu/stats.ts',
    'quantem/gpu/display/colormaps.json',
    'quantem/gpu/display/metal/display.metal',
    'quantem/gpu/dpc/webgpu/kernels.ts',
    'quantem/gpu/dpc/workflow.py',
    'quantem/gpu/io/hdf5/cuda/decode.py',
    'quantem/gpu/io/hdf5/cuda/kernels/bslz4.cu',
    'quantem/gpu/resident/mps/counts.py',
    'quantem/gpu/io/hdf5/mps/decode.py',
    'quantem/gpu/io/hdf5/mps/kernels/bslz4.msl',
    'quantem/gpu/io/hdf5/mps/kernels/qh5idx.metal',
    'quantem/gpu/resident/mps/kernels/runtime_spatial.msl',
    'quantem/gpu/io/hdf5/webgpu/bslz4.ts',
    'quantem/gpu/io/load.py',
    'quantem/gpu/formats/qem/qem-rans-tables-v1.json',
    'quantem/gpu/movie/mps.py',
    'quantem/gpu/ssb/contract.py',
    'quantem/gpu/ssb/cuda/backend.py',
    'quantem/gpu/ssb/mps/backend.py',
    'quantem/gpu/ssb/webgpu/backend.ts',
    'quantem/gpu/ssb/results.py',
    'quantem/gpu/ssb/workflow.py',
    'quantem/gpu/webgpu/sources.json',
}
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
native_sources = sorted(
    name for name in names
    if name.endswith((".swift", ".c", ".h", ".cpp", ".hpp", ".comp"))
)
if native_sources:
    raise SystemExit(f"{wheel} contains native-only files: {native_sources}")
missing = sorted(required - names)
if missing:
    raise SystemExit(f"{wheel} missing required files: {missing}")
license_files = {
    name.rsplit("/", 1)[-1]
    for name in names
    if ".dist-info/licenses/" in name
}
required_license_files = {"LICENSE", "THIRD_PARTY_NOTICES.md"}
missing_license = sorted(required_license_files - license_files)
if missing_license:
    raise SystemExit(
        f"{wheel} missing license files in dist-info/licenses: {missing_license}"
    )
print(f"wheel ok: {wheel}")
PY

echo "ALL LOCAL GPU RELEASE GATES PASS"
