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
    "quantem/gpu/io/load.py",
    "quantem/gpu/io/backends/cuda/decoder.py",
    "quantem/gpu/io/backends/mps/decoder.py",
    "quantem/gpu/io/backends/mps/kernels/bslz4.msl",
    "quantem/gpu/io/qem-rans-tables-v1.json",
    "quantem/gpu/detector/workflow.py",
    "quantem/gpu/detector/backends/webgpu/qem-source.ts",
    "quantem/gpu/dpc/workflow.py",
    "quantem/gpu/ssb/workflow.py",
    "quantem/gpu/webgpu/sources.json",
    "quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/detector.metal",
}
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
development_sources = sorted(
    name for name in names
    if name.startswith(("quantem/gpu/swift/Benchmarks/", "quantem/gpu/swift/Tests/"))
)
if development_sources:
    raise SystemExit(f"{wheel} contains development-only Swift files: {development_sources}")
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
