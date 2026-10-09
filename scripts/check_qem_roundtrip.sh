#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
qem_executable=$(bash scripts/build_qem_roundtrip.sh)
"$qem_executable" "$@"
