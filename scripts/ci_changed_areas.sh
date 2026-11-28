#!/usr/bin/env bash
# Write which test areas a change touches to $GITHUB_OUTPUT (native, qem, python, docs = true/false).
# EVENT, BASE and HEAD come from the Tests workflow. Scheduled and manual runs, a change to the
# workflow or to this script, and a push whose base is unknown (a new branch, a force push) mark
# every area.
set -euo pipefail
out=${GITHUB_OUTPUT:-/dev/stdout}
areas=(native qem python docs)

every_area() {
  for area in "${areas[@]}"; do echo "$area=true" >> "$out"; done
  echo "every area: $1"
  exit 0
}

case "${EVENT:-}" in
  pull_request) range="$BASE...$HEAD" ;;
  push) range="$BASE..$HEAD" ;;
  *) every_area "${EVENT:-no event}" ;;
esac
if [[ -z "${BASE:-}" || "$BASE" =~ ^0+$ ]] || ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
  every_area "no base commit"
fi
files=$(git diff --name-only "$range")
printf '%s\n' "$files"
if grep -qE '^(\.github/workflows/test\.yml|scripts/ci_(changed_areas|restore_mtimes)\.sh)$' <<< "$files"; then
  every_area "the test workflow changed"
fi

# The Swift package, its Metal and MSL resources, and the repository files its tests read.
native='^(native/|Package\.(swift|resolved)$|tests/(hardware/)?metal/|tests/(data|parity)/|scripts/check_(metal|paired)_|src/quantem/gpu/(display/colormaps\.json|parity/|formats/qem/)|docs/performance/kernels\.md$)|\.(metal|msl|swift)$'
qem='^(src/quantem/gpu/(formats|io)/|tests/test_qem_|tests/data/|pyproject\.toml$)'
python='^(src/|tests/|scripts/|pyproject\.toml$|package(-lock)?\.json$|tsconfig[^/]*\.json$)'
docs='^(docs/|README\.md$|CONTRIBUTING\.md$|scripts/check_docs_|tests/infrastructure/test_docs_site\.py$)'
for area in "${areas[@]}"; do
  if grep -qE "${!area}" <<< "$files"; then echo "$area=true"; else echo "$area=false"; fi >> "$out"
done
