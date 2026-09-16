#!/usr/bin/env bash
set -u
# Original inputs remain untouched; discard only our sequential validation copies.
cd "$(dirname "$0")/.."
qem_root=${1:?Provide the local detector testing collection}
qem_executable="$(swift build -c release --show-bin-path)/qem-roundtrip"
qem_directory=$(mktemp -d /tmp/qem-collection-XXXXXX)
qem_output="$qem_directory/roundtrip.qem"
qem_pass=0
qem_fail=0
while IFS= read -r qem_source; do
  echo "SOURCE $qem_source"
  if QGPU_ORIGINAL_READ_AHEAD=0 "$qem_executable" "$qem_source" "$qem_output"; then
    qem_pass=$((qem_pass + 1))
  else
    qem_fail=$((qem_fail + 1))
  fi
  # This exact new file was created by this invocation, never by the user.
  if [ -f "$qem_output" ]; then rm "$qem_output"; fi
done < <(rg --files "$qem_root" | rg '(master\.h5$|/Originals/.*\.dm4$|\.xml$|bg_subtracted/.*\.h5$|/EMPAD/scan_x256_y256\.raw$)' | sort)
rmdir "$qem_directory"
echo "SUMMARY passed=$qem_pass rejected_or_failed=$qem_fail"
test "$qem_fail" -eq 0
