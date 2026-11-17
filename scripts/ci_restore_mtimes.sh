#!/usr/bin/env bash
# Set each tracked file's modification time to its last commit time. A fresh checkout stamps every
# file with the checkout time, so a restored SwiftPM build cache would treat every source as changed.
# Needs the full history (checkout with fetch-depth 0); merges count as the commit that changed a file.
set -euo pipefail
python3 - <<'PY'
import os
import subprocess

pending = set(subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout.decode().split("\0")) - {""}
log = subprocess.run(["git", "log", "--first-parent", "-m", "--format=@%ct", "--name-only", "-z", "--no-renames"],
                     capture_output=True, check=True).stdout.decode()
stamp = None
for field in log.replace("\n", "\0").split("\0"):
    if field.startswith("@"):
        stamp = int(field[1:])
    elif field in pending:
        os.utime(field, (stamp, stamp))
        pending.discard(field)
        if not pending:
            break
print("restored commit times; files without history:", len(pending))
PY
