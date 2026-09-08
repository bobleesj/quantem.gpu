"""Compare serial and bounded asynchronous hashing on one complete original."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

exe, raw, output = map(Path, sys.argv[1:4])
output.mkdir(parents=True, exist_ok=False)
records = []
arms = (("wide-a", "0", "512"), ("bounded", "0", "256"), ("wide-b", "0", "512")) if "--windows" in sys.argv else (
    ("serial-a", "1", "512"), ("pipeline", "0", "512"), ("serial-b", "1", "512"))
for arm, serial, window in arms:
    for repetition in range(3):
        name = f"{arm}-{repetition}"
        target = output / (name + ".bin")
        environment = dict(os.environ, EMPAD_TEST_METAL="1", EMPAD_TEST_BUDGET="6000000000",
                           QGPU_EMPAD_LOAD_PROFILE="1", QGPU_EMPAD_HASH_SERIAL=serial, QGPU_EMPAD_WINDOW=window)
        environment.pop("EMPAD_TEST_HASH_CACHE", None)
        completed = subprocess.run(
            ["/usr/bin/time", "-l", str(exe), str(raw), str(target), "0,32768,65535", "256", "256"],
            env=environment, text=True, capture_output=True, timeout=180,
        )
        (output / (name + ".stdout")).write_text(completed.stdout)
        (output / (name + ".stderr")).write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name} failed: {completed.stderr}")
        match = re.search(r"EMPAD_LOAD (.*)", completed.stderr)
        if match is None:
            raise RuntimeError("Missing full-resident timing")
        fields = {key: float(value) for key, value in re.findall(r"(\w+)=([0-9.]+)", match[1])}
        assert fields["frames_read"] == 65536 and fields["hash_cached"] == 0
        receipt = json.loads(Path(str(target) + ".capabilities.json").read_text())["residentReceipt"]
        record = {"arm": arm, "repetition": repetition, "stages": fields,
                  "receipt": receipt, "rss_bytes": int(re.search(r"(\d+)  maximum resident set size", completed.stderr)[1])}
        records.append(record)
        print(json.dumps(record), flush=True)
        (output / "result.json").write_text(json.dumps(records, indent=2) + "\n")
assert len({record["receipt"]["workingLogicalSHA256"] for record in records}) == 1
