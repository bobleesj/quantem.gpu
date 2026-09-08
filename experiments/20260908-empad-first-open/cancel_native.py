"""Replace an actively loading original source and reject stale publication."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

app, small, large, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(app / "Tests/NativeUI"))
from drive_folder import require_exact_resident
from profile_seven_tilts import connect_navigation, post, wait_state

out.mkdir(parents=True, exist_ok=False)
gui = out / "gui.log"
result = {"cycles": [], "failures": []}
with (out / "console.log").open("w") as console:
    process = subprocess.Popen(
        [str(app / ".build/release/Live4DSTEM"), "--ui-test-hooks", "--ui-test-stdin",
         "--gui-open", str(small), "--gui-benchmark-log", str(gui)],
        stdin=subprocess.PIPE, stdout=console, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "LIVE4DSTEM_ENABLE_EXPERIMENTAL_EMPAD": "1", "LIVE4DSTEM_EMPAD_HASH_CACHE": "0"},
    )
    connect_navigation(process)
    try:
        initial, _ = wait_state(gui, lambda s: s.get("cache_state") == "resident" and not s.get("loading"), 60, "small original")
        require_exact_resident(initial)
        for cycle in range(3):
            post("open", str(large))
            loading, _ = wait_state(gui, lambda s: s.get("loading"), 20, "large source loading")
            post("open", str(small))
            settled, _ = wait_state(gui, lambda s: s.get("dataset_id") == initial["dataset_id"] and
                s.get("cache_state") == "resident" and not s.get("loading"), 60, "newest source visible")
            require_exact_resident(settled)
            time.sleep(1.6)
            final, _ = wait_state(gui, lambda s: True, 5, "no stale large publication")
            assert final["dataset_id"] == initial["dataset_id"] and not final["loading"]
            assert final["selected_diffraction_hash"] == initial["selected_diffraction_hash"]
            result["cycles"].append({"cycle": cycle, "loading_observed": loading["loading"],
                                      "final_generation": final["resident_generation"], "allocated_bytes": final["allocated_bytes"]})
    except Exception as error:
        result["failures"].append(repr(error))
    finally:
        post("quit")
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            result["failures"].append("App did not quit")
        result["exit"] = process.returncode
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
raise SystemExit(bool(result["failures"]) or result["exit"] != 0)
