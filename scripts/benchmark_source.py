"""Verify that a source-checkout benchmark imports the source it fingerprints."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_source() -> dict:
    """Reject stale package imports and record the loaded module file hashes."""
    import quantem.gpu  # noqa: F401

    modules = {}
    for name, module in sorted(sys.modules.copy().items()):
        if name != "quantem.gpu" and not name.startswith("quantem.gpu."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(ROOT / "src"):
            raise RuntimeError(
                f"Benchmark imported {name} from a different checkout: {path}. "
                "Set PYTHONPATH to this checkout's src directory and restart the process."
            )
        modules[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"matched_checkout": True, "modules": modules}
