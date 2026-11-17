"""Package the browser fitting controller alongside calibrated QEM viewers."""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen


def prepare_folder(folder, *, full_calibration, crop_calibration, source):
    """Write a local browser runtime and calibrated full-field and central-crop geometry.

    Parameters
    ----------
    folder : path-like
        Destination inside the standalone viewer bundle.
    full_calibration, crop_calibration : path-like
        ShowPtycho browser calibration JSON for the acquisition and its central
        128 by 128 scan crop, using the same detector crop and orientation.
    source : dict
        Full detector source from the full acquisition's browser manifest.
    """
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).parent
    runtime = json.loads((assets / "runtime.json").read_text())
    for package in runtime["packages"]:
        path = root / package["file"]
        digest = package["sha256"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        with urlopen(package["url"], timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Browser runtime checksum mismatch for {path.name}.")
        path.write_bytes(content)
    policy = root / "policy"
    policy.mkdir(exist_ok=True)
    (policy / "__init__.py").touch()
    for name in ("_gradient_refinement.py", "higher_order.py", "fifth_order_parameters.py"):
        shutil.copyfile(assets.parent / name, policy / name)
    for name in ("solve.py", "load.mjs", "policy.mjs", "fit-worker.mjs"):
        shutil.copyfile(assets / name, root / name)
    shutil.copyfile(full_calibration, root / "full.json")
    shutil.copyfile(crop_calibration, root / "crop.json")
    (root / "extras.json").write_text(json.dumps(runtime["wheels"]))
    (root / "source.json").write_text(json.dumps(source))
