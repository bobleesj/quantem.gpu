"""Time the stages of opening one acquisition in the viewer, on CUDA or MPS, with digests of what each shows.

Run it on any file ``io.load`` reads, with each tree under test, back to back
on the same GPU, then compare the reports::

    CUDA_VISIBLE_DEVICES=0 python tests/benchmarks/viewer_stages.py run scan_master.h5 --out before.json
    CUDA_VISIBLE_DEVICES=0 python tests/benchmarks/viewer_stages.py run scan_master.h5 --out after.json
    python tests/benchmarks/viewer_stages.py compare before.json after.json

On a Mac the same commands run on the Apple GPU. Each timing is the median of
``--reps`` runs after one warm-up, waiting for the GPU after every stage. The
stages are timed separately, each on a freshly loaded acquisition so no stage
reuses another's cached product:

- ``load``: ``io.load(source)`` into the encoded resident;
- ``mean``: the mean diffraction pattern of the detector session, the product
  the bright-field disk is fitted to;
- ``viewer``: ``Show4DSTEM(acquisition)``, which prepares the mean pattern,
  fits the disk, sums the bright-field image and reads the first pattern;
- ``click``, ``detector``, ``scan_roi``: one interaction each on that viewer:
  moving the cursor to another scan position, switching the detector to the
  annular dark field, and averaging the patterns inside a scan circle.

Digests (SHA-256 prefixes of each stage's output) must agree between two
reports unless a change was meant to alter values. The viewer stages need
quantem.widget. This is not a pytest module; it writes only the report named
by ``--out``.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import quantem.gpu
from quantem.gpu import detector, io
from quantem.widget import Show4DSTEM

BACKEND = "cuda" if torch.cuda.is_available() else "mps"


def timed(function):
    """Result and wall time in ms of ``function``, counting the GPU work it queued."""
    synchronize()
    started = time.perf_counter()
    result = function()
    synchronize()
    return result, 1e3 * (time.perf_counter() - started)


def synchronize():
    """Wait for queued GPU work: a stage that returns early would be billed to the next one."""
    if BACKEND == "cuda":
        torch.cuda.synchronize()
    else:
        torch.mps.synchronize()


def digest(*arrays) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        hasher.update(np.ascontiguousarray(np.asarray(array)).tobytes())
    return hasher.hexdigest()[:16]


def measure(source, options):
    """Stage times in ms and output digests of one pass over ``source``, on two fresh loads."""
    times, digests = {}, {}
    loaded, load_first = timed(lambda: io.load(source, verbose=False, **options))
    session = detector.prepare(loaded)
    mean_dp, times["mean"] = timed(session.mean_dp)
    digests["mean"] = digest(mean_dp)
    session.close()
    loaded.close()
    loaded, load_second = timed(lambda: io.load(source, verbose=False, **options))
    times["load"] = (load_first + load_second) / 2
    viewer, times["viewer"] = timed(lambda: Show4DSTEM(loaded))
    digests["viewer"] = digest(viewer.virtual_image(), [viewer.center_row, viewer.center_col, viewer.bright_field_radius_px])
    row, col = viewer.shape_rows // 3, (2 * viewer.shape_cols) // 3

    def click():
        viewer.pos_row, viewer.pos_col = row, col

    _, times["click"] = timed(click)
    digests["click"] = digest(viewer.pattern(row, col))
    _, times["detector"] = timed(lambda: viewer.apply_preset("adf"))
    digests["detector"] = digest(viewer.virtual_image())
    viewer.vi_roi_mode = "circle"
    pattern, times["scan_roi"] = timed(viewer.vi_roi_pattern)
    digests["scan_roi"] = digest(pattern)
    description = f"{'x'.join(map(str, loaded.shape))} {loaded.dtype}, {type(loaded.data).__name__}"
    viewer.close()
    loaded.close()
    return times, digests, description


def run(arguments):
    options = {}
    if arguments.dataset_path:
        options["dataset_path"] = arguments.dataset_path
    if arguments.scan_shape:
        options["scan_shape"] = tuple(arguments.scan_shape)
    device = torch.cuda.get_device_name(0) if BACKEND == "cuda" else "Apple GPU (MPS)"
    report = {"backend": BACKEND, "device": device, "torch": torch.__version__, "tree": str(Path(quantem.gpu.__file__).parents[3]),
              "reps": arguments.reps, "sources": {}}
    print(f"{BACKEND} on {device}, torch {torch.__version__}, quantem.gpu from {report['tree']}")
    # the shortest path tails that tell the sources apart: two acquisitions may share a file name
    depth = 1
    while len({Path(source).parts[-depth:] for source in arguments.sources}) < len(set(arguments.sources)):
        depth += 1
    for source in arguments.sources:
        label = "/".join(Path(source).parts[-depth:])
        measure(source, options)
        passes = [measure(source, options) for _ in range(arguments.reps)]
        digests = {stage: sorted({digests[stage] for _, digests, _ in passes}) for stage in passes[0][1]}
        stages = {
            stage: {
                "median_ms": statistics.median(times[stage] for times, _, _ in passes),
                "min_ms": min(times[stage] for times, _, _ in passes),
                # one digest when every pass gave the same values
                "digest": "/".join(digests[stage]) if stage in digests else "",
            }
            for stage in ("load", "mean", "viewer", "click", "detector", "scan_roi")
        }
        report["sources"][label] = {"data": passes[0][2], "stages": stages}
        print(f"\n{label}: {passes[0][2]}\n| stage | median ms | min ms | digest |\n|---|---|---|---|")
        for stage, values in stages.items():
            print(f"| {stage} | {values['median_ms']:.1f} | {values['min_ms']:.1f} | {values['digest']} |")
    if arguments.out:
        Path(arguments.out).write_text(json.dumps(report, indent=1) + "\n")


def compare(arguments):
    reports = [json.loads(Path(path).read_text()) for path in arguments.reports]
    print("| source | stage | " + " | ".join(f"{path} ({report['backend']})" for path, report in zip(arguments.reports, reports)) + " | same values |")
    print("|---|---|" + "---|" * (len(reports) + 1))
    for source, entry in reports[0]["sources"].items():
        for stage in entry["stages"]:
            present = [report["sources"][source]["stages"][stage] for report in reports if source in report["sources"]]
            cells = [f"{values['median_ms']:.1f}" for values in present]
            same = len({values["digest"] for values in present}) == 1
            print(f"| {source} | {stage} ms | " + " | ".join(cells) + f" | {'yes' if same else 'NO'} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measured = commands.add_parser("run")
    measured.add_argument("sources", nargs="+", help="files io.load reads: a master, .qem, .raw, .npy or HDF5 stack")
    measured.add_argument("--reps", type=int, default=3)
    measured.add_argument("--dataset-path", help="dataset inside an HDF5 file that holds several")
    measured.add_argument("--scan-shape", type=int, nargs=2, metavar=("ROWS", "COLS"), help="scan shape of a flat frame stack")
    measured.add_argument("--out")
    measured.set_defaults(handler=run)
    contrast = commands.add_parser("compare")
    contrast.add_argument("reports", nargs="+")
    contrast.set_defaults(handler=compare)
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
