"""Time the MPS SSB and detector hot paths on the public gold scan and compare them with this chip's baseline.

A change must not silently slow these paths. ``run`` measures them on the
public gold acquisition (``bobleesj/quantem-data``, ``4dstem/gold_512``,
512 x 512 scan, 192 x 192 detector) and writes a JSON report; ``compare``
checks a report against the committed baseline for the same chip and exits
non-zero when any metric is slower than the tolerance allows::

    python tests/benchmarks/mps_ssb.py run --out report.json
    python tests/benchmarks/mps_ssb.py compare report.json
    python tests/benchmarks/mps_ssb.py compare report.json --tolerance find_aberrations=0.3
    python tests/benchmarks/mps_ssb.py run --save-baseline    # re-record this chip's baseline
    python tests/benchmarks/mps_ssb.py merge a.json b.json c.json --save-baseline

On a shared GPU, record the baseline from several runs: ``merge`` keeps, for
each metric, the median run (the faster middle one for an even count), and
refuses runs whose fits differ. ``pytest -m benchmark tests/benchmarks`` runs
the same check as a test. It is
deselected by plain ``pytest`` and skips when the Apple GPU, the cached gold
scan or a baseline for this chip is missing.

Every stage waits for the GPU (torch and MLX) before its clock stops, runs
warm-up calls first (except the single-shot stages) and reports the p50 and
p90 of its repeats. The metrics, all in milliseconds:

- ``io_load``: ``io.load(master)`` into the encoded Metal resident;
- ``ssb_session``: ``SSB(data)``, which estimates the scan-detector rotation;
- ``ssb_prepare``: the first preview, which reads and transforms the
  bright-field evidence (single shot);
- ``preview_native`` / ``preview_2x``: ``ssb.preview`` without the loss, as a
  ShowPtycho slider drag, at upsampling 1 and 2, C10 changing every call;
- ``preview_native_loss``: the same with the variance loss (slider release);
- ``search_candidate``: one candidate of the standard aberration search,
  timed as a TPE pair of the exact phase-variance loss and halved;
- ``find_aberrations``: the default fit, single shot. The rotation is
  estimated, so it runs the tilt-corrected joint fit on both 180-degree
  branches and keeps one;
- ``find_aberrations_standard``: one standard (``tilt=False``) 200-trial
  search and Nelder-Mead at the kept rotation, single shot;
- ``detector_update`` / ``detector_full``: Show4DSTEM's exact masked sum on
  the gold scan as loaded (its uint32 counts fit 16 bits once the flagged
  pixels are replaced, so ``io.load`` stores them as the uint16 count
  resident), when the bright-field disk moves by about half a pixel (the
  incremental exact update) and when the mask alternates between the disk
  and an annulus (a full re-sum). The update must equal a full sum exactly;
- ``detector_full_uint32``: the alternating full sum on the scan loaded with
  ``hot_pixel_correction="none"``, whose kept 0xffffffff bad-pixel markers do
  not fit 16 bits: the counts stay in the generic uint32 bit-lane resident and
  every sum decodes the whole scan.

Memory is measured beside the time, in GiB of Metal driver memory above the
encoded acquisition (``torch.mps.driver_allocated_memory``, which counts Torch's
and MLX's buffers, sampled every few milliseconds for the peaks):

- ``session_steady``: after ``SSB(data)``, which on MPS keeps the crop encoded;
- ``prepare_peak`` / ``prepared_steady``: during and after the first preview,
  which builds G(q, k);
- ``loss_preview_steady``: after the loss previews (MLX keeps their row IFFT
  tile in its cache between slider releases);
- ``fit_peak`` / ``fit_steady``: during and after the default fit;
- ``encoded``: the acquisition itself, absolute.

``compare`` fails a memory metric that grew by more than 10 % and 0.25 GiB.

Run it on a quiet GPU. The report lists every other process that used the
GPU in the two seconds before and after the run (from the IOAccelerator
statistics); ``compare`` warns when either side ran beside one, because
another client (Spotlight's media analysis, a screen-sharing encoder,
another benchmark) slows every Metal kernel and looks like a regression. This is not a pytest module; it writes only the report named by
``--out`` (or the baseline with ``--save-baseline``).
"""

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DATASET = "bobleesj/quantem-data"
REVISION = "5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20"
MASTER = "4dstem/gold_512/gold_004_master.h5"
# The gold tutorial's recorded beam: io.load reads the energy; these three are set on data.calibration.
CALIBRATION = {"semiangle_mrad": 30.0, "scan_sampling_A": 0.5, "detector_sampling_mrad": 0.46}
# As SSB.open and the widget load: the four pixels the masters flag are replaced by a local median.
HOT_PIXELS = "median"
BASELINES = Path(__file__).resolve().parent / "baselines"
DEFAULT_TOLERANCE = 0.20
# Single-shot or disk-bound stages vary more between runs than the GPU kernels.
METRIC_TOLERANCES = {"io_load": 0.35, "ssb_session": 0.30, "ssb_prepare": 0.35}
# A regression must also be this many milliseconds slower: sub-millisecond paths jitter by more than 20 %.
MIN_DELTA_MS = 0.5
BUSY_GPU_PERCENT = 10
# A memory metric regresses when it grows by more than this fraction and this many GiB.
MEMORY_TOLERANCE = 0.10
MEMORY_MIN_DELTA_GIB = 0.25


def synchronize():
    """Wait for queued torch and MLX work, so a stage never bills its GPU time to the next one."""
    import torch

    torch.mps.synchronize()
    try:
        import mlx.core as mx

        mx.synchronize()
    except ImportError:
        pass


def driver_gib():
    """This process's Metal driver memory (Torch's and MLX's buffers), GiB, after the queued GPU work."""
    import torch

    synchronize()
    return torch.mps.driver_allocated_memory() / 2**30


class PeakMemory:
    """The highest Metal driver memory while the ``with`` block runs, sampled every 2 ms, GiB."""

    def __enter__(self):
        import threading

        import torch

        self.peak = driver_gib()
        self._stop = threading.Event()

        def sample():
            while not self._stop.is_set():
                self.peak = max(self.peak, torch.mps.driver_allocated_memory() / 2**30)
                time.sleep(0.002)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        synchronize()
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, driver_gib())


def summary(samples_ms):
    """p50, p90 (nearest rank) and minimum of one stage's repeats."""
    ordered = sorted(samples_ms)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
    return {"n": len(ordered), "p50_ms": round(statistics.median(ordered), 3), "p90_ms": round(p90, 3),
            "min_ms": round(ordered[0], 3)}


def timed(function, reps, warmup=2):
    """Summary of ``function``'s wall time over ``reps`` calls after ``warmup`` calls, waiting for the GPU."""
    for index in range(warmup):
        function(-1 - index)
    synchronize()
    samples = []
    for index in range(reps):
        started = time.perf_counter()
        function(index)
        synchronize()
        samples.append(1e3 * (time.perf_counter() - started))
    return summary(samples)


def _gpu_time_by_process():
    """Accumulated Apple GPU time in ns per (process name, pid) from the IOAccelerator user clients."""
    try:
        text = subprocess.run(["ioreg", "-l", "-w0"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    totals = {}
    for block in text.split("+-o "):
        creator = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"', block)
        times = re.findall(r'"accumulatedGPUTime"=(\d+)', block)
        if creator and times:
            key = (creator.group(2), int(creator.group(1)))
            totals[key] = totals.get(key, 0) + sum(int(value) for value in times)
    return totals


def other_gpu_clients(seconds=2.0):
    """GPU share (% of one GPU's time) of every other process over ``seconds``, busiest first, or None.

    Another client (Spotlight's media analysis, a screen-sharing encoder,
    another benchmark) slows every Metal kernel and looks like a regression.
    """
    first = _gpu_time_by_process()
    if first is None:
        return None
    time.sleep(seconds)
    second = _gpu_time_by_process() or {}
    clients = []
    for (name, pid), total in second.items():
        share = 100 * (total - first.get((name, pid), 0)) / (seconds * 1e9)
        if pid != os.getpid() and share >= 1:
            clients.append({"command": name, "gpu_percent": round(share, 1)})
    return sorted(clients, key=lambda entry: -entry["gpu_percent"])


def chip_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False,
                              timeout=5).stdout.strip() or platform.processor()
    except (OSError, subprocess.SubprocessError):
        return platform.processor()


def chip_slug(chip):
    return re.sub(r"[^a-z0-9]+", "-", chip.lower()).strip("-")


def source_commit():
    """quantem.gpu's checkout commit and whether it has local edits, or None outside a git checkout."""
    import quantem.gpu

    root = Path(quantem.gpu.__file__).resolve().parents[3]

    def git(*arguments):
        return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False,
                              timeout=10).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return None, None
    if not commit:
        return None, None
    return commit[:12], bool(git("status", "--porcelain", "--untracked-files=no"))


def environment():
    import mlx.core as mx
    import torch

    import quantem.gpu

    commit, dirty = source_commit()
    device = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return {
        "chip": chip_name(),
        "memory_GB": round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                              text=True, check=False).stdout or 0) / 2**30),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mlx": getattr(mx, "__version__", None),
        "metal_architecture": device.get("architecture"),
        "quantem_gpu": getattr(quantem.gpu, "__version__", None),
        "quantem_gpu_commit": commit,
        "quantem_gpu_local_edits": dirty,
    }


def gold_master(download=False):
    """The cached gold master file, or None when it is not cached and ``download`` is False."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        root = snapshot_download(DATASET, repo_type="dataset", revision=REVISION,
                                 allow_patterns="4dstem/gold_512/*", local_files_only=not download)
    except LocalEntryNotFoundError:
        return None
    master = Path(root) / MASTER
    return master if master.is_file() else None


def calibrate(data):
    for name, value in CALIBRATION.items():
        setattr(data.calibration, name, value)


def disk_masks(detector_shape, center, radius, steps):
    """The bright-field disk dragged by (0.5, 0.25) px per step, and an annulus outside it."""
    rows, cols = np.indices(detector_shape)
    moved = [np.hypot(rows - center[0] - 0.5 * step, cols - center[1] - 0.25 * step) <= radius
             for step in range(steps)]
    distance = np.hypot(rows - center[0], cols - center[1])
    annulus = (distance >= 1.2 * radius) & (distance <= 2.0 * radius)
    return moved, annulus


def alternating(session, first, second):
    """A call that sums ``first`` and ``second`` in turn, so each sum differs from the last in thousands of pixels."""
    masks = [first, second]

    def call(_index):
        masks.reverse()
        session.masked_sum_exact(masks[0])

    return call


def measure(master, reps=20, load_reps=3, log=print):
    """One report of every stage on ``master`` (the gold scan)."""
    from quantem.gpu import SSB, detector, io
    from quantem.gpu.ssb.mps.hardware import effective_phase_loss_chunk_bf
    from quantem.gpu.ssb.mps.loss import reconstruct_prepared_batch_exact_loss

    report = {"schema": "quantem.gpu.mps-benchmark/1", "dataset": {"id": DATASET, "revision": REVISION,
                                                                   "file": MASTER},
              "environment": environment(), "other_gpu_clients_before": other_gpu_clients(),
              "metrics": {}, "values": {}, "memory_GiB": {}}
    metrics = report["metrics"]
    memory = report["memory_GiB"]
    log(f"chip {report['environment']['chip']}, other GPU clients before the run: "
        f"{report['other_gpu_clients_before']}")

    loads = []
    for index in range(load_reps + 1):
        synchronize()
        started = time.perf_counter()
        data = io.load(str(master), hot_pixel_correction=HOT_PIXELS, verbose=False)
        synchronize()
        if index:
            loads.append(1e3 * (time.perf_counter() - started))
        if index < load_reps:
            data.close()
    metrics["io_load"] = summary(loads)
    calibrate(data)
    encoded = driver_gib()
    memory["encoded"] = round(encoded, 2)
    report["values"]["shape"] = list(data.shape)
    log(f"io_load {metrics['io_load']}")

    sessions = []
    for index in range(3):
        synchronize()
        started = time.perf_counter()
        ssb = SSB(data)
        synchronize()
        sessions.append(1e3 * (time.perf_counter() - started))
        if index < 2:
            ssb.close()
    metrics["ssb_session"] = summary(sessions[1:])
    memory["session_steady"] = round(driver_gib() - encoded, 2)
    assert ssb.backend == "mps", f"this benchmark times the MPS backend, not {ssb.backend}"
    report["values"]["scan_detector_rotation_deg"] = round(float(ssb.scan_detector_rotation_deg), 4)
    log(f"ssb_session {metrics['ssb_session']}")

    base = {"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0}

    def drag(factor, loss):
        def call(index):
            ssb.preview({**base, "C10_nm": 2.0 + 0.05 * index}, compute_loss=loss, upsampling_factor=factor)
        return call

    synchronize()
    with PeakMemory() as prepare_memory:
        started = time.perf_counter()
        ssb.preview(base, compute_loss=False)
        synchronize()
        metrics["ssb_prepare"] = summary([1e3 * (time.perf_counter() - started)])
    memory["prepare_peak"] = round(prepare_memory.peak - encoded, 2)
    memory["prepared_steady"] = round(driver_gib() - encoded, 2)
    report["values"]["num_bright_field_pixels"] = int(ssb.num_bright_field_pixels)
    metrics["preview_native"] = timed(drag(1, False), reps, warmup=3)
    metrics["preview_2x"] = timed(drag(2, False), reps, warmup=3)
    metrics["preview_native_loss"] = timed(drag(1, True), max(5, reps // 2), warmup=2)
    memory["loss_preview_steady"] = round(driver_gib() - encoded, 2)
    for name in ("ssb_prepare", "preview_native", "preview_2x", "preview_native_loss"):
        log(f"{name} {metrics[name]}")

    prepared = ssb._backend()._prepared
    chunk_bf = effective_phase_loss_chunk_bf(16, prepared.scan_shape)

    def pair(index):
        offset = 0.1 * index
        reconstruct_prepared_batch_exact_loss(
            prepared, C10=np.asarray([5.0 + offset, -5.0 - offset], np.float32),
            C12=np.asarray([10.0, 20.0], np.float32), phi12=np.asarray([0.3, 1.1], np.float32), chunk_bf=chunk_bf)

    pairs = timed(pair, max(5, reps // 2), warmup=2)
    metrics["search_candidate"] = {**{key: round(value / 2, 3) if key.endswith("_ms") else value
                                      for key, value in pairs.items()}, "timed_as": "half of one TPE pair"}
    log(f"search_candidate {metrics['search_candidate']}")

    def fit_values(fit):
        return {"C10_nm": round(float(fit.aberrations["C10_nm"]), 4), "C12_nm": round(float(fit.aberrations["C12_nm"]), 4),
                "phi12_deg": round(float(fit.aberrations["phi12_deg"]), 4), "loss": round(float(fit.loss), 7),
                "scan_detector_rotation_deg": round(float(fit.scan_detector_rotation_deg), 4)}

    synchronize()
    with PeakMemory() as fit_memory:
        started = time.perf_counter()
        fit = ssb.find_aberrations(verbose=False)
        synchronize()
        metrics["find_aberrations"] = summary([1e3 * (time.perf_counter() - started)])
    memory["fit_peak"] = round(fit_memory.peak - encoded, 2)
    memory["fit_steady"] = round(driver_gib() - encoded, 2)
    report["values"]["fit"] = fit_values(fit)
    log(f"find_aberrations {metrics['find_aberrations']} -> {report['values']['fit']}")
    log(f"memory above the encoded data, GiB: {memory}")
    rotation = float(fit.scan_detector_rotation_deg)
    ssb.close()

    # One standard (tilt=False) 200-trial search: a given rotation is fitted once, on one branch.
    ssb = SSB(data, scan_detector_rotation_deg=rotation)
    ssb.preview(base, compute_loss=False)
    synchronize()
    started = time.perf_counter()
    fit = ssb.find_aberrations(tilt=False, verbose=False)
    synchronize()
    metrics["find_aberrations_standard"] = summary([1e3 * (time.perf_counter() - started)])
    report["values"]["fit_standard"] = fit_values(fit)
    log(f"find_aberrations_standard {metrics['find_aberrations_standard']} -> {report['values']['fit_standard']}")
    ssb.close()

    # Show4DSTEM on the gold scan as loaded: the uint16 count resident and its incremental exact sum.
    session = detector.prepare(data)
    center, radius = detector.fit_probe(session.mean_dp())
    moved, annulus = disk_masks(session.detector_shape, center, radius, steps=reps + 4)
    images = {}

    def update(index):
        images["update"] = session.masked_sum_exact(moved[index + 2])

    session.masked_sum_exact(moved[0])
    metrics["detector_update"] = timed(update, reps, warmup=2)
    metrics["detector_full"] = timed(alternating(session, annulus, moved[0]), reps, warmup=2)
    session.close()
    session = detector.prepare(data)
    exact = bool(np.array_equal(images["update"], session.masked_sum_exact(moved[reps + 1])))
    session.close()
    report["values"]["detector_update_equals_full_sum"] = exact
    report["values"]["bright_field_disk_px"] = [round(float(center[0]), 2), round(float(center[1]), 2),
                                                round(float(radius), 2)]
    report["values"]["loaded_dtype"] = str(data.dtype)
    log(f"detector_update {metrics['detector_update']} (equals the full sum: {exact})")
    log(f"detector_full {metrics['detector_full']}")
    data.close()

    # The same scan with its bad-pixel markers kept stays uint32: every masked sum decodes the whole scan.
    raw = io.load(str(master), hot_pixel_correction="none", verbose=False)
    session = detector.prepare(raw)
    metrics["detector_full_uint32"] = timed(alternating(session, annulus, moved[0]), 3, warmup=1)
    report["values"]["uint32_dtype"] = str(raw.dtype)
    session.close()
    raw.close()
    log(f"detector_full_uint32 {metrics['detector_full_uint32']}")
    report["other_gpu_clients_after"] = other_gpu_clients()
    return report


def tolerance_for(name, overrides):
    return overrides.get(name, METRIC_TOLERANCES.get(name, DEFAULT_TOLERANCE))


def regressions(report, baseline, overrides=None, min_delta_ms=MIN_DELTA_MS):
    """Rows of (metric, baseline ms, current ms, change, tolerance, verdict) and the list of regressed metrics."""
    overrides = overrides or {}
    rows, failed = [], []
    for name, reference in baseline["metrics"].items():
        current = report["metrics"].get(name)
        if current is None:
            rows.append((name, reference["p50_ms"], None, None, None, "MISSING"))
            failed.append(name)
            continue
        old, new = reference["p50_ms"], current["p50_ms"]
        change = new / old - 1.0
        tolerance = tolerance_for(name, overrides)
        if change > tolerance and new - old > min_delta_ms:
            verdict = "SLOWER"
            failed.append(name)
        elif change < -tolerance and old - new > min_delta_ms:
            verdict = "faster (re-record the baseline?)"
        else:
            verdict = "ok"
        rows.append((name, old, new, change, tolerance, verdict))
    return rows, failed


def memory_regressions(report, baseline, tolerance=MEMORY_TOLERANCE, min_delta_gib=MEMORY_MIN_DELTA_GIB):
    """Rows of (metric, baseline GiB, current GiB, change, verdict) and the list of grown metrics.

    A baseline without memory metrics (recorded before they existed) checks none.
    """
    rows, failed = [], []
    current = report.get("memory_GiB", {})
    for name, old in baseline.get("memory_GiB", {}).items():
        new = current.get(name)
        if new is None:
            rows.append((name, old, None, None, "MISSING"))
            failed.append(f"memory {name}")
            continue
        change = new / old - 1.0 if old else math.inf if new else 0.0
        if change > tolerance and new - old > min_delta_gib:
            verdict = "LARGER"
            failed.append(f"memory {name}")
        elif change < -tolerance and old - new > min_delta_gib:
            verdict = "smaller (re-record the baseline?)"
        else:
            verdict = "ok"
        rows.append((name, old, new, change, verdict))
    return rows, failed


def baseline_path(chip):
    return BASELINES / f"{chip_slug(chip)}.json"


def compare(arguments):
    report = json.loads(Path(arguments.report).read_text())
    chip = report["environment"]["chip"]
    path = Path(arguments.baseline) if arguments.baseline else baseline_path(chip)
    if not path.is_file():
        print(f"No baseline for {chip} ({path.name}). Record one on a quiet GPU with `run --save-baseline`.")
        return 2
    baseline = json.loads(path.read_text())
    if baseline["environment"]["chip"] != chip:
        print(f"warning: baseline chip {baseline['environment']['chip']} differs from {chip}")
    overrides = {}
    for item in arguments.tolerance or []:
        name, _, value = item.partition("=")
        overrides[name] = float(value)
    rows, failed = regressions(report, baseline, overrides, arguments.min_delta_ms)
    print("| metric (p50) | baseline ms | current ms | change | tolerance | verdict |")
    print("|---|---:|---:|---:|---:|---|")
    for name, old, new, change, tolerance, verdict in rows:
        cells = ["-" if new is None else f"{new:.2f}", "-" if change is None else f"{100 * change:+.1f} %",
                 "-" if tolerance is None else f"{100 * tolerance:.0f} %"]
        print(f"| {name} | {old:.2f} | " + " | ".join(cells) + f" | {verdict} |")
    memory_rows, memory_failed = memory_regressions(report, baseline)
    if memory_rows:
        print()
        print("| memory (GiB) | baseline | current | change | tolerance | verdict |")
        print("|---|---:|---:|---:|---:|---|")
        for name, old, new, change, verdict in memory_rows:
            cells = ["-" if new is None else f"{new:.2f}", "-" if change is None else f"{100 * change:+.1f} %"]
            print(f"| {name} | {old:.2f} | " + " | ".join(cells) + f" | {100 * MEMORY_TOLERANCE:.0f} % | {verdict} |")
    else:
        print("note: the baseline has no memory metrics; record them with `run --save-baseline` or `merge`.")
    failed += memory_failed
    for side, data in (("baseline", baseline), ("current", report)):
        busiest = {}
        for entry in (data.get("other_gpu_clients_before") or []) + (data.get("other_gpu_clients_after") or []):
            busiest[entry["command"]] = max(busiest.get(entry["command"], 0), entry["gpu_percent"])
        busy = [f"{command} up to {share:.0f} %" for command, share in busiest.items() if share >= BUSY_GPU_PERCENT]
        if busy:
            print(f"warning: other processes used the GPU around the {side} run ({', '.join(busy)}); "
                  "its times may include their work.")
    old_values, new_values = baseline.get("values", {}), report.get("values", {})
    if old_values.get("fit") != new_values.get("fit"):
        print(f"note: the fit changed ({old_values.get('fit')} -> {new_values.get('fit')}); "
              "find_aberrations may not be timing the same trajectory.")
    if not new_values.get("detector_update_equals_full_sum", True):
        print("FAIL: the incremental detector sum no longer equals the full sum.")
        failed.append("detector_update_equals_full_sum")
    if failed:
        print(f"REGRESSED: {', '.join(failed)}")
        return 1
    print("No metric regressed beyond its tolerance.")
    return 0


def run(arguments):
    master = Path(arguments.master) if arguments.master else gold_master(download=arguments.download)
    if master is None:
        print(f"The gold scan is not cached; pass --download to fetch {MASTER} (about 4.6 GB) from {DATASET}.")
        return 2
    report = measure(master, reps=arguments.reps, load_reps=arguments.load_reps)
    rendered = json.dumps(report, indent=1) + "\n"
    targets = []
    if arguments.out:
        targets.append(Path(arguments.out))
    if arguments.save_baseline:
        targets.append(baseline_path(report["environment"]["chip"]))
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered)
        print(f"wrote {target.name}")
    if not targets:
        print(rendered, end="")
    return 0


def merge(arguments):
    """One report whose every metric is the median run's (the faster middle one for an even count)."""
    reports = [json.loads(Path(path).read_text()) for path in arguments.reports]
    chips = {report["environment"]["chip"] for report in reports}
    if len(chips) != 1:
        print(f"The reports come from different chips: {sorted(chips)}")
        return 2
    fits = {json.dumps(report["values"], sort_keys=True) for report in reports}
    if len(fits) != 1:
        print("The reports' values differ (fit, bright-field pixels or detector disk): they did not time the same work.")
        return 2
    merged = dict(reports[0])
    merged["metrics"] = {}
    for name in reports[0]["metrics"]:
        runs = sorted((report["metrics"][name] for report in reports if name in report["metrics"]),
                      key=lambda summary: summary["p50_ms"])
        merged["metrics"][name] = runs[(len(runs) - 1) // 2]
    memory_names = [name for name in reports[0].get("memory_GiB", {})]
    if memory_names:
        merged["memory_GiB"] = {name: statistics.median_low([report["memory_GiB"][name] for report in reports
                                                             if name in report.get("memory_GiB", {})])
                                for name in memory_names}
    merged["merged_runs"] = len(reports)
    merged["other_gpu_clients_before"] = [client for report in reports
                                          for client in report.get("other_gpu_clients_before") or []]
    merged["other_gpu_clients_after"] = [client for report in reports
                                         for client in report.get("other_gpu_clients_after") or []]
    rendered = json.dumps(merged, indent=1) + "\n"
    target = baseline_path(chips.pop()) if arguments.save_baseline else Path(arguments.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)
    print(f"wrote {target.name} from {len(reports)} runs")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measure_parser = commands.add_parser("run", help="measure every stage and write a JSON report")
    measure_parser.add_argument("--master", help="an io.load input other than the cached gold scan")
    measure_parser.add_argument("--download", action="store_true", help="fetch the gold scan if it is not cached")
    measure_parser.add_argument("--reps", type=int, default=20, help="repeats of the fast stages")
    measure_parser.add_argument("--load-reps", type=int, default=3, help="repeats of io.load")
    measure_parser.add_argument("--out", help="report path")
    measure_parser.add_argument("--save-baseline", action="store_true",
                                help="also write tests/benchmarks/baselines/<chip>.json")
    measure_parser.set_defaults(handler=run)
    compare_parser = commands.add_parser("compare", help="exit 1 when a metric regressed against the baseline")
    compare_parser.add_argument("report")
    compare_parser.add_argument("--baseline", help="baseline JSON (default: this chip's committed baseline)")
    compare_parser.add_argument("--tolerance", action="append", metavar="METRIC=FRACTION",
                                help=f"per-metric slowdown allowed (default {DEFAULT_TOLERANCE:.2f}; "
                                     f"{', '.join(f'{k} {v:.2f}' for k, v in METRIC_TOLERANCES.items())})")
    compare_parser.add_argument("--min-delta-ms", type=float, default=MIN_DELTA_MS,
                                help="a regression must also be this many ms slower")
    compare_parser.set_defaults(handler=compare)
    merge_parser = commands.add_parser("merge", help="combine several runs into one report, metric by metric")
    merge_parser.add_argument("reports", nargs="+")
    target = merge_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", help="merged report path")
    target.add_argument("--save-baseline", action="store_true", help="write this chip's baseline")
    merge_parser.set_defaults(handler=merge)
    arguments = parser.parse_args()
    sys.exit(arguments.handler(arguments))


if __name__ == "__main__":
    main()
