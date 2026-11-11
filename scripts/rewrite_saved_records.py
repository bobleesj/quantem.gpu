"""Carry saved quantem.gpu records over to the one vocabulary of 10-11 Nov 2026, once, offline.

The library reads only the current layouts, with no reader for the old names. This script rewrites what
earlier versions saved:

Every record ends under its direct-ptychography name (11 Nov 2026), one version past its last ``quantem.ssb-*``
version: ``quantem.ssb-result`` is ``quantem.direct-result/15``, ``-stream-frame`` ``/6``, ``-phase`` ``/3``,
``-calibration`` ``/9``, ``-fit`` ``/8``, ``-parameter-slices`` ``/2``, ``-joint-run`` ``/2`` and ``-saved-run`` ``/3``.
Their keys and file names say ``direct`` where they said ``ssb`` (``ssb_tilt`` is ``direct_tilt``,
``ssb-aberrations.json`` is ``direct-aberrations.json``, ``ssb.json`` with ``ssb.npy`` is ``direct.json`` with
``direct.npy``, a ``thickness_method`` of ``ssb_depth`` is ``direct_depth``), and a folder named ``ssb`` or ``ssb-*``
(``ssb-fit``, ``ssb-results``) is renamed ``direct`` or ``direct-*``. Before that, earlier layouts are carried:

- ``quantem.ssb-result/12`` (``ssb-find_aberrations``, ``ssb-reconstruct`` and ``ssb-direct-*`` in a results
  folder) becomes ``/14``: ``ssb-find_aberrations.{json,npz}`` is renamed ``ssb-aberrations``,
  ``ssb-reconstruct.{json,npz}`` is renamed ``ssb-run``, the signature operations follow, and the keys and
  values below take their new names. The arrays are renamed unchanged.
- ``quantem.ssb-result/13`` becomes ``/14``: the result drops ``phase_estimator`` (``reconstruction_method``
  names the method) and ``num_trials`` (the trial records count the trials), and the signature's settings name
  the search budget ``trials`` (was ``num_trials``) and drop ``phase_estimator``.
- ``quantem.ssb-stream-frame/4`` (``<save_dir>/stream/<frame>.json``) becomes ``/5`` with the same renames; the
  chain of previous-record hashes is computed again so the frames still link.
- ``quantem.phase-refinement/1`` neural phase checkpoints (``.pt``) become ``/2``: ``upsampling`` is ``upsample``.
- A ``quantem.ssb-phase/2`` manifest (``ssb.json``): the screening record it carries in
  ``run_metadata`` moves its fit block from ``computed.ssb`` to ``computed.phase_of_mean`` and becomes
  ``quantem.screen-acquisition/3``.

Renamed keys, at any depth: upsampling, upsampling_factor and ssb_upsampling_factor are upsample; upsampling_mode
is upsample_mode; phase_damping, amplitude_damping and input_phase_damping drop their -ing;
nominal_upsampling_requirement is nominal_upsample_requirement. A ``method`` or ``reconstruction_method`` of
``"ssb"`` is ``"phase_of_mean"``.

- A ``quantem.ssb-calibration/8`` file that ``DirectPtychoResult.save_calibration`` wrote with
  ``reconstruction_method`` beside ``reconstruction_options`` holds them as ShowPtycho's one
  ``reconstruction`` dict, ``{"method": ..., "phase_damp": ..., ...}``.

Earlier ShowPtycho calibrations (/4 to /7) are carried by quantem.widget's ``scripts/rewrite_direct_calibration.py``
and QuantEM Live's screening ``config.json`` by its ``tools/rewrite_screen_config.py``.

Every rewritten file keeps its original beside it as ``<name>.before``. A file rewritten before (its
``.before`` already there, from /12 to /13) keeps that older original and is carried on. A renamed file leaves
its original as ``<old name>.before``; a renamed array (``.npz``, ``.npy``) or folder moves unchanged. Records of
any other schema are not touched, so a second run changes nothing.

Usage::

    python scripts/rewrite_saved_records.py --dry-run results/
    python scripts/rewrite_saved_records.py results/ checkpoints/neural-phase.pt
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

KEYS = {
    "upsampling": "upsample",
    "upsampling_factor": "upsample",
    "ssb_upsampling_factor": "upsample",
    "upsampling_mode": "upsample_mode",
    "phase_damping": "phase_damp",
    "amplitude_damping": "amplitude_damp",
    "input_phase_damping": "input_phase_damp",
    "nominal_upsampling_requirement": "nominal_upsample_requirement",
}
METHOD_KEYS = ("method", "reconstruction_method")
OPERATIONS = {"find_aberrations": "aberrations", "reconstruct": "run"}
RESULT = (("quantem.ssb-result/12", "quantem.ssb-result/13"), "quantem.ssb-result/14")
# result fields that repeat another: reconstruction_method names the method, the trial records count the trials
DROPPED_RESULT_FIELDS = ("phase_estimator", "num_trials")
FRAME = ("quantem.ssb-stream-frame/4", "quantem.ssb-stream-frame/5")
CHECKPOINT = ("quantem.phase-refinement/1", "quantem.phase-refinement/2")
PHASE = "quantem.ssb-phase/2"
SCREEN = ("quantem.screen-acquisition/2", "quantem.screen-acquisition/3")
CALIBRATION = "quantem.ssb-calibration/8"
# 11 Nov 2026: the direct-ptychography name of each record, one version past its last quantem.ssb-* version
DIRECT = {
    "quantem.ssb-result/14": "quantem.direct-result/15",
    "quantem.ssb-stream-frame/5": "quantem.direct-stream-frame/6",
    "quantem.ssb-phase/2": "quantem.direct-phase/3",
    "quantem.ssb-calibration/8": "quantem.direct-calibration/9",
    "quantem.ssb-fit/7": "quantem.direct-fit/8",
    "quantem.ssb-parameter-slices/1": "quantem.direct-parameter-slices/2",
    "quantem.ssb-joint-run/1": "quantem.direct-joint-run/2",
    "quantem.ssb-saved-run/2": "quantem.direct-saved-run/3",
}
OLD_WORD = re.compile(r"(?<![A-Za-z])ssb(?![A-Za-z])")
OLD_FILE = re.compile(r"ssb(?=$|[-_.])")


def direct_name(name):
    """A file or folder name with its leading ssb said direct: ssb-fit is direct-fit, ssb.json is direct.json."""
    return OLD_FILE.sub("direct", name, count=1) if OLD_FILE.match(name) else name


def direct_names(value):
    """``value`` with every key saying direct for ssb, every record id under its DIRECT name and every
    ssb-named file or method value (``ssb.npy``, ``ssb_depth``) said direct."""
    if isinstance(value, dict):
        return {OLD_WORD.sub("direct", key): direct_names(item) for key, item in value.items()}
    if isinstance(value, list):
        return [direct_names(item) for item in value]
    if isinstance(value, str):
        return DIRECT.get(value, direct_name(value))
    return value


def renamed(value):
    """``value`` with every dict key of KEYS renamed and every "ssb" method value named "phase_of_mean"."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            item = renamed(item)
            if key in METHOD_KEYS and item == "ssb":
                item = "phase_of_mean"
            out[KEYS.get(key, key)] = item
        return out
    if isinstance(value, list):
        return [renamed(item) for item in value]
    if isinstance(value, tuple):
        return tuple(renamed(item) for item in value)
    return value


def result_record(record):
    """A /12 or /13 result record as /14."""
    record = renamed(record)
    record["schema"] = RESULT[1]
    for name in DROPPED_RESULT_FIELDS:
        (record.get("result") or {}).pop(name, None)
    signature = record.get("signature") or {}
    if signature.get("schema") in RESULT[0]:
        signature["schema"] = RESULT[1]
    if signature.get("operation") in OPERATIONS:
        signature["operation"] = OPERATIONS[signature["operation"]]
    settings = signature.get("settings") or {}
    settings.pop("phase_estimator", None)
    if "num_trials" in settings:
        settings["trials"] = settings.pop("num_trials")
    return record


def screening_record(config):
    """A quantem.screen-acquisition/2 record as /3: the fit block moves from computed.ssb to computed.phase_of_mean."""
    config = dict(config)
    computed = dict(config.get("computed") or {})
    if "ssb" in computed and "phase_of_mean" not in computed:
        computed["phase_of_mean"] = computed.pop("ssb")
    if "computed" in config:
        config["computed"] = computed
    if config.get("schema") == SCREEN[0]:
        config["schema"] = SCREEN[1]
    return config


def calibration_record(record):
    """A two-key /8 calibration with its method and options as one ``reconstruction`` dict, in place of the two."""
    record = dict(record)
    options = record.pop("reconstruction_options", None) or {}
    method = record.pop("reconstruction_method", None) or options.get("method", "phase_of_mean")
    record["reconstruction"] = {**options, "method": method}
    return record


class Rewriter:
    """Plans and applies each file's rewrite, keeping ``<name>.before``."""

    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.changed = self.skipped = 0

    def say(self, path, note):
        print(f"{'would rewrite' if self.dry_run else 'rewrote'} {path}: {note}")

    def keep(self, path):
        """Copy ``path`` to ``<name>.before``; an earlier rewrite's ``.before`` holds the older original and stays."""
        before = path.with_name(path.name + ".before")
        if not self.dry_run and not before.exists():
            shutil.copy2(path, before)
        return True

    def write(self, path, data):
        if self.dry_run:
            return
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stream.name, path)

    def json(self, path, record, indent, note, target=None):
        """Write ``record`` to ``target`` (default ``path``) after keeping the original; a rename moves the old name to .before."""
        target = path if target is None else target
        if target != path and target.exists():
            print(f"skipped {path}: {target.name} already exists")
            self.skipped += 1
            return False
        if not self.keep(path):
            return False
        self.write(target, (json.dumps(record, indent=indent) + "\n").encode())
        if target != path and not self.dry_run:
            path.unlink()
        self.say(path, note)
        self.changed += 1
        return True

    def result(self, path, record):
        stem = path.stem
        name = direct_name({"ssb-find_aberrations": "ssb-aberrations", "ssb-reconstruct": "ssb-run"}.get(stem, stem))
        target = path.with_name(name + ".json")
        arrays, new_arrays = path.with_suffix(".npz"), target.with_suffix(".npz")
        if target != path and new_arrays.exists():
            print(f"skipped {path}: {new_arrays.name} already exists")
            self.skipped += 1
            return
        new = direct_names(result_record(record) if record["schema"] in RESULT[0] else record)
        note = f"{record['schema']} -> {new['schema']}" + (f", renamed {name}.json" if target != path else "")
        if self.json(path, new, 2, note, target) and target != path and arrays.is_file():
            if not self.dry_run:
                os.replace(arrays, new_arrays)
            self.say(arrays, f"renamed {new_arrays.name}, arrays unchanged")

    def frames(self, folder, records):
        """One stream folder's frames, in order, with the previous-record hash chain computed again."""
        previous = {}
        for path, record in sorted(records, key=lambda pair: pair[1]["identity"]["index"]):
            rewritten = renamed(record)
            rewritten["schema"] = FRAME[1]
            rewritten = direct_names(rewritten)
            index = rewritten["identity"]["index"]
            if index - 1 in previous:
                rewritten["identity"]["previous"] = previous[index - 1]
            previous[index] = hashlib.sha256(json.dumps(rewritten, sort_keys=True, allow_nan=False).encode()).hexdigest()
            self.json(path, rewritten, 1, f"{record['schema']} -> {rewritten['schema']}")

    def manifest(self, path, record):
        """A phase manifest (ssb.json) and its phase array, as direct.json and direct.npy."""
        metadata = record.get("run_metadata")
        if isinstance(metadata, dict) and metadata.get("schema") == SCREEN[0]:
            record = {**record, "run_metadata": screening_record(metadata)}
        self.named(path, record, 2)

    def named(self, path, record, indent):
        """Any other record of an ssb schema under its direct names, with an ssb-named phase array beside it."""
        new = direct_names(record)
        target = path.with_name(direct_name(path.name))
        names = record.get("phase_file"), new.get("phase_file")
        moves = isinstance(names[0], str) and names[0] != names[1] and (path.parent / names[0]).is_file()
        phase, new_phase = (path.parent / names[0], path.parent / names[1]) if moves else (None, None)
        if moves and new_phase.exists():
            print(f"skipped {path}: {new_phase.name} already exists")
            self.skipped += 1
            return
        note = f"{record['schema']} -> {new['schema']}" + (f", renamed {target.name}" if target != path else "")
        if self.json(path, new, indent, note, target) and moves:
            if not self.dry_run:
                os.replace(phase, new_phase)
            self.say(phase, f"renamed {new_phase.name}, array unchanged")

    def folders(self, roots):
        """Folders named ssb or ssb-* (ssb-fit, ssb-results) renamed direct or direct-*, deepest first."""
        found = [root for root in roots if root.is_dir()]
        found += [path for root in found for path in root.rglob("*") if path.is_dir()]
        for folder in sorted(set(found), key=lambda path: len(path.parts), reverse=True):
            target = folder.with_name(direct_name(folder.name))
            if target == folder:
                continue
            if target.exists():
                print(f"skipped {folder}: {target.name} already exists")
                self.skipped += 1
                continue
            if not self.dry_run:
                os.replace(folder, target)
            self.say(folder, f"renamed {target.name}")
            self.changed += 1

    def checkpoint(self, path):
        import torch

        # a relabel of the checkpoint's metadata; its weights are written back unchanged
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT[0]:
            return
        if not self.keep(path):
            return
        tensors = {key: checkpoint[key] for key in ("unet", "readout", "directions")}
        rewritten = renamed({key: value for key, value in checkpoint.items() if key not in tensors})
        rewritten = {**rewritten, **tensors, "schema": CHECKPOINT[1]}
        if not self.dry_run:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                torch.save(rewritten, stream)
            os.replace(stream.name, path)
        self.say(path, f"{CHECKPOINT[0]} -> {CHECKPOINT[1]}")
        self.changed += 1


def files(paths):
    for root in paths:
        root = Path(root)
        candidates = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        yield from (path for path in candidates if path.suffix in (".json", ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="results folders, stream folders, ssb.json manifests or .pt checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="print what would change and write nothing")
    arguments = parser.parse_args()
    rewriter = Rewriter(arguments.dry_run)
    frames = {}
    for path in files(arguments.paths):
        if path.suffix == ".pt":
            rewriter.checkpoint(path)
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        schema = record.get("schema") if isinstance(record, dict) else None
        if schema in RESULT[0] or schema == RESULT[1]:
            rewriter.result(path, record)
        elif schema in FRAME:
            frames.setdefault(path.parent, []).append((path, record))
        elif schema == PHASE:
            rewriter.manifest(path, record)
        elif schema == CALIBRATION and ("reconstruction_method" in record or "reconstruction_options" in record):
            rewriter.named(path, calibration_record(record), 2)
        elif schema in DIRECT:
            rewriter.named(path, record, 2)
    for folder, records in frames.items():
        rewriter.frames(folder, records)
    rewriter.folders([Path(root) for root in arguments.paths])
    print(f"{rewriter.changed} file(s) {'to rewrite' if arguments.dry_run else 'rewritten'}, {rewriter.skipped} skipped.")


if __name__ == "__main__":
    main()
