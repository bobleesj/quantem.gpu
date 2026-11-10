"""Carry saved quantem.gpu records over to the one vocabulary of 10 Nov 2026, once, offline.

The library reads only the current layouts, with no reader for the old names. This script rewrites what
earlier versions saved:

- ``quantem.ssb-result/12`` (``ssb-find_aberrations``, ``ssb-reconstruct`` and ``ssb-direct-*`` in a results
  folder) becomes ``/13``: ``ssb-find_aberrations.{json,npz}`` is renamed ``ssb-aberrations``,
  ``ssb-reconstruct.{json,npz}`` is renamed ``ssb-run``, the signature operations follow, and the keys and
  values below take their new names. The arrays are renamed unchanged.
- ``quantem.ssb-stream-frame/4`` (``<save_dir>/stream/<frame>.json``) becomes ``/5`` with the same renames; the
  chain of previous-record hashes is computed again so the frames still link.
- ``quantem.phase-refinement/1`` neural phase checkpoints (``.pt``) become ``/2``: ``upsampling`` is ``upsample``.
- A ``quantem.ssb-phase/2`` manifest (``ssb.json``) keeps its schema; the screening record it carries in
  ``run_metadata`` moves its fit block from ``computed.ssb`` to ``computed.phase_of_mean`` and becomes
  ``quantem.screen-acquisition/3``.

Renamed keys, at any depth: upsampling, upsampling_factor and ssb_upsampling_factor are upsample; upsampling_mode
is upsample_mode; phase_damping, amplitude_damping and input_phase_damping drop their -ing;
nominal_upsampling_requirement is nominal_upsample_requirement. A ``method`` or ``reconstruction_method`` of
``"ssb"`` is ``"phase_of_mean"``.

ShowPtycho's ``calibration.json`` is carried by quantem.widget's ``scripts/rewrite_ssb_calibration.py`` and
QuantEM Live's screening ``config.json`` by its ``tools/rewrite_screen_config.py``.

Every rewritten file keeps its original beside it as ``<name>.before``; a file whose ``.before`` already exists
is left alone. Records of any other schema are not touched.

Usage::

    python scripts/rewrite_saved_records.py --dry-run results/
    python scripts/rewrite_saved_records.py results/ checkpoints/neural-phase.pt
"""

import argparse
import hashlib
import json
import os
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
RESULT = ("quantem.ssb-result/12", "quantem.ssb-result/13")
FRAME = ("quantem.ssb-stream-frame/4", "quantem.ssb-stream-frame/5")
CHECKPOINT = ("quantem.phase-refinement/1", "quantem.phase-refinement/2")
PHASE = "quantem.ssb-phase/2"
SCREEN = ("quantem.screen-acquisition/2", "quantem.screen-acquisition/3")


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
    """A /12 result record as /13."""
    record = renamed(record)
    record["schema"] = RESULT[1]
    signature = record.get("signature") or {}
    if signature.get("schema") == RESULT[0]:
        signature["schema"] = RESULT[1]
    if signature.get("operation") in OPERATIONS:
        signature["operation"] = OPERATIONS[signature["operation"]]
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


class Rewriter:
    """Plans and applies each file's rewrite, keeping ``<name>.before``."""

    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.changed = self.skipped = 0

    def say(self, path, note):
        print(f"{'would rewrite' if self.dry_run else 'rewrote'} {path}: {note}")

    def keep(self, path):
        """Copy ``path`` to ``<name>.before``; False when an earlier backup is already there."""
        before = path.with_name(path.name + ".before")
        if before.exists():
            print(f"skipped {path}: {before.name} already exists")
            self.skipped += 1
            return False
        if not self.dry_run:
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
        name = {"ssb-find_aberrations": "ssb-aberrations", "ssb-reconstruct": "ssb-run"}.get(stem, stem)
        target = path.with_name(name + ".json")
        arrays, new_arrays = path.with_suffix(".npz"), target.with_suffix(".npz")
        if target != path and new_arrays.exists():
            print(f"skipped {path}: {new_arrays.name} already exists")
            self.skipped += 1
            return
        note = f"{RESULT[0]} -> {RESULT[1]}" + (f", renamed {name}.json" if target != path else "")
        if self.json(path, result_record(record), 2, note, target) and target != path and arrays.is_file():
            if not self.dry_run:
                os.replace(arrays, new_arrays)
            self.say(arrays, f"renamed {new_arrays.name}, arrays unchanged")

    def frames(self, folder, records):
        """One stream folder's frames, in order, with the previous-record hash chain computed again."""
        previous = {}
        for path, record in sorted(records, key=lambda pair: pair[1]["identity"]["index"]):
            rewritten = renamed(record)
            rewritten["schema"] = FRAME[1]
            index = rewritten["identity"]["index"]
            if index - 1 in previous:
                rewritten["identity"]["previous"] = previous[index - 1]
            previous[index] = hashlib.sha256(json.dumps(rewritten, sort_keys=True, allow_nan=False).encode()).hexdigest()
            self.json(path, rewritten, 1, f"{FRAME[0]} -> {FRAME[1]}")

    def manifest(self, path, record):
        metadata = record.get("run_metadata")
        if not isinstance(metadata, dict) or metadata.get("schema") != SCREEN[0]:
            return
        record = {**record, "run_metadata": screening_record(metadata)}
        self.json(path, record, 2, f"run_metadata {SCREEN[0]} -> {SCREEN[1]}, computed.ssb -> computed.phase_of_mean")

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
        if schema == RESULT[0]:
            rewriter.result(path, record)
        elif schema == FRAME[0]:
            frames.setdefault(path.parent, []).append((path, record))
        elif schema == PHASE:
            rewriter.manifest(path, record)
    for folder, records in frames.items():
        rewriter.frames(folder, records)
    print(f"{rewriter.changed} file(s) {'to rewrite' if arguments.dry_run else 'rewritten'}, {rewriter.skipped} skipped.")


if __name__ == "__main__":
    main()
