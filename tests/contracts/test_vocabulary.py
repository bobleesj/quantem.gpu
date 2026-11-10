"""Public names follow the maintainer's naming vocabulary: one name per concept, no retired spelling.

Every public signature (the functions, classes, methods, parameters and dataclass fields that the public
namespaces export), every command-line flag and every key a saved-record writer emits is checked against the
names below. A hit fails unless it is listed in ``EXCEPTIONS`` with a one-line reason.
"""

import ast
import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src" / "quantem" / "gpu"

# The retired spelling of each concept; the comment names the one name used instead.
BANNED = {
    "num_trials", "n_trials",                                   # trials
    "phase_estimator",                                          # method
    "upsampling", "upsampling_factor", "upsample_factor",       # upsample
    "upsampling_mode",                                          # upsample_mode
    "phase_damping", "amplitude_damping",                       # phase_damp, amplitude_damp
    "phase_regularization", "amplitude_regularization",
    "mean_dp", "mean_diffraction_pattern", "dp_mean",           # mean_pattern
    "det_mask",                                                 # mask
    "gpu", "gpus", "gpu_id", "gpu_ids",                         # device, devices
    "cols", "ncols", "max_cols", "compare_cols",                # columns
    "save_to", "output_dir", "out_dir", "output_folder",        # save_dir (a folder), path (a file)
    "ui_mode", "show_static",                                   # show(mode=...)
    "energy_kV", "energy_keV",                                  # voltage_kV
    "alpha_mrad", "convergence_angle_mrad",                     # semiangle_mrad
    "scan_size", "detector_size", "det_size",                   # scan_shape, detector_shape
    "binned", "binning",                                        # detector_bin
    "random_state",                                             # seed
    "remove_hot_pixels",                                        # hot_pixel_correction
    "fit_aberrations", "find_aberrations",                      # aberrations()
    "fit_bright_field_disk", "fit_detector_sampling", "fit_probe",  # bright_field_disk(), detector_sampling()
    "reconstruct",                                              # run()
    "SSB", "SSBResult",                                         # DirectPtycho, DirectPtychoResult
    "Dataset4dstemGPU",                                         # Dataset4dstem
    "dwell_us", "settle_sec",                                   # dwell_s, settle_s
    "figsize",                                                  # size
    "backend",                                                  # device
}
# Counts are plain plurals (trials, frames, columns); a public name never starts with one of these.
BANNED_PREFIXES = ("num_", "n_")

# name -> one-line reason it stays. Keep this list short; a new entry needs the same scrutiny as a rename.
EXCEPTIONS = {
    "num_bright_field_pixels": "in result, calibration session_identity, WebGPU and native records; its own pass",
    "num_frames": "DetectorSession.num_frames: `frames` would read as the frames beside frame(index); needs a word",
    "backend": "DirectPtycho, run_series, screening.prepare, detector.prepare and io.save; folding into device= is open",
}


def _public_modules():
    import quantem.gpu as package

    names = [name for name in package.__all__ if name not in {"DirectPtycho", "DirectPtychoResult", "__version__"}]
    return [package] + [importlib.import_module(f"quantem.gpu.{name}") for name in
                        (*names, "direct_ptycho", "drift", "remote", "streaming")]


def _names_of(obj, owner):
    """(name, where) for a public callable or class: its name, its parameters, its public methods and fields."""
    found = [(obj.__name__, owner)]
    if inspect.isclass(obj):
        if dataclasses.is_dataclass(obj):
            found += [(field.name, f"{owner}.{field.name}") for field in dataclasses.fields(obj)
                      if not field.name.startswith("_")]
        for name, member in vars(obj).items():
            if name.startswith("_") and name != "__init__":
                continue
            if isinstance(member, property):
                found.append((name, f"{owner}.{name}"))
                continue
            function = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
            if inspect.isfunction(function):
                found += [(item, place) for item, place in _signature_names(function, f"{owner}.{name}")
                          if name != "__init__" or item != "__init__"]
        return found
    if callable(obj):
        found += _signature_names(obj, owner)[1:]
    return found


def _signature_names(function, owner):
    found = [(function.__name__, owner)]
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return found
    return found + [(name, f"{owner}({name}=)") for name in parameters if name not in ("self", "cls")
                    and parameters[name].kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]


def _banned(name):
    return name not in EXCEPTIONS and (name in BANNED or name.startswith(BANNED_PREFIXES))


def test_public_signatures_use_the_vocabulary():
    hits = set()
    for module in _public_modules():
        for exported in getattr(module, "__all__", ()):
            obj = getattr(module, exported)
            if inspect.ismodule(obj) or not (callable(obj) or inspect.isclass(obj)):
                continue
            hits.update(f"{where}: {name}" for name, where in _names_of(obj, f"{module.__name__}.{exported}")
                        if _banned(name))
    assert not hits, "\n".join(sorted(hits))


def test_command_line_flags_use_the_vocabulary():
    flags = set()
    for path in SOURCE.rglob("*.py"):
        flags.update(re.findall(r"add_argument\(\s*[\"']--([a-z0-9-]+)[\"']", path.read_text(encoding="utf-8")))
    hits = sorted(flag for flag in flags if _banned(flag.replace("-", "_")) or flag in {"gpu", "gpus", "out", "output-dir"})
    assert not hits, hits


# The writers of saved records, as (module, function): every string key of a dict they build is a record key.
WRITERS = [
    ("direct_ptycho/calibration.py", "save_calibration"),
    ("direct_ptycho/calibration.py", "capture_calibration"),
    ("direct_ptycho/workflow.py", "_result_signature"),
    ("direct_ptycho/persistence.py", "save_result"),
    ("direct_ptycho/stream.py", None),
    ("screening/workflow.py", None),
    ("formats/ssb_phase.py", None),
]


def _record_keys(path, function):
    tree = ast.parse((SOURCE / path).read_text(encoding="utf-8"))
    if function is not None:
        tree = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            yield from (key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str))


@pytest.mark.parametrize(("path", "function"), WRITERS)
def test_saved_record_writers_use_the_vocabulary(path, function):
    hits = sorted({key for key in _record_keys(path, function) if _banned(key)})
    assert not hits, hits


def test_result_record_fields_use_the_vocabulary():
    """A saved result record holds every field of DirectPtychoResult."""
    from quantem.gpu import DirectPtychoResult

    hits = [field.name for field in dataclasses.fields(DirectPtychoResult) if _banned(field.name)]
    assert not hits, hits
