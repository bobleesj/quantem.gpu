"""Aberration units at the direct ptychography boundary: the public API is nm and degrees, the engines work in Angstrom and radians.

Every backend evaluates chi = (pi / lambda[A]) alpha^2 (C10 + C12 cos 2(phi - phi12)) with no conversion, so the
numbers it takes and returns are Angstrom and radians. Before 2026-09-24 they were passed through and reported as "nm"
(known-defocus check: abTEM C10 = -100 A came back as C10 = -100.26 "nm"). The direct ptychography session converts here, so every
aberration a user or a stored record sees carries its unit in its key (``C10_nm``, ``C12_nm``, ``phi12_deg``), while
kernels, backend parity tests and fixtures keep Angstrom and radians.
"""

import math

import numpy as np

from quantem.gpu.direct_ptycho.results import DirectPtychoResult

ENGINE_PER_NM = 10.0
PRIMARY = ("C10_nm", "C12_nm", "phi12_deg")
# Slot of each higher-order coefficient in the engines' 14-slot arrays, Krivanek order after C10 and C12. A round
# aberration (C30, C50) has a magnitude and no azimuth.
HIGHER_ORDER_SLOTS = {"C21": 2, "C23": 3, "C30": 4, "C32": 5, "C34": 6, "C41": 7, "C43": 8, "C45": 9,
                      "C50": 10, "C52": 11, "C54": 12, "C56": 13}
HIGHER_ORDER = tuple(f"{name}_nm" for name in HIGHER_ORDER_SLOTS) + tuple(
    f"phi{name[1:]}_deg" for name in HIGHER_ORDER_SLOTS if name[2] != "0"
)


def sampling_pair(name: str, value) -> tuple[float, float]:
    """A sampling as the (row, col) Python floats the backends read; one number means both axes.

    ``scan_sampling_A=0.264`` and ``scan_sampling_A=(0.264, 0.264)`` describe the same scan, as
    ``data.calibration.scan_sampling_A = 0.264`` does. The value is always stored and reported as the pair, so a
    record never holds a scalar that could be read as a pair that lost an axis. NumPy values (``np.float32``) become
    Python floats, which the backends index safely. Direct ptychography and parallax read their samplings through here, so one rule
    and one message cover both.
    """
    if np.ndim(value) == 0 and value is not None and not isinstance(value, (str, bytes, bool)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = None
        if number is not None:
            return (number, number)
    if np.ndim(value) != 1 or len(value) != 2 or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} is one number for both axes or a (row, col) pair; got {value!r}.")
    return (float(value[0]), float(value[1]))


def validate_aberrations(aberrations: dict[str, float] | None, *, higher_order: bool = False) -> dict[str, float]:
    """Return a complete, owned C10_nm/C12_nm/phi12_deg mapping (nm, nm, degrees); None means no aberrations.

    ``higher_order=True`` also keeps the keys of ``HIGHER_ORDER`` (``C21_nm``, ``phi21_deg``, ...), which only the
    fixed-parameter reconstruction applies. A key outside the accepted set raises: a coefficient that is silently dropped, or read in another
    unit, reconstructs a different image with no error. The message names the accepted keys, and for a sample tilt
    or depth spread key (they are not aberration coefficients) where those are given instead.
    """
    if aberrations is None:
        return dict.fromkeys(PRIMARY, 0.0)
    accepted = PRIMARY + (HIGHER_ORDER if higher_order else ())
    missing = set(PRIMARY) - aberrations.keys()
    extra = aberrations.keys() - set(accepted)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        names = "C10_nm, C12_nm and phi12_deg (nm, nm, degrees)"
        if higher_order:
            names += (", and by name the higher orders " + ", ".join(f"{name}_nm" for name in HIGHER_ORDER_SLOTS)
                      + " (nm) with their phi.._deg (degrees; the round C30 and C50 have none)")
        message = f"Direct ptychography aberrations take {names}; {', '.join(details)}."
        if any("tilt" in str(name) or "depth" in str(name) for name in extra):
            message += (" Sample tilt and depth spread are not aberration coefficients: pass the fit that measured "
                        "them, run(fit), or set ptycho.sample_tilt_mrad = (row, col) and ptycho.depth_spread_nm "
                        "on the session before run(); preview() takes them as keywords.")
        raise ValueError(message)
    return {name: float(aberrations[name]) for name in accepted if name in aberrations}


def override_aberrations(held: dict[str, float], overrides: dict[str, float] | None, *, from_fit: bool) -> dict[str, float]:
    """The coefficients ``run(aberrations=overrides)`` applies: ``held`` with every named coefficient replaced.

    ``held`` is the fit passed to ``run`` (``from_fit``) or the session's ``known_aberrations``, which its latest
    fit, ``run()`` or assignment set. A coefficient the override does not name stays as held, so
    ``run(fit, aberrations={"C10_nm": 12.5})`` changes the defocus of a fit and nothing else. The same rule keeps the
    higher orders of an earlier ``run(aberrations=fifth_order)`` when a later override names only C10_nm, C12_nm and
    phi12_deg: both calls then reconstruct one image. One line names the non-zero higher orders an override leaves
    applied and says how to reconstruct without them.
    """
    merged = validate_aberrations({**held, **(overrides or {})}, higher_order=True)
    if overrides:
        kept = [f"{name}_nm" for name in HIGHER_ORDER_SLOTS
                if f"{name}_nm" not in overrides and merged.get(f"{name}_nm", 0.0) != 0.0]
        if kept:
            terms, stay, each, them = ("term", "stays", "it", "it") if len(kept) == 1 else ("terms", "stay", "each", "them")
            if from_fit:
                holder, without = "of the fit passed to run()", f"Name {each} as 0.0 in aberrations="
            else:
                holder = "the session holds from its latest fit, run() or assignment"
                without = ("Assign the complete mapping to ptycho.known_aberrations and call run() without "
                           f"aberrations=, or name {each} as 0.0,")
            print(f"Direct ptycho: aberrations= names {', '.join(overrides)}; {len(kept)} higher-order {terms} {holder} "
                  f"{stay} applied ({', '.join(kept)}). {without} to reconstruct without {them}.")
    return merged


def aberrations_to_engine(aberrations: dict[str, float]) -> dict[str, float]:
    """nm -> engine Angstrom for C10 / C12, degrees -> radians for phi12; the engines' own key names."""
    return {"C10": float(aberrations["C10_nm"]) * ENGINE_PER_NM, "C12": float(aberrations["C12_nm"]) * ENGINE_PER_NM,
            "phi12": math.radians(aberrations["phi12_deg"])}


def aberrations_from_engine(aberrations: dict[str, float]) -> dict[str, float]:
    """engine Angstrom -> nm for C10 / C12, radians -> degrees for phi12; the public key names."""
    return {"C10_nm": float(aberrations["C10"]) / ENGINE_PER_NM, "C12_nm": float(aberrations["C12"]) / ENGINE_PER_NM,
            "phi12_deg": math.degrees(aberrations["phi12"])}


def higher_order_arrays(aberrations: dict[str, float]) -> tuple[np.ndarray, np.ndarray] | None:
    """The 14-slot float32 magnitudes (nm) and azimuths (radians) of a mapping that names a higher order; else None.

    Slots 0 and 1 repeat C10, C12 and phi12, as the full aberration kernel reads all 14 coefficients from these
    arrays. A mapping with only the three primary keys returns None, so it keeps the two-term kernels.

    The magnitudes stay in nm here because the session still decides from them which kernel runs; ``DirectPtycho.preview``
    multiplies all 14 by ``ENGINE_PER_NM`` once, in float32, before it calls ``preview`` on either backend. CUDA
    and MPS therefore both receive Angstrom (their ``magnitudes_A`` arguments), never these nm values.
    """
    if not any(name in aberrations for name in HIGHER_ORDER):
        return None
    magnitudes = np.zeros(14, np.float32)
    angles = np.zeros(14, np.float32)
    magnitudes[0], magnitudes[1], angles[1] = aberrations["C10_nm"], aberrations["C12_nm"], math.radians(aberrations["phi12_deg"])
    for name, slot in HIGHER_ORDER_SLOTS.items():
        magnitudes[slot] = aberrations.get(f"{name}_nm", 0.0)
        angles[slot] = math.radians(aberrations.get(f"phi{name[1:]}_deg", 0.0))
    return magnitudes, angles


ENGINE_KEYS = {"C10_nm": ("C10_A", ENGINE_PER_NM), "C12_nm": ("C12_A", ENGINE_PER_NM), "phi12_deg": ("phi12_deg", 1.0)}


def search_ranges_to_engine(search_ranges: dict[str, object] | None) -> dict[str, object] | None:
    """Optuna search ranges in engine units: a ``(low, high)`` tuple (a list is read as one) or a fixed value per key.

    ``C10_nm`` and ``C12_nm`` become ``C10_A`` and ``C12_A`` in Angstrom; ``phi12_deg`` stays in degrees. Another key
    raises: a range under a name the engines do not read would be dropped and the parameter searched over its default.
    """
    if search_ranges is None:
        return None
    unknown = set(search_ranges) - set(ENGINE_KEYS)
    if unknown:
        raise ValueError(f"search_ranges keys are C10_nm, C12_nm and phi12_deg; got {sorted(unknown)}.")
    scaled = {}
    for key, value in search_ranges.items():
        name, scale = ENGINE_KEYS[key]
        if isinstance(value, tuple | list):
            scaled[name] = tuple(float(bound) * scale for bound in value)
        else:
            scaled[name] = None if value is None else float(value) * scale
    return scaled


def refine_lock_to_engine(refine_lock: list[str] | None) -> list[str] | None:
    """The engines' names (``C10``, ``C12``, ``phi12``) of the public coefficients a refinement keeps fixed."""
    if refine_lock is None:
        return None
    unknown = set(refine_lock) - set(ENGINE_KEYS)
    if unknown:
        raise ValueError(f"refine_lock names C10_nm, C12_nm and phi12_deg; got {sorted(unknown)}.")
    return [{"C10_nm": "C10", "C12_nm": "C12", "phi12_deg": "phi12"}[name] for name in refine_lock]


def result_from_engine(result: DirectPtychoResult) -> DirectPtychoResult:
    """Convert a result computed by a backend (Angstrom, radians) to the public nm and degrees, in place."""
    result.aberrations = aberrations_from_engine(result.aberrations)
    if result.trial_records:
        converted = []
        public = {"C10_A": "C10_nm", "C12_A": "C12_nm"}
        for trial in result.trial_records:
            params = {public.get(key, key): float(value) / ENGINE_PER_NM if key in public else value
                      for key, value in (trial.get("params") or {}).items()}
            converted.append({**trial, "params": params})
        result.trial_records = converted
    return result
