"""Aberration units at the SSB boundary: the public API is nm, the engines work in Angstrom.

Every backend evaluates chi = (pi / lambda[A]) alpha^2 (C10 + C12 cos 2(phi - phi12)) with no conversion, so the
numbers it takes and returns are Angstrom. Before 2026-09-24 they were passed through and reported as "nm"
(known-defocus check: abTEM C10 = -100 A came back as C10 = -100.26 "nm"). The SSB session converts here, so C10, C12
and the depth spread are true nm everywhere users see them, while kernels, backend parity tests and fixtures keep
Angstrom. Saved records must declare nm explicitly; older records require a fresh fit.
"""

from quantem.gpu.ssb.results import SSBResult

ENGINE_PER_NM = 10.0
ABERRATION_UNIT = "nm"


def validate_aberrations(aberrations: dict[str, float] | None) -> dict[str, float]:
    """Return a complete, owned C10/C12/phi12 mapping (nm, nm, radians); None means no aberrations."""
    if aberrations is None:
        return {"C10": 0.0, "C12": 0.0, "phi12": 0.0}
    required = {"C10", "C12", "phi12"}
    missing = required - aberrations.keys()
    extra = aberrations.keys() - required
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        raise ValueError(
            "SSB aberrations must contain exactly C10, C12, and phi12 "
            f"(nm, nm, radians); {', '.join(details)}."
        )
    return {name: float(aberrations[name]) for name in ("C10", "C12", "phi12")}


def aberrations_to_engine(aberrations: dict[str, float]) -> dict[str, float]:
    """nm -> engine Angstrom for C10 / C12; phi12 (rad) unchanged."""
    return {**aberrations, "C10": float(aberrations["C10"]) * ENGINE_PER_NM, "C12": float(aberrations["C12"]) * ENGINE_PER_NM}


def aberrations_from_engine(aberrations: dict[str, float]) -> dict[str, float]:
    """engine Angstrom -> nm for C10 / C12; phi12 (rad) unchanged."""
    return {**aberrations, "C10": float(aberrations["C10"]) / ENGINE_PER_NM, "C12": float(aberrations["C12"]) / ENGINE_PER_NM}


def search_ranges_to_engine(search_ranges: dict[str, object] | None) -> dict[str, object] | None:
    """Scale the nm keys of an Optuna search-range dict (``C10_nm``, ``C12_nm``: range tuple or fixed value) into Angstrom."""
    if search_ranges is None:
        return None
    scaled = dict(search_ranges)
    for key in ("C10_nm", "C12_nm"):
        if key in scaled and scaled[key] is not None:
            value = scaled[key]
            scaled[key] = tuple(float(bound) * ENGINE_PER_NM for bound in value) if isinstance(value, (tuple, list)) else float(value) * ENGINE_PER_NM
    return scaled


def result_from_engine(result: SSBResult) -> SSBResult:
    """Convert a result computed by a backend (Angstrom) to the public nm units, in place."""
    result.aberrations = aberrations_from_engine(result.aberrations)
    if result.trial_records:
        converted = []
        for trial in result.trial_records:
            params = dict(trial.get("params") or {})
            for key in ("C10_nm", "C12_nm"):
                if key in params:
                    params[key] = float(params[key]) / ENGINE_PER_NM
            converted.append({**trial, "params": params})
        result.trial_records = converted
    return result
