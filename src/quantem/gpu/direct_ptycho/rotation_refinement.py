"""Refine an estimated scan-detector rotation inside the SSB fit: a 1-D search over the angle, one polish per angle.

G(q, k) does not depend on the rotation; only the bright-field k vectors do, so a session changes the angle with
``set_rotation`` (the k geometry is retargeted, G is not rebuilt) and each angle costs one polish of the fit
(``find_aberrations(trials=0)``: Newton from the nearest angle's fit, same objective and tilt mode). The search
walks from the estimate in 1-degree steps until the objective falls on both sides (at most 10 degrees away), then
bounded Brent narrows the bracket to 0.05 degrees.
"""

import time
from collections.abc import Callable

from quantem.gpu.optics.physics import wrap_degrees

STEP_DEG = 1.0
LIMIT_DEG = 10.0
TOLERANCE_DEG = 0.05


def fit_score(result) -> float:
    """Higher is better: the model agreement, or minus the phase variance for a phase-variance fit (no agreement)."""
    if result.sample_agreement is not None:
        return float(result.sample_agreement)
    return -float(result.loss)


def refine_rotation(start, polish: Callable[[float, object], object], *, step_deg: float = STEP_DEG,
                    limit_deg: float = LIMIT_DEG, tolerance_deg: float = TOLERANCE_DEG) -> tuple[object, dict]:
    """Return the best polished fit within ``limit_deg`` of ``start``'s rotation, and a record of the search.

    ``polish(rotation_deg, from_fit)`` fits at ``rotation_deg`` starting from ``from_fit`` and returns the result.
    Every angle polishes from the fit at the nearest angle already evaluated, so the search follows one basin of the
    fit as the angle changes. The record holds the estimate and refined angles, the agreement and loss at both, the
    angles evaluated in order and the time.
    """
    from scipy.optimize import minimize_scalar

    started = time.perf_counter()
    origin = float(start.scan_detector_rotation_deg)
    fits = {0.0: start}

    def objective(offset: float) -> float:
        offset = float(offset)
        if offset not in fits:
            nearest = min(fits, key=lambda known: (abs(known - offset), known))
            fits[offset] = polish(origin + offset, fits[nearest])
        return -fit_score(fits[offset])

    centre, up, down = objective(0.0), objective(step_deg), objective(-step_deg)
    if centre <= min(up, down):
        low, high = -step_deg, step_deg
    else:
        direction = step_deg if up < down else -step_deg
        best, value = direction, min(up, down)
        while abs(best + direction) <= limit_deg + 1e-9:
            trial = objective(best + direction)
            if trial >= value:
                break
            best, value = best + direction, trial
        low, high = max(-limit_deg, best - step_deg), min(limit_deg, best + step_deg)
    minimize_scalar(objective, bounds=(low, high), method="bounded", options={"xatol": tolerance_deg})
    offset = max(fits, key=lambda known: fit_score(fits[known]))
    refined = fits[offset]
    record = {
        "estimate_deg": wrap_degrees(origin, 360.0),
        "refined_deg": float(refined.scan_detector_rotation_deg),
        "estimate_agreement": start.sample_agreement,
        "refined_agreement": refined.sample_agreement,
        "estimate_loss": start.loss,
        "refined_loss": refined.loss,
        "evaluated_deg": [wrap_degrees(origin + known, 360.0) for known in fits if known != 0.0],
        "evaluations": len(fits) - 1,
        "seconds": time.perf_counter() - started,
    }
    return refined, record


def refinement_summary(record: dict) -> str:
    """One line: the estimate, the refined angle, the change and the gain of the objective."""
    change = (record["refined_deg"] - record["estimate_deg"] + 180.0) % 360.0 - 180.0
    if record["estimate_agreement"]:
        gain = f"agreement {100.0 * (record['refined_agreement'] / record['estimate_agreement'] - 1.0):+.2f}%"
    else:
        gain = f"phase variance {100.0 * (record['refined_loss'] / record['estimate_loss'] - 1.0):+.2f}%"
    return (f"refined rotation {record['estimate_deg']:.2f}° → {record['refined_deg']:.2f}° ({change:+.2f}°, {gain}; "
            f"{record['evaluations']} polishes, {record['seconds']:.1f} s)")


__all__ = ["LIMIT_DEG", "STEP_DEG", "TOLERANCE_DEG", "fit_score", "refine_rotation", "refinement_summary"]
