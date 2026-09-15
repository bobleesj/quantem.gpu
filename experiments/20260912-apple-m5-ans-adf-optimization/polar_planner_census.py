"""Census exact polar-index coverage for translated 192 by 192 ADF masks."""

from __future__ import annotations

import json
import numpy as np

from quantem.gpu._compact.paired import polar_layout


SHAPE = (192, 192)
INNER_RADIUS = 48
OUTER_RADIUS = 94


def _annulus(scan_row_offset: int, scan_col_offset: int) -> np.ndarray:
    """Return the benchmark's binary ADF mask in detector row, col order."""
    detector_row, detector_col = np.indices(SHAPE)
    detector_row = detector_row - SHAPE[0] // 2 - scan_row_offset
    detector_col = detector_col - SHAPE[1] // 2 - scan_col_offset
    radius_squared = detector_row**2 + detector_col**2
    return (
        (radius_squared >= INNER_RADIUS**2)
        & (radius_squared <= OUTER_RADIUS**2)
    ).astype(np.int32)


def _reconstruct_fields(
    selected_fields: np.ndarray, coefficients: np.ndarray, leaf_pixels: int
) -> np.ndarray:
    """Expand selected exact root and leaf coefficients to detector pixels."""
    permutation, _, _ = polar_layout(SHAPE)
    leaves = permutation.size // leaf_pixels
    roots = leaves // 16
    leaf_coefficients = np.zeros(leaves, dtype=np.int32)
    root_coefficients = np.zeros(roots, dtype=np.int32)
    for field, coefficient in zip(selected_fields, coefficients, strict=True):
        if field < leaves:
            leaf_coefficients[field] = coefficient
        else:
            root_coefficients[field - leaves] = coefficient
    leaf_coefficients += np.repeat(root_coefficients, 16)[:leaves]
    expanded = np.repeat(leaf_coefficients, leaf_pixels)
    result = np.zeros(np.prod(SHAPE), dtype=np.int32)
    valid = permutation >= 0
    result[permutation[valid]] = expanded[valid]
    return result.reshape(SHAPE)


def _plan(delta: np.ndarray, leaf_pixels: int, valid: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mirror the Swift majority/tie planner for one exact leaf width."""
    permutation, _, _ = polar_layout(SHAPE)
    leaves = permutation.size // leaf_pixels
    roots = leaves // 16
    flat = delta.ravel().copy()
    flat[~valid.ravel()] = 0
    values = flat[permutation]
    valid_ordered = valid.ravel()[permutation]
    leaf = np.zeros(leaves, dtype=np.int32)
    for index in range(leaves):
        block = values[index * leaf_pixels : (index + 1) * leaf_pixels]
        block_valid = valid_ordered[index * leaf_pixels : (index + 1) * leaf_pixels]
        counts = [int(np.count_nonzero(block[block_valid] == value)) for value in (0, 1, -1)]
        leaf[index] = (0, 1, -1)[int(np.argmax(counts))]
    residual = np.zeros(flat.size, dtype=np.int32)
    residual[permutation[valid_ordered]] = (
        values - np.repeat(leaf, leaf_pixels)
    )[valid_ordered]
    root = np.zeros(roots, dtype=np.int32)
    for index in range(roots):
        block = leaf[index * 16 : (index + 1) * 16]
        counts = [int(np.count_nonzero(block == value)) for value in (0, 1, -1)]
        root[index] = (0, 1, -1)[int(np.argmax(counts))]
    leaf -= np.repeat(root, 16)
    fields = np.concatenate((leaf, root))
    selected_fields = np.flatnonzero(fields).astype(np.uint32)
    selected_pixels = np.flatnonzero(residual).astype(np.uint32)
    return selected_fields, fields[selected_fields], selected_pixels, residual[selected_pixels]


def _measure(
    name: str, delta: np.ndarray, leaf_pixels: int
) -> dict[str, int | float | str]:
    """Measure one exact planner decomposition and verify reconstruction."""
    permutation, _, _ = polar_layout(SHAPE)
    leaves = permutation.size // leaf_pixels
    selected_fields, field_coefficients, selected_pixels, residual_coefficients = _plan(
        delta, leaf_pixels, np.ones(SHAPE, dtype=bool)
    )
    indexed = _reconstruct_fields(selected_fields, field_coefficients, leaf_pixels)
    residual = np.zeros(np.prod(SHAPE), dtype=np.int32)
    residual[selected_pixels] = residual_coefficients
    reconstructed = indexed + residual.reshape(SHAPE)
    if not np.array_equal(reconstructed, delta):
        raise RuntimeError(f"Polar planner reconstruction changed {name}.")

    changed = delta != 0
    residual_mask = residual.reshape(SHAPE) != 0
    changed_pixels = int(changed.sum())
    residual_pixels = int(residual_mask.sum())
    covered_changed = int((changed & ~residual_mask).sum())
    leaf_fields = int((selected_fields < leaves).sum())
    root_fields = int((selected_fields >= leaves).sum())
    direct_cost = 4 * changed_pixels
    planner_cost = int(selected_fields.size + 4 * selected_pixels.size)
    return {
        "transition": name,
        "leaf_pixels": leaf_pixels,
        "changed_pixels": changed_pixels,
        "selected_leaf_fields": leaf_fields,
        "selected_root_fields": root_fields,
        "selected_fields_total": int(selected_fields.size),
        "residual_pixels_total": residual_pixels,
        "residual_pixels_inside_delta": int((changed & residual_mask).sum()),
        "residual_pixels_outside_delta": int((~changed & residual_mask).sum()),
        "changed_pixels_covered_without_residual": covered_changed,
        "changed_coverage_fraction": covered_changed / changed_pixels,
        "planner_cost": planner_cost,
        "direct_residual_cost": direct_cost,
        "planner_to_direct_cost_ratio": planner_cost / direct_cost,
        "estimated_cost_reduction_fraction": 1 - planner_cost / direct_cost,
        "reconstruction": "exact",
    }


def main() -> None:
    """Write the reproducible CPU-only geometry census beside this script."""
    base = _annulus(0, 0)
    center1 = _annulus(0, 1)
    center8 = _annulus(5, 8)
    center20 = _annulus(12, 20)
    transitions = [
        ("zero-to-adf-base", base),
        ("adf-base-to-center-1", center1 - base),
        ("adf-center-1-to-center-8", center8 - center1),
        ("adf-center-8-to-center-20", center20 - center8),
    ]
    permutation, _, _ = polar_layout(SHAPE)
    sentinel_delta = center20 - center8
    sentinel_valid = np.ones(SHAPE, dtype=bool)
    sentinel_pixel = int(permutation[1234])
    sentinel_valid.ravel()[sentinel_pixel] = False
    sentinel_delta.ravel()[sentinel_pixel] = 1_000_000_000
    sentinel_exact = {}
    for leaf_pixels in (16, 32, 64):
        selected_fields, coefficients, selected_pixels, residuals = _plan(
            sentinel_delta, leaf_pixels, sentinel_valid
        )
        reconstructed = _reconstruct_fields(selected_fields, coefficients, leaf_pixels)
        reconstructed.ravel()[selected_pixels] += residuals
        expected = sentinel_delta.copy()
        expected[~sentinel_valid] = 0
        sentinel_exact[str(leaf_pixels)] = bool(np.array_equal(reconstructed, expected))
    result = {
        "schema": "quantem-gpu-polar-planner-census/v1",
        "scope": "CPU geometry only; no count data, GPU timing, or 120 Hz claim",
        "source": {
            "layout": "src/quantem/gpu/_compact/paired.py::polar_layout",
            "planner": "src/quantem/gpu/_compact/paired.py::polar_planner",
            "kernels": "src/quantem/gpu/_compact/kernels/paired.cu::pm_fields/pm_index_sum",
        },
        "geometry": {
            "detector_shape": list(SHAPE),
            "inner_radius": INNER_RADIUS,
            "outer_radius": OUTER_RADIUS,
            "leaf_pixel_variants": [16, 32, 64],
            "root_leaves": 16,
            "fields_by_leaf_pixels": {
                str(width): permutation.size // width + permutation.size // width // 16
                for width in (16, 32, 64)
            },
            "padded_permutation_pixels": int(permutation.size),
        },
        "cost_model": "one selected field plus four units per residual pixel",
        "command": (
            "PYTHONPATH=src python "
            "experiments/20260912-apple-m5-ans-adf-optimization/polar_planner_census.py"
        ),
        "results": [
            _measure(name, delta, leaf_pixels)
            for leaf_pixels in (16, 32, 64)
            for name, delta in transitions
        ],
        "invalid_sentinel": {
            "pixel": sentinel_pixel,
            "value": 1_000_000_000,
            "projected_to_zero_exact_by_leaf_pixels": sentinel_exact,
        },
        "interpretation": (
            "Planner cost is a geometric work proxy from the existing CUDA implementation, "
            "not a Metal timing prediction. Kernel-only and indexed paths remain hypotheses."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
