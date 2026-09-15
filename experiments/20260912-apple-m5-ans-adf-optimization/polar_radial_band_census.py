"""Compare exact 64-pixel polar layouts with narrower radial bands."""

from __future__ import annotations

import json
import numpy as np


SHAPE = (192, 192)
ROOT_LEAVES = 16


def annulus(row_offset: int, column_offset: int) -> np.ndarray:
    """Return the benchmark ADF mask."""
    row, column = np.indices(SHAPE)
    radius2 = (row - 96 - row_offset) ** 2 + (column - 96 - column_offset) ** 2
    return ((radius2 >= 48**2) & (radius2 <= 94**2)).astype(np.int32)


def permutation(layout: str) -> np.ndarray:
    """Return deterministic radial-band, angle, radius, pixel ordering."""
    row, column = np.indices(SHAPE)
    row = row - (SHAPE[0] - 1) / 2
    column = column - (SHAPE[1] - 1) / 2
    radius = np.hypot(row, column)
    angle = np.arctan2(row, column)
    if layout == "morton_z":
        pixel = np.arange(np.prod(SHAPE), dtype=np.int32)
        row_bits = (pixel // SHAPE[1]).astype(np.uint32)
        column_bits = (pixel % SHAPE[1]).astype(np.uint32)
        code = np.zeros(pixel.size, dtype=np.uint32)
        for bit in range(8):
            code |= ((column_bits >> bit) & 1) << (2 * bit)
            code |= ((row_bits >> bit) & 1) << (2 * bit + 1)
        return np.lexsort((pixel, code)).astype(np.int32)
    if layout == "r_pow_1_5_over_45":
        band = np.floor(radius**1.5 / 45)
    elif layout == "radius_1px":
        band = np.floor(radius)
    elif layout == "radius_half_px":
        band = np.floor(radius * 2)
    else:
        raise ValueError(layout)
    pixel = np.arange(np.prod(SHAPE), dtype=np.int32)
    return np.lexsort((pixel, radius.ravel(), angle.ravel(), band.ravel())).astype(np.int32)


def plan(
    delta: np.ndarray, order: np.ndarray, valid: np.ndarray, leaf_pixels: int
) -> tuple[np.ndarray, ...]:
    """Decompose delta exactly with the production majority and tie rules."""
    values = delta.ravel().copy()
    values[~valid.ravel()] = 0
    ordered = values[order]
    ordered_valid = valid.ravel()[order]
    leaves = order.size // leaf_pixels
    leaf = np.zeros(leaves, np.int32)
    for index in range(leaves):
        block = ordered[index * leaf_pixels : (index + 1) * leaf_pixels]
        block_valid = ordered_valid[index * leaf_pixels : (index + 1) * leaf_pixels]
        counts = [np.count_nonzero(block[block_valid] == value) for value in (0, 1, -1)]
        leaf[index] = (0, 1, -1)[int(np.argmax(counts))]
    residual = np.zeros(order.size, np.int32)
    residual[order[ordered_valid]] = (
        ordered - np.repeat(leaf, leaf_pixels)
    )[ordered_valid]
    root = np.zeros(leaves // ROOT_LEAVES, np.int32)
    for index in range(root.size):
        block = leaf[index * ROOT_LEAVES : (index + 1) * ROOT_LEAVES]
        counts = [np.count_nonzero(block == value) for value in (0, 1, -1)]
        root[index] = (0, 1, -1)[int(np.argmax(counts))]
    leaf -= np.repeat(root, ROOT_LEAVES)
    fields = np.concatenate((leaf, root))
    return fields, residual, values


def measure(
    name: str, delta: np.ndarray, layout: str, order: np.ndarray, leaf_pixels: int
) -> dict[str, object]:
    """Measure proxy cost and assert exact reconstruction."""
    valid = np.ones(SHAPE, dtype=bool)
    fields, residual, expected = plan(delta, order, valid, leaf_pixels)
    leaves = order.size // leaf_pixels
    expanded = np.repeat(
        fields[:leaves] + np.repeat(fields[leaves:], ROOT_LEAVES), leaf_pixels
    )
    reconstructed = np.zeros(order.size, np.int32)
    reconstructed[order] = expanded
    reconstructed += residual
    if not np.array_equal(reconstructed, expected):
        raise RuntimeError(f"non-exact reconstruction: {layout} {name}")
    changed = int(np.count_nonzero(expected))
    field_count = int(np.count_nonzero(fields))
    residual_count = int(np.count_nonzero(residual))
    cost = field_count + 4 * residual_count
    return {
        "layout": layout,
        "leaf_pixels": leaf_pixels,
        "transition": name,
        "changed_pixels": changed,
        "selected_fields": field_count,
        "residual_pixels": residual_count,
        "changed_pixels_without_residual": int(np.count_nonzero((expected != 0) & (residual == 0))),
        "planner_cost": cost,
        "direct_cost": 4 * changed,
        "planner_to_direct_cost_ratio": cost / (4 * changed),
        "reconstruction": "exact",
    }


def main() -> None:
    """Print the reproducible CPU-only census as JSON."""
    base, center1, center8, center20 = annulus(0, 0), annulus(0, 1), annulus(5, 8), annulus(12, 20)
    transitions = [
        ("zero-to-base", base),
        ("base-to-center1", center1 - base),
        ("center1-to-center8", center8 - center1),
        ("center8-to-center20", center20 - center8),
        ("base-to-center8", center8 - base),
        ("base-to-center20", center20 - base),
    ]
    layouts = ("r_pow_1_5_over_45", "radius_1px", "radius_half_px", "morton_z")
    # Four pixels is a geometry lower-bound probe; eight is the smallest
    # currently plausible retained-index candidate. Neither changes production.
    leaf_variants = (4, 8, 16, 64)
    orders = {name: permutation(name) for name in layouts}
    invalid_exact = {}
    for leaf_pixels in leaf_variants:
        for name, order in orders.items():
            delta = center20 - base
            valid = np.ones(SHAPE, dtype=bool)
            invalid = int(order[1234])
            valid.ravel()[invalid] = False
            delta.ravel()[invalid] = 1_000_000_000
            fields, residual, expected = plan(delta, order, valid, leaf_pixels)
            leaves = order.size // leaf_pixels
            expanded = np.repeat(
                fields[:leaves] + np.repeat(fields[leaves:], ROOT_LEAVES), leaf_pixels
            )
            reconstructed = np.zeros(order.size, np.int32)
            reconstructed[order] = expanded
            reconstructed[~valid.ravel()] = 0
            reconstructed += residual
            invalid_exact[f"{name}-leaf{leaf_pixels}"] = bool(
                np.array_equal(reconstructed, expected)
            )
    print(json.dumps({
        "schema": "quantem-gpu-polar-radial-band-census/v1",
        "scope": "CPU geometry proxy only; exact integer masks, no Metal timing",
        "geometry": {"shape": list(SHAPE), "leaf_pixel_variants": list(leaf_variants),
                     "root_leaves": 16},
        "cost_model": "one unit per selected field plus four per residual pixel",
        "results": [measure(label, delta, layout, orders[layout], leaf_pixels)
                    for leaf_pixels in leaf_variants for layout in layouts
                    for label, delta in transitions],
        "invalid_value_1000000000_projected_exactly": invalid_exact,
        "command": "PYTHONPATH=src python experiments/20260912-apple-m5-ans-adf-optimization/polar_radial_band_census.py",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
