"""How many residual pixels each candidate index layout leaves for a real drag.

The residual pixels of a query are the ones the index groups cannot cover, and each one
costs a full 512-scan stream decode: 78% of a query's device time. The groups come from
`polar_layout`, which orders pixels by radial band then angle, with bands at
``floor(radius ** exponent / divisor)``. That ordering is stored with the data, so this
counts residuals for candidate orderings in host memory before anything is re-encoded.
"""

import itertools
import json
import math
import sys

import numpy as np

sys.path.insert(0, "/home/owner/worktrees/sep-11-encoded-detector-speed/src")

from quantem.gpu.detector import detector_mask

SHAPE = (192, 192)
LEAF, ROOT = 64, 16


def layout(shape, exponent, divisor, angular):
    """Order pixels by radial band, then angle, then radius; return the permutation."""
    rows, cols = shape
    row, col = np.indices(shape)
    row = row - (rows - 1) / 2
    col = col - (cols - 1) / 2
    radius = np.hypot(row, col)
    angle = np.arctan2(row, col)
    band = np.floor(radius**exponent / divisor)
    # `angular` buckets the angle so a band's leaves are arcs of a bounded angular width.
    sector = np.floor((angle + math.pi) / (2 * math.pi) * angular) if angular else angle
    order = np.lexsort((radius.ravel(), sector.ravel(), band.ravel())).astype(np.int32)
    leaves = math.ceil(rows / 8) * math.ceil(cols / 8)
    permutation = np.full(leaves * LEAF, -1, np.int32)
    permutation[: order.size] = order
    return permutation, leaves


def residuals(permutation, leaves, mask):
    """Residual pixels this layout leaves for one mask, the way the planner counts them."""
    real = permutation >= 0
    pixels = permutation[real]
    values = np.zeros(permutation.size, np.int32)
    values[real] = mask.ravel()[pixels]
    tiles = values.reshape(leaves, LEAF)
    options = np.array([0, 1, -1], np.int32)
    counts = np.stack([(tiles == value).sum(axis=1) for value in options])
    leaf = options[counts.argmax(axis=0)]
    residual = np.zeros(mask.size, np.int32)
    residual[pixels] = (values - np.repeat(leaf, LEAF))[real]
    roots = math.ceil(leaves / ROOT)
    padded = np.zeros(roots * ROOT, np.int32)
    padded[:leaves] = leaf
    counts = np.stack([(padded.reshape(roots, ROOT) == value).sum(axis=1) for value in options])
    root = options[counts.argmax(axis=0)]
    fields = np.concatenate((leaf - np.repeat(root, ROOT)[:leaves], root))
    return int(np.count_nonzero(residual)), int(np.count_nonzero(fields))


def drag(shape, inner, outer, amplitude, steps):
    """A sweep of detector geometries like an operator dragging the annulus."""
    centre = [(value - 1) / 2 for value in shape]
    for step in range(steps):
        yield detector_mask((centre[0] + amplitude * math.sin(step * 0.29),
                             centre[1] + amplitude * 1.1 * math.sin(step * 0.23)),
                            inner, outer, shape, dtype=np.float64)


def main():
    masks = {f"adf {inner}-{outer}, drag {amplitude}px":
             list(drag(SHAPE, inner, outer, amplitude, 12))
             for inner, outer, amplitude in ((40, 80, 12), (40, 80, 2), (20, 90, 12), (55, 95, 12))}
    rows = []
    for exponent, divisor, angular in itertools.product(
            (1.0, 1.25, 1.5, 1.75, 2.0), (15, 30, 45, 70, 110), (0, 64, 128, 256)):
        permutation, leaves = layout(SHAPE, exponent, divisor, angular)
        entry = dict(exponent=exponent, divisor=divisor, angular=angular)
        total = 0
        for name, sequence in masks.items():
            counted = [residuals(permutation, leaves, mask) for mask in sequence]
            entry[name] = dict(residual=int(np.median([c[0] for c in counted])),
                               fields=int(np.median([c[1] for c in counted])))
            total += entry[name]["residual"]
        entry["total_residual"] = total
        rows.append(entry)
    rows.sort(key=lambda row: row["total_residual"])
    stored = next(r for r in rows if (r["exponent"], r["divisor"], r["angular"]) == (1.5, 45, 0))
    print("stored layout (exponent 1.5, divisor 45, no angular bucket):")
    print(json.dumps(stored, indent=1))
    print("\nbest eight:")
    for row in rows[:8]:
        detail = " ".join(f"{k.split(',')[0]}={v['residual']}" for k, v in row.items() if isinstance(v, dict))
        print(f"  exp {row['exponent']:<5} div {row['divisor']:<4} angular {row['angular']:<4} "
              f"total {row['total_residual']:<6} {detail}")
    print(f"\nbest total {rows[0]['total_residual']} against stored {stored['total_residual']}: "
          f"{stored['total_residual'] / max(1, rows[0]['total_residual']):.2f}x fewer residual pixels")


if __name__ == "__main__":
    main()
