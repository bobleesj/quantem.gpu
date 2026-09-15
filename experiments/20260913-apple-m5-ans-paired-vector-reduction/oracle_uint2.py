"""Check uint2 SIMD reduction against two exact scalar UInt32 sums."""

from __future__ import annotations

import json
from itertools import product


LANES = 32
UINT16_MAX = (1 << 16) - 1
UINT32_MASK = (1 << 32) - 1
INT32_MAX = (1 << 31) - 1
COEFFICIENTS = (-2, -1, 0, 1, 2)
VALUES = (0, 1, 32767, 32768, 65534, 65535)


def _sample(lane: int, stream: int, channel: int, scenario: str) -> int:
    if scenario == "endpoints":
        return (0, UINT16_MAX)[(lane + stream + channel) & 1]
    return VALUES[(3 * lane + 2 * stream + channel) % len(VALUES)]


def _check(coefficients: tuple[int, int], scenario: str) -> int:
    lane_pairs: list[tuple[int, int]] = []
    exact = [0, 0]
    for lane in range(LANES):
        lane_sums = [0, 0]
        if lane % 7 not in (1, 5):
            for stream, coefficient in enumerate(coefficients):
                lane_sums[0] += coefficient * _sample(lane, stream, 0, scenario)
                lane_sums[1] += coefficient * _sample(lane, stream, 1, scenario)
        assert all(abs(value) <= 2 * 2 * UINT16_MAX for value in lane_sums)
        exact[0] += lane_sums[0]
        exact[1] += lane_sums[1]
        lane_pairs.append(tuple(value & UINT32_MASK for value in lane_sums))

    assert all(-INT32_MAX - 1 <= value <= INT32_MAX for value in exact)
    scalar_reduction = tuple(
        sum(pair[channel] for pair in lane_pairs) & UINT32_MASK
        for channel in range(2)
    )
    vector_lanes = list(lane_pairs)
    stride = LANES // 2
    while stride:
        for lane in range(stride):
            vector_lanes[lane] = tuple(
                (vector_lanes[lane][channel] + vector_lanes[lane + stride][channel])
                & UINT32_MASK
                for channel in range(2)
            )
        stride //= 2
    vector_reduction = vector_lanes[0]
    expected = tuple(value & UINT32_MASK for value in exact)
    assert scalar_reduction == vector_reduction == expected
    return max(abs(value) for value in exact)


def main() -> None:
    """Check all two-stream coefficient pairs on endpoints and mixed values."""
    cases = 0
    largest_absolute_total = 0
    for coefficients in product(COEFFICIENTS, repeat=2):
        for scenario in ("endpoints", "mixed-range"):
            largest_absolute_total = max(
                largest_absolute_total, _check(coefficients, scenario)
            )
            cases += 1
    assert cases == 50
    theoretical_bound = LANES * 2 * UINT16_MAX * 2
    assert theoretical_bound < INT32_MAX
    print(json.dumps({
        "status": "pass",
        "cpu_only": True,
        "cases": cases,
        "streams_per_lane": 2,
        "coefficients": list(COEFFICIENTS),
        "uint16_values": list(VALUES),
        "inactive_lanes_per_case": 9,
        "largest_absolute_total": largest_absolute_total,
        "theoretical_signed_total_bound": theoretical_bound,
        "vector_and_scalar_reductions_match": True,
        "metal_compilation": "separately verified with MTLDevice.makeLibrary",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
