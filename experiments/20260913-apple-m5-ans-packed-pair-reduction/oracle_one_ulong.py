"""Check the bias-packed ulong reduction against exact signed sums.

Run with ``python3 experiments/20260913-apple-m5-ans-packed-pair-reduction/oracle_one_ulong.py``.
This CPU-only preflight does not import Metal code or launch a GPU workload.
"""

from __future__ import annotations

import json
from itertools import product

LANE_COUNT = 32
UINT16_MAX = (1 << 16) - 1
UINT32_LIMIT = 1 << 32
UINT32_MASK = UINT32_LIMIT - 1
UINT64_LIMIT = 1 << 64
COEFFICIENTS = (-2, -1, 0, 1, 2)
VALUE_DOMAIN = (0, 1, 32767, 32768, 65534, 65535)


def _value(scenario: str, lane: int, stream: int, channel: int) -> int:
    """Return a deterministic endpoint or mixed-range uint16 sample."""
    if scenario == "endpoints":
        return (0, UINT16_MAX)[(lane + stream + channel) & 1]
    return VALUE_DOMAIN[(3 * lane + 2 * stream + channel) % len(VALUE_DOMAIN)]


def _check_case(stream_count: int, coefficients: tuple[int, ...], scenario: str) -> int:
    """Check one 32-lane reduction and return its largest packed field sum."""
    lane_bound = stream_count * UINT16_MAX * 2
    lane_bias = lane_bound
    packed_words: list[int] = []
    reference_a = 0
    reference_b = 0
    biased_a_values: list[int] = []
    biased_b_values: list[int] = []

    for lane in range(LANE_COUNT):
        active = lane % 7 not in (1, 5)
        lane_a = 0
        lane_b = 0
        if active:
            for stream in range(stream_count):
                coefficient = coefficients[(stream + lane) % stream_count]
                value_a = _value(scenario, lane, stream, 0)
                value_b = _value(scenario, lane, stream, 1)
                assert 0 <= value_a <= UINT16_MAX
                assert 0 <= value_b <= UINT16_MAX
                lane_a += coefficient * value_a
                lane_b += coefficient * value_b
        assert -lane_bound <= lane_a <= lane_bound
        assert -lane_bound <= lane_b <= lane_bound
        reference_a += lane_a
        reference_b += lane_b

        biased_a = lane_a + lane_bias
        biased_b = lane_b + lane_bias
        assert 0 <= biased_a <= 2 * lane_bound
        assert 0 <= biased_b <= 2 * lane_bound
        biased_a_values.append(biased_a)
        biased_b_values.append(biased_b)
        packed_words.append(biased_a | (biased_b << 32))

    packed_sum = sum(packed_words)
    assert packed_sum < UINT64_LIMIT
    sum_a_biased = sum(biased_a_values)
    sum_b_biased = sum(biased_b_values)
    assert sum_a_biased < UINT32_LIMIT
    assert sum_b_biased < UINT32_LIMIT

    # This equality proves that summing packed lane words introduced no carry
    # from the low 32-bit channel into the high 32-bit channel.
    assert packed_sum == (sum_a_biased | (sum_b_biased << 32))
    reduced_a = (packed_sum & UINT32_MASK) - LANE_COUNT * lane_bias
    reduced_b = ((packed_sum >> 32) & UINT32_MASK) - LANE_COUNT * lane_bias
    assert (reduced_a, reduced_b) == (reference_a, reference_b)
    return max(sum_a_biased, sum_b_biased)


def main() -> None:
    """Exhaust coefficient tuples for one, two, and four streams per lane."""
    case_count = 0
    pattern_counts: dict[str, int] = {}
    largest_field_sum = 0
    inactive_lanes = sum(lane % 7 in (1, 5) for lane in range(LANE_COUNT))

    for stream_count in (1, 2, 4):
        patterns = tuple(product(COEFFICIENTS, repeat=stream_count))
        pattern_counts[str(stream_count)] = len(patterns)
        for coefficients in patterns:
            for scenario in ("endpoints", "mixed-range"):
                largest_field_sum = max(
                    largest_field_sum,
                    _check_case(stream_count, coefficients, scenario),
                )
                case_count += 1

    assert pattern_counts == {"1": 5, "2": 25, "4": 625}
    assert case_count == 2 * sum(pattern_counts.values()) == 1310
    max_theoretical_field_sum = LANE_COUNT * 2 * 4 * UINT16_MAX * 2
    assert max_theoretical_field_sum == 33_553_920
    assert max_theoretical_field_sum < UINT32_LIMIT
    print(
        json.dumps(
            {
                "status": "pass",
                "cpu_only": True,
                "cases": case_count,
                "coefficient_patterns_by_stream_count": pattern_counts,
                "stream_counts_per_lane": [1, 2, 4],
                "coefficient_domain": list(COEFFICIENTS),
                "uint16_value_domain": list(VALUE_DOMAIN),
                "inactive_lanes_per_case": inactive_lanes,
                "largest_observed_biased_channel_sum": largest_field_sum,
                "four_stream_worst_case_biased_channel_sum": max_theoretical_field_sum,
                "uint32_limit": UINT32_LIMIT,
                "low_to_high_carry": False,
                "gpu_pipeline_wired": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
