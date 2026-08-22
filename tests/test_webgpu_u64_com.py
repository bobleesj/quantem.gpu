"""CPU proof model for the paired-u32 WebGPU center-of-mass contract."""

from __future__ import annotations

import random

import pytest

U32_MASK = (1 << 32) - 1
U64_MASK = (1 << 64) - 1
UINT16_MAX = (1 << 16) - 1

Words = tuple[int, int]


def _words(value: int) -> Words:
    """Split an unsigned 64-bit integer into low and high words."""

    assert 0 <= value <= U64_MASK
    return value & U32_MASK, value >> 32


def _value(words: Words) -> int:
    """Join low and high words into an unsigned 64-bit integer."""

    return words[0] | (words[1] << 32)


def _add_words(left: Words, right: Words) -> Words:
    """Mirror the WGSL paired-word addition and carry."""

    low_sum = left[0] + right[0]
    low = low_sum & U32_MASK
    carry = int(low_sum > U32_MASK)
    high = (left[1] + right[1] + carry) & U32_MASK
    return low, high


def _subtract_words(left: Words, right: Words) -> Words:
    """Mirror the WGSL paired-word subtraction and borrow."""

    borrow = int(left[0] < right[0])
    return (
        (left[0] - right[0]) & U32_MASK,
        (left[1] - right[1] - borrow) & U32_MASK,
    )


def _shift_left_one(words: Words) -> Words:
    """Mirror a one-bit logical left shift modulo 2^64."""

    return (words[0] << 1) & U32_MASK, (((words[1] << 1) | (words[0] >> 31)) & U32_MASK)


def _multiply_u32(left: int, right: int) -> Words:
    """Mirror the WGSL 16-bit-limb 32 x 32 -> 64 multiplication."""

    left_low, left_high = left & 0xFFFF, left >> 16
    right_low, right_high = right & 0xFFFF, right >> 16
    product = (left_low * right_low, left_high * right_high)
    cross = left_low * right_high
    product = _add_words(product, ((cross << 16) & U32_MASK, cross >> 16))
    cross = left_high * right_low
    return _add_words(product, ((cross << 16) & U32_MASK, cross >> 16))


def _double_step(remainder: Words, denominator: Words) -> tuple[Words, int]:
    """Double a remainder and return its exact quotient bit."""

    overflow = bool(remainder[1] & (1 << 31))
    doubled = _shift_left_one(remainder)
    if overflow or _value(doubled) >= _value(denominator):
        return _subtract_words(doubled, denominator), 1
    return doubled, 0


def _compare_twice(remainder: Words, denominator: Words) -> int:
    """Compare the mathematical value 2 * remainder with denominator."""

    if remainder[1] & (1 << 31):
        return 1
    doubled = _value(_shift_left_one(remainder))
    denominator_value = _value(denominator)
    return (doubled > denominator_value) - (doubled < denominator_value)


def _double_step_u32(remainder: int, denominator: int) -> tuple[int, int]:
    """Mirror the narrow WGSL remainder step without paired words."""

    overflow = bool(remainder & (1 << 31))
    doubled = (remainder << 1) & U32_MASK
    if overflow or doubled >= denominator:
        return (doubled - denominator) & U32_MASK, 1
    return doubled, 0


def _compare_twice_u32(remainder: int, denominator: int) -> int:
    """Compare 2 * remainder with denominator using u32 operations."""

    if remainder & (1 << 31):
        return 1
    doubled = remainder << 1
    return (doubled > denominator) - (doubled < denominator)


def _ratio_bits_u32(numerator: int, denominator: int) -> int:
    """Mirror the specialized WGSL u32 rational-to-float32 conversion."""

    assert 0 <= numerator <= U32_MASK
    assert 0 <= denominator <= U32_MASK
    if numerator == 0 or denominator == 0:
        return 0

    exponent = 0
    if numerator >= denominator:
        scaled_denominator = denominator
        while not scaled_denominator & (1 << 31):
            candidate = scaled_denominator << 1
            if candidate > numerator:
                break
            scaled_denominator = candidate
            exponent += 1
        remainder = numerator - scaled_denominator
        normalized_denominator = scaled_denominator
    else:
        remainder = numerator
        while True:
            remainder, quotient_bit = _double_step_u32(remainder, denominator)
            exponent -= 1
            if quotient_bit:
                break
        normalized_denominator = denominator

    significand = 1 << 23
    for bit in range(23, 0, -1):
        remainder, quotient_bit = _double_step_u32(remainder, normalized_denominator)
        significand |= quotient_bit << (bit - 1)
    round_comparison = _compare_twice_u32(remainder, normalized_denominator)
    if round_comparison > 0 or (round_comparison == 0 and significand & 1):
        significand += 1
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    return ((exponent + 127) << 23) | (significand & 0x7FFFFF)


def _ratio_bits_words(numerator_value: int, denominator_value: int) -> int:
    """Mirror the WGSL exact paired-word rational-to-float32 conversion."""

    if numerator_value == 0 or denominator_value == 0:
        return 0
    numerator = _words(numerator_value)
    denominator = _words(denominator_value)
    exponent = 0
    if numerator_value >= denominator_value:
        scaled_denominator = denominator
        while not (scaled_denominator[1] & (1 << 31)):
            candidate = _shift_left_one(scaled_denominator)
            if _value(candidate) > numerator_value:
                break
            scaled_denominator = candidate
            exponent += 1
        remainder = _subtract_words(numerator, scaled_denominator)
        normalized_denominator = scaled_denominator
    else:
        remainder = numerator
        while True:
            remainder, quotient_bit = _double_step(remainder, denominator)
            exponent -= 1
            if quotient_bit:
                break
        normalized_denominator = denominator

    significand = 1 << 23
    for bit in range(23, 0, -1):
        remainder, quotient_bit = _double_step(remainder, normalized_denominator)
        significand |= quotient_bit << (bit - 1)
    round_comparison = _compare_twice(remainder, normalized_denominator)
    if round_comparison > 0 or (round_comparison == 0 and significand & 1):
        significand += 1
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    return ((exponent + 127) << 23) | (significand & 0x7FFFFF)


def _ratio_bits_oracle(numerator: int, denominator: int) -> int:
    """Round one exact rational to float32 without binary64 intermediates."""

    if numerator == 0 or denominator == 0:
        return 0
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    shift = 23 - exponent
    if shift >= 0:
        scaled_numerator, scaled_denominator = numerator << shift, denominator
    else:
        scaled_numerator, scaled_denominator = numerator, denominator << -shift
    significand, remainder = divmod(scaled_numerator, scaled_denominator)
    assert 1 << 23 <= significand < 1 << 24
    twice_remainder = remainder * 2
    if twice_remainder > scaled_denominator or (
        twice_remainder == scaled_denominator and significand & 1
    ):
        significand += 1
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    return ((exponent + 127) << 23) | (significand & 0x7FFFFF)


def _selection_bounds(
    detector_indices: list[int], detector_columns: int, value_max: int
) -> tuple[int, int, int]:
    """Return exact total, row-moment, and column-moment worst-case bounds."""

    total = len(detector_indices) * value_max
    row = sum(index // detector_columns for index in detector_indices) * value_max
    column = sum(index % detector_columns for index in detector_indices) * value_max
    return total, row, column


def _full_detector_bounds(
    detector_rows: int, detector_columns: int, value_max: int
) -> tuple[int, int, int]:
    """Return closed-form bounds for a full rectangular detector."""

    total = value_max * detector_rows * detector_columns
    row = value_max * detector_columns * detector_rows * (detector_rows - 1) // 2
    column = value_max * detector_rows * detector_columns * (detector_columns - 1) // 2
    return total, row, column


def test_paired_words_match_exact_u64_arithmetic() -> None:
    """Exercise carries, borrows, and 32 x 32 multiplication adversarially."""

    values = [0, 1, (1 << 32) - 1, 1 << 32, (1 << 63) - 1, U64_MASK]
    for left in values:
        for right in values:
            assert (
                _value(_add_words(_words(left), _words(right)))
                == (left + right) & U64_MASK
            )
            assert (
                _value(_subtract_words(_words(left), _words(right)))
                == (left - right) & U64_MASK
            )

    rng = random.Random(20260822)
    u32_values = [0, 1, 0xFFFF, 1 << 16, U32_MASK]
    u32_values.extend(rng.randrange(1 << 32) for _ in range(10_000))
    for left, right in zip(u32_values, reversed(u32_values), strict=True):
        assert _value(_multiply_u32(left, right)) == left * right


def test_full_192_uint16_max_count_accumulation_is_exact() -> None:
    """Prove the known 192 x 192 high-count case crosses u32 only in moments."""

    detector_indices = list(range(192 * 192))
    expected = _full_detector_bounds(192, 192, UINT16_MAX)
    assert expected == (2_415_882_240, 230_716_753_920, 230_716_753_920)
    assert expected[0] <= U32_MASK
    assert expected[1] > U32_MASK
    assert expected[2] > U32_MASK

    accumulated = [_words(0), _words(0), _words(0)]
    for index in detector_indices:
        row_term = (index // 192) * UINT16_MAX
        column_term = (index % 192) * UINT16_MAX
        assert row_term <= U32_MASK
        assert column_term <= U32_MASK
        accumulated[0] = _add_words(accumulated[0], _words(UINT16_MAX))
        accumulated[1] = _add_words(accumulated[1], _words(row_term))
        accumulated[2] = _add_words(accumulated[2], _words(column_term))
    assert tuple(_value(words) for words in accumulated) == expected
    assert _ratio_bits_words(expected[1], expected[0]) == _ratio_bits_oracle(
        expected[1], expected[0]
    )
    assert _ratio_bits_words(expected[2], expected[0]) == _ratio_bits_oracle(
        expected[2], expected[0]
    )


def test_known_failing_192_uint16_ratios_are_correctly_rounded() -> None:
    """Cover retained packed-uint16 frames that exposed the mode-0 defect."""

    retained_total_row_column = [
        (75_317, 7_277_182, 7_323_996),
        (74_890, 7_250_347, 7_218_508),
        (75_933, 7_341_839, 7_356_692),
        (74_156, 7_125_662, 7_135_601),
        (75_951, 7_229_374, 7_345_866),
    ]
    for total, row_moment, column_moment in retained_total_row_column:
        assert _ratio_bits_words(row_moment, total) == _ratio_bits_oracle(
            row_moment, total
        )
        assert _ratio_bits_words(column_moment, total) == _ratio_bits_oracle(
            column_moment, total
        )


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (1, U64_MASK),
        ((1 << 63) - 1, U64_MASK),
        ((1 << 63) + 123, U64_MASK - 4),
        (U64_MASK - 1, U64_MASK),
        (U64_MASK, U64_MASK),
        (U64_MASK, 1),
        ((1 << 53) + 1, (1 << 53) - 1),
    ],
)
def test_ratio_conversion_handles_high_word_and_rounding_edges(
    numerator: int, denominator: int
) -> None:
    """Exercise normalization overflow and round-to-nearest-even edges."""

    assert _ratio_bits_words(numerator, denominator) == _ratio_bits_oracle(
        numerator, denominator
    )


def test_ratio_conversion_matches_exact_oracle_over_random_u64_pairs() -> None:
    """Compare the paired-word converter with an independent exact oracle."""

    rng = random.Random(0)
    for _ in range(50_000):
        denominator = rng.randrange(1, U64_MASK + 1)
        numerator = rng.randrange(0, U64_MASK + 1)
        assert _ratio_bits_words(numerator, denominator) == _ratio_bits_oracle(
            numerator, denominator
        )


def test_narrow_ratio_conversion_matches_exact_oracle_over_random_u32_pairs() -> None:
    """Prove the u32 fast path preserves the paired path's exact rounding."""

    rng = random.Random(20260822)
    for _ in range(50_000):
        denominator = rng.randrange(1, U32_MASK + 1)
        numerator = rng.randrange(0, U32_MASK + 1)
        expected = _ratio_bits_oracle(numerator, denominator)
        assert _ratio_bits_u32(numerator, denominator) == expected
        assert _ratio_bits_words(numerator, denominator) == expected


@pytest.mark.parametrize(
    ("detector_size", "expected_moment", "fits_u32"),
    [
        (192, 230_716_753_920, False),
        (96, 28_688_601_600, False),
        (48, 3_548_327_040, True),
        (24, 434_103_840, True),
    ],
)
def test_uint16_full_detector_bounds_select_narrow_or_wide_path(
    detector_size: int, expected_moment: int, fits_u32: bool
) -> None:
    """Freeze the geometry-dependent one-word versus paired-word decision."""

    total, row, column = _full_detector_bounds(detector_size, detector_size, UINT16_MAX)
    assert row == expected_moment
    assert column == expected_moment
    assert (max(total, row, column) <= U32_MASK) is fits_u32
    assert max(total, row, column) <= U64_MASK


def test_mask_specific_bounds_preserve_row_column_order() -> None:
    """Check arbitrary active pixels use public (row, column) coordinates."""

    indices = [0, 2, 7, 11]
    total, row, column = _selection_bounds(indices, 4, UINT16_MAX)
    assert total == 4 * UINT16_MAX
    assert row == (0 + 0 + 1 + 2) * UINT16_MAX
    assert column == (0 + 2 + 3 + 3) * UINT16_MAX


def test_unsupported_geometry_exceeds_two_word_contract() -> None:
    """Show that a declared geometry can exceed the exact 64-bit admission bound."""

    total, row, column = _full_detector_bounds(1 << 16, 1 << 16, U32_MASK)
    assert total <= U64_MASK
    assert row > U64_MASK
    assert column > U64_MASK
