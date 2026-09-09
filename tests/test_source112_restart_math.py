"""Independent bit-position oracle for the optional GPU tANS restart cache."""

import random

import pytest


def _decode_integer_bits(words, table):
    encoded = sum(word << (32 * i) for i, word in enumerate(words))
    state, cursor = words[0] & 1023, 10
    values, checkpoints = [], []
    for pair_index in range(256):
        code = table[state]
        count = (code >> 12) & 15
        low = (encoded >> cursor) & ((1 << count) - 1)
        cursor += count
        state = (code >> 16) + low
        pair = code & 4095
        if pair == 4095:
            pair = (encoded >> cursor) & 4095
            cursor += 12
        assert cursor <= len(words) * 32
        values.extend((pair & 63, pair >> 6))
        if pair_index in (63, 127, 191):
            checkpoints.append(0x80000000 | (cursor << 10) | state)
    return values, checkpoints


def _restart_reservoir(words, table, checkpoint, segment):
    cursor = (checkpoint >> 10) & 8191 if segment else 10
    state = checkpoint & 1023 if segment else words[0] & 1023
    word_index, shift = divmod(cursor, 32)
    reservoir, available = 0, 0
    if word_index < len(words):
        reservoir, available = words[word_index] >> shift, 32 - shift
        word_index += 1

    def take(count):
        nonlocal reservoir, available, word_index
        if not count:
            return 0
        mask = (1 << count) - 1
        if available >= count:
            value = reservoir & mask
            reservoir >>= count
            available -= count
            return value
        if word_index >= len(words):
            raise ValueError("Truncated checkpoint stream")
        word = words[word_index]
        word_index += 1
        value = (reservoir | (word << available)) & mask
        consumed = count - available
        reservoir, available = word >> consumed, 32 - consumed
        return value

    values = []
    for _ in range(64):
        code = table[state]
        state = (code >> 16) + take((code >> 12) & 15)
        pair = code & 4095
        if pair == 4095:
            pair = take(12)
        values.extend((pair & 63, pair >> 6))
    return values


@pytest.mark.parametrize("seed", range(20))
def test_four_restart_segments_equal_independent_serial_oracle(seed):
    rng = random.Random(seed)
    words = [rng.getrandbits(32) for _ in range(256)]
    table = []
    for state in range(1024):
        count = state % 9
        symbol = 4095 if state % 17 == 0 else state * 37 % 4095
        table.append(((1024 - (1 << count)) << 16) | (count << 12) | symbol)
    expected, checkpoints = _decode_integer_bits(words, table)
    actual = []
    for segment, checkpoint in enumerate([0, *checkpoints]):
        actual.extend(_restart_reservoir(words, table, checkpoint, segment))
    assert actual == expected
    assert all(checkpoint & 0x80000000 for checkpoint in checkpoints)
    assert all(((checkpoint >> 10) & 8191) >= 10 for checkpoint in checkpoints)


def test_word_aligned_end_allows_zero_bit_transitions():
    table = [((state + 1) % 1024) << 16 | (state % 4095) for state in range(1024)]
    # Consume exactly the header's 22 reservoir bits, then use zero-bit transitions.
    for state in (20, 21):
        table[state] |= 11 << 12
    expected, checkpoints = _decode_integer_bits([0], table)
    assert all(((checkpoint >> 10) & 8191) == 32 for checkpoint in checkpoints)
    actual = sum((_restart_reservoir([0], table, checkpoint, segment)
                  for segment, checkpoint in enumerate([0, *checkpoints])), [])
    assert actual == expected


def test_full_native_checkpoint_budget_and_cursor_bound():
    assert 1056 * 32 * 17466 * 3 * 4 == 7_082_532_864
    assert 10 + 192 * (15 + 12) < 8192
    assert 26 * (12 + 32 * 17466 * 3) * 4 < 256 * 1024**2


@pytest.mark.parametrize("seed", range(10))
def test_cached_absolute_stream_offsets_match_compact_block_prefixes(seed):
    rng = random.Random(seed)
    lengths = [rng.randrange(1, 257) for _ in range(32 * 20)]
    base = 16 + seed * 123_456
    expected, block_starts, first = [], [], 0
    for stream, length in enumerate(lengths):
        if stream % 32 == 0:
            block_starts.append(first)
        expected.append(base + first)
        first += length
    packed_lengths = [0] * (len(lengths) // 4)
    for stream, length in enumerate(lengths):
        packed_lengths[stream // 4] |= (length - 1) << (8 * (stream % 4))
    actual = []
    for stream in range(len(lengths)):
        start = block_starts[stream // 32]
        for preceding in range(stream & ~31, stream):
            start += ((packed_lengths[preceding // 4] >> (8 * (preceding % 4))) & 255) + 1
        actual.append(base + start)
    assert actual == expected
    for stream, absolute in enumerate(actual):
        length = ((packed_lengths[stream // 4] >> (8 * (stream % 4))) & 255) + 1
        relative = absolute - base
        assert absolute >= base
        assert relative < first
        assert length <= first - relative
        assert absolute + length == (actual[stream + 1] if stream + 1 < len(actual) else base + first)


def test_stream_offset_cache_budget_and_bounds():
    streams = 1056 * 32 * 17466
    assert streams * 4 == 2_360_844_288
    assert streams * 16 + 1056 * 48 == 9_443_427_840
    base, words, length = 1234, 256, 97

    def valid(absolute):
        if absolute < base:
            return False
        relative = absolute - base
        return relative < words and length <= words - relative

    assert valid(base)
    assert valid(base + words - length)
    assert not valid(base - 1)
    assert not valid(base + words)
    assert not valid(base + words - 1)
    assert not valid(0xFFFFFFFF)
