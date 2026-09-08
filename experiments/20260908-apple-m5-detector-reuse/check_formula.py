"""Check shared-header addressing and vector detector sums against counts."""

import json
import random


def _transpose(words):
    for distance in (16, 8, 4, 2, 1):
        mask = 0xFFFFFFFF // ((1 << distance) + 1)
        words = [(((words[lane ^ distance] >> distance) & mask)
                  | (value & ~mask)) if lane & distance else
                 ((value & mask) | ((words[lane ^ distance] & mask) << distance))
                 for lane, value in enumerate(words)]
    return words


def check():
    """Compare full uint16 signed detector updates without changing counts."""
    rng = random.Random(20260908)
    headers = 0
    for tiles in (1, 4, 5, 7, 8, 9, 31, 32, 33, 64):
        for _ in range(32):
            widths = [rng.choice(tuple(range(15)) + (16,)) for _ in range(tiles)]
            encoded = [(15 if width == 16 else width) for width in widths]
            for group_size in (4, 8):
                for first in range(0, tiles, group_size):
                    word_start = (first // 8) * 8
                    word = sum(value << (4 * lane) for lane, value in
                               enumerate(encoded[word_start:word_start + 8]))
                    word >>= (first % 8) * 4
                    offset = sum(widths[:first])
                    for tile in range(group_size):
                        if first + tile >= tiles:
                            break
                        width = (word >> (4 * tile)) & 15
                        width = 16 if width == 15 else width
                        assert offset == sum(widths[:first + tile])
                        assert width == widths[first + tile]
                        offset += width
                        headers += 1
    frames = 0
    for case in range(12):
        entry_count = (1, 17, 31, 32)[case % 4]
        coefficients = [rng.choice((-1, 1)) for _ in range(32)]
        positive = sum((1 << lane) for lane, sign in enumerate(coefficients) if sign > 0)
        for tile in range(4):
            counts = [[rng.randrange(1 << rng.randrange(17))
                       if entry < entry_count else 0 for _ in range(32)]
                      for entry in range(32)]
            counts[0][(case + tile) % 32] = 65535
            observed = [0] * 32
            for plane in range(16):
                words = [sum(((value >> plane) & 1) << scan
                             for scan, value in enumerate(column)) for column in counts]
                for scan, word in enumerate(_transpose(words)):
                    observed[scan] += (bin(word & positive).count('1')
                                       - bin(word & ~positive).count('1')) << plane
            expected = [sum(coefficients[entry] * counts[entry][scan] for entry in range(32))
                        for scan in range(32)]
            assert observed == expected
            frames += 32
    print(json.dumps({'exact_header_addresses': headers, 'exact_signed_scan_sums': frames,
                      'maximum_count_preserved': 65535}))


if __name__ == '__main__':
    check()
