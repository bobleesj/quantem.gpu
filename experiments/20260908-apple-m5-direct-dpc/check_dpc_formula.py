"""Check exact DPC weighted sums against direct integer counts."""

import json
import random


def check():
    """Exercise sparse counts, every uint16 width, and saturated pixels."""
    rng = random.Random(20260908)
    masks = (0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000)
    checked = 0
    for width in range(17):
        for repetition in range(64):
            values = [rng.randrange(1 << width) for _ in range(32)]
            if repetition == 0:
                values = [(1 << width) - 1] * 32
            if repetition == 1:
                values[rng.randrange(32)] = 65535
            detector_columns = rng.choice((32, 64, 96, 192, 256))
            pixel = rng.randrange(128) * 32
            detector_row, detector_col = divmod(pixel, detector_columns)
            observed = [0, 0, 0]
            for plane in range(16):
                word = sum(((value >> plane) & 1) << lane
                           for lane, value in enumerate(values))
                count = bin(word).count('1')
                weighted_col = sum((1 << bit) * bin(word & mask).count('1')
                                   for bit, mask in enumerate(masks))
                observed[0] += count << plane
                observed[1] += (count * detector_row) << plane
                observed[2] += (count * detector_col + weighted_col) << plane
            expected = [sum(values), sum(values) * detector_row,
                        sum(value * (detector_col + lane)
                            for lane, value in enumerate(values))]
            assert observed == expected
            checked += 1
    print(json.dumps({'exact_word_cases': checked, 'preserved_maximum': 65535}))


if __name__ == '__main__':
    check()
