"""Check direct source-plane transposition and exact maximum without GPU data.

The subsequent real-HDF5 kernel run remains the required hardware parity gate.
"""

import random


def transpose(words):
    for shift in [1, 2, 4, 8, 16]:
        mask = 0xFFFFFFFF // ((1 << shift) + 1)
        words = [((value & ~mask) | ((words[lane ^ shift] & ~mask) >> shift))
                 if lane & shift else
                 ((value & mask) | ((words[lane ^ shift] & mask) << shift))
                 for lane, value in enumerate(words)]
    return words


def check(counts):
    output = [[0] * 32 for _ in range(32)]
    sums = [0] * 32
    maxima = [0] * 32
    candidates = [0xFFFFFFFF] * 32
    for plane in reversed(range(16)):
        source_words = [sum(((counts[scan][pixel] >> plane) & 1) << pixel
                            for pixel in range(32)) for scan in range(32)]
        planes = transpose(source_words)
        for pixel, bits in enumerate(planes):
            sums[pixel] += bin(bits).count('1') << plane
            matches = candidates[pixel] & bits
            if matches:
                maxima[pixel] |= 1 << plane
                candidates[pixel] = matches
            for scan in range(32):
                output[scan][pixel] |= ((bits >> scan) & 1) << plane
    assert output == counts
    assert sums == [sum(row[pixel] for row in counts) for pixel in range(32)]
    assert maxima == [max(row[pixel] for row in counts) for pixel in range(32)]


if __name__ == '__main__':
    rng = random.Random(20260908)
    for width in range(17):
        check([[rng.randrange(1 << width) for _ in range(32)] for _ in range(32)])
    check([[65535 if (scan, pixel) == (31, 17) else 0 for pixel in range(32)]
           for scan in range(32)])
    check([[127 if scan % 2 else 128 for _ in range(32)] for scan in range(32)])
    print('19 independent 32x32 fixtures: counts, sums and exact maxima pass')
