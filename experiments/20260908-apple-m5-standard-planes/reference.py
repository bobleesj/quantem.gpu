"""Independent raw-count checks of low-width register transposition.

This checks the integer derivation; the actual Metal/HDF5 experiment and
full-volume hash gates remain required before enabling the representation.
"""

import random

MASK = (1 << 32) - 1


def even(value):
    value &= 0x55555555
    for shift, mask in [(1, 0x33333333), (2, 0x0F0F0F0F), (4, 0x00FF00FF)]:
        value = (value | (value >> shift)) & mask
    return (value | (value >> 8)) & 0xFFFF


def third(value):
    high = (value >> 20) & 0x400
    value &= 0x09249249
    for shift, mask in [(2, 0x030C30C3), (4, 0x0300F00F), (8, 0x030000FF), (16, 0x000003FF)]:
        value = (value ^ (value >> shift)) & mask
    return value | high


def fourth(value):
    value &= 0x11111111
    value = (value | (value >> 3)) & 0x03030303
    value = (value | (value >> 6)) & 0x000F000F
    return (value | (value >> 12)) & 0xFF


def convert(values, width):
    assert len(values) == 32 and max(values) < (1 << width)
    packed = sum(value << (sample * width) for sample, value in enumerate(values))
    words = [(packed >> (word * 32)) & MASK for word in range(width)]
    output = []
    for plane in range(width):
        if width == 1:
            value = words[0]
        elif width == 2:
            value = even(words[0] >> plane) | (even(words[1] >> plane) << 16)
        elif width == 3:
            second, last = (plane + 1) % 3, (plane + 2) % 3
            count, following = (34 - plane) // 3, (34 - second) // 3
            value = third(words[0] >> plane) | (third(words[1] >> second) << count)
            value |= third(words[2] >> last) << (count + following)
        elif width == 4:
            value = sum(fourth(words[word] >> plane) << (word * 8) for word in range(4))
        else:
            value = sum(((v >> plane) & 1) << scan for scan, v in enumerate(values))
        output.append(value & MASK)
    restored = [sum(((word >> scan) & 1) << plane for plane, word in enumerate(output))
                for scan in range(32)]
    assert restored == values


def main():
    rng = random.Random(20260908)
    cases = 0
    for width in range(17):
        for _ in range(100):
            convert([rng.randrange(1 << width) for _ in range(32)], width)
            cases += 1
    for width in range(1, 17):
        for scan in range(32):
            values = [0] * 32
            values[scan] = (1 << width) - 1
            convert(values, width)
            cases += 1
    print(f'PASS {cases} cells with32 exact counts, widths0–16 including65535; CPU only')


if __name__ == '__main__':
    main()
