"""Independent integer checks of four-tile packed-header reuse.

This validates the proposed address arithmetic, not the Metal compiler or
GPU result. The separate seven-source benchmark is required for those claims.
"""

import random


def make_headers(widths, initial=0):
    checkpoints = (len(widths) + 31) // 32
    result = [initial] + [sum(widths[:32 * n]) for n in range(1, checkpoints)]
    for first in range(0, len(widths), 8):
        word = 0
        for lane, width in enumerate(widths[first:first + 8]):
            assert 0 <= width <= 16 and width != 15
            word |= (15 if width == 16 else width) << (lane * 4)
        result.append(word)
    return result


def width_from_nibble(word, shift):
    value = (word >> shift) & 15
    return 16 if value == 15 else value


def first_descriptor(headers, tile_count, tile):
    checkpoints = (tile_count + 31) // 32
    checkpoint = tile // 32
    offset = headers[0] + (headers[checkpoint] if checkpoint else 0)
    for before in range(checkpoint * 32, tile):
        offset += width_from_nibble(headers[checkpoints + before // 8], (before % 8) * 4)
    width = width_from_nibble(headers[checkpoints + tile // 8], (tile % 8) * 4)
    return offset, width


def quad_descriptors(headers, tile_count, first):
    checkpoints = (tile_count + 31) // 32
    offset, _ = first_descriptor(headers, tile_count, first)
    widths = headers[checkpoints + first // 8] >> ((first % 8) * 4)
    output = []
    for tile in range(min(4, tile_count - first)):
        width = width_from_nibble(widths, tile * 4)
        output.append((offset, width))
        offset += width
    return output


def main():
    rng = random.Random(20260908)
    cases = 0
    for tiles in [1, 3, 4, 5, 7, 8, 9, 31, 32, 33, 63, 64, 65, 128]:
        for kind in range(20):
            widths = [rng.choice(list(range(15)) + [16]) for _ in range(tiles)]
            if kind < 16:
                widths = [16 if kind == 15 else kind] * tiles
            initial = rng.randrange(1 << 20)
            headers = make_headers(widths, initial)
            for first in range(0, tiles, 4):
                expected = [(initial + sum(widths[:tile]), widths[tile])
                            for tile in range(first, min(first + 4, tiles))]
                assert quad_descriptors(headers, tiles, first) == expected
                cases += 1
    print(f'PASS {cases} quad address groups, widths0–16 and partial tiles; CPU only')


if __name__ == '__main__':
    main()
