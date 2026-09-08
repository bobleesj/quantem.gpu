"""Validate four-column SIMD address mapping independently of GPU execution."""

import importlib.util
from pathlib import Path


def main():
    source = Path(__file__).parents[1] / '20260908-apple-m5-plane-gather/reference.py'
    spec = importlib.util.spec_from_file_location('plane_reference', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    checked = 0
    for pixels in [4096, 8192, 36864]:
        coverage = set()
        for index in range(pixels // 4):
            lane = index % 32
            group = (index // 32) * 128
            for part in range(4):
                pixel = group + lane + part * 32
                assert pixel not in coverage
                coverage.add(pixel)
                for plane in [0, 7, 15]:
                    vector_word = (plane * 32 + (group % 4096) // 128) * 4 + part
                    scalar_word = plane * 128 + (pixel % 4096) // 32
                    assert vector_word == scalar_word
                    assert group // 4096 == pixel // 4096
                checked += 1
        assert coverage == set(range(pixels))
    for part in range(4):
        # Independent uint16 component, with distinct hot-pixel coordinates.
        counts = [[65535 if (scan, pixel) == (31 - part, part * 3)
                   else ((scan * 41 + pixel * 17 + part * 7) % (1 << (part + 1)))
                   for pixel in range(32)] for scan in range(32)]
        reference.check(counts)
    print(f'PASS {checked} vector column addresses and four independent full-uint16 transposes; CPU only')


if __name__ == '__main__':
    main()
