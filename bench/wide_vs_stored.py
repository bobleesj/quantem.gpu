"""Coded size of a wider symbol against the bytes actually stored, on real streams.

Both earlier size estimates charged an escaped group its counts at their own entropy, which
only holds if escapes are coded well. Here an escaped group falls back to the stored pair
alphabet, which is what an implementation would really do, and the result is compared with
the bytes the file actually holds for the same streams. No model stands in for either side.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/owner/worktrees/sep-11-encoded-detector-speed/src")

STATES = 1024
FORMS = Path.home() / ".config/live4dstem/gate-forms.txt"


def poisson(rate, size):
    counts = np.arange(size)
    p = np.exp(counts * np.log(max(rate, 1e-9))
               - np.array([math.lgamma(k + 1) for k in counts]) - rate)
    return p / p.sum()


def quantise(probability, escape_mass):
    """Give the commonest symbols the table's states; return bits per symbol by index."""
    order = np.argsort(-probability)
    support = min(STATES - 2, int((probability[order] > 1e-12).sum()))
    keep = order[:support]
    frequency = np.zeros(probability.size + 1, np.int64)
    budget = STATES - (support + 1)
    frequency[keep] = np.floor(probability[keep] * budget).astype(np.int64) + 1
    frequency[-1] = max(1, int(round(escape_mass * budget)))
    spare = STATES - int(frequency.sum())
    if spare > 0:
        frequency[-1] += spare
    else:
        for index in keep[::-1]:
            while spare < 0 and frequency[index] > 1:
                frequency[index] -= 1
                spare += 1
            if spare == 0:
                break
    cost = np.full(probability.size + 1, np.inf)
    live = frequency > 0
    cost[live] = np.log2(STATES / frequency[live])
    return cost


def pair_cost(rate):
    """Bits for each pair symbol under the stored alphabet, plus the escape's cost."""
    single = poisson(rate, 33)
    joint = np.outer(single, single).ravel()
    joint = joint / joint.sum()
    return quantise(joint, escape_mass=1e-3)


def group_cost(rate, group, small):
    """Bits for each group symbol, and the escape's cost, for a group of `group` counts."""
    single = poisson(rate, 64)
    inside = single[:small]
    conditional = inside / inside.sum()
    probability = np.ones(1)
    for _ in range(group):
        probability = np.outer(probability, conditional).ravel()
    probability = probability * inside.sum() ** group
    return quantise(probability, escape_mass=max(1e-3, 1 - probability.sum()))


def stream_bits(counts, rate, group, small):
    """Bits to code one stream with `group`-count symbols, escaping to the pair alphabet."""
    cost = group_cost(rate, group, small)
    pairs = pair_cost(rate)
    escape_index = small**group
    total = 0.0
    lookups = 0
    for start in range(0, len(counts), group):
        chunk = counts[start:start + group]
        symbol = escape_index
        if len(chunk) == group and all(int(v) < small for v in chunk):
            symbol = 0
            for value in reversed(chunk):
                symbol = symbol * small + int(value)
            if symbol >= escape_index or not np.isfinite(cost[symbol]):
                symbol = escape_index
        if symbol != escape_index:
            total += cost[symbol]
            lookups += 1
            continue
        total += cost[escape_index]        # the escape symbol itself
        lookups += 1
        for at in range(0, len(chunk), 2):   # its counts fall back to stored pair symbols
            a = int(chunk[at])
            b = int(chunk[at + 1]) if at + 1 < len(chunk) else 0
            index = a * 33 + b if a < 32 and b < 32 else 1089
            total += pairs[index] if index < pairs.size and np.isfinite(pairs[index]) else pairs[-1] + 13
            lookups += 1
    return total, lookups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=192)
    parser.add_argument("--inner", type=float, default=40.0)
    parser.add_argument("--outer", type=float, default=92.0)
    args = parser.parse_args()

    import cupy as cp
    from quantem.gpu.io._paired import load_paired_file

    path = [line.strip() for line in FORMS.read_text().splitlines() if line.strip()][0]
    source = getattr(load_paired_file(path, device=None, verbose=False), "data", None)
    shape = tuple(int(v) for v in source.shape[2:])
    centre = [(v - 1) / 2 for v in shape]
    row, col = np.indices(shape)
    radius = np.hypot(row - centre[0], col - centre[1])
    edge = (np.abs(radius - args.inner) < 1.0) | (np.abs(radius - args.outer) < 1.0)
    candidates = np.flatnonzero(edge.ravel())
    chosen = candidates[np.linspace(0, candidates.size - 1, min(args.streams, candidates.size)).astype(int)]

    interval = int(source.interval)
    frames = cp.asarray(source.decode_blocks(0, interval)).reshape(interval, -1)
    block = frames[:, cp.asarray(chosen)].get().T          # one row is one stream
    chunk = source.chunks[0]
    records, models = chunk.arrays[1].get(), chunk.arrays[2].get()

    def stored_bytes(stream):
        group_index, lane = stream >> 5, stream & 31
        base = int(records[group_index * 17])
        relative = records[group_index * 17 + 1: group_index * 17 + 17].view(np.uint16)
        begin = base + int(relative[lane])
        if lane == 31:
            end = int(records[(group_index + 1) * 17])
        else:
            end = base + int(relative[lane + 1])
        return max(0, end - begin)

    variants = ((16, 2), (12, 2), (8, 2), (4, 4), (3, 12), (2, 33))
    stored_total = 0.0
    stored_lookups = 0
    totals = {variant: [0.0, 0] for variant in variants}
    compared = 0
    for index, stream in enumerate(chosen):
        counts = block[index]
        mode = int(models[int(stream)])
        if not (64 <= mode < 96):
            continue          # only the paired tANS streams are the ones this would replace
        rate = float(counts.mean())
        stored_total += stored_bytes(int(stream)) * 8
        stored_lookups += len(counts) / 2
        compared += 1
        for variant in variants:
            bits, lookups = stream_bits(counts, rate, *variant)
            totals[variant][0] += bits
            totals[variant][1] += lookups
    print(f"{compared} paired streams, stored {stored_total / 8 / 1024:.1f} KiB, "
          f"{stored_total / (compared * len(block[0])):.3f} bits per count, "
          f"{stored_lookups:.0f} table lookups")
    print(f"{'group of counts':>22}  {'size':>7}  {'lookups':>8}  {'counts/lookup':>13}")
    for variant in variants:
        bits, lookups = totals[variant]
        counts_each = compared * len(block[0]) / max(1, lookups)
        print(f"{f'{variant[0]} of 0 to {variant[1] - 1}':>22}  "
              f"{bits / stored_total:6.3f}x  {lookups / stored_lookups:7.3f}x  {counts_each:12.2f}")


if __name__ == "__main__":
    main()
