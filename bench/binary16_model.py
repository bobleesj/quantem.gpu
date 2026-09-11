"""Real coded size for a 16-count binary symbol, under the same kind of model family.

The stored coder fits 32 Poisson means and gives each the best of 12 alphabet supports, so
its symbols are independent counts. A symbol covering sixteen counts of 0 or 1 saves table
lookups; whether it also saves bytes depends on whether the model can pay for its alphabet.
This builds the frequencies the same way for both alphabets and codes real streams with
them, so the size comparison is against a coder that could actually be built.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/owner/worktrees/sep-11-encoded-detector-speed/src")

FORMS = Path.home() / ".config/live4dstem/gate-forms.txt"
STATES = 1024
MODELS = 32
SUPPORTS = (4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768)


def quantised_bits(probability, escape_bits):
    """Bits per symbol after giving the commonest symbols the table's states.

    A tANS symbol needs at least one of the model's states, so only the commonest are
    codable; the rest share one escape symbol and pay a literal. This returns the best cost
    over the same alphabet supports the stored coder tries.
    """
    best = None
    order = np.argsort(-probability)
    for support in SUPPORTS:
        keep = order[:support]
        mass = probability[keep].sum()
        escape = max(0.0, 1.0 - mass)
        allocation = np.zeros_like(probability)
        allocation[keep] = probability[keep]
        frequency = np.zeros(probability.size, np.int64)
        budget = STATES - (support + 1)
        frequency[keep] = np.floor(allocation[keep] * budget).astype(np.int64) + 1
        escape_frequency = max(1, int(round(escape * budget)))
        spare = STATES - frequency.sum() - escape_frequency
        if spare > 0:
            frequency[keep[: min(spare, keep.size)]] += 1
        total = frequency.sum() + escape_frequency
        if total > STATES:
            continue
        bits = sum(probability[s] * math.log2(STATES / frequency[s])
                   for s in keep if frequency[s] > 0 and probability[s] > 0)
        bits += escape * (math.log2(STATES / escape_frequency) + escape_bits)
        best = bits if best is None else min(best, bits)
    return best


def pair_bits(rate):
    """Bits per count for the stored pair alphabet at one Poisson mean."""
    counts = np.arange(33)
    single = np.exp(counts * np.log(max(rate, 1e-9)) - np.array(
        [math.lgamma(k + 1) for k in counts]) - rate)
    single = single / single.sum()
    joint = np.outer(single, single).ravel()
    return quantised_bits(joint, escape_bits=13) / 2


def group_bits(rate, group, small):
    """Bits per count for a symbol covering `group` counts, each below `small`.

    A group is codable only when every count in it is below `small`; otherwise it escapes
    and pays a literal for each of its counts at the single-count entropy, which is what
    stops a long group from looking free on a bright stream.
    """
    counts = np.arange(64)
    single = np.exp(counts * np.log(max(rate, 1e-9)) - np.array(
        [math.lgamma(k + 1) for k in counts]) - rate)
    single = single / single.sum()
    inside = single[:small]
    mass = inside.sum()
    if mass <= 0:
        return math.inf
    codable = mass**group                       # every count in the group is representable
    literal = -sum(p * math.log2(p) for p in single if p > 0)
    # The alphabet is the product distribution over the group, conditioned on all being
    # representable. Its symbols number small ** group, so only the commonest earn states.
    shape = (small,) * group
    probability = np.ones(1)
    for _ in range(group):
        probability = np.outer(probability, inside / mass).ravel()
    bits = quantised_bits(probability, escape_bits=group * literal)
    if bits is None:
        return math.inf
    # Groups that are not codable at all pay the escape plus their literals.
    return (codable * bits + (1 - codable) * (group * literal + 2)) / group


def binary16_bits(rate):
    return group_bits(rate, 16, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="*",
                        default=[0.002, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0])
    args = parser.parse_args()
    variants = (("pairs 33", 2, 33), ("triples 12", 3, 12), ("twelve 3", 12, 3),
                ("sixteen 2", 16, 2))
    rows = []
    for rate in args.rates:
        entry = dict(mean_count=rate)
        base = None
        for label, group, small in variants:
            if small**group > 4_000_000:
                continue
            bits = group_bits(rate, group, small)
            base = bits if label == "pairs 33" else base
            entry[label] = dict(bits_per_count=round(bits, 4),
                                size=round(bits / base, 3) if base else None,
                                counts_per_lookup=group)
        rows.append(entry)
    print(json.dumps(rows, indent=1))
    print("\nmean  " + "  ".join(f"{label:>22}" for label, _, _ in
                                 (("pairs 33", 2, 33), ("triples 12", 3, 12), ("sixteen 2", 16, 2))))
    for row in rows:
        cells = []
        for label in ("pairs 33", "triples 12", "sixteen 2"):
            cell = row.get(label)
            cells.append(f"{cell['bits_per_count']:8.4f} ({cell['size']:.2f}x)" if cell else " " * 22)
        print(f"{row['mean_count']:<6}" + "  ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
