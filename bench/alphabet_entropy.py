"""Would a four-count symbol cost bytes? Measure it on the real count distributions.

The residual decode is 93% of a detector query and is bound by the serial chain through
`table[state]`, one lookup per coded pair. A symbol covering four counts instead of two
halves both the lookups and the chain length. The only thing that can kill it is size: 70
acquisitions already take 91.7 of 96 GiB, so an alphabet that compresses worse is unusable.

This decodes real streams, then compares the order-0 entropy of the stored pair alphabet
against a quad alphabet on the same counts. Entropy is the floor a tANS coder approaches,
so the comparison is the right one to make before writing any kernel.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/owner/worktrees/sep-11-encoded-detector-speed/src")

FORMS = Path.home() / ".config/live4dstem/gate-forms.txt"
SMALL = 33          # counts 0..32 ride inside a pair symbol, as the stored format does
ESCAPE_BITS = 13    # what the stored escape costs for one out-of-range pair


def entropy_bits(counter):
    """Order-0 entropy of a symbol stream, in bits."""
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum(n * math.log2(n / total) for n in counter.values() if n)


def stream_bits(counts, group, small, states=1024):
    """Bits to code one stream's counts in symbols of `group` counts each.

    The stored format codes each stream on its own with one model, so entropy is taken per
    stream and summed, not pooled across the detector.

    A symbol holds its group when every count is below `small`. Two things push a group onto
    the escape path: a count at or above `small`, and a symbol too rare to earn one of the
    model's `states` table slots, since a tANS symbol with zero frequency cannot be coded.
    Escaped groups pay one escape symbol plus a literal per count, charged their own
    entropy, so a hot pixel is not made to look artificially cheap.
    """
    usable = counts[: counts.size // group * group].reshape(-1, group)
    inside = (usable < small).all(axis=1)
    keys = np.zeros(usable.shape[0], np.int64)
    for column in range(group):
        keys = keys * small + usable[:, column].astype(np.int64)
    frequency = Counter(int(k) for k in keys[inside])
    # The table has `states` slots and the escape needs one, so only the commonest symbols
    # are codable; everything else joins the escape.
    codable = {key for key, _ in frequency.most_common(states - 1)}
    symbols = Counter({key: n for key, n in frequency.items() if key in codable})
    escaped = ~inside
    if len(codable) < len(frequency):
        escaped = escaped | np.isin(keys, np.fromiter(set(frequency) - codable, np.int64, len(frequency) - len(codable)))
    escapes = int(escaped.sum())
    if escapes:
        symbols[-1] = escapes
    literals = Counter(int(v) for v in usable[escaped].ravel())
    return entropy_bits(symbols) + entropy_bits(literals), usable.size, escapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", type=int, default=1)
    parser.add_argument("--pixels", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--ring", action="store_true")
    parser.add_argument("--inner", type=float, default=40.0)
    parser.add_argument("--outer", type=float, default=92.0)
    args = parser.parse_args()

    import cupy as cp
    from quantem.gpu.io._paired import load_paired_file

    # A model's table has 1024 states, so the alphabet has to stay near the stored 1089
    # symbols. Quads therefore cover only small counts and escape the rest.
    variants = (("pairs 33", 2, 33), ("triples 12", 3, 12), ("quads 7", 4, 7),
                ("quints 5", 5, 5), ("sextets 3", 6, 3), ("sextets 4", 6, 4),
                ("octets 2", 8, 2), ("octets 3", 8, 3), ("ten 2", 10, 2),
                ("twelve 2", 12, 2), ("sixteen 2", 16, 2), ("twelve 3", 12, 3))

    paths = [line.strip() for line in FORMS.read_text().splitlines() if line.strip()][: args.acquisitions]
    rows = []
    per_stream = []
    for path in paths:
        item = load_paired_file(path, device=None, verbose=False)
        source = getattr(item, "data", item)
        detector = int(np.prod(source.shape[2:]))
        # Sample pixels across the whole detector so both bright and dark streams are in.
        if args.ring:
            # The pixels a query actually decodes are the annulus edges the index groups
            # cannot cover, not the whole detector.
            shape = tuple(int(v) for v in source.shape[2:])
            centre = [(v - 1) / 2 for v in shape]
            row, col = np.indices(shape)
            radius = np.hypot(row - centre[0], col - centre[1])
            edge = (np.abs(radius - args.inner) < 1.0) | (np.abs(radius - args.outer) < 1.0)
            candidates = np.flatnonzero(edge.ravel())
            chosen = candidates[np.linspace(0, candidates.size - 1, min(args.pixels, candidates.size)).astype(int)]
        else:
            chosen = np.linspace(0, detector - 1, args.pixels).astype(int)
        scans = args.blocks * source.interval
        frames = source.decode_blocks(0, scans)
        block = cp.asarray(frames).reshape(scans, -1)[:, cp.asarray(chosen)].get()
        for label, group, small in variants:
            bits = counted = escapes = 0
            for stream in block.T:   # one row is one detector pixel across the scans
                stream_total, size, escaped = stream_bits(stream, group, small)
                bits += stream_total
                counted += size
                escapes += escaped
            rows.append(dict(file=Path(path).name, alphabet=label, group=group, small=small,
                             symbols=small ** group + 1,
                             bits_per_count=round(bits / counted, 4), counts=int(counted),
                             escape_groups=escapes))
        # The stored format already selects a mode per stream, so each stream may choose the
        # group size that codes it smallest: dark streams take long groups, bright ones stay
        # on pairs. That is one extra mode byte value, not a new container.
        best_bits = best_counts = 0
        lookups = 0
        picked = Counter()
        for stream in block.T:
            options = []
            for label, group, small in variants:
                stream_total, size, _ = stream_bits(stream, group, small)
                options.append((stream_total, group, label, size))
            stream_total, group, label, size = min(options)
            best_bits += stream_total
            best_counts += size
            lookups += size / group
            picked[label] += 1
        per_stream.append(dict(file=Path(path).name,
                               bits_per_count=round(best_bits / best_counts, 4),
                               counts_per_lookup=round(best_counts / lookups, 3),
                               chosen=dict(picked.most_common())))

    for label, _, _ in variants:
        chosen = [r for r in rows if r["alphabet"] == label]
        bits = float(np.mean([r["bits_per_count"] for r in chosen]))
        base = float(np.mean([r["bits_per_count"] for r in rows if r["alphabet"] == "pairs 33"]))
        print(f"{label:12s} symbols {chosen[0]['symbols']:>8}  "
              f"{bits:.4f} bits/count  {bits / base:.3f}x stored size  "
              f"escape groups {int(np.mean([r['escape_groups'] for r in chosen]))}")
    base = float(np.mean([r["bits_per_count"] for r in rows if r["alphabet"] == "pairs 33"]))
    bits = float(np.mean([r["bits_per_count"] for r in per_stream]))
    lookups = float(np.mean([r["counts_per_lookup"] for r in per_stream]))
    print(f"\nper-stream choice: {bits:.4f} bits/count  {bits / base:.3f}x stored size  "
          f"{lookups:.2f} counts per table lookup against 2.00 stored "
          f"({lookups / 2:.2f}x fewer lookups)")
    print("chosen:", json.dumps(per_stream[0]["chosen"]))


if __name__ == "__main__":
    main()
