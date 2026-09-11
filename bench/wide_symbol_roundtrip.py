"""Round-trip a wider symbol alphabet: host encoder, device decoder, bit-exact or nothing.

The stored coder gives one table entry two counts. A stream whose counts are nearly all
zero or one can put sixteen counts in one entry instead, which is an eighth of the table
lookups and, at those rates, fewer bytes as well. This builds the tables, encodes counts on
the host, decodes them with a CUDA kernel, and asserts every count comes back unchanged.

Bits are written and read most significant first through a forward cursor, which is this
prototype's own convention; matching the stored layout's backward reader comes when the
format moves. Nothing here touches a stored file.

    python bench/wide_symbol_roundtrip.py --group 16 --small 2 --rate 0.08
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/owner/worktrees/sep-11-encoded-detector-speed/src")

STATES = 1024
SPREAD = 43          # inverse of the layout's spread step, as the stored tables use
FORMS = Path.home() / ".config/live4dstem/gate-forms.txt"

DECODE_SOURCE = r"""
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// One table entry carries a whole group: bits 0-31 hold `group` counts of `width` bits
// each, 32-47 the tANS rank offset, 48-51 what the state step consumes, 52 the escape flag.
extern "C" __global__ void ws_decode(const u8* payload, const u32* offsets, const u16* first_state,
                                     const u64* table, u32* out, u32 streams, u32 length,
                                     u32 group, u32 width, u32 escape_bits) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= streams) return;
    u32 cursor = offsets[s], end = offsets[s + 1], state = first_state[s], produced = 0;
    u64 buffer = 0; u32 available = 0;
    while (produced < length) {
        while (available < 48 && cursor < end) { buffer = (buffer << 8) | payload[cursor++]; available += 8; }
        u64 code = table[state];
        u32 bits = u32((code >> 48) & 15u);
        u32 low = 0;
        if (bits) { available -= bits; low = u32(buffer >> available) & ((1u << bits) - 1u); }
        state = u32((code >> 32) & 0xffffu) + low;
        if ((code >> 52) & 1u) {   // escape: this group's counts follow as literals
            for (u32 k = 0; k < group && produced < length; ++k, ++produced) {
                while (available < escape_bits && cursor < end) { buffer = (buffer << 8) | payload[cursor++]; available += 8; }
                available -= escape_bits;
                out[u64(s) * length + produced] = u32(buffer >> available) & ((1u << escape_bits) - 1u);
            }
            continue;
        }
        u32 packed = u32(code & 0xffffffffu);
        for (u32 k = 0; k < group && produced < length; ++k, ++produced)
            out[u64(s) * length + produced] = (packed >> (k * width)) & ((1u << width) - 1u);
    }
}
"""


def model_frequencies(rate, group, small):
    """Frequencies over group symbols for one Poisson mean, summing to the table's states.

    A symbol is a group of counts each below `small`, so its probability is the product of
    the conditional single-count probabilities; the leftover mass is the escape's.
    """
    counts = np.arange(64)
    single = np.exp(counts * np.log(max(rate, 1e-9))
                    - np.array([math.lgamma(k + 1) for k in counts]) - rate)
    single = single / single.sum()
    inside = single[:small]
    conditional = inside / inside.sum()
    probability = np.ones(1)
    for _ in range(group):
        probability = np.outer(probability, conditional).ravel()
    probability = probability * inside.sum() ** group
    order = np.argsort(-probability)
    support = min(STATES - 2, int((probability[order] > 1e-12).sum()))
    keep = order[:support]
    frequency = np.zeros(probability.size + 1, np.int64)   # the last entry is the escape
    budget = STATES - (support + 1)
    frequency[keep] = np.floor(probability[keep] * budget).astype(np.int64) + 1
    frequency[-1] = 1
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
    assert int(frequency.sum()) == STATES, int(frequency.sum())
    return frequency


def build_tables(frequency, group, width, escape_index):
    """Spread symbols over the states; return the decode table and each symbol's states."""
    starts = np.cumsum(frequency) - frequency
    decode = np.zeros(STATES, np.uint64)
    slots = {symbol: [] for symbol, count in enumerate(frequency) if count}
    window = {}
    for symbol, count in enumerate(frequency):
        if count:
            window[symbol] = (int(starts[symbol]), int(starts[symbol]) + int(count))
    for state in range(STATES):
        spread = (state * SPREAD) & (STATES - 1)
        symbol = next(s for s, (lo, hi) in window.items() if lo <= spread < hi)
        rank = len(slots[symbol])
        slots[symbol].append(state)
        total = int(frequency[symbol]) + rank
        bits = 10 - (total.bit_length() - 1)
        base = (total << bits) - STATES
        packed = 0
        if symbol != escape_index:
            remaining, limit = symbol, 1 << width
            for position in range(group):
                packed |= (remaining % limit) << (position * width)
                remaining //= limit
        decode[state] = (np.uint64(packed) | (np.uint64(base) << np.uint64(32))
                         | (np.uint64(bits) << np.uint64(48))
                         | (np.uint64(symbol == escape_index) << np.uint64(52)))
    return decode, slots


def encode_stream(counts, frequency, slots, group, small, escape_index, escape_bits):
    """Encode one stream backwards so the device reads it forwards.

    The encoder is the exact inverse of the table: for a state the decoder must produce, it
    finds the rank and the low bits that reconstruct it, emits those bits, and steps to the
    state that carries this symbol at that rank.
    """
    groups = [list(counts[i:i + group]) for i in range(0, len(counts), group)]
    emitted = []   # (value, bits) in reverse decode order
    state = 0
    for chunk in reversed(groups):
        codable = len(chunk) == group and all(int(v) < small for v in chunk)
        symbol = escape_index
        if codable:
            symbol, limit = 0, 1 << max(1, (small - 1).bit_length())
            for value in reversed(chunk):
                symbol = symbol * small + int(value)
            if symbol >= escape_index or int(frequency[symbol]) == 0:
                symbol, codable = escape_index, False
        if not codable:
            for value in reversed(chunk):
                emitted.append((int(value), escape_bits))
        count = int(frequency[symbol])
        target = STATES + state
        bits = 10 - (count.bit_length() - 1)
        if target < (count << bits):
            bits -= 1
        rank = (target >> bits) - count
        low = target & ((1 << bits) - 1)
        emitted.append((low, bits))
        state = slots[symbol][rank]
    stream, length = 0, 0
    for value, width_bits in reversed(emitted):
        stream = (stream << width_bits) | value
        length += width_bits
    pad = (-length) % 8
    stream <<= pad
    length += pad
    return stream.to_bytes(length // 8, "big") if length else b"", state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, default=16)
    parser.add_argument("--small", type=int, default=2)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--rate", type=float, default=0.08)
    args = parser.parse_args()

    import cupy as cp

    width = max(1, (args.small - 1).bit_length())
    escape_bits = 16
    escape_index = args.small**args.group
    rng = np.random.default_rng(7)
    counts = rng.poisson(args.rate, size=(args.streams, args.length)).astype(np.uint32)
    frequency = model_frequencies(args.rate, args.group, args.small)
    decode, slots = build_tables(frequency, args.group, width, escape_index)

    blobs, states = [], []
    for row in counts:
        data, state = encode_stream(row, frequency, slots, args.group, args.small,
                                    escape_index, escape_bits)
        blobs.append(data)
        states.append(state)
    offsets = np.zeros(args.streams + 1, np.uint32)
    offsets[1:] = np.cumsum([len(b) for b in blobs])
    payload = np.frombuffer(b"".join(blobs), np.uint8)

    module = cp.RawModule(code=DECODE_SOURCE, options=("--std=c++17",))
    kernel = module.get_function("ws_decode")
    out = cp.zeros((args.streams, args.length), cp.uint32)
    kernel(((args.streams + 127) // 128,), (128,),
           (cp.asarray(payload), cp.asarray(offsets), cp.asarray(np.array(states, np.uint16)),
            cp.asarray(decode), out, np.uint32(args.streams), np.uint32(args.length),
            np.uint32(args.group), np.uint32(width), np.uint32(escape_bits)))
    got = out.get()
    exact = np.array_equal(got, counts)
    bits_per_count = len(payload) * 8 / counts.size
    print(f"group {args.group} of counts below {args.small}, Poisson mean {args.rate}")
    print(f"  codable symbols {int((frequency[:-1] > 0).sum())} of {escape_index}, "
          f"escape frequency {int(frequency[-1])}")
    print(f"  counts at or above {args.small}: {int((counts >= args.small).sum())} of {counts.size}")
    print(f"  coded {bits_per_count:.4f} bits per count, "
          f"{1 / args.group:.4f} table lookups per count")
    print(f"  round trip exact: {exact}")
    if not exact:
        wrong = np.argwhere(got != counts)
        print(f"  first difference at {tuple(wrong[0])}: got {got[tuple(wrong[0])]} "
              f"want {counts[tuple(wrong[0])]}, {len(wrong)} differ")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
