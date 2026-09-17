#!/usr/bin/env python3
"""Render the candidate-independence table from one or more hoist probe reports.

usage: hoist-table.py <report.json> [<report.json> ...]
"""
import json
import sys

PLANE = 512 * 512
HALF_PLANE = 512 * 257
BF = 8937
BATCH = 8
BATCHES = -(-BF // BATCH)
HALF_BYTES = HALF_PLANE * 8


def cpu_digest(words):
    low = high = 0
    start = 0
    while start < len(words):
        end = min(start + 16, len(words))
        h = 1469598103934665603
        for i in range(start, end):
            h ^= words[i]
            h = (h * 1099511628211) & ((1 << 64) - 1)
            h ^= i + 1
        low = (low + h) & 0xFFFFFFFF
        high = (high + (h >> 32)) & 0xFFFFFFFF
        start += 16
    return low | (high << 32)


# bytes the pipeline moves per evaluation, and whether the write is per batch
SIZES = {
    "Gchunk": (BF * HALF_BYTES, "read once per evaluation, produced by prepare()"),
    "columnPassOutput": (BF * HALF_BYTES, f"{BATCH * HALF_BYTES} B x {BATCHES} batches, written and read"),
    "nyquistCorrection": (BATCHES * BATCH * 2 * 512 * 8, f"written and read"),
    "chiTrig": (PLANE * 8, "written once per evaluation"),
    "chiTrigRepeat": (0, "control: repeat digest of chiTrig"),
    "crossTrig": (BF * 2 * 512 * 8, "written once per evaluation"),
    "bfGeometry": (BF * 16, "written only when the rotation changes"),
    "phaseSum": (PLANE * 4, "read-modify-write accumulator"),
    "phaseSumSquared": (PLANE * 4, "read-modify-write accumulator"),
    "twiddle": (512 * 8, "constant"),
    "qRow": (512 * 4, "constant"),
    "qCol": (512 * 4, "constant"),
    "knownAnswer": (0, "control: digest kernel known answer"),
}


def group(name):
    return "Gchunk" if name.startswith("Gchunk") else name


def main():
    reports = [json.load(open(path)) for path in sys.argv[1:]]
    names = sorted(reports[0]["digests"]["candidateA"])
    by_group = {}
    for name in names:
        by_group.setdefault(group(name), []).append(name)

    print(f"{'buffer':20s} {'slots':>5s} {'bytes/eval':>15s} {'differing':>9s} {'identical':>10s}  verdict")
    totals = {"independent": 0, "dependent": 0}
    for key, members in by_group.items():
        differing = same_zero = total_slots = 0
        for report in reports:
            for member in members:
                a = report["digests"]["candidateA"][member]
                b = report["digests"]["candidateB"][member][len(a):]
                total_slots += len(a)
                differing += sum(1 for x, y in zip(a, b) if x != y)
                zero = cpu_digest([0] * (BATCH * 2 * 512 * 8 // 4))
                same_zero += sum(1 for x, y in zip(a, b) if x == y == zero)
        bytes_per_eval, note = SIZES.get(key, (0, ""))
        verdict = "candidate-independent" if differing == 0 else "candidate-dependent"
        totals["independent" if differing == 0 else "dependent"] += bytes_per_eval
        label = f"{key} (x{len(members)})" if len(members) > 1 else key
        print(f"{label:20s} {total_slots // len(reports):5d} {bytes_per_eval:15,d} "
              f"{differing // len(reports):9d} {str(differing == 0):>10s}  {verdict}")
        if note:
            print(f"{'':20s} {'':>5s} {note}")
        if key == "nyquistCorrection" and differing:
            print(f"{'':20s} {'':>5s} {same_zero // len(reports)} of {total_slots // len(reports)} "
                  f"batches are an all-zero correction in both runs (digest {cpu_digest([0] * (BATCH * 2 * 512 * 8 // 4)):#018x})")
    print()
    # Device traffic per evaluation, split by whether the bytes can be reused
    # across candidates. Every dependent buffer is written and read once.
    traffic = [
        ("G half-plane read", BF * HALF_BYTES, True),
        ("column-pass output write", BF * HALF_BYTES, False),
        ("column-pass output read", BF * HALF_BYTES, False),
        ("chi table write+read", 2 * PLANE * 8, False),
        ("cross table write+read", 2 * BF * 2 * 512 * 8, False),
        ("Nyquist correction write+read", 2 * BATCHES * BATCH * 2 * 512 * 8, False),
        ("phase moments write+read", 4 * PLANE * 4, False),
    ]
    independent = sum(b for _, b, flag in traffic if flag)
    dependent = sum(b for _, b, flag in traffic if not flag)
    for label, value, flag in traffic:
        print(f"  {label:32s} {value:15,d} B  "
              f"{'already produced once per session' if flag else 'recomputed per candidate'}")
    grand = independent + dependent
    print(f"  {'candidate-independent':32s} {independent:15,d} B  {independent / grand * 100:.1f}%")
    print(f"  {'candidate-dependent':32s} {dependent:15,d} B  {dependent / grand * 100:.1f}%")
    print(f"  {'total per evaluation':32s} {grand:15,d} B  ({grand / 1e9:.3f} GB)")
    print(f"  hoistable saving: 0 B per evaluation — the {independent / 1e9:.3f} GB that is "
          f"candidate-independent is a read of the cache prepare() already built")
    for report in reports:
        print("process: losses=%s prepare=%.3fs known_answer_match=%s mutated_detected=%s"
              % (report["losses"], report["prepare_seconds"],
                 report["known_answer"]["match"], report["known_answer"]["mutated_detected"]))


if __name__ == "__main__":
    main()
