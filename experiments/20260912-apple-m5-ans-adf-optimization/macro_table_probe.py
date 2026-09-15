#!/usr/bin/env python3
"""CPU census for a three-pair, four-lookahead-bit paired-tANS macro table."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "experiments/20260912-metal-paired-runtime-ans-prototype/paired_runtime_ans_oracle.py"
spec = importlib.util.spec_from_file_location("paired_runtime_ans_oracle", ORACLE)
assert spec and spec.loader
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)

MODELS = 32
STATES = 1024
LOOKAHEADS = 16
MAX_PAIRS = 3
ESCAPE = 4095


def scalar_prefix(decoding: np.ndarray, model: int, state: int, nibble: int,
                  limit: int, remaining: int = 4) -> tuple[tuple[int, ...], int, int]:
    """Decode ordinary transitions while they fit in the available prefix."""
    pairs: list[int] = []
    consumed = 0
    while len(pairs) < limit:
        code = int(decoding[model, state])
        pair, bits, base = code & 4095, (code >> 12) & 15, code >> 16
        if pair == ESCAPE or consumed + bits > remaining:
            break
        low = 0 if bits == 0 else (nibble >> (4 - consumed - bits)) & ((1 << bits) - 1)
        state = base + low
        pairs.append(pair)
        consumed += bits
    return tuple(pairs), state, consumed


def pack_entry(pairs: tuple[int, ...], state: int, consumed: int) -> np.uint64:
    """Pack 3x12-bit pairs, 10-bit state, 3-bit consumed and 2-bit count."""
    value = 0
    for index, pair in enumerate(pairs):
        value |= pair << (12 * index)
    value |= state << 36
    value |= consumed << 46
    value |= len(pairs) << 49
    return np.uint64(value)


def unpack_entry(value: np.uint64) -> tuple[tuple[int, ...], int, int]:
    word = int(value)
    count = (word >> 49) & 3
    pairs = tuple((word >> (12 * index)) & 4095 for index in range(count))
    return pairs, (word >> 36) & 1023, (word >> 46) & 7


def build_macros(decoding: np.ndarray) -> np.ndarray:
    macros = np.empty((MODELS, STATES, LOOKAHEADS), dtype="<u8")
    for model in range(MODELS):
        for state in range(STATES):
            for nibble in range(LOOKAHEADS):
                macros[model, state, nibble] = pack_entry(
                    *scalar_prefix(decoding, model, state, nibble, MAX_PAIRS)
                )
    return macros


def validate(decoding: np.ndarray, macros: np.ndarray) -> int:
    checked = 0
    for model in range(MODELS):
        for state in range(STATES):
            for nibble in range(LOOKAHEADS):
                expected = scalar_prefix(decoding, model, state, nibble, MAX_PAIRS)
                assert unpack_entry(macros[model, state, nibble]) == expected
                checked += 1
                # At a short tail, the runtime may use the entry only if its complete
                # bit consumption fits; otherwise it takes the checked scalar path.
                for remaining in range(5):
                    if nibble & ((1 << (4 - remaining)) - 1):
                        continue  # padding below the meaningful prefix must be zero
                    actual = unpack_entry(macros[model, state, nibble])
                    chosen = actual if actual[2] <= remaining else scalar_prefix(
                        decoding, model, state, nibble, MAX_PAIRS, remaining
                    )
                    assert chosen == scalar_prefix(
                        decoding, model, state, nibble, MAX_PAIRS, remaining
                    )
                    checked += 1
    return checked


def stationary_states(decoding: np.ndarray, model: int) -> np.ndarray:
    """Stationary state weights assuming equiprobable renormalization bit strings."""
    probability = np.full(STATES, 1 / STATES)
    for _ in range(20_000):
        following = np.zeros(STATES)
        for state, weight in enumerate(probability):
            code = int(decoding[model, state])
            bits, base = (code >> 12) & 15, code >> 16
            following[base : base + (1 << bits)] += weight / (1 << bits)
        if np.max(np.abs(following - probability)) < 1e-15:
            break
        probability = following
    assert abs(float(probability.sum()) - 1) < 1e-12
    return probability


def census(decoding: np.ndarray, macros: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for model in range(MODELS):
        counts = np.empty((STATES, LOOKAHEADS), np.uint8)
        consumed = np.empty_like(counts)
        for state in range(STATES):
            for nibble in range(LOOKAHEADS):
                pairs, _, bits = unpack_entry(macros[model, state, nibble])
                counts[state, nibble] = len(pairs)
                consumed[state, nibble] = bits
        stationary = stationary_states(decoding, model)
        stationary_count = float(np.sum(counts.mean(axis=1) * stationary))
        rows.append({
            "model": model,
            "mode": model + 64,
            "uniform_average_pairs": float(counts.mean()),
            "stationary_average_pairs": stationary_count,
            "stationary_macro_lookup_reduction": stationary_count,
            "entry_count_0_1_2_3": np.bincount(counts.ravel(), minlength=4).tolist(),
            "uniform_average_bits_consumed": float(consumed.mean()),
        })
    return rows


def main() -> None:
    frequencies = oracle.build_frequencies()
    _, decoding = oracle.build_tables(frequencies)
    macros = build_macros(decoding)
    interleaved = np.empty(MODELS * (STATES + STATES * LOOKAHEADS * 2), dtype="<u4")
    cursor = 0
    macro_words = macros.view("<u4").reshape(MODELS, STATES * LOOKAHEADS * 2)
    for model in range(MODELS):
        interleaved[cursor : cursor + STATES] = decoding[model]
        cursor += STATES
        interleaved[cursor : cursor + STATES * LOOKAHEADS * 2] = macro_words[model]
        cursor += STATES * LOOKAHEADS * 2
    checked = validate(decoding, macros)
    rows = census(decoding, macros)
    hot = [row for row in rows if 69 <= row["mode"] <= 90]
    core = [row for row in rows if 75 <= row["mode"] <= 80]
    ordinary_bytes = MODELS * STATES * 4
    macro_bytes = macros.nbytes
    report = {
        "layout": {
            "per_model": "ordinary[1024] uint32, then macro[1024][16] uint64",
            "ordinary_bytes": ordinary_bytes,
            "macro_bytes": macro_bytes,
            "total_bytes": ordinary_bytes + macro_bytes,
            "total_mib": (ordinary_bytes + macro_bytes) / (1024 * 1024),
            "macro_sha256": hashlib.sha256(macros.tobytes()).hexdigest(),
            "interleaved_sha256": hashlib.sha256(interleaved.tobytes()).hexdigest(),
        },
        "validation_cases": checked,
        "hot_modes_69_90_stationary_average_pairs_range": [
            min(row["stationary_average_pairs"] for row in hot),
            max(row["stationary_average_pairs"] for row in hot),
        ],
        "core_modes_75_80_stationary_average_pairs_range": [
            min(row["stationary_average_pairs"] for row in core),
            max(row["stationary_average_pairs"] for row in core),
        ],
        "models": rows,
        "stationary_assumption": "equiprobable renormalization bit strings; actual ADF weighting still requires a state/nibble trace",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
