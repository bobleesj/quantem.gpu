"""Frozen parity checks for the standalone paired runtime tANS oracle."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).parent))

from paired_runtime_ans_oracle import (  # noqa: E402
    INTERVAL,
    EncodedStream,
    InvalidStream,
    build_frequencies,
    build_tables,
    decode_stream,
    encode_stream,
    run_oracle,
)


def test_frozen_tables_and_exact_fixtures() -> None:
    """Freeze deterministic tables and require exact paired transition count."""

    result = run_oracle()
    assert result["frequency_sha256"] == "74a39b914a3d563cec122294bf3d29c4972e089ae64ae42c6500eaaafde4145f"
    assert result["encoding_sha256"] == "b76355bf9e4d4c9bf4949faf412671d7ec881b232df7a303250a5071342253d8"
    assert result["decoding_sha256"] == "44966937a0352082f194c121b37cbd0dd6883cf65919f2d73f94a893c303becd"
    assert result["transition_cases"] == 2_572_288
    entropy_rows = [row for row in result["round_trip_cases"] if row["paired_transitions"]]
    assert entropy_rows
    assert all(row["paired_transitions"] == 256 for row in entropy_rows)
    assert all(row["scalar_transition_reference"] == 512 for row in entropy_rows)


def test_randomized_uint8_and_uint16_round_trips() -> None:
    """Exercise exact randomized streams, including wide uint16 escapes."""

    frequencies = build_frequencies()
    encoding, decoding = build_tables(frequencies)
    rng = np.random.default_rng(20260912)
    for dtype, high in ((np.uint8, 256), (np.uint16, 65536)):
        for _ in range(32):
            values = rng.integers(0, high, INTERVAL, dtype=dtype)
            encoded = encode_stream(values, frequencies, encoding)
            actual = decode_stream(encoded, INTERVAL, values.dtype, decoding)
            np.testing.assert_array_equal(actual, values)


def test_invalid_sparse_stream_is_rejected() -> None:
    """Reject duplicate sparse positions rather than silently overwriting."""

    _, decoding = build_tables(build_frequencies())
    duplicate = ((7 << 7) | 2).to_bytes(2, "little") * 2
    with np.testing.assert_raises(InvalidStream):
        decode_stream(EncodedStream(252, duplicate, 0), INTERVAL, np.dtype("uint16"), decoding)
