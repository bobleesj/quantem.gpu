"""Corrected packed counts participate in complete-detector summaries."""
import os

import pytest


def test_real_corrected_packed_and_encoded_summaries_match():
    """Compare full summaries without host scientific arithmetic."""
    path = os.environ.get("MAPED_PARITY_FILE")
    if not path:
        pytest.skip("Set MAPED_PARITY_FILE to a masked count HDF5 acquisition.")
    import torch
    from quantem.gpu import detector, io

    sources = []
    try:
        for representation in ("encoded", "packed"):
            sources.append(io.load(
                path, backend="cuda", representation=representation,
                dtype="native", apply_mask=False, verbose=False,
            ))
        assert all(s.metadata["hot_pixel_correction"]["applied"] for s in sources)
        session = detector.prepare(sources)
        means = torch.from_dlpack(session.mean_dp(output="native"))
        assert means.device.type == "cuda"
        assert torch.equal(means[0], means[1])
    finally:
        for source in sources:
            source.close()
