"""Portable object and probe waves from a synthetic reconstruction."""

import numpy as np

from quantem.gpu import DirectPtycho
from quantem.gpu.direct_ptycho.archive import SavedDirectPtychoResult, save_review


def test_reviewed_waves_round_trip(tmp_path):
    wave = np.exp(1j * np.arange(64).reshape(8, 8) / 64)
    reference = SavedDirectPtychoResult(object_wave=wave, backend="cuda", aberrations={"C10_nm": 1.5},
                                       object_sampling_A=(0.2, 0.3), upsample=2)
    reference.real_probe = wave[::-1].copy()
    reference.real_probe_sampling_A = (0.1, 0.2)
    reference.fourier_probe_sampling_mrad = (1.0, 2.0)
    save_review(reference, tmp_path)
    saved = DirectPtycho.load(tmp_path)
    np.testing.assert_array_equal(saved.object, wave)
    np.testing.assert_array_equal(saved.probe("real"), reference.real_probe)
    np.testing.assert_allclose(saved.probe("fourier"), reference.probe(space="fourier"))
    np.testing.assert_allclose(saved.object_sampling_A, (0.2, 0.3))
    assert saved.aberrations == reference.aberrations
