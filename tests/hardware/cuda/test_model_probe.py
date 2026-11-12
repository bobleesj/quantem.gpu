"""The displayed wavefront agrees with the reconstruction's GPU polynomial."""

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quantem.gpu.direct_ptycho.higher_order import CHI_SOURCE, ORDERS, pack_coefficients
from quantem.gpu.direct_ptycho.probe import model_probe, probe_grid
from quantem.gpu.direct_ptycho.results import DirectPtychoResult
from quantem.gpu.optics.physics import wavelength_A_from_kV


@pytest.mark.parametrize("term", range(14))
def test_probe_phase_matches_reconstruction_polynomial(term):
    n, m = ORDERS[term]
    wavelength = wavelength_A_from_kV(300)
    magnitude_A = wavelength * 0.2 / 0.03 ** (n + 1)
    azimuth = 0.37
    result = DirectPtychoResult(
        object_wave=cp.ones((2, 2), cp.complex64), backend="cuda",
        voltage_kV=300, semiangle_mrad=30,
        aberrations={f"C{n}{m}_nm": magnitude_A / 10,
                     f"phi{n}{m}_deg": math.degrees(azimuth)},
    )
    wave = model_probe(result, space="fourier")
    shape, _, steps = probe_grid(result)
    rows = cp.broadcast_to((cp.arange(shape[0], dtype=cp.float32) - shape[0] // 2)[:, None]
                           * steps[0] * 1e-3 / wavelength, shape).copy()
    cols = cp.broadcast_to((cp.arange(shape[1], dtype=cp.float32) - shape[1] // 2)[None, :]
                           * steps[1] * 1e-3 / wavelength, shape).copy()
    magnitudes = np.zeros(14, np.float32)
    angles = np.zeros(14, np.float32)
    magnitudes[term], angles[term] = magnitude_A, azimuth
    coefficients = cp.asarray(pack_coefficients(magnitudes, angles))
    source = ("#define DIRECT_DEVICE __device__\n#define DIRECT_COEFFICIENTS const float*\n"
              + CHI_SOURCE + r"""
    extern "C" __global__ void phase(const float* r, const float* c, float wl,
                                     const float* coefficients, float* out, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) out[i] = direct_chi_full(r[i], c[i], wl, coefficients);
    }
    """)
    kernel = cp.RawKernel(source, "phase")
    chi = cp.empty(shape, cp.float32)
    kernel(((chi.size + 255) // 256,), (256,),
           (rows, cols, np.float32(wavelength), coefficients, chi, np.int32(chi.size)))
    mask = cp.abs(wave) > cp.max(cp.abs(wave)) * 0.99
    cp.testing.assert_allclose(wave[mask] / cp.abs(wave[mask]),
                               cp.exp(-1j * chi[mask]), atol=2e-6, rtol=2e-6)
    assert float(cp.max(cp.abs(cp.angle(wave[mask])))) > 0.05
    assert float(cp.sum(cp.abs(wave) ** 2)) == pytest.approx(1, abs=1e-6)


def test_live_probe_follows_optics_but_not_damping():
    import json
    from .test_direct_thick_sample import _synthetic_session

    session, _ = _synthetic_session()
    viewer = session.show("regularization", show_probe=True, compute_loss=False)
    try:
        initial_probe = viewer.probe._data.copy()
        initial_phase = viewer.reconstruction.phase_bytes
        # the browser sends slider requests as a "direct-preview" message, not a trait
        request = dict(id=1, C10_nm=5, C12_nm=0, phi12_deg=0, committed=True)
        viewer.reconstruction._on_preview_message(viewer.reconstruction, {"type": "direct-preview", "request": request}, [])
        assert not np.array_equal(initial_probe[0], viewer.probe._data[0])
        assert viewer.reconstruction.phase_bytes != initial_phase
        changed_probe = viewer.probe._data.copy()
        changed_phase = viewer.reconstruction.phase_bytes
        viewer.reconstruction.phase_damp = 0.3
        np.testing.assert_array_equal(viewer.probe._data, changed_probe)
        assert viewer.reconstruction.phase_bytes != changed_phase
        viewer.reconstruction.higher_order_aberrations_json = json.dumps({"C30_nm": 10000})
        assert not np.array_equal(changed_probe[0], viewer.probe._data[0])
    finally:
        viewer.close()
        viewer.close()
        session.close()
