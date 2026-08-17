from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_compact_active_redraw_avoids_sparse_logical_scan() -> None:
    """Packed active BF rows should launch exactly the stored row count."""
    from quantem.gpu.ssb.compute.mps.engine import (
        _object_redraw_storage_topology,
    )

    compact = SimpleNamespace(
        num_bf=12,
        g_qk=np.zeros((5, 1, 2), dtype=np.complex64),
        bf_storage_indices_np=np.asarray([0, 3, 4, 7, 11], dtype=np.int32),
    )
    assert _object_redraw_storage_topology(compact) == (5, False)

    noncompact = SimpleNamespace(
        num_bf=12,
        g_qk=np.zeros((12, 1, 2), dtype=np.complex64),
        bf_storage_indices_np=None,
    )
    assert _object_redraw_storage_topology(noncompact) == (12, False)

    inconsistent_legacy = SimpleNamespace(
        num_bf=12,
        g_qk=np.zeros((6, 1, 2), dtype=np.complex64),
        bf_storage_indices_np=np.asarray([0, 3, 4, 7, 11], dtype=np.int32),
    )
    assert _object_redraw_storage_topology(inconsistent_legacy) == (12, True)


def test_compact_active_redraw_preserves_logical_dc_and_parity(monkeypatch) -> None:
    """Compact active accumulation must match the logical sparse reference."""
    from quantem.gpu.ssb.compute.mps import engine as mps

    class FakeFFT:
        @staticmethod
        def ifft2(value):
            return np.fft.ifft2(value)

    class FakeMlx:
        float32 = np.float32
        int32 = np.int32
        complex64 = np.complex64
        fft = FakeFFT()
        array = staticmethod(np.asarray)
        zeros = staticmethod(np.zeros)
        sum = staticmethod(np.sum)

        @staticmethod
        def eval(*_values):
            pass

    active = np.asarray([0, 3, 4, 7, 11], dtype=np.int32)
    stored = np.zeros((5, 1, 2), dtype=np.complex64)
    stored[:, 0, 1] = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    prepared = SimpleNamespace(
        mx=FakeMlx,
        scan_shape=(1, 2),
        g_qk=stored,
        q_row=np.zeros(1, dtype=np.float32),
        q_col=np.zeros(2, dtype=np.float32),
        kx=np.zeros(5, dtype=np.float32),
        ky=np.zeros(5, dtype=np.float32),
        factor=1.0,
        dc_value=complex(6.0, -2.0),
        wavelength=1.0,
        semiangle_rad=1.0,
        ang_y_rad=1.0,
        ang_x_rad=1.0,
        num_bf=12,
        bf_storage_indices_np=active,
    )
    launches = []

    def fake_kernel(num_bf, logical_num_bf, chunk_bf, ny, nx, _cols, sparse):
        launches.append((num_bf, logical_num_bf, sparse))

        def launch(*, inputs, output_shapes, **_kwargs):
            g_qk, *_middle, storage_map = inputs
            partial = np.zeros(output_shapes[0], dtype=np.complex64)
            groups = output_shapes[0][0]
            if num_bf == logical_num_bf:
                for group in range(groups):
                    valid = min(chunk_bf, num_bf - group * chunk_bf)
                    partial[group, 0, 0] = prepared.dc_value * valid
            else:
                partial[0, 0, 0] = prepared.dc_value * logical_num_bf
            for group in range(groups):
                stop = min((group + 1) * chunk_bf, num_bf)
                for bf in range(group * chunk_bf, stop):
                    stored_bf = int(storage_map[bf]) if sparse else bf
                    if stored_bf >= 0:
                        partial[group, 0, 1] += g_qk[stored_bf, 0, 1]
            return [partial]

        return launch

    monkeypatch.setattr(mps, "_object_fourier_sum_dynamic_kernel", fake_kernel)
    monkeypatch.setattr(
        mps,
        "_pk_from_prepared",
        lambda *_args, **_kwargs: np.zeros(5, dtype=np.complex64),
    )
    compact = mps._object_fourier_sum_dynamic(
        prepared, C10=0.0, C12=0.0, phi12=0.0, chunk_bf=4
    )
    compact_launch = launches[-1]

    monkeypatch.setattr(
        mps,
        "_object_redraw_storage_topology",
        lambda candidate: (candidate.num_bf, True),
    )
    logical = mps._object_fourier_sum_dynamic(
        prepared, C10=0.0, C12=0.0, phi12=0.0, chunk_bf=4
    )

    assert compact_launch == (5, 12, False)
    assert launches[-1] == (12, 12, True)
    np.testing.assert_allclose(compact, logical, rtol=0.0, atol=1e-7)
    expected_fourier = np.asarray([[prepared.dc_value, 15.0 / 12.0]])
    np.testing.assert_allclose(
        compact,
        np.fft.ifft2(expected_fourier),
        rtol=0.0,
        atol=1e-7,
    )
