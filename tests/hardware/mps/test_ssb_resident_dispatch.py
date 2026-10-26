"""Changing Newton batch sizes preserves the fixed-specialization GPU scores."""

import numpy as np
import pytest

from .test_ssb_thick_sample_mps import _require_mps
from .test_ssb_upsampling import _disk_session


@pytest.mark.parametrize("batch", [1, 8, 13, 28, 84])
def test_runtime_batch_matches_specialized_shader(batch, monkeypatch):
    """Search and changing Newton active sets retain every score bit."""
    _require_mps()
    from quantem.gpu.direct_ptycho._resident_search import _Device, _trial
    from quantem.gpu.direct_ptycho.mps import resident_fit

    device = _Device("mps")
    reference = device.xp.fast.metal_kernel(
        name="ssb_specialized_batch_reference",
        input_names=["G", "band_flat", "band_qx", "band_qy", "kx", "ky",
                     "trial", "scalars", "probe_geometry", "neighbour", "step_k"],
        output_names=["numer_real", "numer_imag", "denom"],
        source=resident_fit._ANGLE_SUM_SOURCE,
        header=resident_fit._THICK_FIT_BATCH_HEADER + resident_fit._SINC_HEADER,
    )

    def specialized(**kwargs):
        kwargs["inputs"] = kwargs["inputs"][:-1]
        kwargs["template"] = [*kwargs["template"], ("TOTAL", batch)]
        return reference(**kwargs)

    with _disk_session(n=32) as session:
        backend = session._backend()
        backend.cache_rotation(0.0)
        rows = np.tile([80.0, 10.0, 0.3, 2.0, -1.0, 120.0], (batch, 1))
        rows[:, 0] += np.arange(batch)
        trial = _trial(device, device.pack(rows))
        actual = device.host(resident_fit.evaluate(backend._prepared, trial))
        monkeypatch.setattr(resident_fit, "_angle_sum_kernel", lambda: specialized)
        expected = device.host(resident_fit.evaluate(backend._prepared, trial))
        np.testing.assert_array_equal(actual, expected)


def test_cached_probe_geometry_matches_trial_kernel_and_rotation():
    """Reuse exact detector geometry, rebuilding it after rotation changes."""
    _require_mps()
    from quantem.gpu.direct_ptycho._resident_search import _Device, _trial
    from quantem.gpu.direct_ptycho.mps import resident_fit

    device = _Device("mps")
    mx = device.xp
    trial = _trial(device, device.pack([[80.0, 10.0, 0.3, 2.0, -1.0, 120.0]]))
    with _disk_session(n=32) as session:
        backend = session._backend()
        previous = None
        for rotation in (0.0, 0.2, 0.0):
            backend.cache_rotation(rotation)
            prepared = backend._prepared
            assert prepared.fit_probe_geometry is None
            scalars = mx.array([
                prepared.wavelength, prepared.semiangle_rad,
                prepared.ang_y_rad, prepared.ang_x_rad, prepared.factor,
            ], dtype=mx.float32)
            count = int(prepared.kx.size)
            expected, _ = resident_fit._trial_probe_kernel()(
                inputs=[prepared.kx, prepared.ky, scalars, trial.reshape(-1)],
                template=[("NUM_BF", count), ("B", 1)],
                grid=(count, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(count, 4), (count, 1)],
                output_dtypes=[mx.float32] * 2,
            )
            actual = resident_fit._probe_geometry(prepared, scalars)
            mx.eval(actual, expected)
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            assert resident_fit._probe_geometry(prepared, scalars) is actual
            assert actual is not previous
            previous = actual

        # A separate BF selection cannot inherit geometry indexed by the full
        # detector selection, even when its source has already been fitted.
        from quantem.gpu.direct_ptycho.mps.backend import _subset_prepared

        count = int(prepared.num_bright_field_pixels) // 2
        subset = _subset_prepared(prepared, count)
        assert subset.fit_probe_geometry is None
        reduced = resident_fit._probe_geometry(subset, scalars)
        mx.eval(reduced)
        np.testing.assert_array_equal(
            np.asarray(reduced), np.asarray(actual)[::2][:count],
        )
        assert prepared.fit_probe_geometry is actual
