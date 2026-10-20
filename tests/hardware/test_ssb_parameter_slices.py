"""Review saved joint-fit diagnostics without changing the reconstructed phase."""

import platform

import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_measure_show_and_reopen_joint_parameters(tmp_path, backend):
    if backend == "mps":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            pytest.skip("Requires an Apple silicon GPU")
        pytest.importorskip("mlx.core")
    else:
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("Requires a CUDA GPU")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("Requires a CUDA runtime")
    import matplotlib.pyplot as plt

    from quantem.gpu import SSB
    from quantem.gpu.ssb.persistence import load_result, result_paths
    from quantem.gpu.ssb.results import host_array

    counts = np.random.default_rng(7).poisson(40, size=(128, 128, 32, 32)).astype(np.uint16)
    rows, columns = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rows, columns) < 10] += 200
    source = tmp_path / "counts.npy"
    np.save(source, counts)
    with SSB(
        counts,
        backend=backend,
        voltage_kV=300,
        semiangle_mrad=30,
        scan_sampling_A=(0.3, 0.3),
        detector_sampling_mrad=(3.0, 3.0),
        source_path=str(source),
    ) as ssb:
        fit = ssb.find_aberrations(
            tilt=True,
            num_trials=8,
            refinement=None,
            compare_standard=False,
            check_rotation=False,
            verbose=False,
            save_dir=tmp_path,
        )
        phase = host_array(fit.phase).copy()
        aberrations = dict(fit.aberrations)
        evidence = ssb.measure_parameters(fit)
        np.testing.assert_array_equal(host_array(fit.phase), phase)
        assert fit.aberrations == aberrations
        for data in evidence["slices"].values():
            column = data["column_values"].index(data["fitted_point"][0])
            row = data["row_values"].index(data["fitted_point"][1])
            np.testing.assert_allclose(data["relative_agreement"][row][column], 1.0, rtol=1e-6)
        figure = fit.show()
        figure.savefig(tmp_path / "parameters.png", dpi=100)
        assert figure.number not in plt.get_fignums()
        restored = load_result(
            paths=result_paths(tmp_path, "find_aberrations"), signature=fit.metadata["signature"], backend=backend
        )
        assert restored.parameter_slices == evidence
        np.testing.assert_array_equal(host_array(restored.phase), phase)
        assert len(restored.show("parameters").axes) == 3
        assert len(restored.show(show_parameters=False).axes) == 2
