"""Review saved joint-fit diagnostics without changing the reconstructed phase."""

import platform

import numpy as np
import pytest


def _record_static_arrays(monkeypatch):
    """Inspect reconstructed values before the standard plotter colormaps them."""
    from quantem.core import visualization

    arrays = []
    original = visualization.show_2d

    def show(values, **options):
        arrays.append(values)
        return original(values, **options)

    monkeypatch.setattr(visualization, "show_2d", show)
    return arrays


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_measure_show_and_reopen_joint_parameters(tmp_path, backend, monkeypatch):
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
        # Review one diagnostic at a time without another GPU evaluation.
        ssb.measure_parameters = lambda *args: pytest.fail("Recorded slices must be reused")
        depth = ssb.show("depth", size="small")
        tilt = restored.show("tilt", size="large")
        assert np.all(depth.get_size_inches() < tilt.get_size_inches())
        assert len(depth.axes) == len(tilt.axes) == 2
        for figure, key in ((depth, "focus_depth"), (tilt, "tilt")):
            np.testing.assert_array_equal(
                figure.axes[0].collections[0].get_array(), evidence["slices"][key]["relative_agreement"])
            np.testing.assert_array_equal(
                figure.axes[0].lines[0].get_xydata()[0], evidence["slices"][key]["fitted_point"])
            assert figure.number not in plt.get_fignums()

        # Image comparisons keep every fitted correction, including nonzero tilt/depth.
        shown = ssb.reconstruct(fit, upsampling_factor="auto", compute_loss=False)
        # Matplotlib consumes host images even when both fitted waves stay on CUDA.
        from quantem.gpu.ssb.rotation_defocus import RotationDefocusComparison

        comparison = RotationDefocusComparison((fit,) * 4, (shown,) * 4, ((-30, 0), (0, 30)))
        review = comparison.show(view_fraction=1)
        expected = host_array(shown.phase)
        np.testing.assert_array_equal(
            review.axes[0].images[0].get_array(), expected - np.median(expected)
        )
        captured = _record_static_arrays(monkeypatch)
        focus = ssb.show("focus", mode="static", focus_offsets_nm=(0, 5),
                         depth_spread_nm=fit.depth_spread_nm, size="small")
        assert focus.number not in plt.get_fignums()
        images = captured[-1][0]
        np.testing.assert_allclose(images[0], host_array(shown.phase), atol=1e-6, rtol=0)
        assert not np.allclose(images[0], images[1], atol=1e-6, rtol=0)
        assert ssb.sample_tilt_mrad == fit.sample_tilt_mrad
        assert ssb.depth_spread_nm == fit.depth_spread_nm
        assert ssb.aberrations == fit.aberrations


def test_show_selected_region_preserves_full_field_joint_fit(monkeypatch):
    """Inspect a region with shared rotation without replacing the full-field solution."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple silicon GPU")
    from quantem.gpu import SSB

    counts = np.random.default_rng(7).poisson(40, size=(128, 128, 32, 32)).astype(np.uint16)
    rows, columns = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rows, columns) < 10] += 200
    with SSB(counts, backend="mps", voltage_kV=300, semiangle_mrad=30,
             scan_sampling_A=0.3, detector_sampling_mrad=3.0, scan_detector_rotation_deg=17.0) as ssb:
        fit = ssb.find_aberrations(compare_standard=False, compute_loss=False, verbose=False)
        phase = fit.phase.copy()
        figure = ssb.show("tilt", region=(16, 80, 32, 96), size="small")
        assert "Rows 16:80, columns 32:96" in figure.get_suptitle()
        assert ssb.aberrations == fit.aberrations
        assert ssb.scan_detector_rotation_deg == fit.scan_detector_rotation_deg
        np.testing.assert_array_equal(fit.phase, phase)
        assert not fit.parameter_slices  # regional evidence was not attached to the full field

        # A regional image view corrects the full parent scan before cropping.
        captured = _record_static_arrays(monkeypatch)
        with ssb.select_region((16, 80, 32, 96)) as region:
            regional_fit = region.find_aberrations(check_rotation=False, compare_standard=False,
                                                   compute_loss=False, verbose=False)
            focus = region.show("focus", mode="static", focus_offsets_nm=(0,),
                                depth_spread_nm=regional_fit.depth_spread_nm, size="small")
            assert "rows 16:80, columns 32:96" in focus.get_suptitle()
            factor = ssb._resolve_upsampling_factor("auto")
            expected, _ = ssb.preview(dict(regional_fit), sample_tilt_mrad=regional_fit.sample_tilt_mrad,
                                      depth_spread_nm=regional_fit.depth_spread_nm,
                                      upsampling_factor=factor, compute_loss=False)
            actual = captured[-1][0][0]
            np.testing.assert_array_equal(actual, expected[16 * factor:80 * factor, 32 * factor:96 * factor])
            assert ssb.aberrations == fit.aberrations
            assert ssb.sample_tilt_mrad == fit.sample_tilt_mrad


def test_focus_grid_stack_and_static_preserve_images(monkeypatch):
    """Browse identical GPU reconstructions in the existing 2D and 3D viewers."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple silicon GPU")
    qw = pytest.importorskip("quantem.widget")
    from quantem.gpu import SSB

    counts = np.random.default_rng(7).poisson(40, size=(128, 128, 32, 32)).astype(np.uint16)
    rows, columns = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rows, columns) < 10] += 200
    with SSB(counts, backend="mps", voltage_kV=300, semiangle_mrad=30,
             scan_sampling_A=0.3, detector_sampling_mrad=3.0) as ssb:
        fit = ssb.find_aberrations(num_trials=8, refinement=None, check_rotation=False,
                                   compare_standard=False, compute_loss=False, verbose=False)
        options = {"focus_offsets_nm": (-3, 0, 7), "depth_spread_nm": (5, fit.depth_spread_nm)}
        grid = ssb.show("focus", **options)
        stack = ssb.show("focus", layout="stack", **options)
        assert isinstance(grid, qw.Show2D) and isinstance(stack, qw.Show3D)
        assert (stack.n_panels, stack.n_slices, stack.slice_idx) == (2, 3, 1)
        assert stack.dim_label == "Defocus"
        assert stack.labels[1] == f"C₁₀ {fit.aberrations['C10_nm']:+.2f} nm"
        assert "fitted depth" in stack.panel_titles[1]
        assert stack.display_bin == 1 and stack.avg_window == 1
        assert stack.link_contrast and stack.link_panels
        assert stack.vmin == grid.vmin and stack.vmax == grid.vmax
        expected = np.concatenate([grid._data[:3], grid._data[3:]], axis=2)
        np.testing.assert_array_equal(stack._data, expected)
        captured = _record_static_arrays(monkeypatch)
        figure = ssb.show("focus", mode="static", layout="stack", **options)
        np.testing.assert_array_equal(np.concatenate(captured[-1]), grid._data)
        assert all(len(row) == 2 for row in captured[-1])
        figure.draw_without_rendering()
        for ax in figure.axes:
            image = ax.images[0]
            assert ax.get_aspect() == 1
            assert np.all(ax.get_window_extent().size[::-1] >= image.get_array().shape[:2])
        single_column = ssb.show("focus", mode="static", columns=1,
                                 focus_offsets_nm=(0,), depth_spread_nm=fit.depth_spread_nm)
        assert len(single_column.axes) == 1
        reflowed = ssb.show("focus", rows=2, columns=3, **options)
        np.testing.assert_array_equal(reflowed._data, grid._data)
        assert reflowed.columns == 3
        with pytest.warns(DeprecationWarning, match="use columns"):
            legacy = ssb.show("focus", mode="static", max_cols=2, **options)
        np.testing.assert_array_equal(np.concatenate(captured[-1]), grid._data)
        assert legacy.get_size_inches().tolist() == figure.get_size_inches().tolist()
        reflowed.close()
        assert ssb._aberration_search is fit
        assert ssb.aberrations == fit.aberrations
        assert ssb.sample_tilt_mrad == fit.sample_tilt_mrad
        assert ssb.depth_spread_nm == fit.depth_spread_nm
        grid.close()
        stack.close()
