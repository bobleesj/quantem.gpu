"""Translation polarity and causal feedback on a small synthetic crystal image."""

import numpy as np
import pytest
import torch

cp = pytest.importorskip("cupy")

from quantem.gpu.drift import register, shifted, simulate_drift, simulate_feedback

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="registration runs on CUDA"
)


def crystal():
    sampling_A = (0.2, 0.25)
    # Match a full scan's field: a 256-pixel field's Hann taper biases this texture by 0.08 px.
    noise = cp.random.default_rng(4).standard_normal((512, 512))
    q_row = cp.fft.fftfreq(512, sampling_A[0])[:, None]
    q_col = cp.fft.fftfreq(512, sampling_A[1])[None, :]
    q = cp.hypot(q_row, q_col)
    background = cp.fft.ifft2(cp.fft.fft2(noise) * (q > 0.02) * (q < 0.2)).real
    row = cp.arange(512)[:, None]
    col = cp.arange(512)[None, :]
    lattice = cp.cos(2 * cp.pi * (9 * row + 16 * col) / 256)
    return (3 * background / background.std() + lattice).astype(cp.float32), sampling_A


def test_translation_sign_including_null_and_wrong_correction():
    reference, sampling_A = crystal()
    images, truth_nm = simulate_drift(reference, sampling_A=sampling_A, num_scans=2)
    for image, expected_nm in zip(images, truth_nm):
        found = register(reference, image, sampling_A=sampling_A)
        np.testing.assert_allclose(
            np.asarray(found.shift_px) * sampling_A / 10, expected_nm, atol=0.001
        )
    corrected = shifted(images[-1], -truth_nm[-1] * 10 / sampling_A)
    # Two float32 FFTs and a stored float32 image incur rounding; measure relative image error.
    assert (
        float(cp.linalg.norm(corrected - reference) / cp.linalg.norm(reference)) < 1e-6
    )
    wrong_sign = shifted(images[-1], truth_nm[-1] * 10 / sampling_A)
    found = register(reference, wrong_sign, sampling_A=sampling_A)
    np.testing.assert_allclose(
        np.asarray(found.shift_px) * sampling_A / 10, 2 * truth_nm[-1], atol=0.001
    )


def test_feedback_corrects_crossed_axes_on_the_next_scan():
    reference, sampling_A = crystal()
    images, truth_nm = simulate_drift(reference, sampling_A=sampling_A, num_scans=6)
    matrix = np.array([[0.0, -2.0], [1.0, 0.0]])
    _, report = simulate_feedback(
        images,
        sampling_A=sampling_A,
        image_from_piezo=matrix,
        gain=1.0,
        max_step_nm=1.0,
    )
    applied = report[["applied_axis1_nm", "applied_axis2_nm"]].to_numpy()
    next_command = report[["next_axis1_nm", "next_axis2_nm"]].to_numpy()
    np.testing.assert_array_equal(applied[1:], next_command[:-1])
    np.testing.assert_allclose(
        report[["row_nm", "col_nm"]], truth_nm + applied @ matrix.T, atol=0.001
    )
    assert report.accepted.all()
    assert report.residual_nm.iloc[-1] < 0.3 * np.linalg.norm(truth_nm[-1])


@pytest.mark.parametrize("missing", [0.0, np.nan])
def test_missing_scans_hold_position_and_keep_the_last_valid_reference(missing):
    reference, sampling_A = crystal()
    images, _ = simulate_drift(reference, sampling_A=sampling_A, num_scans=6)
    images[2:5] = missing
    _, report = simulate_feedback(
        images, sampling_A=sampling_A, image_from_piezo=np.eye(2)
    )
    assert not report.accepted.iloc[2:5].any() and report.accepted.iloc[5]
    np.testing.assert_array_equal(
        report.loc[2:4, ["next_axis1_nm", "next_axis2_nm"]].to_numpy(),
        report.loc[2:4, ["applied_axis1_nm", "applied_axis2_nm"]].to_numpy(),
    )
    assert report.residual_nm.iloc[2:5].isna().all()


@pytest.mark.parametrize("noise_ratio", [0.1, 1.0])
def test_noisy_feedback_requires_both_accuracy_and_match_strength(noise_ratio):
    reference, sampling_A = crystal()
    images, truth_nm = simulate_drift(reference, sampling_A=sampling_A, num_scans=6)
    # Phase-domain noise, not detector dose. The last image is clean to test recovery.
    noise = cp.random.default_rng(21).standard_normal(images.shape, dtype=cp.float32)
    noise[-1] = 0
    images += noise * reference.std() * noise_ratio
    matrix = np.array([[0.9, -0.1], [0.15, 1.05]])
    _, report = simulate_feedback(
        images, sampling_A=sampling_A, image_from_piezo=matrix
    )
    applied = report[["applied_axis1_nm", "applied_axis2_nm"]].to_numpy()
    expected = truth_nm + applied @ matrix.T
    error_nm = np.linalg.norm(
        report[["row_nm", "col_nm"]].to_numpy() - expected, axis=1
    )
    assert report.accepted.iloc[-1]
    if noise_ratio == 0.1:
        assert report.accepted.all()
        assert error_nm.max() < 0.0025  # Less than 10% of the imposed 0.025 nm step.
    else:
        # This used to accept errors up to 0.069 nm because the peak margin alone passed.
        assert not report.accepted.iloc[1:-1].any()
        assert (report.confidence.iloc[1:-1] >= 0.04).all()
        assert (report.correlation.iloc[1:-1] < 0.5).all()
        np.testing.assert_array_equal(applied, 0)
        assert np.isnan(error_nm[1:-1]).all()


def test_step_limit_cannot_cancel_faster_drift():
    reference, sampling_A = crystal()
    images, truth_nm = simulate_drift(
        reference, sampling_A=sampling_A, num_scans=6, step_nm=(0.04, -0.03)
    )
    _, report = simulate_feedback(
        images,
        sampling_A=sampling_A,
        image_from_piezo=np.eye(2),
        gain=1,
        max_step_nm=0.01,
    )
    applied = report[["applied_axis1_nm", "applied_axis2_nm"]].to_numpy()
    commands = report[["next_axis1_nm", "next_axis2_nm"]].to_numpy()
    assert report.accepted.all() and report.limited.iloc[1:].all()
    np.testing.assert_allclose(np.linalg.norm(commands[1:] - applied[1:], axis=1), 0.01)
    residual = np.linalg.norm(truth_nm + applied, axis=1)
    assert np.all(np.diff(residual) > 0)
    np.testing.assert_allclose(report.residual_nm, residual, atol=0.001)
