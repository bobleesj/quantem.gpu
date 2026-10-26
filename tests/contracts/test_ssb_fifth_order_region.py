"""Closing a regional fifth-order fit preserves its scope and diagnostic panels."""

import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from quantem.gpu.direct_ptycho.results import DirectPtychoResult
from quantem.gpu.direct_ptycho.workflow import SSB


@pytest.mark.parametrize("search", ["standard", "broad"])
@pytest.mark.parametrize("saved", [False, True])
def test_regional_fit_keeps_scope_panels_and_parent_state(tmp_path, search, saved):
    region = (64, 192, 128, 256)
    events = []
    initial = DirectPtychoResult(
        object_wave=np.ones((128, 128), dtype=np.complex64), backend="mps",
        aberrations={"C10_nm": 2.0, "C12_nm": 0.0, "phi12_deg": 0.0},
        sample_tilt_mrad=(1.0, -2.0), depth_spread_nm=12.0,
    )
    result = DirectPtychoResult(
        object_wave=np.ones((128, 128), dtype=np.complex64), backend="mps",
        aberrations={"C10_nm": 2.0, "C12_nm": 0.0, "phi12_deg": 0.0, "C50_nm": 3.0},
        sample_tilt_mrad=(1.0, -2.0), depth_spread_nm=12.0,
        trial_records=[{"trial": 0}], optimizer={"model": "broad_fifth_v1"},
        saved_path=tmp_path / "fit.npz" if saved else None,
        metadata={"signature": {"data": {"scan_region": list(region)}}} if saved else {},
    )
    phases = (np.ones((256, 256)), np.ones((256, 256)))
    def panels():
        assert "closed" not in events
        events.append("panels")
        return phases
    result.rotation_branches = SimpleNamespace(upsampled_phases=Mock(side_effect=panels))
    received = []

    def fit(**options):
        assert options["search"] == search
        options["on_initial_fit"](initial)
        assert initial.fit_scan_region is None
        events.append("fit")
        return result

    def save(updated, *, paths, signature):
        assert "closed" not in events
        assert updated.fit_scan_region == region
        assert paths == (result.saved_path, result.saved_path.with_suffix(".json"))
        assert signature == result.metadata["signature"]
        events.append("save")
        return updated

    child = SimpleNamespace(find_aberrations=fit, _save_result=Mock(side_effect=save))
    @contextmanager
    def select_region(chosen):
        assert chosen == region
        yield child
        events.append("closed")

    parent = SimpleNamespace(input_scan_shape=(512, 512), select_region=select_region)
    def accept(updated):
        assert "closed" in events
        assert updated.fit_scan_region == region
        parent.aberrations = dict(updated.aberrations)
        parent._reconstruction = updated
        return updated
    parent._accept_result = Mock(side_effect=accept)
    fitted = inspect.unwrap(SSB.find_aberrations)(
        parent, search=search, scan_region=region, show=False,
        on_initial_fit=received.append, save_dir=tmp_path if saved else None,
    )
    assert fitted is result
    assert result.fit_scan_region == region
    assert parent._aberration_search is result
    assert parent._reconstruction is None
    assert parent.aberrations["C50_nm"] == 3.0
    assert parent.trial_history == result.trial_records
    assert len(received) == 1 and received[0].fit_scan_region == region
    assert received[0].object_wave is initial.object_wave
    assert child._save_result.call_count == int(saved)
    assert events == ["fit", "panels", *(["save"] if saved else []), "closed"]


@pytest.mark.parametrize("region", [
    (0, 128, 0, 64), (0, 100, 0, 100), (400, 528, 0, 128), (0, 128.0, 0, 128),
])
def test_regional_fifth_order_rejects_invalid_or_padded_fit_regions(region):
    parent = SimpleNamespace(input_scan_shape=(512, 512), select_region=Mock())
    with pytest.raises(ValueError, match="scan_region"):
        inspect.unwrap(SSB.find_aberrations)(parent, scan_region=region, show=False)
    parent.select_region.assert_not_called()
