from __future__ import annotations

import pytest

from quantem.gpu import device_report, select_device


def test_select_device_cpu_is_always_available() -> None:
    assert select_device("cpu") == "cpu"


def test_device_report_has_backend_fields() -> None:
    report = device_report()

    assert report.selected in {"cuda", "mps", "cpu"}
    assert isinstance(report.cuda_available, bool)
    assert isinstance(report.cuda_device_count, int)
    assert isinstance(report.mps_available, bool)
    assert report.cpu_available is True


def test_unknown_device_backend_errors() -> None:
    with pytest.raises(ValueError, match="Unknown device backend"):
        select_device("vulkan")
