from __future__ import annotations

import subprocess
import sys
import textwrap


def test_quantem_gpu_root_import_without_cupy() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCupy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cupy" or fullname.startswith("cupy."):
                    raise ModuleNotFoundError("blocked cupy for import smoke")
                return None

        sys.meta_path.insert(0, BlockCupy())

        import quantem.gpu as qg
        import quantem.gpu.io as qgio

        report = qg.device_report("cpu")
        assert report.selected == "cpu"
        assert qg.CalibrationMemoryPlan.__module__ == "quantem.gpu.calibration"
        assert qg.calibration_memory_plan.__module__ == "quantem.gpu.calibration"
        assert qg.load_calibration_products.__module__ == "quantem.gpu.calibration"
        assert qg.dp_mean.__module__ == "quantem.gpu.detector"
        assert qg.load_scan_indices.__module__ == "quantem.gpu.io.hdf5"
        assert "load_scan_region" not in qg.__all__
        assert not hasattr(qg, "load_scan_region")
        assert "load_scan_region" not in qgio.__all__
        assert not hasattr(qgio, "load_scan_region")
        assert qg.random_scan_indices.__module__ == "quantem.gpu.io.hdf5"
        assert qg.SSB.__module__ == "quantem.gpu.ssb.workflow"
        assert qg.ssb.__name__ == "quantem.gpu.ssb"
        assert not any(
            name.lower().endswith(("_cuda", "_mps", "_webgpu"))
            for name in qg.__all__
        )
        assert "load_mps_4dstem" in qgio.__all__
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "ok"
