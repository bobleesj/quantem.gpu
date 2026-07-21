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
        import quantem.gpu.ssb.mps as ssb_mps
        import quantem.gpu.ssb.preprocess as ssb_preprocess

        report = qg.device_report("cpu")
        assert report.selected == "cpu"
        assert qg.CalibrationMemoryPlan.__module__ == "quantem.gpu.calibration"
        assert qg.calibration_memory_plan.__module__ == "quantem.gpu.calibration"
        assert qg.load_calibration_products.__module__ == "quantem.gpu.calibration"
        assert qg.dp_mean.__module__ == "quantem.gpu.detector"
        assert qg.bf_df_dpc.__module__ == "quantem.gpu.ssb.preprocess"
        assert qg.load_scan_indices.__module__ == "quantem.gpu.io.hdf5"
        assert qg.load_scan_region.__module__ == "quantem.gpu.io.hdf5"
        assert qg.random_scan_indices.__module__ == "quantem.gpu.io.hdf5"
        assert qg.ssb_fit_mps.__module__ == "quantem.gpu.ssb.mps"
        assert qg.ssb_preview_mps.__module__ == "quantem.gpu.ssb.mps"
        assert ssb_mps.ssb_preview.__module__ == "quantem.gpu.ssb.mps"
        assert ssb_preprocess.Preview.__module__ == "quantem.gpu.ssb.preprocess"
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
