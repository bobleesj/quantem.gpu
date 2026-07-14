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

        report = qg.device_report("cpu")
        assert report.selected == "cpu"
        assert qg.dp_mean.__module__ == "quantem.gpu.detector"
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
