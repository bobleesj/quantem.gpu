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
        assert qg.detector.__name__ == "quantem.gpu.detector"
        assert qg.device.__name__ == "quantem.gpu.device"
        assert qg.dpc.__name__ == "quantem.gpu.dpc"
        assert qg.io.__name__ == "quantem.gpu.io"
        assert qg.parallax.__name__ == "quantem.gpu.parallax"
        assert qg.screening.__name__ == "quantem.gpu.screening"
        assert qg.SSB.__module__ == "quantem.gpu.ssb.workflow"
        assert "load" not in qg.__all__
        assert "bf" not in qg.__all__
        assert "dpc" in qg.__all__
        assert "webgpu" not in qg.__all__
        assert not any(
            name.lower().endswith(("_cuda", "_mps", "_webgpu"))
            for name in qg.__all__
        )
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
