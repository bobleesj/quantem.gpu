"""quantem.gpu.ssb stays importable as an alias of quantem.gpu.direct_ptycho, with no second module copies."""

import pickle
import subprocess
import sys
import textwrap
import warnings
from importlib import import_module

import numpy as np
import pytest


def _run(script: str) -> str:
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], check=True, text=True,
                            capture_output=True)
    return result.stdout.strip()


def test_old_package_is_the_new_package_and_warns_once() -> None:
    output = _run(
        """
        import sys
        import warnings

        import quantem.gpu as qg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import quantem.gpu.ssb as old
            from quantem.gpu.ssb import DEPTH_SPREAD_RANGE_NM, DirectPtycho, SSBResult
            import quantem.gpu.ssb.results
            from quantem.gpu.ssb.persistence import load_result
            import quantem.gpu.ssb

        import quantem.gpu.direct_ptycho as new
        from quantem.gpu.direct_ptycho import persistence, results

        assert [(type(item.message), str(item.message)) for item in caught] == [
            (DeprecationWarning, "quantem.gpu.ssb is now quantem.gpu.direct_ptycho")]
        assert caught[0].filename == "<string>"  # attributed to the importer, not to importlib
        assert old is new and qg.ssb is new and sys.modules["quantem.gpu.ssb"] is new
        assert DirectPtycho is qg.DirectPtycho is new.DirectPtycho and SSBResult is qg.SSBResult
        assert DEPTH_SPREAD_RANGE_NM is new.DEPTH_SPREAD_RANGE_NM
        assert quantem.gpu.ssb.results is results and load_result is persistence.load_result
        assert sys.modules["quantem.gpu.ssb.results"] is results
        assert results.__name__ == results.__spec__.name == "quantem.gpu.direct_ptycho.results"
        assert DirectPtycho.__module__ == "quantem.gpu.direct_ptycho.workflow"

        # one module object per source file, whichever name loaded it
        seen = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if path and "direct_ptycho" in path:
                seen.setdefault(path, set()).add(id(module))
        assert all(len(ids) == 1 for ids in seen.values()), seen
        for name in [name for name in sys.modules if name.startswith("quantem.gpu.ssb.")]:
            assert sys.modules[name] is sys.modules["quantem.gpu.direct_ptycho" + name[len("quantem.gpu.ssb"):]]
        print("ok")
        """
    )
    assert output == "ok"


def test_unknown_old_submodule_is_not_found_under_its_old_name() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import_module("quantem.gpu.ssb")
    with pytest.raises(ModuleNotFoundError, match="No module named 'quantem.gpu.ssb.no_such_module'"):
        import_module("quantem.gpu.ssb.no_such_module")


def test_pickles_naming_the_old_module_load_the_same_class() -> None:
    from quantem.gpu.direct_ptycho.results import SSBResult

    # protocol 0 GLOBAL opcode: what a pickle written before the rename stores for the class
    assert pickle.loads(b"cquantem.gpu.ssb.results\nSSBResult\n.") is SSBResult
    assert b"quantem.gpu.direct_ptycho.results" in pickle.dumps(SSBResult, protocol=0)


def test_result_saved_by_the_new_package_loads_through_the_old_one(tmp_path) -> None:
    from quantem.gpu.direct_ptycho.persistence import save_result
    from quantem.gpu.direct_ptycho.results import SSBResult

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = import_module("quantem.gpu.ssb.persistence")
        OldResult = import_module("quantem.gpu.ssb").SSBResult
    paths = (tmp_path / "ssb-reconstruct.npz", tmp_path / "ssb-reconstruct.json")
    signature = {"source": "synthetic"}
    wave = np.exp(1j * np.arange(4, dtype=np.float32).reshape(2, 2)).astype(np.complex64)
    save_result(SSBResult(object_wave=wave, backend="mps", aberrations={"C10": -101.0}),
                paths=paths, signature=signature, input_metadata={})

    loaded = old.load_result(paths=paths, signature=signature, backend="mps")

    assert isinstance(loaded, OldResult) and isinstance(loaded, SSBResult) and loaded.reused
    np.testing.assert_array_equal(loaded.object_wave, wave)


@pytest.mark.parametrize(("submodule", "name", "library"), [
    ("cuda.direct", "PreparedDirect", "cupy"),
    ("mps.direct", "PreparedDirect", "mlx.core"),
])
def test_backend_submodules_resolve_to_the_same_objects(submodule: str, name: str, library: str) -> None:
    pytest.importorskip(library)
    output = _run(
        f"""
        import sys
        import warnings

        warnings.simplefilter("ignore", DeprecationWarning)
        import quantem.gpu.ssb.{submodule} as old
        from quantem.gpu.ssb.{submodule} import {name} as OldClass
        import quantem.gpu.direct_ptycho.{submodule} as new

        assert old is new and OldClass is new.{name}
        assert sys.modules["quantem.gpu.ssb.{submodule}"] is new
        assert new.__spec__.name == "quantem.gpu.direct_ptycho.{submodule}"
        print("ok")
        """
    )
    assert output == "ok"
