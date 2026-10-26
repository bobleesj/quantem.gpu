"""Deprecated alias of ``quantem.gpu.direct_ptycho``.

``quantem.gpu.ssb`` and every ``quantem.gpu.ssb.<name>`` resolve to the module objects of
``quantem.gpu.direct_ptycho``, so classes, ``isinstance`` checks and module paths stored by older
code are the same objects under either name. Importing it warns once with a DeprecationWarning.
"""

import sys
import warnings
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader

_OLD = __name__
_NEW = "quantem.gpu.direct_ptycho"


class _Alias(MetaPathFinder, Loader):
    """Resolve ``quantem.gpu.ssb.<name>`` to the module object ``quantem.gpu.direct_ptycho.<name>``."""

    is_ssb_alias = True

    def __init__(self) -> None:
        self._specs = {}

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + ".") or find_spec(_NEW + fullname[len(_OLD):]) is None:
            return None
        return spec_from_loader(fullname, self)

    def create_module(self, spec):
        module = import_module(_NEW + spec.name[len(_OLD):])
        self._specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        # the import system set __spec__ on the module create_module returned; give it its own back
        module.__spec__ = self._specs.pop(module.__spec__.name)


# first, so the path finder never loads a second copy of a module under the old name
if not any(getattr(finder, "is_ssb_alias", False) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
warnings.warn(f"{_OLD} is now {_NEW}", DeprecationWarning, stacklevel=2)
sys.modules[_OLD] = import_module(_NEW)
