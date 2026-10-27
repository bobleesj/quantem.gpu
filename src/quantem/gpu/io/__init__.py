"""Accelerated 4D-STEM storage workflows.

The public API contains four operations: :func:`load`, :func:`save`,
:func:`report`, and :func:`discover`; the :class:`Dataset4dstemGPU` that
``load`` returns; the :class:`DataRepresentation` names a load selects; and the
:class:`PairedLoader` that streams paired residents in an application. Device
decoders, metadata parsers and scheduling helpers remain private to the I/O
domain.
"""

from quantem.gpu.io.discover import discover
from quantem.gpu.io.inspect import inspect
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.paired import PairedLoader
from quantem.gpu.io.load import load
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.io.report import report
from quantem.gpu.io.save import save

# Keep inspect available for existing readiness/metadata consumers.
__all__ = ["DataRepresentation", "Dataset4dstemGPU", "PairedLoader", "discover", "load", "report", "save"]
