"""Accelerated 4D-STEM storage workflows.

The public API contains five operations: :func:`load`, :func:`save`,
:func:`report`, :func:`discover` and :func:`session` (a streamed session's scans by arrival index); the :class:`Dataset4dstem` that
``load`` returns; the :class:`DataRepresentation` names a load selects; and the
:class:`PairedLoader` that streams paired residents in an application. Device
decoders, metadata parsers and scheduling helpers remain private to the I/O
domain.
"""

from quantem.gpu.io.discover import discover, newest
from quantem.gpu.io.inspect import inspect
from quantem.gpu.io.dataset import Dataset4dstem
from quantem.gpu.io.paired import PairedLoader
from quantem.gpu.io.load import load
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.io.report import report
from quantem.gpu.io.save import save
from quantem.gpu.io.session import Session, session

# Keep inspect available for existing readiness/metadata consumers.
__all__ = ["DataRepresentation", "Dataset4dstem", "PairedLoader", "discover", "load", "newest", "report", "save", "Session", "session"]
