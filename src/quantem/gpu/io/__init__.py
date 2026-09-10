"""Accelerated 4D-STEM storage workflows.

The public API intentionally contains four operations: :func:`load`,
:func:`save`, :func:`inspect`, and :func:`discover`. Device decoders, metadata
parsers, scheduling helpers, and storage representations remain private to the
I/O domain.
"""

from .discover import discover
from .inspect import inspect
from .integrity import SourceIntegrity as SourceIntegrity
from .models import FourDSTEMData as FourDSTEMData
from .models import LoadResult as LoadResult
from ._paired import PairedLoader as PairedLoader
from .load import load
from .representation import DataRepresentation as DataRepresentation
from .save import save

__all__ = ["discover", "inspect", "load", "save"]
