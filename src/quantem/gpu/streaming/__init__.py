"""GPU backend for live acquisition: encode detector batches into ANS datasets as they arrive.

Receivers, queues and the acquisition lifecycle live with the caller (live4d);
this package only encodes and serves partial-scan queries.
"""

from quantem.gpu.streaming.encoder import IncompleteAcquisition, LiveEncoder
from quantem.gpu.streaming.parts import chunks_of, dataset_of

__all__ = ["IncompleteAcquisition", "LiveEncoder", "chunks_of", "dataset_of"]
