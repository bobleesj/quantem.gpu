"""Acquisition metadata tables without decoding detector values."""

from os import PathLike
from typing import TYPE_CHECKING

from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.inspect import inspect

if TYPE_CHECKING:
    import pandas as pd


def report(
    source: str | PathLike[str] | Dataset4dstemGPU,
    *,
    scan_shape: tuple[int, int] | None = None,
    dataset_path: str | None = None,
) -> "pd.DataFrame":
    """Return acquisition geometry and calibration as a table.

    Parameters
    ----------
    source
        A file path or an already-loaded acquisition. Files are read through
        their headers; detector values are not decoded or loaded onto a GPU.
    scan_shape
        Expected ``(row, col)`` scan shape for a file, when needed.
    dataset_path
        Measurement dataset to select in a file with multiple acquisitions.

    Returns
    -------
    pandas.DataFrame
        Properties indexed by name, with Value, Unit and Metadata columns.
        Missing calibration remains explicitly unrecorded. A header report
        does not authenticate a saved payload; ``io.load`` validates it.

    Examples
    --------
    >>> io.report("acquisition.qem")  # doctest: +SKIP
    >>> io.report(data)[["Value", "Unit"]]  # doctest: +SKIP
    """
    if isinstance(source, Dataset4dstemGPU):
        if scan_shape is not None or dataset_path is not None:
            raise ValueError("scan_shape and dataset_path select file data; omit them for a loaded acquisition.")
        return source.metadata_table()
    return inspect(source, scan_shape=scan_shape, dataset_path=dataset_path)._summary()
