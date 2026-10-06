"""Result schema for differential phase contrast."""
from dataclasses import dataclass

import numpy as np


@dataclass
class DPCResult:
    """Float32 DPC products in public ``(row, col)`` order."""

    phase: np.ndarray
    com_row: np.ndarray
    com_col: np.ndarray
    com_row_aligned: np.ndarray
    com_col_aligned: np.ndarray
    rotation_deg: float
    use_transpose: bool
    elapsed: float

    def report(self):
        """Return the rotation and axis choice as a labeled parameter table.

        The curl-based estimate has a 180-degree ambiguity. SSB's optional
        phase-polarity check can choose between those two branches. Detector
        angular sampling is a separate microscope calibration.

        Returns
        -------
        pandas.DataFrame
            Scan-detector rotation, axis swap and elapsed time in one row.

        Examples
        --------
        >>> dpc_result = dpc.run(data)
        >>> dpc_result.report()
        """
        import pandas as pd

        return pd.DataFrame({
            "Scan-detector rotation (deg)": [self.rotation_deg],
            "Detector-axis swap": [self.use_transpose],
            "Time (s)": [self.elapsed],
        }, index=pd.Index(["DPC"], name="method")).round(3)
