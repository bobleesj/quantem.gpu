"""Result schema for differential phase contrast."""
from dataclasses import dataclass

import numpy as np


@dataclass
class DPCResult:
    """Float32 DPC products in public ``(row, col)`` order.

    Exactly one rotation is set. ``scan_detector_rotation_mod180_deg`` is the
    curl fit, which fixes the scan-to-detector rotation only modulo 180
    degrees, in [0, 180) (the fit grid runs from 0 to 180; its end point is
    returned half a turn back, with the phase of that angle).
    ``scan_detector_rotation_deg`` is the physical rotation the caller gave
    ``dpc.run``, in [0, 360). ``detector_transposed`` says whether the detector
    axes are exchanged before the rotation.
    """

    phase: np.ndarray
    com_row: np.ndarray
    com_col: np.ndarray
    com_row_aligned: np.ndarray
    com_col_aligned: np.ndarray
    scan_detector_rotation_mod180_deg: float | None
    scan_detector_rotation_deg: float | None
    detector_transposed: bool
    elapsed_s: float

    def report(self):
        """Return the rotation and axis choice as a labeled parameter table.

        The curl-based estimate has a 180-degree ambiguity. SSB's optional
        phase-polarity check can choose between those two branches. Detector
        angular sampling is a separate microscope calibration.

        Returns
        -------
        pandas.DataFrame
            Scan-detector rotation, axis swap and elapsed_s time in one row.

        Examples
        --------
        >>> dpc_result = dpc.run(data)
        >>> dpc_result.report()
        """
        import pandas as pd

        fitted = self.scan_detector_rotation_deg is None
        return pd.DataFrame({
            "Scan-detector rotation (deg)": [self.scan_detector_rotation_mod180_deg if fitted else self.scan_detector_rotation_deg],
            "Known modulo 180": [fitted],
            "Detector transposed": [self.detector_transposed],
            "Time (s)": [self.elapsed_s],
        }, index=pd.Index(["DPC"], name="method")).round(3)
