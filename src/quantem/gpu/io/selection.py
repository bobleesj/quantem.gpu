"""Scan-order unflattening for the CPU reference loader."""

from typing import Literal

ScanOrder = Literal["row-major", "serpentine"]


def _apply_scan_shape(
    data,
    explicit: tuple[int, int] | None,
    meta: dict,
    scan_order: str = "row-major",
):
    """Reshape a NumPy 3D ``(N, det_r, det_c)`` → 4D ``(scan_r, scan_c, det_r, det_c)``.

    Uses ``explicit`` when the caller passed ``scan_shape=``, else
    ``meta["scan_shape"]`` (auto-derived from ``ntrigger``). No-op when
    no shape is available. ``scan_order="serpentine"`` reverses odd scan rows
    after unflattening so downstream code sees normal ``(row, col)`` order.
    """
    order = _normalize_scan_order(scan_order)
    shape = explicit if explicit is not None else meta.get("scan_shape")
    if shape is None:
        return data
    scan_r, scan_c = shape
    if data.ndim == 3:
        if scan_r * scan_c != data.shape[0]:
            raise ValueError(
                f"scan_shape {shape} incompatible with frame count {data.shape[0]}"
            )
        det_rows, det_cols = data.shape[-2:]
        data = data.reshape(scan_r, scan_c, det_rows, det_cols)
    elif data.ndim == 4:
        if tuple(int(v) for v in data.shape[:2]) != (int(scan_r), int(scan_c)):
            return data
    else:
        return data
    if order == "row-major":
        return data
    # Reverse one scan row at a time to avoid materializing a full second
    # 4D array for no-bin 512/1024 acquisitions.
    for row in range(1, int(data.shape[0]), 2):
        data[row] = data[row, ::-1].copy()
    return data


def _normalize_scan_order(scan_order: str | None) -> ScanOrder:
    """Normalize accepted flattened scan order names."""
    key = "row-major" if scan_order is None else str(scan_order).lower()
    key = key.replace("_", "-").replace(" ", "-")
    aliases: dict[str, ScanOrder] = {
        "row-major": "row-major",
        "raster": "row-major",
        "serpentine": "serpentine",
        "snake": "serpentine",
        "boustrophedon": "serpentine",
    }
    if key not in aliases:
        raise ValueError(
            "scan_order must be 'row-major' or 'serpentine' "
            f"(got {scan_order!r})"
        )
    return aliases[key]
