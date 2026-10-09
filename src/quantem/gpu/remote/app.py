"""The loopback HTTP application native 4D-STEM viewers talk to.

The service owns no browser UI and no reconstruction orchestration. It exposes
the smallest versioned contract a native client needs: catalog discovery,
acquisition readiness, exact virtual-detector products, custom detector masks,
selected diffraction patterns and saved SSB phases. Raw detector data stays on
the compute host.
"""

import asyncio
import math
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response

from quantem.gpu.remote.browse import (
    CENTER_OF_MASS_MODES,
    DETECTOR_MODES,
    BrowseService,
)
from quantem.gpu.remote.saved_ssb import SavedSSBResults


def create_app(
    data_folder: str | os.PathLike[str],
    *,
    gpus: Sequence[int] | str = (0,),
    implementation_revision: str = version("quantem.gpu"),
) -> FastAPI:
    """Create the loopback remote-viewer application.

    The app serves the catalog, exact detector products and saved SSB phases
    for one data folder. Bind it to 127.0.0.1 and reach it through an SSH
    tunnel; it has no authentication of its own.

    Parameters
    ----------
    data_folder : str or os.PathLike
        Root of the served acquisitions.
    gpus : sequence of int or "auto"
        CUDA devices of the pool; ``"auto"`` uses every visible device.
    implementation_revision : str, optional
        quantem.gpu revision reported to clients; defaults to the installed
        package version.

    Returns
    -------
    fastapi.FastAPI
        The application, with its :class:`BrowseService` on
        ``app.state.browse_service``.

    Examples
    --------
    >>> import uvicorn
    >>> app = create_app("/data", gpus="auto", implementation_revision=revision)
    >>> uvicorn.run(app, host="127.0.0.1", port=8765)
    """
    browse = BrowseService(data_folder, gpus=gpus)
    saved_ssb = SavedSSBResults(Path(data_folder))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await asyncio.to_thread(browse.close)

    app = FastAPI(
        title="QuantEM GPU Remote Browse",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.browse_service = browse

    @app.get("/api/browse/capabilities")
    async def capabilities() -> dict[str, object]:
        result = browse.capabilities()
        result["implementation_revision"] = implementation_revision
        return result

    @app.get("/api/ssb/saved-results")
    async def ssb_saved_results(session: str, file: str) -> dict:
        try:
            master = await asyncio.to_thread(browse.catalog.resolve_master, session, file)
            return await asyncio.to_thread(saved_ssb.read, master)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(409, "Could not verify saved SSB results: " + str(exc)) from exc

    @app.get("/api/browse/sessions")
    async def sessions(refresh: bool = False) -> dict[str, object]:
        return await asyncio.to_thread(browse.catalog.sessions, refresh=refresh)

    @app.get("/api/browse/acquisitions")
    async def acquisitions() -> dict[str, object]:
        return await asyncio.to_thread(browse.catalog.acquisitions)

    @app.get("/api/browse/cbed")
    async def cbed(
        session: str,
        file: str,
        sx: int = 0,
        sy: int = 0,
        det_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        column_start: int | None = None,
        column_stop: int | None = None,
        ensure_resident: bool = False,
    ) -> Response:
        path, plan = await asyncio.to_thread(
            browse.plan,
            session,
            file,
            det_bin=det_bin,
            scan_bin=scan_bin,
            scan_region=_scan_region(row_start, row_stop, column_start, column_stop),
        )

        def pattern(entry):
            return browse.selected_diffraction(entry, plan, scan_row=sx, scan_column=sy)

        if ensure_resident:
            image = await asyncio.to_thread(browse.compute, path, pattern)
        else:
            image = await asyncio.to_thread(lambda: pattern(browse.residency.resident(path)))
        return _image_response(image, cache_control="max-age=60")

    @app.get("/api/browse/realspace")
    async def realspace(
        session: str,
        file: str,
        mode: str = "BF",
        inner: float = 0.0,
        outer: float = 1.0,
        cx: float | None = None,
        cy: float | None = None,
        det_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        column_start: int | None = None,
        column_stop: int | None = None,
        ensure_resident: bool = False,
    ) -> Response:
        if mode not in DETECTOR_MODES | CENTER_OF_MASS_MODES:
            raise HTTPException(400, f"unknown virtual-image mode: {mode!r}")
        region = _scan_region(row_start, row_stop, column_start, column_stop)
        image_key = (
            session,
            file,
            max(1, det_bin),
            scan_bin,
            region,
            mode,
            round(inner, 5),
            round(outer, 5),
            None if cy is None else round(cy, 3),
            None if cx is None else round(cx, 3),
        )
        cached = browse.cached_image(image_key)
        if cached is not None and not ensure_resident:
            return _image_response(cached)
        path, plan = await asyncio.to_thread(
            browse.plan,
            session,
            file,
            det_bin=det_bin,
            scan_bin=scan_bin,
            scan_region=region,
        )

        def virtual_image(entry):
            if cached is not None:
                return cached
            return browse.virtual_image(
                entry,
                plan,
                mode=mode,
                inner=inner,
                outer=outer,
                center_row=cy,
                center_column=cx,
            )

        image = await asyncio.to_thread(browse.compute, path, virtual_image)
        if browse.cached_image(image_key) is None:
            browse.store_image(image_key, image)
        return _image_response(image)

    @app.get("/api/browse/realspace-shape")
    async def realspace_shape(
        session: str,
        file: str,
        shape: str = "annulus",
        cx: float = 0.0,
        cy: float = 0.0,
        inner: float = 0.0,
        outer: float = 0.0,
        det_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        column_start: int | None = None,
        column_stop: int | None = None,
    ) -> Response:
        if shape not in {"circle", "square", "annulus"}:
            raise HTTPException(400, "detector shape must be circle, square, or annulus")
        if not all(math.isfinite(value) for value in (cx, cy, inner, outer)):
            raise HTTPException(400, "detector center and radii must be finite")
        if outer <= 0 or (shape == "annulus" and (inner < 0 or outer <= inner)):
            raise HTTPException(400, "detector radii must satisfy 0 <= inner < outer")
        path, plan = await asyncio.to_thread(
            browse.plan,
            session,
            file,
            det_bin=det_bin,
            scan_bin=scan_bin,
            scan_region=_scan_region(row_start, row_stop, column_start, column_stop),
        )
        image = await asyncio.to_thread(
            browse.compute,
            path,
            lambda entry: browse.custom_detector(
                entry,
                plan,
                center_row=cy,
                center_column=cx,
                inner_radius=inner,
                outer_radius=outer,
                shape=shape,
            ),
        )
        return _image_response(image)

    return app


# --- Responses and request parsing


def _image_response(image: np.ndarray, *, cache_control: str = "max-age=300") -> Response:
    """Send one image as raw little-endian bytes with its shape and dtype in headers."""
    payload, dtype = _wire_image(image)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Width": str(image.shape[1]),
            "X-Height": str(image.shape[0]),
            "X-Dtype": dtype,
            "Cache-Control": cache_control,
        },
    )


def _wire_image(image: np.ndarray) -> tuple[bytes, str]:
    """Encode one image without changing exact detector counts."""
    if image.ndim != 2:
        raise HTTPException(500, f"Remote compute produced a non-2-D image: {image.shape}.")
    if np.issubdtype(image.dtype, np.integer):
        if np.issubdtype(image.dtype, np.signedinteger) and image.size and image.min() < 0:
            raise HTTPException(500, "Remote compute produced negative detector counts.")
        maximum = int(image.max()) if image.size else 0
        if maximum > np.iinfo(np.uint32).max:
            raise HTTPException(
                409,
                "Exact detector counts exceed uint32 display capacity. "
                "Use a smaller scan bin or detector aperture.",
            )
        wire = np.ascontiguousarray(image, dtype="<u4")
        return wire.tobytes(), "<u4"
    wire = np.ascontiguousarray(image, dtype="<f4")
    return wire.tobytes(), "<f4"


def _scan_region(
    row_start: int | None,
    row_stop: int | None,
    column_start: int | None,
    column_stop: int | None,
) -> tuple[int, int, int, int] | None:
    """Return the half-open (row, column) scan crop, or None for the full scan."""
    values = (row_start, row_stop, column_start, column_stop)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise HTTPException(
            400,
            "scan crop requires row_start, row_stop, column_start, and column_stop",
        )
    region = tuple(int(value) for value in values)
    if region[0] < 0 or region[2] < 0 or region[1] <= region[0] or region[3] <= region[2]:
        raise HTTPException(
            400,
            "scan crop must be a nonempty half-open (row, column) region",
        )
    return region
