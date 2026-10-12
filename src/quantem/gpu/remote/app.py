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
from fastapi import Depends, FastAPI, HTTPException, Request, Response

from quantem.gpu.remote.browse import (
    CENTER_OF_MASS_MODES,
    DETECTOR_MODES,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
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
        dependencies=[Depends(_refuse_undeclared_parameters)],
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
        scan_row: int = 0,
        scan_col: int = 0,
        detector_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        col_start: int | None = None,
        col_stop: int | None = None,
        ensure_resident: bool = False,
    ) -> Response:
        path, plan = await asyncio.to_thread(
            browse.plan,
            session,
            file,
            detector_bin=detector_bin,
            scan_bin=scan_bin,
            scan_region=_scan_region(row_start, row_stop, col_start, col_stop),
        )

        def pattern(entry):
            return browse.selected_diffraction(entry, plan, scan_row=scan_row, scan_col=scan_col)

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
        inner_bright_field_fraction: float = 0.0,
        outer_bright_field_fraction: float = 1.0,
        center_row_px: float | None = None,
        center_col_px: float | None = None,
        detector_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        col_start: int | None = None,
        col_stop: int | None = None,
        ensure_resident: bool = False,
    ) -> Response:
        if mode not in DETECTOR_MODES | CENTER_OF_MASS_MODES:
            raise HTTPException(400, f"unknown virtual-image mode: {mode!r}")
        region = _scan_region(row_start, row_stop, col_start, col_stop)
        image_key = (
            session,
            file,
            max(1, detector_bin),
            scan_bin,
            region,
            mode,
            round(inner_bright_field_fraction, 5),
            round(outer_bright_field_fraction, 5),
            None if center_row_px is None else round(center_row_px, 3),
            None if center_col_px is None else round(center_col_px, 3),
        )
        cached = browse.cached_image(image_key)
        if cached is not None and not ensure_resident:
            return _image_response(cached)
        path, plan = await asyncio.to_thread(
            browse.plan,
            session,
            file,
            detector_bin=detector_bin,
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
                inner_bright_field_fraction=inner_bright_field_fraction,
                outer_bright_field_fraction=outer_bright_field_fraction,
                center_row_px=center_row_px,
                center_col_px=center_col_px,
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
        center_row_px: float = 0.0,
        center_col_px: float = 0.0,
        inner_radius_px: float = 0.0,
        outer_radius_px: float = 0.0,
        detector_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        col_start: int | None = None,
        col_stop: int | None = None,
    ) -> Response:
        if shape not in {"circle", "square", "annulus"}:
            raise HTTPException(400, "detector shape must be circle, square, or annulus")
        if not all(math.isfinite(value) for value in (center_row_px, center_col_px, inner_radius_px, outer_radius_px)):
            raise HTTPException(400, "detector center and radii must be finite")
        if outer_radius_px <= 0 or (shape == "annulus" and (inner_radius_px < 0 or outer_radius_px <= inner_radius_px)):
            raise HTTPException(400, "detector radii must satisfy 0 <= inner_radius_px < outer_radius_px")
        path, plan = await asyncio.to_thread(
            browse.plan,
            session,
            file,
            detector_bin=detector_bin,
            scan_bin=scan_bin,
            scan_region=_scan_region(row_start, row_stop, col_start, col_stop),
        )
        image = await asyncio.to_thread(
            browse.compute,
            path,
            lambda entry: browse.custom_detector(
                entry,
                plan,
                center_row_px=center_row_px,
                center_col_px=center_col_px,
                inner_radius_px=inner_radius_px,
                outer_radius_px=outer_radius_px,
                shape=shape,
            ),
        )
        return _image_response(image)

    return app


# --- Responses and request parsing


def _refuse_undeclared_parameters(request: Request) -> None:
    """Refuse a query parameter the matched route does not declare, naming it and the protocol version.

    FastAPI ignores a parameter it does not know. A client of an earlier protocol version (``sx``, ``cx``,
    ``det_bin``, ``inner``) then received HTTP 200 with the image of the default scan position or mask, which
    cannot be told from the image it asked for. One dependency of the application covers every route.
    """
    declared = {parameter.alias for parameter in request.scope["route"].dependant.query_params}
    unknown = sorted(set(request.query_params) - declared)
    if unknown:
        raise HTTPException(
            400,
            f"unknown query parameters {unknown} for {request.url.path}: {PROTOCOL_NAME} protocol version "
            f"{PROTOCOL_VERSION} takes {sorted(declared)}",
        )


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
    col_start: int | None,
    col_stop: int | None,
) -> tuple[int, int, int, int] | None:
    """Return the half-open (row, col) scan crop, or None for the full scan."""
    values = (row_start, row_stop, col_start, col_stop)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise HTTPException(
            400,
            "scan crop requires row_start, row_stop, col_start, and col_stop",
        )
    region = tuple(int(value) for value in values)
    if region[0] < 0 or region[2] < 0 or region[1] <= region[0] or region[3] <= region[2]:
        raise HTTPException(
            400,
            "scan crop must be a nonempty half-open (row, col) region",
        )
    return region
