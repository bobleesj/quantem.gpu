# Protocol and integration

The service implementation lives in `src/quantem/gpu/remote`. It composes the
public IO, detector, and DPC functions rather than duplicating their kernels.

## Implementation map

| Layer | Source | Responsibility |
|---|---|---|
| CLI entry | `src/quantem/gpu/cli.py` | parse `quantem-gpu serve` arguments |
| HTTP application | `src/quantem/gpu/remote/app.py::create_app` | versioned routes, payload headers, lifecycle |
| browse service | `src/quantem/gpu/remote/browse.py::BrowseService` | capabilities, request plans, exact products |
| catalog | `src/quantem/gpu/remote/catalog.py::Catalog` | session folders, master discovery, readiness |
| residency | `src/quantem/gpu/remote/residency.py::Residency` | GPU pool, admission, placement, eviction |
| bin and crop plan | `src/quantem/gpu/remote/plan.py::BrowsePlan` | binned and cropped views of one resident |
| saved SSB phases | `src/quantem/gpu/remote/saved_ssb.py` | acquisition-verified saved phase transfer |
| transport tests | `tests/contracts/remote` | routes, headers, errors, admission, lifecycle, dense parity |

## Version negotiation

The capabilities route reports protocol name `quantem-gpu-browse`, protocol
version `1`, backend/device information, feature capabilities, per-device
admission telemetry, and `implementation_revision`. Clients must validate this
response before assuming an endpoint or field exists. New optional response
fields are additive; changed scientific meaning requires a protocol version
change.

## Endpoints

| Route | Contract |
|---|---|
| `GET /api/browse/capabilities` | protocol, GPU pool, live admission capacity, features |
| `GET /api/browse/sessions` | catalog of session folders and their masters |
| `GET /api/browse/acquisitions` | acquisitions being written and ready, with a change token |
| `GET /api/browse/realspace` | BF, ABF, ADF, HAADF, DF exact count images; CoMx, CoMy, CoMmag, DPC, iCoM |
| `GET /api/browse/realspace-shape` | exact counts inside a circle, square, or annulus detector |
| `GET /api/browse/cbed` | exact diffraction pattern at one scan position |
| `GET /api/ssb/saved-results` | saved SSB phases verified against the acquisition identity |

Binary 2D images use `application/octet-stream` with width, height, and dtype
headers. Integer count images are encoded as little-endian unsigned 32-bit
values after an overflow check; clients must not infer native counts from
display-scaled data.

## Bins and crops

Every product request carries the plan `det_bin`, `scan_bin`, and an optional
half-open crop `row_start, row_stop, column_start, column_stop`. The plan is
checked before any residency change: `det_bin` must divide both detector sizes,
`scan_bin` is 1, 2, 4, 8, or 16, and the crop must lie inside the scan. A
rejected plan returns 400 and leaves the resident acquisition untouched.

All plans of one acquisition read the same encoded resident, so changing bin
or crop never reloads it. Binning adds exact integer counts. Scan bins keep the
partial bins at the bottom and right edges of the crop, where the missing
positions count as zero. Detector geometry (`cx`, `cy`, radii, and the fitted
bright-field disk) is in binned detector pixels; `sx` and `sy` are in binned,
cropped scan positions. Centre-of-mass products divide exact
integer moments in binned detector coordinates by the exact total count. These
products equal, value for value, the products of a dense volume binned and
cropped the same way (`tests/contracts/remote/test_browse_plans.py`).

## Client integration contract

A client persists the scan and detector regions, scan and detector bins, output
shapes and dtypes, backend/device, and implementation revision with every
derived product. It uses `(row, column) ≡ (r, c)` for all public coordinates.

The service may add scheduling, cache reuse, or a faster kernel without
changing this contract. It may not silently alter coverage, detector geometry,
precision, masks, calibration, or reconstruction parameters.
