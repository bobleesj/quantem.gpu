# Protocol and integration

The service implementation lives in `src/quantem/gpu/remote`. It composes the
public IO, detector, DPC, and reconstruction engines rather than duplicating
their kernels.

## Implementation map

| Layer | Source | Responsibility |
|---|---|---|
| CLI entry | `src/quantem/gpu/cli.py` | parse `quantem-gpu serve` arguments |
| HTTP application | `src/quantem/gpu/remote/server.py::create_app` | versioned routes, payload headers, lifecycle |
| browse/residency service | `src/quantem/gpu/remote/server.py::BrowseService` | discovery, readiness, placement, cache ownership |
| reconstruction protocol | `src/quantem/gpu/remote/ssb_api.py` | source identity, prepare/reconstruct jobs, typed state |
| advanced protocol | `src/quantem/gpu/remote/maped_api.py` | inventory, payload, job, and cache-validation contracts |
| transport tests | `tests/remote` | routes, headers, errors, admission, and lifecycle |

## Version negotiation

The capabilities route reports protocol name `quantem-gpu-browse`, protocol
version `1`, backend/device information, feature capabilities, and per-device
admission telemetry. Clients must validate this response before assuming an
endpoint or field exists. New optional response fields are additive; changed
scientific meaning requires a protocol version change.

## Endpoint groups

| Prefix | Contract |
|---|---|
| `/api/browse` | capabilities, sessions, acquisitions, selected diffraction, real-space products |
| `/api/ssb` | source identity, preparation, reconstruction, interactive and queued jobs |
| `/api/maped` | inventory, previews, selected diffraction, payloads, cache validation, jobs |

Binary 2D images use `application/octet-stream` with width, height, and dtype
headers. Integer count images are encoded as little-endian unsigned 32-bit
values after an overflow check. Any value divisor is explicit in the response;
clients must not infer native counts from display-scaled data.

## Client integration contract

A client persists the response's source identity, source/output shapes,
source/output dtypes, scan and detector regions, scan and detector bins,
backend/device, and implementation revision with every derived product. It
uses `(row, column) ≡ (r, c)` for all public coordinates.

The service may add scheduling, cache reuse, or a faster kernel without
changing this contract. It may not silently alter coverage, detector geometry,
precision, masks, calibration, or reconstruction parameters.
