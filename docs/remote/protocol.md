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
| `/api/browse` | capabilities, sessions, acquisitions, residency telemetry, selected diffraction, real-space products |
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

## Compact exact residency

The server can bind a catalogued `*_master.h5` acquisition to one immutable
QGIX compact artifact with `CompactBrowseSource`. Clients continue to send the
same session and master filename to the existing browse routes. The compact
path is trusted server configuration and is never accepted from a client.
The normal CLI loads these bindings from `--compact-sources`; each registry
entry names the catalogued master, compact artifact, and required whole-file
SHA-256. Relative master paths are rooted at the served data folder, while
relative compact paths are rooted at the registry directory.

Compact bindings are full-coverage plans with scan bin 1, detector bin 1, no
crop, and exact integer output. A request for a transformed plan fails rather
than silently loading a different representation. Preset BF, ABF, ADF, HAADF,
and DF require source-bound detector calibration. Custom detector masks and
selected diffraction use the resident packed source. Scan-ROI diffraction and
center-of-mass products remain explicitly unsupported until exact compact
reducers are implemented.

`GET /api/browse/residency` reports whether the requested plan is resident, its
source kind, CUDA device, measured resident bytes, exact shapes and dtype,
source identity, load-phase metrics, and whether the catalogued source changed
after loading. It never triggers a load. A stale resident entry fails closed on
the next scientific request.
