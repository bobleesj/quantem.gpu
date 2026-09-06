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

Packaged Live4DSTEM Windows uses one explicit ordered compatibility chain:

```text
quantem-live-browse/3 -> live4dstem-standalone/3 -> quantem-gpu-browse/1
```

The raw CUDA service remains `quantem-gpu-browse/1`. Its capabilities response
includes `packaged_service` with schema
`quantem.gpu.packaged-browse-service/v2`, the exact implementation revision,
the ordered chain, and the field contracts consumed through the loopback
adapter. The adapter must reject any other upstream protocol/version and the
client must reject any other client protocol, adapter, or upstream declaration.
The distributable JSON Schema is
`src/quantem/gpu/remote/packaged_service.schema.json`.

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

## Lossless-packed exact residency

The server can bind a catalogued `*_master.h5` acquisition to one immutable
Lossless Pack Format v1 artifact. Clients continue to send the
same session and master filename to the existing browse routes. The compact
path is trusted server configuration and is never accepted from a client.
The normal CLI loads these bindings from `--compact-sources`; each registry
entry names the catalogued master, compact artifact, and required whole-file
SHA-256. Relative master paths are rooted at the served data folder, while
relative compact paths are rooted at the registry directory.

Both ordinary and packed sources enter through `quantem.gpu.io.load`.
The server's trusted `CompactBrowseSource` may carry a `SourceIntegrity` value
from an externally sealed byte-range manifest. Registry preparation is exposed
through `prepare_browse_source` and the matching `prepare-browse` CLI command;
see [deployment](deployment.md).

Compact bindings are full-coverage plans with scan bin 1, detector bin 1, no
crop, and exact integer output. A request for a transformed plan fails rather
than silently loading a different representation. Preset BF, ABF, ADF, HAADF,
and DF require source-bound detector calibration. Custom detector masks and
selected diffraction use the resident packed source. CoM-row, CoM-column,
CoM magnitude/DPC, and integrated CoM are available when the source contains
authenticated exact total and detector-coordinate moments. A source without
that extension receives a corrective unsupported response; there is no silent
dense expansion or invented calibration. Scan-ROI diffraction remains
unsupported on this compact browse route.

The service owns and releases each packed allocation on eviction and shutdown.
In-process callers own the returned `FourDSTEMData` and must close it after
their scientific operations. Device result views may borrow source-owned
storage; copy a result that must outlive its next update or source closure.

`GET /api/browse/residency` reports whether the requested plan is resident, its
source kind, CUDA device, measured resident bytes, exact shapes and dtype,
source identity, load-phase metrics, and whether the catalogued source changed
after loading. It never triggers a load. A stale resident entry fails closed on
the next scientific request.

The response schema is `quantem.gpu.browse-residency/v2`. Its canonical
`representation` is `packed` or `dense`; `storage_kind` remains a
legacy alias during client migration. For a lossless-packed resident source it
reports `logical_tensor_bytes` separately from `physical_resident_bytes`, the
storage schema, complete source
and working dtype/shape, compact whole-file and logical-source hashes, the
served implementation revision, and the exact plan. `time_to_resident_ready_ms`
is server-owned and is the complete authenticated CUDA-residency interval.
First resident-backed presentation, detector-ready presentation, and
p50/p95/max/sample-count summaries are client-owned. The service reports those
presentation fields as unavailable rather than substituting upload or kernel
completion.
