# Connect locally or over SSH

QuantEM.GPU Remote always speaks the same loopback HTTP protocol. The client
location determines whether an SSH transport is needed.

## Client on the CUDA host

Start the service on loopback and connect directly:

```text
http://127.0.0.1:8780
```

This is useful for a local process boundary, integration tests, and service
profiling. It does not make the service a different CUDA backend.

## Client on another machine

Keep the service bound to loopback on the CUDA host. From the client machine,
create a local SSH port forward:

```bash
ssh -N -L 8780:127.0.0.1:8780 cuda-host
```

The client still connects to `http://127.0.0.1:8780`; SSH transports that
connection to the service host and provides authentication and encryption.
Choose a different local port if 8780 is already occupied.

## Connection ownership

The transport owns authentication, encryption, reconnect behavior, and port
forwarding. The QuantEM.GPU protocol owns scientific requests, typed errors,
array payloads, and provenance. Do not place credentials in dataset URLs,
request bodies, logs, or cache identities.

A client should first read `/api/browse/capabilities`, validate the protocol
name/version and implementation revision, then enable only the operations the
service advertises. A network success is not scientific compatibility.

## Failure behavior

Treat these conditions separately:

| Condition | Client response |
|---|---|
| tunnel or service unavailable | preserve current result and offer reconnect |
| protocol/version mismatch | stop before issuing scientific work |
| source not ready | show the typed readiness reason |
| GPU capacity rejection | retain the requested plan; ask for an explicit policy change |
| stale source identity | invalidate dependent cached products |

Never convert a connectivity or capacity failure into silent CPU execution,
scan cropping, detector binning, or dtype reduction.
