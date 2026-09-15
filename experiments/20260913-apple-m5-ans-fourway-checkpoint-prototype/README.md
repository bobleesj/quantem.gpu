# Four-way paired runtime tANS checkpoint prototype

This is an opt-in diagnostic for the paired-runtime tANS resident. The
experiment Metal source, `fourway_checkpoint.metal`, splits a 512-count
entropy stream into four independently decodable 64-pair segments. The
`fw_capture_selected_checkpoints` kernel walks each selected stream in decoder
order and captures the state and unread meaningful-bit count after pairs 64,
128, and 192. Each checkpoint occupies three little-endian bytes: ten state
bits and fourteen unread-bit bits; capture rejects a position that cannot fit
that field. `fw_decode_fourway_segment_values` returns
all decoded `UInt16` values, in selected-stream order, and validates each
segment boundary and terminal state.

The additive SPI method
`diagnoseFourWayCheckpoint(selectedStreamIndices:residualDetectorPixels:shaderSourceURL:)`
is the only bridge into the resident's private payload, offsets, modes, and
decoding table. It runtime-compiles this file only when explicitly called.
First it dispatches mode inspection and reports all 256 mode counts plus
coverage against the caller's complete residual detector-pixel set. If any
selected stream is a fallback mode (252–255) or unsupported mode, it rejects
before capture and returns the counts; fallback is never represented by a
zero-valued decode. Selections are limited to 4096 streams to bound diagnostic
output memory. A successful call reports separate mode-inspection, checkpoint
capture, and segmented-decode wall/GPU times; checkpoint, scratch, resident,
compact-offset, and Metal allocation-size bytes; status histograms; and the
exact decoded value vector. Checkpoint capture is a separate full decode pass,
so its time is part of any cache-miss cost. The outcome explicitly says that
parity is unchecked: consumers must compare every value with the serial
reference before making any correctness claim. This selected-stream diagnostic
is not a complete ADF update or a qualified timing path.

Only entropy modes 64 through 95 are supported. Macro/interleaved modes and
fallback modes stay on their existing route and are not decoded by this
prototype. No production default or API behavior changes; the bridge is
available only through the `FourWayCheckpointPrototype` SPI.

The CPU test imports the checkpoint oracle from
`20260913-apple-m5-ans-checkpoint-restart-proof` and covers ordinary entropy,
escape values including 65,535, three-byte checkpoint packing, four-way
restart, and an odd 509-value stream. Run it from this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v
```

The CPU oracle passed (2 tests), and the updated seven-source benchmark product
build passed. The runtime MSL compilation, bindings, and GPU dispatch were not
started; the user asked to stop before the controlled runtime gate. This host
has Command Line Tools but no `metal` command-line compiler, so MSL compilation
must happen only through the resident diagnostic at runtime. Original-HDF5
parity, Metal allocation impact, and all timing remain unverified. The next
step is still the planned one-stream controlled gate, not an optimization
result.

Memory remains a qualification gate. Each stream adds nine checkpoint bytes,
four capture-status bytes, 1,024 decoded-value bytes, and 48 bytes of terminal
state/bit/status storage, plus the selected-ID and mode-inspection buffers.
The earlier campaign estimates a full ADF-selected checkpoint cache at about
201 MB across seven sources and a residual-only checkpoint cache at about
34.4 MB. Compact offsets free about 247.7 MB but had a small measured
slowdown; the bridge reports whether they are active, their allocation, and
the total diagnostic allocations. All measurements must remain under the
existing GPU-memory ceiling.

## Validation boundary

- **CPU checkpoint proof:** previously passed; exercises checkpoint direction,
  packing and exact value reconstruction on adversarial fixtures.
- **Swift package build:** previously passed before the SPI bridge was added;
  it only checked the surrounding package layer.
- **Runtime Metal compilation/dispatch:** pending; root-owned controlled gate.
- **Exact selected-stream GPU parity:** pending comparison of every returned
  `UInt16` against the serial decoder.
- **Full ADF update parity, memory qualification and timing:** not established
  by this selected-stream harness; separate acceptance gates remain required.
