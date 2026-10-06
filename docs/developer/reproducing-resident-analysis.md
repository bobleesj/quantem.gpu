# Reproducing resident count analysis

Use the installed `quantem.gpu` package for scientific computation. Keep exact
study settings, private source locations, accepted results and figure rendering
in the study workspace. A manuscript does not need to be public for its code
dependency to be reusable.

For ordinary scientific work, use `io.load`, array indexing and
`detector.bf` / `detector.adf` as shown in the [README](https://github.com/bobleesj/quantem.gpu/blob/main/README.md).
This guide covers advanced native and exact-integer integration. Its entry
points are `io.load`, `detector.prepare`, `session.frame`
and `session.masked_sum_exact`. There is no separate paper API, codec registry
or common codec base class. Preserve the implementations until evidence
justifies combining them.

## Select the existing path

| Input and purpose | Entry point | Retained implementation | Boundary |
|---|---|---|---|
| Ordinary complete uint8/uint16 H5 into an indexed resident | `io.load(path, backend="cuda", representation="encoded", apply_mask=False)` | Runtime encoded streams in `resident/cuda/counts.py`; H5 loading in `io/encoded.py` | Save portable `.qem` with `io.save`; native format/backend acceptance remains separately gated |
| Portable QuantEM/ANS file | `io.load(path, backend="cuda")` | Container reader in `io/qem.py`, CUDA decoder in `resident/cuda/counts.py`, Metal decoder in `resident/mps/counts.py` | Different format/model from historical storage experiments; MPS also supports its documented subset |
| Older prepared packed source or prepared series folder | Re-export the original acquisition as `.qem` | None; these readers were removed | `io.load` does not open them |
| Original HDF5 into native Metal integer packing | Native Swift `MetalOriginalHDF5Packing` workflow | Swift/Metal sources in the same package repository | Separate native lifecycle; not an automatic Python H5-to-packed conversion |

The `encoded` selector names the representation family. The format/profile selects
the compatible decoder. rANS and tANS streams are not interchangeable.

See [native Metal original-HDF5 packing](../api/original-hdf5-metal-packing.md)
for its entry points. Existing experimental storage codecs remain distinct from the
portable file writer. Do not substitute the public writer into a historical
compression comparison and assume that it reproduces its archive bytes.

## Run through the public API

For general H5 inputs, load each complete acquisition explicitly and prepare
the returned list. Supply the study's ordered source list; acquisition order
is part of reproducibility.

```python
from contextlib import ExitStack

import numpy as np

from quantem.gpu import detector, io

paths = ["acquisition-0.h5", "acquisition-1.h5"]
with ExitStack() as owners:
    acquisitions = [
        owners.enter_context(io.load(
            path, backend="cuda", representation="encoded", apply_mask=False,
            dtype="native", device=0,
        ))
        for path in paths
    ]
    session = detector.prepare(acquisitions)
    mask = np.ones(session.detector_shape, dtype=bool)
    # Host-returning exact outputs are convenient for independent comparisons.
    images = session.masked_sum_exact(mask)
    patterns = session.frame(0)
    np.savez("detector-products.npz", mask=mask, images=images, patterns=patterns)
    del session
```

An all-true mask still respects the stored detector-validity policy. Raw point
patterns retain original counts. Record the validity mask alongside the
requested mask when constructing independent reference sums. Neither example
bins, crops or narrows the source. Weighted masks, mean DP and other operations
have separate support boundaries; a successful binary detector query does not
qualify them.

Keep sources and sessions alive while native consumers use them. Complete
consumer work before closing loaded owners. Native outputs avoid the host
copy; NumPy output is appropriate for reference comparisons and saved results.
Do not report NumPy-save time as GPU kernel time or query time as display FPS.

## Pin code and preserve evidence

Install a recorded release or exact source revision in the study environment.
Record `git rev-parse HEAD` and `git status --short` for a source install, plus
`python -m pip freeze`. A dirty checkout requires an archived patch and any
untracked runtime inputs; a commit identifier alone does not describe it.
Package version alone is insufficient when multiple revisions share a version.

Keep a small study-owned record with:

- Ordered dataset identifiers, complete input checksums (including H5 external
  shards or prepared packs), calibration, dtype, shape and validity policy.
- Package revision/environment and, where supplied,
  `session.backend_metadata` including implementation digest and query ABI.
- `loaded.metadata["resident_profile"]` when supplied, representation, logical
  bytes and reported resident bytes. Missing metadata means unreported, not zero.
- Exact detector masks, scan indices, output arrays and their checksums, plus
  independent reference comparisons. For integer sums use exact comparison;
  float32 workflows need their documented bit-preservation and reduction rules.
- Hardware, driver, timing boundary, repetitions and memory-accounting scope.
  Resident payload/index accounting is not full application peak memory.

Store frozen products separately from figure code. Figure rendering reads the
accepted result bundle; it must not silently rerun loading or scientific
computation. Changing a codec, model, layout or query policy creates a new
study result, not an updated label on an old measurement.

The scientific owner package owns reusable algorithms and their tests.
QuantEM.GPU owns reusable accelerator infrastructure and its tests. The private
study owns
manuscript text, exact publication protocols, failed investigations and figure
layout. Large inputs and outputs belong in durable data storage. Do not copy
private manuscript sources, raw data or workstation credentials into the
software repository, or copy production kernels back into the paper workspace.

## Verify the boundary

The host tests exercise reported representation, native/NumPy products and
portable-file reference round trips:

```bash
python -m pytest -q tests/detector tests/contracts/io/test_ans_public_api.py
```

With an available CUDA device, run the existing general-source workflows:

```bash
python -m pytest -q tests/hardware/cuda/test_streamed_h5.py \
    tests/hardware/cuda/test_native_series.py
```

Host tests with device stand-ins verify Python routing and output semantics;
they do not execute CUDA kernels. Full-series performance and exact reference
arrays remain separate study evidence. A refactor that changes scientific code
must repeat the affected real-data parity endpoint before reusing its claim.

## Revisit implementation overlap later

Python/MPS and Swift retain separate Metal ANS reader sources. Runtime indexed
rANS and paired tANS remain independent implementations with different input
contracts. Future consolidation should start with matching
format fixtures and unchanged native outputs, not with a shared class hierarchy.
Keep old evidence pinned while comparing any replacement through the same public
scientific entry points.
