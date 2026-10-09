# Scientific contract

Accelerator parity begins with meaning, not speed. Every backend must preserve
the same source geometry, coordinate convention, arithmetic, and provenance
before its performance can be compared.

## Coordinates and regions

Spatial coordinates use

$$
(\text{row},\text{column}) \equiv (r,c)
$$

throughout Python, Swift, Metal, CUDA, and WebGPU. Real-space probe/scan
coordinates are $\mathbf R=(R_r,R_c)$, detector coordinates are
$\mathbf k=(k_r,k_c)$, and logical 4D-STEM array order is
$I[R_r,R_c,k_r,k_c]$. Scan and detector regions are half-open:

```text
(row_start, row_stop, column_start, column_stop)
```

Regions are scientific choices. They are never introduced automatically to
make a dataset fit or a benchmark pass.

## Binning

Scan and detector bins are explicit positive integers. Integer detector binning
sums counts and widens the accumulation dtype before overflow. Partial edge
bins are retained using ceiling output dimensions.

Automatic resource policy may recommend or select detector binning only when
the consuming application makes that choice visible and records:

- the requested and selected bin;
- source and output detector shape;
- source, accumulation, and output dtype; and
- the memory reason for the choice.

A binned result must never be labeled or cached as native detector resolution.

## Precision and masking

Native detector counts remain integer evidence. A bad-pixel mask is applied in
the same order on every backend and is part of source identity. Converting to
`uint8` is lossless only when an exact value-range audit proves every corrected
count fits. Reconstruction workflows retain the precision required by their
objective.

## Provenance

Every reusable result or parity bundle records at least:

- source identity and source revision;
- source scan/detector shape and dtype;
- scan and detector region;
- scan and detector bin;
- bad-pixel policy and mask identity;
- output shape, output dtype, and accumulation dtype;
- backend, device, and kernel revision; and
- whether the evidence is native, cropped, binned, cached, or reconstructed.

## Failure behavior

An unsupported accelerated path fails with a corrective error. Production APIs
do not silently move scientific work to CPU, crop a scan, bin a detector,
reduce a mask, or change precision. The CPU implementation is available only
when explicitly selected as a reference.

See [Cross-backend parity](../performance/parity.md) for numerical gates and
[Kernel architecture](kernel-architecture.md) for the implementation boundary.

(resident-integer-detector-products-v1)=
## Resident integer detector products v1

**Contract ID: `quantem.gpu.resident-integer-products/v1`.** A scientist selects
detector pixels to form a scan image, then inspects diffraction at chosen scan
positions without changing the source or the next detector result.

The advanced Python operation handle is `detector.prepare(data)`, followed by
`masked_sum_exact(mask)` and `frame(index)`. This contract specifies observable
counts, not a common kernel, storage layout, buffer class, or performance target.
Ordinary users call `detector.bf(data)`, `detector.adf(data)` or `data[row, column]`;
the session API is for exact integer totals and native-buffer integration.
It is a narrow profile of `detector.integer-products` in
`tests/parity/backend_matrix.json`, not a new capability or evidence registry.

### Inputs, outputs, and ordering

1. The logical input is a complete `uint16` working array in
   `(scan_row, scan_column, detector_row, detector_column)` order. The fixture
   retains values `0`, `255`, `256`, `32768`, and `65535`; no downcast, bin, crop,
   or normalization is implicit. Already declared source exclusions are zeroed
   before detector selection and remain part of source identity.
2. The detector mask is an explicit binary array with the full detector shape.
   Python's exact session API rejects wrong-shape and nonbinary masks before
   dispatch; boolean and numeric `0`/`1` masks have the same selection meaning.
   At each scan position, sum exactly the selected working counts. Empty masks
   return zero. Integer accumulation and an integer result must preserve every
   count, including sums above the exact `float32` range. Python
   `masked_sum_exact` returns `uint64`; a native `uint32` product is equivalent
   only when its complete sum is proven to fit. Canonical fixture comparison
   widens integer outputs to little-endian uint64 without changing values.
3. A selected DP is the complete working detector pattern at one scan
   coordinate. Its output retains the working integer dtype or an explicitly
   documented wider integer dtype. Do not route scientific counts through a
   display conversion. Repeated requests retain order and duplicates. Python's
   public `frame` returns an independent small copy: editing that output does
   not mutate resident counts or invalidate cached detector products.
4. Python `frame` takes a logical row-major flat index in
   `0 <= index < scan_rows * scan_columns`. A caller holding `(row, column)`
   checks both coordinate bounds before using `row * scan_columns + column`.
   Native coordinate APIs check the two bounds themselves. Negative indexing
   is not part of this scientific selection contract.
5. A selected-frame **sum** is a separate optional operation: request order is
   irrelevant to addition, but duplicates contribute repeatedly. Python dense
   `reduce_frames_exact` covers it. Lack of packed support must be reported,
   not replaced by a hidden dense expansion.

The existing `bf`, `adf`, `df`, `masked_sum`, and `mean_dp` display-oriented
results are not promises of integer output. In particular, converting an exact
sum to `float32` can round a count above 2^24. Their established behavior and
floating tolerances are unchanged by this contract.

### Geometry is not yet one interchangeable radius API

Mask bytes are the core contract. Radius-to-mask conversion has an existing
incompatibility that must not be hidden by a backend adapter:

| Existing entry point | Edge convention |
| --- | --- |
| Python `detector.detector_mask` and convenience BF/ADF | Both inner and outer boundaries included; distance is evaluated through the existing float32 square-root path. |
| WebGPU `detector/webgpu/geometry.ts` | Both boundaries included, using JavaScript squared-distance arithmetic. |
| Native Swift/Metal mask-driven reductions | Consume caller-supplied masks; the caller's geometry must be declared independently. |

The frozen counterexample is a 3 by 3 detector, center `(1, 1)`, inner radius
zero, and outer radius one. The closed-edge mask selects **five** pixels; a
half-open mask (inner included, outer excluded) selects **one**. These are different scientific
requests even if the displayed radii look identical. The fixture includes both
literal masks and names its moving-aperture provenance
`binary32-inner-inclusive-outer-exclusive`. That name does not change the
Python or browser convenience APIs or make them radius-conformant.

Any future unification needs an explicit geometry/version migration and
independent edge fixtures. Existing radius expectations must not be rewritten
to make a backend appear to agree.

### Lifecycle and failure requirements

An invalid request must not mutate the working source or a previously valid
detector result. Failed source authentication, incomplete loading, or allocation
failure must not publish a partially ready resident source. Closing an owned
resident source releases its resources once and invalidates subsequent source
operations; closing a borrowed NumPy array does not destroy caller-owned data.
Interleaved selected-DP requests must not overwrite the detector generation or
its source identity. An A-B-A file sequence must give source A's original
counts when A is reopened, not a result retained from B.

These requirements do not prescribe a common session state machine. Existing
tests retain the specific ownership and publication checks:

- `tests/hardware/test_native_ssb.py`: SSB refuses encoded storage, never
  releases borrowed data, and releases the acquisition `SSB.open` loads.

### Frozen vectors and executed scope

`tests/parity/fixtures/resident_integer_products_v1.json` contains literal raw
and corrected integer patterns, masks, sums, selected-DP ordering with duplicate
positions, the radius-edge counterexample, and a full-uint16 large-sum case.
`tests/parity/resident_integer_oracle.py` checks them using plain NumPy and
Python integer addition without importing production geometry or kernels.
The fixture SHA-256 is pinned in the test; changing its scientific expectations
requires a new contract version, not recapturing a failing backend's output.

Run the narrow public workflows on explicit non-accelerated runners:

```bash
PYTHONPATH=src python -m pytest -q tests/parity/test_resident_integer_contract.py
```

The suite uses NumPy and an explicit CPU Torch tensor for the same mask and
selection vectors. Its A-B-A HDF5 test performs actual file IO with small
synthetic uint16 inputs. It is not real-acquisition parity, cold-load timing,
or a GPU-residency benchmark.

The same frozen values and test-only assertions are also wired into
`tests/hardware/cuda/test_resident_integer_contract.py` and
`tests/hardware/mps/test_resident_integer_contract.py`. These request real CuPy
or raw-Metal native uint16 storage, respectively; neither casts the source to
float32 or substitutes a CPU reducer. The raw-Metal runner explicitly rejects
the unsupported exact selected-frame sum instead of claiming conformance for
it. Their presence and successful collection do not establish hardware parity.

Native Swift/Metal uses
`native/swift/Tests/Native4DSTEMIOTests/ResidentIntegerContractTests.swift`.
Its private uint16 shards force GPU reduction rather than the shared-memory CPU
delta route. The tests include poisoned-output empty-mask reset, nonbinary-mask
rejection, cancellation, and recovery. An initial empty detector must clear the
small result buffer; an unchanged previous/next detector remains a no-op.

No WebGPU adapter runs these vectors. Dense WebGPU `maskedSum` returns display
float32, not an exact integer product, and selected-frame reduction cannot
preserve duplicate scan selections.

Each backend still needs retained execution and scope-specific adjudication
through its genuine entry point; building an adapter is not that evidence.
Passing these small cases does not qualify complete real acquisitions,
application switching, cold loading, minimum-memory hardware, or presentation.
Keep these gaps separate from
existing independent backend tests and from the canonical capability/evidence
matrices.
