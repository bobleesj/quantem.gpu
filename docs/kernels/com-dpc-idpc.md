# CoM, DPC, and iDPC

Center of mass converts each diffraction pattern into a detector-space vector.
For optional detector mask $M[q_r,q_c]$, first compute

```text
4D counts → fused intensity/row-moment/column-moment reduction
          → CoM row/column fields → center and rotation selection
          → Fourier integration → scan-shaped iDPC phase
```

```{admonition} Executable reference, not pseudocode
:class: note
The PyTorch functions below are ordinary executable reference code. They state
the same row/column convention, rotation search, Fourier factor, DC handling,
and final sign used by the maintained implementation. Production CUDA,
MPS/Metal, and WebGPU kernels fuse or stream these operations for performance.
```

## PyTorch reference: CoM

### Step 1 — Define the tensor axes

Start with a 4D count tensor `counts_R_q` whose axes are
`(scan_row, scan_column, detector_row, detector_column)`, plus an optional
detector mask `mask_q` with shape `(detector_row, detector_column)`:

```python
import torch
```

### Step 2 — Apply the detector mask

$$
S[R_r,R_c]=\sum_{q_r,q_c}M[q_r,q_c]I[R_r,R_c,q_r,q_c].
$$

The detector axes are explicitly dimensions 2 and 3. The mask is applied
before all three reductions:

```python
def masked_counts(
    counts_R_q: torch.Tensor,
    mask_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return float32 counts after applying one detector-space mask."""
    counts_float_R_q = counts_R_q.to(torch.float32)
    if mask_q is None:
        return counts_float_R_q
    return counts_float_R_q * mask_q.to(
        device=counts_R_q.device,
        dtype=torch.float32,
)
```

### Step 3 — Compute intensity and both CoM components

Then

$$
\mu_r[R_r,R_c]
=\frac{\sum_{q_r,q_c}q_rM[q_r,q_c]I[R_r,R_c,q_r,q_c]}
{S[R_r,R_c]},
$$

$$
\mu_c[R_r,R_c]
=\frac{\sum_{q_r,q_c}q_cM[q_r,q_c]I[R_r,R_c,q_r,q_c]}
{S[R_r,R_c]}.
$$

The matching reference computes intensity and both moments together, makes
zero-intensity frames finite, and subtracts the scan-field means just as the
public CoM workflow does:

```python
def center_of_mass_reference(
    counts_R_q: torch.Tensor,
    mask_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean-centered CoM in public detector (row, column) order."""
    weighted_R_q = masked_counts(counts_R_q, mask_q)
    detector_dimensions = (2, 3)
    intensity_R = weighted_R_q.sum(dim=detector_dimensions)
    safe_intensity_R = torch.where(
        intensity_R > 0,
        intensity_R,
        torch.ones_like(intensity_R),
    )

    detector_rows, detector_columns = counts_R_q.shape[2:4]
    q_row = torch.arange(
        detector_rows,
        device=counts_R_q.device,
        dtype=torch.float32,
    ).reshape(1, 1, detector_rows, 1)
    q_column = torch.arange(
        detector_columns,
        device=counts_R_q.device,
        dtype=torch.float32,
    ).reshape(1, 1, 1, detector_columns)

    com_row_R = (weighted_R_q * q_row).sum(
        dim=detector_dimensions
    ) / safe_intensity_R
    com_column_R = (weighted_R_q * q_column).sum(
        dim=detector_dimensions
    ) / safe_intensity_R

    valid_R = intensity_R > 0
    com_row_R = torch.where(valid_R, com_row_R, torch.zeros_like(com_row_R))
    com_column_R = torch.where(
        valid_R,
        com_column_R,
        torch.zeros_like(com_column_R),
    )
    return com_row_R - com_row_R.mean(), com_column_R - com_column_R.mean()
```

`com_row` is $\mu_r$ and `com_col` is $\mu_c$:

$$
(\text{row},\text{column})\equiv(r,c).
$$

In plain terms, `(row, column)` is `(r, c)`.

This explicit naming is required at Python, Swift, CUDA, Metal, and WebGPU
boundaries; a backend may not swap components to match launch coordinates.

## Coordinate, shape, dtype, unit, and provenance contract

The detector input is
`I[scan_row, scan_column, detector_row, detector_column]`. `com_row`,
`com_col`, their aligned components, and `phase` are float32 arrays with shape
`(scan_row, scan_column)`. Detector coordinates are expressed in detector
pixels unless calibrated units are explicitly provided. Zero-intensity or
fully masked frames produce finite zero moments rather than division-by-zero
values.

Provenance records source identity and loaded geometry, detector bin/crop,
source and moment dtypes, detector mask/checksum, detector and scan calibration,
rotation search configuration, selected `rotation_deg`, `use_transpose`,
backend/device, and package revision. The phase convention and FFT
normalization are part of the result contract, not display choices.

## From CoM to DPC and iDPC

The CoM field is centered and rotated into a DPC field
$\mathbf g=(g_r,g_c)$. When automatic rotation is requested, the chosen angle
minimizes the configured curl criterion on the scan-shaped vector field.

### Step 4 — Rotate the DPC vector field

For one angle $\theta$, including the optional component-order test used by the
automatic search,

$$
g_r=\cos\theta\,\mu_r-\sin\theta\,\mu_c,
\qquad
g_c=\sin\theta\,\mu_r+\cos\theta\,\mu_c.
$$

```python
def rotate_dpc_reference(
    com_row_R: torch.Tensor,
    com_column_R: torch.Tensor,
    angle_deg: float,
    *,
    use_transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a scan-shaped CoM field without changing spatial axes."""
    source_row_R = com_column_R if use_transpose else com_row_R
    source_column_R = com_row_R if use_transpose else com_column_R
    angle_rad = torch.deg2rad(
        torch.tensor(angle_deg, device=com_row_R.device, dtype=torch.float32)
    )
    cosine = torch.cos(angle_rad)
    sine = torch.sin(angle_rad)
    aligned_row_R = cosine * source_row_R - sine * source_column_R
    aligned_column_R = sine * source_row_R + cosine * source_column_R
    return aligned_row_R, aligned_column_R
```

Here `use_transpose` means testing the exchanged vector components; it does not
transpose the scan image. The readable automatic search evaluates the same
central-difference curl objective for both component orderings:

### Step 5 — Score each candidate by curl

```python
def curl_score(dpc_row_R: torch.Tensor, dpc_column_R: torch.Tensor) -> torch.Tensor:
    """Return the mean squared interior curl of one DPC vector field."""
    scan_rows, scan_columns = dpc_row_R.shape
    row_change_of_column = 0.5 * (
        dpc_column_R[2:scan_rows, 1:scan_columns - 1]
        - dpc_column_R[0:scan_rows - 2, 1:scan_columns - 1]
    )
    column_change_of_row = 0.5 * (
        dpc_row_R[1:scan_rows - 1, 2:scan_columns]
        - dpc_row_R[1:scan_rows - 1, 0:scan_columns - 2]
    )
    curl_R = row_change_of_column - column_change_of_row
    return torch.mean(curl_R * curl_R)
```

### Step 6 — Select the minimum-curl rotation

```python
def select_dpc_rotation_reference(
    com_row_R: torch.Tensor,
    com_column_R: torch.Tensor,
    rotation_steps: int = 180,
) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Choose the angle and component order with the smallest curl score."""
    angles_deg = torch.linspace(
        0.0,
        180.0,
        rotation_steps,
        device=com_row_R.device,
        dtype=torch.float32,
    )
    candidates: list[tuple[torch.Tensor, float, bool]] = []
    for use_transpose in (False, True):
        for angle_deg in angles_deg:
            dpc_row_R, dpc_column_R = rotate_dpc_reference(
                com_row_R,
                com_column_R,
                float(angle_deg),
                use_transpose=use_transpose,
            )
            candidates.append(
                (curl_score(dpc_row_R, dpc_column_R), float(angle_deg), use_transpose)
            )

    _, selected_angle_deg, selected_transpose = min(
        candidates,
        key=lambda candidate: float(candidate[0]),
    )
    selected_row_R, selected_column_R = rotate_dpc_reference(
        com_row_R,
        com_column_R,
        selected_angle_deg,
        use_transpose=selected_transpose,
    )
    return selected_row_R, selected_column_R, selected_angle_deg, selected_transpose
```

The production search evaluates this objective from precomputed
curl/divergence moments instead of materializing every rotated candidate. The
selected angle and component order remain identical to the readable reference.

### Step 7 — Fourier-integrate the aligned field

Integrated DPC reconstructs a scalar phase-like field in Fourier space. With
scan frequency $\mathbf k=(k_r,k_c)$, a standard least-squares integration is

$$
\hat\phi(\mathbf k)
 =\frac{-0.25i\,[k_r\hat g_r(\mathbf k)+k_c\hat g_c(\mathbf k)]}
{k_r^2+k_c^2+\epsilon},
$$

with the zero-frequency value and normalization fixed by the shared contract.
The inverse two-dimensional FFT returns $\phi[R_r,R_c]$.

```python
def integrate_idpc_reference(
    dpc_row_R: torch.Tensor,
    dpc_column_R: torch.Tensor,
) -> torch.Tensor:
    """Fourier-integrate DPC using the maintained iDPC sign convention."""
    scan_rows, scan_columns = dpc_row_R.shape
    k_row = torch.fft.fftfreq(
        scan_rows,
        device=dpc_row_R.device,
        dtype=torch.float32,
    )
    k_column = torch.fft.fftfreq(
        scan_columns,
        device=dpc_row_R.device,
        dtype=torch.float32,
    )
    k_row_R, k_column_R = torch.meshgrid(k_row, k_column, indexing="ij")

    dpc_row_k = torch.fft.fft2(dpc_row_R.to(torch.float32))
    dpc_column_k = torch.fft.fft2(dpc_column_R.to(torch.float32))
    frequency_squared_R = k_row_R * k_row_R + k_column_R * k_column_R
    safe_frequency_squared_R = frequency_squared_R.clone()
    safe_frequency_squared_R[0, 0] = 1.0

    phase_k = (-0.25j) * (
        k_row_R * dpc_row_k + k_column_R * dpc_column_k
    ) / safe_frequency_squared_R
    phase_k[0, 0] = 0.0
    phase_R = torch.fft.ifft2(phase_k).real.to(torch.float32)
    return -(phase_R - phase_R.mean())
```

### Step 8 — Assemble the complete readable workflow

A complete readable reference is therefore:

```python
com_row_R, com_column_R = center_of_mass_reference(counts_R_q, mask_q)
aligned_row_R, aligned_column_R, angle_deg, use_transpose = (
    select_dpc_rotation_reference(com_row_R, com_column_R)
)

# The production contract swaps the selected components back before integration.
gradient_row_R = aligned_column_R if use_transpose else aligned_row_R
gradient_column_R = aligned_row_R if use_transpose else aligned_column_R
phase_R = integrate_idpc_reference(gradient_row_R, gradient_column_R)
```

These functions are explanatory reference code, not a promise that PyTorch is
the optimized production path. The maintained public workflow is:

### Use the maintained public API

```python
from quantem.gpu import dpc, io

loaded = io.load("scan_master.h5", backend="auto", det_bin=1)
result = dpc.run(loaded.data)

print(result.com_row.shape, result.com_col.shape)
print(result.rotation_deg, result.use_transpose)
```

## Optimization model

CoM should not require three full detector-volume traversals. A fused moment
kernel accumulates $S$, $\sum q_rI$, and $\sum q_cI$ in one pass, with masks
and bad-pixel treatment applied in the same order as the reference. The large
source remains accelerator-resident; only the small scan-shaped moment fields
continue to rotation and FFT integration.

Rotation search operates on those scan-shaped fields. Batched analytic
curl/divergence moments avoid allocating a full rotated field for every
candidate angle. iDPC keeps both vector components and FFT intermediates on the
same device until the final result is requested.

## Source map and gates

| Layer | Source |
|---|---|
| Public workflow and result | `src/quantem/gpu/dpc` |
| CUDA CoM/DPC | `src/quantem/gpu/dpc/compute/cuda` and detector CUDA moment kernels |
| Python MPS/Metal | `src/quantem/gpu/dpc/compute/mps` |
| WebGPU | `src/quantem/gpu/dpc/compute/webgpu` |
| Native Metal and FFT | `Metal4DSTEMKernels` and `MetalImageFFT` |

Parity reports compare `com_row`, `com_col`, centered/rotated DPC components,
`rotation_deg`, transpose convention, and iDPC phase. They include asymmetric
detector patterns and rectangular scans specifically to catch row/column swaps.
