"""Reuse optical coefficients and unchanged chi(k) across CUDA candidates."""

import cupy as cp

from .broad_objective import code as original_source

source = original_source.replace(
    "__launch_bounds__(256, 4)", "__launch_bounds__(256, 3)"
)
source = source.replace(
    "__shared__ float4 geometry_k[256];",
    "__shared__ float chi_k[MAXB * 256];\n    __shared__ float4 geometry_k[256];",
)
anchor = "geometry_k[offset] = tf_geometry(kx[k], ky[k], wl, semiangle, ang_y, ang_x, inv_ang);"
source = source.replace(
    anchor,
    anchor
    + "\n            for (int b = 0; b < B; ++b) chi_k[b * 256 + offset] = ssb_chi_full(kx[k], ky[k], wl, coefficients + b * 28);",
)
source = source.replace(
    "ssb_chi_full(kxv,kyv,wl,coefficients+b*28)", "chi_k[b * 256 + k - k0]"
)
kernel = cp.RawKernel(
    source + "\n// Independent module for cache allocation experiment.\n",
    "thick_fit_batch",
)
kernel.preferred_shared_memory_carveout = 50


base_source = source
source = base_source.replace(
    "    // per parameter set:",
    """    __shared__ float shared_coefficients[MAXB * 28];
    for (int i=threadIdx.x; i<B*28; i+=blockDim.x)
        shared_coefficients[i]=coefficients[i];
    __syncthreads();
    // per parameter set:""",
)
assert source != base_source
assert source.count("coefficients+b*28") == 2
assert source.count("coefficients + b * 28") == 1
source = source.replace("coefficients+b*28", "shared_coefficients+b*28")
source = source.replace("coefficients + b * 28", "shared_coefficients + b * 28")
kernel = cp.RawKernel(source, "thick_fit_batch")
kernel.preferred_shared_memory_carveout = 50
