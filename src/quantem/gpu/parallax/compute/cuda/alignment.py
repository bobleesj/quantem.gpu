"""Compatibility imports; canonical implementation: ``quantem.gpu.parallax.backends.cuda.alignment``."""

from quantem.gpu.parallax.backends.cuda.alignment import (
    _bin_mapping_only as _bin_mapping_only,
    align_vbf_stack_multiscale_cp as align_vbf_stack_multiscale_cp,
    compute_pairwise_shifts_cp as compute_pairwise_shifts_cp,
    compute_reference_shifts_cp as compute_reference_shifts_cp,
    make_periodic_pairs_cp as make_periodic_pairs_cp,
    synchronize_shifts_cp as synchronize_shifts_cp,
)
