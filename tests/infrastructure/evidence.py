"""Skip marker for tests that read the private experiment evidence archive.

Raw experiment records live in the quantem.gpu-experiments archive, not in
this repository. ``scripts.benchmark_registry.EVIDENCE_ROOT`` resolves it from
``QUANTEM_GPU_EXPERIMENTS`` or a sibling checkout; without it, tests that need
evidence file contents are skipped instead of failing.
"""

import pytest

from scripts.benchmark_registry import EVIDENCE_ROOT

requires_evidence = pytest.mark.skipif(
    EVIDENCE_ROOT is None,
    reason=(
        "needs the private experiment evidence archive; set "
        "QUANTEM_GPU_EXPERIMENTS to a quantem.gpu-experiments checkout"
    ),
)
