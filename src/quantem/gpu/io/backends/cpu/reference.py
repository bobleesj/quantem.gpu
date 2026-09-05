"""Compatibility imports for the canonical ``dense`` backend.

New internal callers use the representation-named module. These aliases
retain the same functions/classes without a second implementation.
"""

from .dense import (
    _bin_sum as _bin_sum,
    load_master as load_master,
)
