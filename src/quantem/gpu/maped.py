"""GPU-resident MAPED merge backend used by the QuantEM workflow package."""

from ._maped import merge_to_scaled_h5

__all__ = ["merge_to_scaled_h5"]
