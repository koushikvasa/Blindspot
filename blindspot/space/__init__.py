"""Encoding, Gower distance, nearest neighbours, 2-D map coordinates (H2)."""
from .gower import encode, family_weights, gower

__all__ = ["encode", "family_weights", "gower", "map_coords"]


def map_coords(*args, **kwargs):  # lazy: avoids importing umap/numba unless maps are requested
    from .maps import map_coords as _map_coords
    return _map_coords(*args, **kwargs)
