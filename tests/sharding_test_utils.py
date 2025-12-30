"""
Sharding Test Utilities
=======================

Minimal helpers for creating test data.
"""

import numpy as np


def make_array(*shape: int, dtype=np.float32, scale: float = 1.0) -> np.ndarray:
    """Create test array using arange + reshape.
    
    Example:
        make_array(4, 8)  # values 0..31 in shape (4, 8)
        make_array(2, 3, 4, scale=0.1)  # values 0..23 * 0.1
    """
    size = 1
    for s in shape:
        size *= s
    return np.arange(size, dtype=dtype).reshape(shape) * scale


def make_randn(*shape: int, dtype=np.float32, seed: int = 42) -> np.ndarray:
    """Create random normal array with fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(dtype)
