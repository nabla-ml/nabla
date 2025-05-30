#!/usr/bin/env python3
"""Test squeeze and unsqueeze operations with negative axes."""

import numpy as np


def test_squeeze_negative_axes():
    """Test squeeze with negative axes."""
    print("Testing squeeze with negative axes...")

    a = np.ones((2, 3, 4))
    # b = np.unsquueze(a, [2]) # what is this op in numpy?
    b = np.expand_dims(a, axis=-3)  # Equivalent to unsqueeze at axis 2
    print(b.shape)


if __name__ == "__main__":
    test_squeeze_negative_axes()
