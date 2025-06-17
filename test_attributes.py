#!/usr/bin/env python3
"""Test script to verify that batch_dims and tensor_value are properly recognized."""

import os
import sys

# Add the nabla directory to the path so we can import it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from max.dtype import DType

from nabla.core.array import Array

# Create a test array
test_array = Array(shape=(2, 3), dtype=DType.float32, batch_dims=(4,))

# These should now work without Pylance errors
print(f"Shape: {test_array.shape}")
print(f"Batch dims: {test_array.batch_dims}")
print(f"Tensor value: {test_array.tensor_value}")

print("✅ All attributes are accessible without errors!")
