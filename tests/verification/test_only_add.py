#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Test with only add operations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import nabla

print("   Creating arrays...")
x = nabla.arange((2, 2))
y = nabla.randn((2, 2), seed=42)

print("   Testing function calls...")
# Test function calls - only add operations
z1 = nabla.add(x, y)
z2 = nabla.add(x, y)
z3 = nabla.add(x, y)
z4 = nabla.add(x, y)

print("   Realizing results...")
# Realize results
print("     Realizing z1 (add)...")
z1.realize()
print("     Realizing z2 (add)...")
z2.realize()
print("     Realizing z3 (add)...")
z3.realize()
print("     Realizing z4 (add)...")
z4.realize()

print(f"   z1 result shape: {z1.shape}")
print(f"   z2 result shape: {z2.shape}")
print(f"   z3 result shape: {z3.shape}")
print(f"   z4 result shape: {z4.shape}")

print("✅ All operations completed successfully!")
