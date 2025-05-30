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

"""Utility functions for formatting and displaying array shapes and dtypes."""

from __future__ import annotations

from typing import Any

from ..core.array import Array

# ANSI color codes
light_purple = "\033[94m"
purple = "\033[95m"
reset = "\033[0m"


def format_dtype(dtype: Any) -> str:
    """Format dtype for display."""
    # Convert DType to string representation
    dtype_str = str(dtype).lower()
    if "float32" in dtype_str:
        return "f32"
    elif "float64" in dtype_str:
        return "f64"
    elif "int32" in dtype_str:
        return "i32"
    elif "int64" in dtype_str:
        return "i64"
    else:
        return dtype_str


def format_shape_and_dtype(array: Array) -> str:
    """Format shape and dtype in JAX style with batch_dims in light purple and shape in purple."""
    dtype_str = format_dtype(array.dtype)

    # Build the dimension string with different colors
    dims_parts = []

    # Add batch dimensions in light purple
    if array.batch_dims:
        batch_dims_str = ",".join(map(str, array.batch_dims))
        dims_parts.append(f"{light_purple}{batch_dims_str}{reset}")

    # Add shape dimensions in purple
    if array.shape:
        shape_str = ",".join(map(str, array.shape))
        dims_parts.append(f"{purple}{shape_str}{reset}")

    # Combine dimensions with comma separator
    if dims_parts:
        all_dims = ",".join(dims_parts)
        return f"{dtype_str}[{all_dims}]"
    else:
        return f"{dtype_str}[]"
