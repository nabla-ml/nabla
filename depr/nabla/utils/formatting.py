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

"""Utility functions for formatting and displaying tensor shapes and dtypes."""

from __future__ import annotations

from typing import Any

from ..core.tensor import Tensor

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


def format_shape_dtype_device(tensor: Tensor) -> str:
    """Format shape and dtype in JAX style with batch_dims numbers in blue and everything else in purple."""
    dtype_str = format_dtype(tensor.dtype)

    # Build the dimension string
    dims_parts = []

    # Add batch dimensions with blue numbers
    if tensor.batch_dims:
        batch_dims_colored = []
        for dim in tensor.batch_dims:
            batch_dims_colored.append(f"{light_purple}{dim}{purple}")
        batch_dims_str = f"{purple},{purple}".join(batch_dims_colored)
        dims_parts.append(batch_dims_str)

    # Add shape dimensions in purple
    if tensor.shape:
        shape_str = f"{purple},{purple}".join(map(str, tensor.shape))
        dims_parts.append(shape_str)

    # Add device information
    device_info = ":" + str(tensor.impl.device.label) + "(" + str(tensor.impl.device.id) + ")"

    # Combine dimensions with comma separator and wrap everything in purple
    if dims_parts:
        all_dims = f"{purple},{purple}".join(dims_parts)
        return f"{purple}{dtype_str}[{all_dims}]{device_info}{reset}"
    else:
        return f"{purple}{dtype_str}[]{device_info}{reset}"

