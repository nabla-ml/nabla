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

"""Utilities for gradient computation and pytree handling."""

from collections.abc import Sequence
from typing import Any, Union


def select_gradients_by_argnums(
    all_gradients: Any, args: tuple, argnums: Union[int, Sequence[int]]
) -> Any:
    """
    Select gradients based on argnums, matching JAX behavior exactly.

    JAX behavior:
    1. Single argument: grad(func)(x) -> returns gradient with same structure as x
    2. Multiple arguments: grad(func, argnums=i)(x, y, z) -> returns gradient w.r.t. arg i
    3. Multiple arguments with multiple argnums: returns tuple of gradients

    The key insight: argnums refers to the FUNCTION ARGUMENTS, not elements within
    a single argument structure.

    Parameters:
        all_gradients: The gradients returned by VJP (preserves input structure)
        args: The original function arguments
        argnums: Which function arguments to compute gradients for

    Returns:
        Selected gradients matching JAX behavior exactly
    """
    num_inputs = len(args)

    # Normalize argnums to sequence for uniform handling
    if isinstance(argnums, int):
        argnums_seq = [argnums]
        return_single = True
    else:
        argnums_seq = list(argnums)
        return_single = False

    # Validate argnums are within bounds
    for idx in argnums_seq:
        if idx < 0 or idx >= num_inputs:
            raise ValueError(
                f"argnum {idx} is out of bounds for function with {num_inputs} arguments"
            )

    if num_inputs == 1:
        # Single input case - return full gradient structure for argnum=0
        # This matches JAX: grad(func)(pytree) -> returns gradient with same structure
        if argnums_seq == [0]:
            return all_gradients if return_single else (all_gradients,)
        else:
            # Invalid argnum for single input
            invalid = [idx for idx in argnums_seq if idx != 0]
            raise ValueError(
                f"argnums {invalid} are out of bounds for function with 1 argument"
            )
    else:
        # Multiple input case - select gradients by argument position
        # all_gradients is a tuple with one gradient per argument
        selected = [all_gradients[i] for i in argnums_seq]

        return selected[0] if return_single else tuple(selected)


def validate_scalar_output(obj: Any) -> None:
    """
    Validate that the function output is scalar-like for gradient computation.

    Parameters:
        obj: The function output to validate

    Raises:
        ValueError: If the output is not scalar-like
    """
    from ..core.tensor import Tensor

    if isinstance(obj, Tensor):
        # JAX behavior: allow both () and (1,) shapes as "scalar-like"
        if obj.shape != () and obj.shape != (1,):
            raise ValueError(
                f"Gradient only defined for scalar-output functions. "
                f"Found tensor with shape: {obj.shape}"
            )
    elif isinstance(obj, list | tuple):
        for item in obj:
            validate_scalar_output(item)
    elif isinstance(obj, dict):
        for value in obj.values():
            validate_scalar_output(value)
    else:
        # Handle non-Tensor outputs (like numpy tensors, Python scalars)
        import numpy as np

        test_tensor = np.astensor(obj)
        if test_tensor.shape != () and test_tensor.shape != (1,):
            raise ValueError(
                f"Gradient only defined for scalar-output functions. "
                f"Found non-scalar with shape: {test_tensor.shape}"
            )


def create_ones_like_cotangent(obj: Any) -> Any:
    """
    Create a cotangent with ones_like for each Tensor leaf in the structure.

    Parameters:
        obj: The object to create cotangent for

    Returns:
        Cotangent with same structure but ones_like for Tensor leaves
    """
    from ..core.tensor import Tensor
    from ..ops.creation import ones_like

    if isinstance(obj, Tensor):
        return ones_like(obj)
    elif isinstance(obj, list | tuple):
        return type(obj)(create_ones_like_cotangent(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: create_ones_like_cotangent(v) for k, v in obj.items()}
    else:
        # For non-Tensor leaves, we don't need cotangents
        return obj
