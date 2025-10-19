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

from collections.abc import Callable
from typing import Any, Literal, overload

from ..core.tensor import Tensor
from .utils import (
    _convert_scalars_to_tensors,
    _convert_to_scalar_if_needed,
    _extract_tensors_from_pytree,
    make_traced_pytree,
    make_untraced_pytree,
    pullback,
)


@overload
def vjp(
    func: Callable[..., Any], *primals, has_aux: Literal[False] = False
) -> tuple[Any, Callable]: ...


@overload
def vjp(
    func: Callable[..., Any], *primals, has_aux: Literal[True]
) -> tuple[Any, Callable, Any]: ...


def vjp(
    func: Callable[..., Any], *primals, has_aux: bool = False
) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
    """Compute vector-Jacobian product (reverse-mode autodiff).

    Args:
        func: Function to differentiate (should take positional arguments)
        *primals: Positional arguments to the function (can be arbitrary pytrees)
        has_aux: Optional, bool. Indicates whether `func` returns a pair where the
            first element is considered the output of the mathematical function to be
            differentiated and the second element is auxiliary data. Default False.

    Returns:
        If has_aux is False:
            Tuple of (outputs, vjp_function) where vjp_function computes gradients.
        If has_aux is True:
            Tuple of (outputs, vjp_function, aux) where aux is the auxiliary data.

        The vjp_function always returns gradients as a tuple (matching JAX behavior):
        - Single argument: vjp_fn(cotangent) -> (gradient,)
        - Multiple arguments: vjp_fn(cotangent) -> (grad1, grad2, ...)

        Note:
        This follows JAX's vjp API exactly:
        - Only accepts positional arguments
        - Always returns gradients as tuple
        - For functions requiring keyword arguments, use functools.partial or lambda
    """
    # Keep a reference to the original primals before conversion
    original_primals = primals

    # Convert all scalar numbers to Nabla tensors to ensure they are traceable
    primals = _convert_scalars_to_tensors(primals)

    # Handle the input structure based on number of arguments
    if len(primals) == 1:
        inputs_pytree = primals[0]
        is_single_arg = True
    else:
        inputs_pytree = primals
        is_single_arg = False

    any_arg_traced = any(
        getattr(arg, "traced", False)
        for arg in _extract_tensors_from_pytree(inputs_pytree)
    )

    # Make traced copies of all inputs
    traced_inputs_pytree = make_traced_pytree(inputs_pytree)

    # Extract traced args based on the structure
    traced_args = (traced_inputs_pytree,) if is_single_arg else traced_inputs_pytree

    # Execute the function with traced inputs
    full_outputs = func(*traced_args)

    # Handle has_aux: separate main outputs from auxiliary data
    if has_aux:
        if not isinstance(full_outputs, tuple) or len(full_outputs) != 2:
            raise ValueError(
                "Function with has_aux=True must return a tuple (output, aux)"
            )
        outputs, aux = full_outputs
    else:
        outputs = full_outputs
        aux = None

    def vjp_fn(cotangents: Any) -> Any:
        """VJP function that computes gradients."""
        # Always ensure cotangents are traced for composability
        traced_cotangents = make_traced_pytree(cotangents)

        # Use the unified pullback function with pytree support
        gradients = pullback(traced_inputs_pytree, outputs, traced_cotangents)

        # Check if original cotangents were traced - if so, keep gradients traced
        cotangent_tensors = _extract_tensors_from_pytree(cotangents)
        any_cotangent_traced = any(
            getattr(arr, "traced", False) for arr in cotangent_tensors
        )

        # Only make gradients untraced if original cotangents were not traced
        if not any_cotangent_traced and not any_arg_traced:
            make_untraced_pytree(gradients)

        # Convert gradients back to scalars if the original primal was a scalar
        original_inputs_pytree = original_primals[0] if is_single_arg else original_primals
        final_gradients = _convert_to_scalar_if_needed(original_inputs_pytree, gradients)

        return final_gradients

    # Make outputs untraced before returning
    if not any_arg_traced:
        make_untraced_pytree(outputs)

    # Return based on has_aux
    if has_aux:
        return outputs, vjp_fn, aux
    else:
        return outputs, vjp_fn
