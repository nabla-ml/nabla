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

from .utils import (
    _extract_arrays_from_pytree,
    make_traced_pytree,
    make_untraced_pytree,
    pushfwd,
)


@overload
def jvp(
    func: Callable[..., Any], primals, tangents, has_aux: Literal[False] = False
) -> tuple[Any, Any]: ...


@overload
def jvp(
    func: Callable[..., Any], primals, tangents, has_aux: Literal[True]
) -> tuple[Any, Any, Any]: ...


def jvp(
    func: Callable[..., Any], primals, tangents, has_aux: bool = False
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """Compute Jacobian-vector product (forward-mode autodiff).

    Args:
        func: Function to differentiate (should take positional arguments)
        primals: Positional arguments to the function (can be arbitrary pytrees)
        tangents: Tangent vectors for directional derivatives (matching structure of primals)
        has_aux: Optional, bool. Indicates whether func returns a pair where the first element
            is considered the output of the mathematical function to be differentiated and the
            second element is auxiliary data. Default False.

    Returns:
        If has_aux is False, returns a (outputs, output_tangents) pair.
        If has_aux is True, returns a (outputs, output_tangents, aux) tuple where aux is the
        auxiliary data returned by func.

    Note:
        This follows JAX's jvp API:
        - Only accepts positional arguments
        - For functions requiring keyword arguments, use functools.partial or lambda
    """
    # Handle inputs correctly based on structure
    # Follow JAX convention: if primals is a tuple with length > 1, treat as multiple arguments
    # If primals is a tuple with length 1, treat as single argument
    is_multi_arg = isinstance(primals, tuple) and len(primals) > 1

    any_primal_traced = any(
        getattr(arg, "traced", False) for arg in _extract_arrays_from_pytree(primals)
    )
    any_tangent_traced = any(
        getattr(arg, "traced", False) for arg in _extract_arrays_from_pytree(tangents)
    )

    # Validate primals and tangents match
    if is_multi_arg:
        if not isinstance(tangents, tuple) or len(primals) != len(tangents):
            raise ValueError(
                f"primals and tangents must have the same structure and length, "
                f"got {len(primals)} primals and {len(tangents) if isinstance(tangents, tuple) else 1} tangents"
            )
    elif isinstance(primals, tuple) and len(primals) == 1:
        # Single argument case wrapped in tuple: primals = (arg,), tangents = (tangent,)
        if not isinstance(tangents, tuple) or len(tangents) != 1:
            raise ValueError(
                "For single argument wrapped in tuple, tangents must also be wrapped in tuple of length 1"
            )
    elif isinstance(tangents, tuple):
        raise ValueError(
            "If primal is a single argument, tangent should also be a single argument"
        )

    # Make traced copies of all inputs
    traced_inputs_pytree = make_traced_pytree(primals)

    # Extract traced args based on structure
    if is_multi_arg:
        # Multiple arguments: func(arg1, arg2, ...)
        traced_args = traced_inputs_pytree
    elif isinstance(primals, tuple) and len(primals) == 1:
        # Single argument wrapped in tuple: func(arg)
        traced_args = (traced_inputs_pytree[0],)
    else:
        # Single argument not wrapped: func(arg)
        traced_args = (traced_inputs_pytree,)

    # Execute the function with traced inputs
    outputs = func(*traced_args)

    # Compute output tangents
    output_tangents = pushfwd(traced_inputs_pytree, outputs, tangents)

    # Make everything untraced before returning
    if not any_primal_traced and not any_tangent_traced:
        make_untraced_pytree(outputs)
        make_untraced_pytree(output_tangents)

    return outputs, output_tangents
