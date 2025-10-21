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

    Examples:
        Basic forward-mode differentiation:

        ```python
        import nabla as nb

        def f(x):
            return x ** 2

        # Compute derivative at x=3.0 in direction v=1.0
        primals = (nb.tensor(3.0),)
        tangents = (nb.tensor(1.0),)
        y, y_dot = nb.jvp(f, primals, tangents)
        # y = 9.0, y_dot = 6.0 (derivative of x^2 at x=3 is 2*3=6)
        ```

        Multiple inputs:

        ```python
        import nabla as nb

        def f(x, y):
            return x * y + x ** 2

        primals = (nb.tensor(3.0), nb.tensor(4.0))
        tangents = (nb.tensor(1.0), nb.tensor(0.0))  # Differentiate w.r.t. x only
        output, tangent_out = nb.jvp(f, primals, tangents)
        ```

        With auxiliary data:

        ```python
        import nabla as nb

        def f_with_aux(x):
            y = x ** 2
            aux = {"intermediate": x * 2}
            return y, aux

        primals = (nb.tensor(3.0),)
        tangents = (nb.tensor(1.0),)
        y, y_dot, aux = nb.jvp(f_with_aux, primals, tangents, has_aux=True)
        ```
    """
    # Convert scalars to tensors for traceability
    primals = _convert_scalars_to_tensors(primals)
    tangents = _convert_scalars_to_tensors(tangents)

    # Handle inputs correctly based on structure
    is_multi_arg = isinstance(primals, tuple) and len(primals) > 1

    any_primal_traced = any(
        getattr(arg, "traced", False) for arg in _extract_tensors_from_pytree(primals)
    )
    any_tangent_traced = any(
        getattr(arg, "traced", False) for arg in _extract_tensors_from_pytree(tangents)
    )

    # Validate primals and tangents match
    if is_multi_arg:
        if not isinstance(tangents, tuple) or len(primals) != len(tangents):
            raise ValueError(
                f"primals and tangents must have the same structure and length, "
                f"got {len(primals)} primals and {len(tangents) if isinstance(tangents, tuple) else 1} tangents"
            )
    elif isinstance(primals, tuple) and len(primals) == 1:
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
        traced_args = traced_inputs_pytree
    elif isinstance(primals, tuple) and len(primals) == 1:
        traced_args = (traced_inputs_pytree[0],)
    else:
        traced_args = (traced_inputs_pytree,)

    # Execute the function with traced inputs
    full_outputs = func(*traced_args)

    # Handle has_aux
    if has_aux:
        if not isinstance(full_outputs, tuple) or len(full_outputs) != 2:
            raise ValueError(
                "Function with has_aux=True must return a tuple (output, aux)"
            )
        outputs, aux = full_outputs
    else:
        outputs = full_outputs
        aux = None

    # Compute output tangents
    output_tangents = pushfwd(traced_inputs_pytree, outputs, tangents)

    # Convert output tangents to scalars if the corresponding output is a scalar
    final_tangents = _convert_to_scalar_if_needed(outputs, output_tangents)

    # Make everything untraced before returning
    if not any_primal_traced and not any_tangent_traced:
        make_untraced_pytree(outputs)
        make_untraced_pytree(final_tangents)

    if has_aux:
        return outputs, final_tangents, aux
    else:
        return outputs, final_tangents
