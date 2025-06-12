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

from collections.abc import Callable, Sequence
from typing import Any

from ..utils.grad_utils import (
    create_ones_like_cotangent,
    select_gradients_by_argnums,
    validate_scalar_output,
)
from .vjp import vjp


def value_and_grad(
    fun: Callable = None,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence = (),
) -> Callable[..., Any]:
    """
    Creates a function that evaluates both the value and gradient of fun.

    This function uses VJP (Vector-Jacobian Product) directly with a cotangent
    of ones_like(output) to compute gradients for scalar-valued functions.
    This is simpler and more efficient than using jacrev/jacfwd for scalar outputs.

    Parameters:
        fun: Function to be differentiated. Should return a scalar.
        argnums: Which positional argument(s) to differentiate with respect to (default 0).
        has_aux: Whether fun returns (output, aux) pair (default False).
        holomorphic: Whether fun is holomorphic - currently ignored (default False).
        allow_int: Whether to allow integer inputs - currently ignored (default False).
        reduce_axes: Axes to reduce over - currently ignored (default ()).

    Returns:
        A function that computes both the value and gradient of fun.

    Examples:
        Basic usage as a function call::

            value_and_grad_fn = value_and_grad(my_loss)
            value, grads = value_and_grad_fn(x)

        Usage as a decorator::

            @value_and_grad
            def my_loss(x):
                return x**2

            value, grads = my_loss(3.0)
    """

    # Handle being used as a decorator without arguments
    if fun is None:
        return lambda f: value_and_grad(
            f,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        )

    def value_and_grad_fn(*args: Any) -> Any:
        """
            The actual value_and_grad function that gets returned.

            Validates that the function returns a scalar output, then computes
            both the value and gradient using VJP with ones_like cotangent.
        """
        # Compute VJP to get both output and pullback function
        if has_aux:
            output, vjp_fn, aux = vjp(fun, *args, has_aux=True)
        else:
            output, vjp_fn = vjp(fun, *args, has_aux=False)

        # Validate scalar output
        validate_scalar_output(output)

        # Create cotangent of ones_like(output) and compute gradients
        cotangent = create_ones_like_cotangent(output)

        # VJP computes gradients for all inputs, select based on argnums
        all_gradients = vjp_fn(cotangent)

        # Use utility function to handle argnums selection
        selected_gradients = select_gradients_by_argnums(all_gradients, args, argnums)

        # Return based on has_aux - JAX returns ((value, aux), grad) when has_aux=True
        if has_aux:
            return (output, aux), selected_gradients
        else:
            return output, selected_gradients

    return value_and_grad_fn


def grad(
    fun: Callable = None,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence = (),
    mode: str = "reverse",  # Kept for API compatibility but ignored
) -> Callable[..., Any]:
    """
    Creates a function that evaluates the gradient of fun.

    This is implemented as a special case of value_and_grad that only returns
    the gradient part. Uses VJP directly for efficiency with scalar outputs.

    Parameters:
        fun: Function to be differentiated. Should return a scalar.
        argnums: Which positional argument(s) to differentiate with respect to (default 0).
        has_aux: Whether fun returns (output, aux) pair (default False).
        holomorphic: Whether fun is holomorphic - currently ignored (default False).
        allow_int: Whether to allow integer inputs - currently ignored (default False).
        reduce_axes: Axes to reduce over - currently ignored (default ()).
        mode: Kept for API compatibility but ignored (always uses reverse-mode VJP).

        Returns:
            A function that computes the gradient of fun.

        Examples:
            Basic usage as a function call::

                grad_fn = grad(my_loss)
                grads = grad_fn(x)

            Usage as a decorator::

                @grad
                def my_loss(x):
                    return x**2

                grads = my_loss(3.0)  # Returns gradient, not function value
    """
    # Handle decorator pattern: if fun is None, return a decorator
    if fun is None:
        return lambda f: grad(
            f, argnums, has_aux, holomorphic, allow_int, reduce_axes, mode
        )

    # Get the value_and_grad function
    value_and_grad_fn = value_and_grad(
        fun=fun,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    def grad_fn(*args: Any) -> Any:
        """
            The actual gradient function that gets returned.

            Just calls value_and_grad and returns only the gradient part.
        """
        if has_aux:
            (value, aux), gradients = value_and_grad_fn(*args)
            return gradients, aux
        else:
            value, gradients = value_and_grad_fn(*args)
            return gradients

    return grad_fn
