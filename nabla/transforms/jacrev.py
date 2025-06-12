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
from typing import Any

from .utils import (
    _extract_arrays_from_pytree,
    _std_basis,
    make_traced_pytree,
    make_untraced_pytree,
)
from .vjp import vjp
from .vmap import vmap


def jacrev(
    func: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable[..., Any]:
    """Compute the Jacobian of a function using reverse-mode autodiff.

    Args:
        func: Function to differentiate (should take positional arguments)
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate with respect to (default 0).
        has_aux: Optional, bool. Indicates whether `func` returns a pair where the
            first element is considered the output of the mathematical function to be
            differentiated and the second element is auxiliary data. Default False.
        holomorphic: Optional, bool. Indicates whether `func` is promised to be
            holomorphic. Default False. Currently ignored.
        allow_int: Optional, bool. Whether to allow differentiating with
            respect to integer valued inputs. Currently ignored.

    Returns:
        A function with the same arguments as `func`, that evaluates the Jacobian of
        `func` using reverse-mode automatic differentiation. If `has_aux` is True
        then a pair of (jacobian, auxiliary_data) is returned.

    Note:
        This follows JAX's jacrev API:
        - Only accepts positional arguments
        - For functions requiring keyword arguments, use functools.partial or lambda
        - Returns the Jacobian as a pytree structure matching the input structure
    """

    def jacrev_fn(*args: Any) -> Any:
        # print("\nSTART JACREV FN")
        # Normalize argnums to a tuple of integers
        selected_argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        # Validate argnums
        for argnum in selected_argnums:
            if argnum >= len(args) or argnum < -len(args):
                raise ValueError(
                    f"argnum {argnum} is out of bounds for function with {len(args)} arguments"
                )

        # Normalize negative indices
        normalized_argnums = tuple(
            argnum if argnum >= 0 else len(args) + argnum for argnum in selected_argnums
        )

        # Extract the arguments to differentiate with respect to
        diff_args = tuple(args[i] for i in normalized_argnums)

        # Create a function that takes only the differentiated arguments
        def partial_func(*diff_args_inner):
            # Reconstruct the full argument list
            full_args = list(args)
            for i, arg in zip(normalized_argnums, diff_args_inner, strict=False):
                full_args[i] = arg
            return func(*full_args)

        # Compute VJP - delegate has_aux handling to vjp
        vjp_result = vjp(partial_func, *diff_args, has_aux=has_aux)

        if has_aux:
            y, pullback, aux = vjp_result
        else:
            y, pullback = vjp_result

        # Flatten output arrays for std_basis generation
        flat_y = _extract_arrays_from_pytree(y)
        if not isinstance(flat_y, list):
            flat_y = [flat_y]

        # Generate standard basis vectors and get sizes for split operations
        sizes, std_basis_vectors = _std_basis(flat_y)

        std_basis_flat = _extract_arrays_from_pytree(std_basis_vectors)
        if not isinstance(std_basis_flat, list):
            std_basis_flat = [std_basis_flat]

        # Handle mixed scalar/tensor outputs by creating appropriate in_axes specification
        if all(arr.shape == () for arr in std_basis_flat):
            # All outputs are scalar - use in_axes=None to broadcast
            grads = vmap(pullback, in_axes=None)(std_basis_vectors)
        elif any(arr.shape == () for arr in std_basis_flat):
            # Mixed scalar/tensor outputs - create in_axes specification for each element
            # Note: std_basis_vectors is a list/tuple, so in_axes should match that structure
            if isinstance(std_basis_vectors, list | tuple):
                in_axes_spec = [
                    None if arr.shape == () else 0 for arr in std_basis_flat
                ]
                grads = vmap(pullback, in_axes=in_axes_spec)(std_basis_vectors)
            else:
                # Single element case - shouldn't happen with mixed outputs, but handle for completeness
                in_axes_spec = None if std_basis_flat[0].shape == () else 0
                grads = vmap(pullback, in_axes=in_axes_spec)(std_basis_vectors)
        else:
            # All outputs are tensors - use in_axes=0 to vectorize along the first axis
            grads = vmap(pullback)(std_basis_vectors)

        # CRITICAL: Check if std_basis_vectors were traced (indicating composition with other transformations)
        std_basis_arrays = _extract_arrays_from_pytree(std_basis_vectors)
        any_std_basis_traced = any(
            getattr(arr, "traced", False) for arr in std_basis_arrays
        )

        # Make grads traced to capture subsequent operations in the computation graph
        if not any_std_basis_traced:
            # Only make traced if original std_basis wasn't traced (avoid double tracing)
            grads = make_traced_pytree(grads)

        # Import split function for proper jacobian structuring
        from ..ops.view import reshape, split

        # Extract flat input arguments for reshaping
        flat_diff_args = _extract_arrays_from_pytree(diff_args)

        splits = []
        for i in range(len(flat_diff_args)):  # For each input argument
            if isinstance(grads, list) and len(grads) > 0:
                if isinstance(grads[0], tuple):
                    # Multiple inputs: extract i-th input's gradients from each batch
                    input_grads = grads[0][i]  # All batched gradients for input i
                else:
                    # Single input case
                    input_grads = grads[0] if len(flat_diff_args) == 1 else grads[i]
            else:
                # Direct case
                input_grads = grads[i] if isinstance(grads, tuple) else grads

            # Split this input's gradients by output components (now traced!)
            splits.append(split(input_grads, sizes=sizes, axis=0))

        # Reshape jacobian components to proper out_shape + arg_shape format (now traced!)
        cotangents = []
        for j in range(len(flat_y)):  # For each output component
            arg_jacs = []
            for i in range(len(flat_diff_args)):  # For each input argument
                grad = splits[i][j]  # j-th output component for i-th input
                batch_dims = flat_y[j].batch_dims
                out_shape = flat_y[j].shape
                arg_shape = flat_diff_args[i].shape

                # print("out_shape:", out_shape, "in_shape:", arg_shape)

                # Only remove (1,) from output shape when we have batch dimensions (from vmap)
                # This handles the case where scalar functions return (1,) instead of ()
                if len(batch_dims) > 0 and len(out_shape) == 1 and out_shape[0] == 1:
                    out_shape = ()
                # Never remove (1,) from arg_shape - it represents valid jacobian structure

                # Jacobian shape should be output_shape + input_shape
                target_shape = out_shape + arg_shape
                reshaped_grad = reshape(grad, target_shape)  # Now traced!
                arg_jacs.append(reshaped_grad)

            if len(arg_jacs) == 1:
                arg_jacs = arg_jacs[0]  # Single input case, return single jacobian

            cotangents.append(arg_jacs)

        final_jac = cotangents
        # print(len(cotangents))

        if len(cotangents) == 1:
            final_jac = cotangents[0]

        # Make final jacobian untraced unless we're in a composition context
        if not any_std_basis_traced:
            make_untraced_pytree(final_jac)

        # print("\nEND JACREV FN\n")

        if not has_aux:
            return final_jac
        else:
            return final_jac, aux

    return jacrev_fn
