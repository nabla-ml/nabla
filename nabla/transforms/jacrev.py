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
    _create_jacobian_helpers,
    _extract_tensors_from_pytree,
    _std_basis,
    make_traced_pytree,
    make_untraced_pytree,
)
from .vjp import vjp
from .vmap import vmap


def jacrev(
    func: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] | None = None,
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

    Examples:
        Basic Jacobian computation:

        ```python
        import nabla as nb

        def f(x):
            return x ** 2

        x = nb.tensor([1.0, 2.0, 3.0])
        jac_fn = nb.jacrev(f)
        jacobian = jac_fn(x)
        # jacobian will be a diagonal matrix with [2.0, 4.0, 6.0] on the diagonal
        ```

        Vector-valued function:

        ```python
        import nabla as nb

        def f(x):
            return nb.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])

        x = nb.tensor([3.0, 4.0])
        jacobian = nb.jacrev(f)(x)
        # jacobian shape: (3, 2) - 3 outputs, 2 inputs
        ```

        Multiple arguments with argnums:

        ```python
        import nabla as nb

        def f(x, y):
            return x * y + x ** 2

        x = nb.tensor([1.0, 2.0])
        y = nb.tensor([3.0, 4.0])
        
        # Differentiate w.r.t. first argument
        jac_x = nb.jacrev(f, argnums=0)(x, y)
        
        # Differentiate w.r.t. both arguments
        jac_both = nb.jacrev(f, argnums=(0, 1))(x, y)
        ```

        With auxiliary data:

        ```python
        import nabla as nb

        def f_with_aux(x):
            y = x ** 2
            aux = {"loss": nb.sum(y)}
            return y, aux

        x = nb.tensor([1.0, 2.0, 3.0])
        jacobian, aux = nb.jacrev(f_with_aux, has_aux=True)(x)
        ```
    """

    def jacrev_fn(*args: Any) -> Any:
        # Use the helper to handle arg parsing and partial function creation
        diff_args, partial_func = _create_jacobian_helpers(func, argnums, args)

        # Compute VJP - delegate has_aux handling to vjp
        vjp_result = vjp(partial_func, *diff_args, has_aux=has_aux)

        if has_aux:
            y, pullback, aux = vjp_result
        else:
            y, pullback = vjp_result

        # Flatten output tensors for std_basis generation
        flat_y = _extract_tensors_from_pytree(y)
        if not isinstance(flat_y, list):
            flat_y = [flat_y]

        # Generate standard basis vectors and get sizes for split operations
        sizes, std_basis_vectors = _std_basis(flat_y)

        std_basis_flat = _extract_tensors_from_pytree(std_basis_vectors)
        if not isinstance(std_basis_flat, list):
            std_basis_flat = [std_basis_flat]

        # Handle mixed scalar/tensor outputs by creating appropriate in_axes for vmap
        if all(arr.shape == () for arr in std_basis_flat):
            in_axes_spec = None
        elif any(arr.shape == () for arr in std_basis_flat):
            if isinstance(std_basis_vectors, (list, tuple)):
                in_axes_spec = [None if arr.shape == () else 0 for arr in std_basis_flat]
            else:
                in_axes_spec = None if std_basis_flat[0].shape == () else 0
        else:
            in_axes_spec = 0
        
        grads = vmap(pullback, in_axes=in_axes_spec)(std_basis_vectors)

        # Make gradients traceable for further composition
        any_std_basis_traced = any(
            getattr(arr, "traced", False) for arr in _extract_tensors_from_pytree(std_basis_vectors)
        )
        if not any_std_basis_traced:
            grads = make_traced_pytree(grads)

        from ..ops.view import reshape, split

        flat_diff_args = _extract_tensors_from_pytree(diff_args)

        # Split batched gradients based on output components
        splits = []
        for i in range(len(flat_diff_args)):
            input_grads = grads[i] if isinstance(grads, tuple) else grads
            splits.append(split(input_grads, sizes=sizes, axis=0))

        # Reshape gradients into Jacobian components
        jacobian_parts = []
        for j, out_tensor in enumerate(flat_y):
            arg_jacs = []
            for i, in_tensor in enumerate(flat_diff_args):
                grad_slice = splits[i][j]
                
                out_shape = out_tensor.shape
                # Handle scalar outputs that get an extra dimension from vmap
                if len(out_tensor.batch_dims) > 0 and out_shape == (1,):
                    out_shape = ()
                
                target_shape = out_shape + in_tensor.shape
                reshaped_grad = reshape(grad_slice, target_shape)
                arg_jacs.append(reshaped_grad)
            
            jacobian_parts.append(arg_jacs[0] if len(arg_jacs) == 1 else tuple(arg_jacs))

        final_jac = jacobian_parts[0] if len(jacobian_parts) == 1 else tuple(jacobian_parts)

        if not any_std_basis_traced:
            make_untraced_pytree(final_jac)

        return (final_jac, aux) if has_aux else final_jac
    return jacrev_fn
