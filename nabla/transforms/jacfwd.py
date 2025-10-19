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

from .jvp import jvp
from .utils import (
    _create_jacobian_helpers,
    _extract_tensors_from_pytree,
    _std_basis,
)
from .vmap import vmap


def jacfwd(
    func: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] | None = None,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable[..., Any]:
    """
    Prototype implementation of jacfwd using forward-mode autodiff.

    This computes the Jacobian using the pattern:
    vmap(jvp(func, primals, tangents), in_axes=(primal_axes, tangent_axes))

    where primal_axes are None (broadcast) and tangent_axes are 0 (vectorize).

    Args:
        func: Function to differentiate
        argnums: Which arguments to differentiate with respect to
        has_aux: Whether function returns auxiliary data
        holomorphic: Ignored (for JAX compatibility)
        allow_int: Ignored (for JAX compatibility)

    Returns:
        Function that computes the Jacobian using forward-mode autodiff
    """

    def jacfwd_fn(*args: Any) -> Any:
        diff_args, partial_func = _create_jacobian_helpers(func, argnums, args)

        # Generate standard basis vectors for the INPUT arguments
        flat_diff_args = _extract_tensors_from_pytree(diff_args)
        if not isinstance(flat_diff_args, list):
            flat_diff_args = [flat_diff_args]

        sizes, std_basis_vectors = _std_basis(flat_diff_args)

        # Create the JVP function that we'll vmap over
        def jvp_func(*jvp_args):
            num_primals = len(diff_args)
            primals = jvp_args[:num_primals]
            tangent_vectors = jvp_args[num_primals:]

            jvp_primals = primals[0] if len(primals) == 1 else tuple(primals)
            jvp_tangents = tangent_vectors[0] if len(tangent_vectors) == 1 else tuple(tangent_vectors)
            
            _, tangent_out = jvp(partial_func, jvp_primals, jvp_tangents)
            return tangent_out

        # Create in_axes for vmap
        primals_axes = tuple(None for _ in diff_args)
        tangents_axes = tuple(0 for _ in std_basis_vectors)
        vmap_in_axes = primals_axes + tangents_axes

        # Apply vmap to vectorize the JVP computation
        output_tangents = vmap(jvp_func, in_axes=vmap_in_axes)(*diff_args, *std_basis_vectors)

        from nabla.ops.view import reshape, split, permute

        # Get output structure by running the function once
        test_output = partial_func(*diff_args)
        flat_output = _extract_tensors_from_pytree(test_output)
        if not isinstance(flat_output, list):
            flat_output = [flat_output]

        # Handle multiple outputs by splitting each component
        if isinstance(output_tangents, (list, tuple)):
            all_split_tangents = [split(comp, sizes=sizes, axis=0) for comp in output_tangents]
            # Transpose the splits to be per-input instead of per-output
            split_tangents = list(zip(*all_split_tangents))
        else:
            split_tangents = split(output_tangents, sizes=sizes, axis=0)

        # Reshape and permute to create Jacobian components
        jacobian_components = []
        for i, (arg, tangents_for_arg) in enumerate(zip(flat_diff_args, split_tangents)):
            arg_shape = arg.shape
            if isinstance(tangents_for_arg, (list, tuple)): # Multiple outputs
                output_jacobians = []
                for j, tangent_for_output in enumerate(tangents_for_arg):
                    output_shape = flat_output[j].shape
                    target_shape = arg_shape + output_shape
                    jacobian_comp = reshape(tangent_for_output, target_shape)
                    
                    perm_axes = tuple(range(len(arg_shape), len(target_shape))) + tuple(range(len(arg_shape)))
                    output_jacobians.append(permute(jacobian_comp, perm_axes))
                jacobian_components.append(output_jacobians)
            else: # Single output
                output_shape = flat_output[0].shape
                target_shape = arg_shape + output_shape
                jacobian_comp = reshape(tangents_for_arg, target_shape)

                perm_axes = tuple(range(len(arg_shape), len(target_shape))) + tuple(range(len(arg_shape)))
                jacobian_components.append(permute(jacobian_comp, perm_axes))

        jacobian = jacobian_components[0] if len(jacobian_components) == 1 else jacobian_components

        return (jacobian, None) if has_aux else jacobian

    return jacfwd_fn
