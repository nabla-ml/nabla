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
    _extract_arrays_from_pytree,
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
        # print(f"\n=== JACFWD PROTOTYPE ===")
        # print(f"Input args shapes: {[arg.shape if hasattr(arg, 'shape') else type(arg).__name__ for arg in args]}")

        # Handle default case: if argnums is None, differentiate w.r.t. all arguments
        if argnums is None:
            selected_argnums = tuple(range(len(args)))
        else:
            # Normalize argnums to a tuple of integers (same as jacrev)
            selected_argnums = (
                (argnums,) if isinstance(argnums, int) else tuple(argnums)
            )

        # Validate argnums (same as jacrev)
        for argnum in selected_argnums:
            if argnum >= len(args) or argnum < -len(args):
                raise ValueError(
                    f"argnum {argnum} is out of bounds for function with {len(args)} arguments"
                )

        # Normalize negative indices (same as jacrev)
        normalized_argnums = tuple(
            argnum if argnum >= 0 else len(args) + argnum for argnum in selected_argnums
        )
        # print(f"Differentiating w.r.t. arguments: {normalized_argnums}")

        # Extract the arguments to differentiate with respect to (same as jacrev)
        diff_args = tuple(args[i] for i in normalized_argnums)
        # print(f"Diff args shapes: {[arg.shape for arg in diff_args]}")

        # Create a function that takes only the differentiated arguments (same as jacrev)
        def partial_func(*diff_args_inner):
            # Reconstruct the full argument list
            full_args = list(args)
            for i, arg in zip(normalized_argnums, diff_args_inner, strict=False):
                full_args[i] = arg
            return func(*full_args)

        # Generate standard basis vectors for the INPUT arguments (key difference from jacrev)
        flat_diff_args = _extract_arrays_from_pytree(diff_args)
        if not isinstance(flat_diff_args, list):
            flat_diff_args = [flat_diff_args]

        # print(f"Flat diff args shapes: {[arg.shape for arg in flat_diff_args]}")

        # Create standard basis vectors for inputs (this is the key difference from jacrev)
        sizes, std_basis_vectors = _std_basis(flat_diff_args)  # type: ignore

        # print(f"Standard basis sizes: {sizes}")
        # print(f"Standard basis vectors shape: {std_basis_vectors[0].shape if std_basis_vectors else 'None'}")        # Create the JVP function that we'll vmap over
        # This function takes the individual arguments from diff_args + one tangent per input
        def jvp_func(*args):
            """
            JVP function that computes output tangents.

            For single input: args = (primal, tangent_vector)
            For multi-input: args = (primal1, primal2, ..., tangent1, tangent2, ...)

            The tangent vectors come from _std_basis and are already properly shaped.
            """
            num_primals = len(diff_args)
            primals = args[:num_primals]  # First N arguments are primals
            tangent_vectors = args[num_primals:]  # Last N arguments are tangents

            if len(primals) == 1:
                # Single input case - reconstruct tangent structure to match primal structure
                primals_tuple = primals[0]
                if isinstance(primals_tuple, (list, tuple)):
                    # If primal is a container, wrap it in tuple for jvp and reconstruct tangent
                    jvp_primals = (primals_tuple,)
                    jvp_tangents = (type(primals_tuple)(tangent_vectors),)
                else:
                    # If primal is not a container, pass directly
                    jvp_primals = primals_tuple
                    jvp_tangents = tangent_vectors[0]
            else:
                # Multi-input case - pass as tuple for multiple arguments
                jvp_primals = tuple(primals)
                jvp_tangents = tuple(tangent_vectors)

            # Compute JVP with proper calling convention
            jvp_result = jvp(partial_func, jvp_primals, jvp_tangents)
            primal_out, tangent_out = jvp_result  # type: ignore

            return tangent_out  # Return tangent output directly

        # Create in_axes: None for each primal argument, 0 for each tangent vector
        primals_axes = tuple(None for _ in diff_args)  # Broadcast all primal arguments
        tangents_axes = tuple(
            0 for _ in std_basis_vectors
        )  # Vectorize all tangent arguments
        vmap_in_axes = primals_axes + tangents_axes

        # Apply vmap to vectorize the JVP computation
        # print(f"vmap in_axes: {vmap_in_axes}")
        vmap_jvp = vmap(jvp_func, in_axes=vmap_in_axes)

        output_tangents = vmap_jvp(*diff_args, *std_basis_vectors)

        from nabla.ops.view import reshape, split

        # Get output structure by running the function once
        test_output = partial_func(*diff_args)
        flat_output = _extract_arrays_from_pytree(test_output)
        if not isinstance(flat_output, list):
            flat_output = [flat_output]

        # Handle output_tangents structure - it may be a container if function returns multiple outputs
        if isinstance(output_tangents, (list, tuple)):
            # Function returns multiple outputs, so output_tangents is also a container
            # We need to split each output component separately
            all_split_tangents = []
            for output_component in output_tangents:
                split_tangents_component = split(output_component, sizes=sizes, axis=0)
                all_split_tangents.append(split_tangents_component)

            # Reorganize: instead of [split_for_output0, split_for_output1],
            # we want [split_for_input0, split_for_input1] where each contains all outputs
            split_tangents = []
            for input_idx in range(len(sizes)):
                tangents_for_this_input = []
                for output_idx in range(len(all_split_tangents)):
                    tangents_for_this_input.append(
                        all_split_tangents[output_idx][input_idx]
                    )
                split_tangents.append(tangents_for_this_input)
        else:
            # Function returns single output, handle normally
            split_tangents = split(output_tangents, sizes=sizes, axis=0)

        # print("\n\nSPLIT TANGENTS")
        # print(split_tangents)
        # print("\n\n")

        jacobian_components = []
        for _j, (arg, tangents_for_arg) in enumerate(
            zip(flat_diff_args, split_tangents, strict=False)
        ):
            arg_shape = arg.shape

            if isinstance(tangents_for_arg, list):
                # Multiple outputs case - create jacobian for each output
                output_jacobians = []
                for output_idx, tangent_for_output in enumerate(tangents_for_arg):
                    output_shape = flat_output[output_idx].shape

                    # Reshape to proper Jacobian format: output_shape + input_shape
                    target_shape = arg_shape + output_shape
                    jacobian_component = reshape(tangent_for_output, target_shape)

                    # Permute axes to get output_shape + input_shape format
                    perm_axes = []
                    for k in range(len(output_shape)):
                        perm_axes.append(k + len(arg_shape))
                    for k in range(len(arg_shape)):
                        perm_axes.append(k)

                    from ..ops.view import permute

                    jacobian_component = permute(jacobian_component, tuple(perm_axes))
                    output_jacobians.append(jacobian_component)

                jacobian_components.append(output_jacobians)
            else:
                # Single output case
                output_shape = flat_output[0].shape

                # Reshape to proper Jacobian format: output_shape + input_shape
                target_shape = arg_shape + output_shape
                jacobian_component = reshape(tangents_for_arg, target_shape)

                # Permute axes to get output_shape + input_shape format
                perm_axes = []
                for k in range(len(output_shape)):
                    perm_axes.append(k + len(arg_shape))
                for k in range(len(arg_shape)):
                    perm_axes.append(k)

                from ..ops.view import permute

                jacobian_component = permute(jacobian_component, tuple(perm_axes))
                jacobian_components.append(jacobian_component)

        # Return as tuple for multiple inputs
        if len(jacobian_components) == 1:
            jacobian_components = jacobian_components[0]

        jacobian = jacobian_components

        if not has_aux:
            return jacobian
        else:
            # TODO: Handle auxiliary data properly
            return jacobian, None

    return jacfwd_fn
