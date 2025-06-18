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

"""Core transformations for automatic differentiation and tracing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.array import Array


def tree_flatten(tree: Any) -> tuple[list[Array], Any]:
    """Flatten a pytree into a list of Arrays and structure info.

    Args:
        tree: A pytree containing Arrays and other structures

    Returns:
        A tuple of (list of Array leaves, structure info for reconstruction)
    """
    leaves = []

    def _flatten(obj: Any) -> Any:
        if isinstance(obj, Array):
            leaves.append(obj)
            return None  # Placeholder for Array
        elif isinstance(obj, dict):
            keys = sorted(obj.keys())  # Deterministic ordering
            return {k: _flatten(obj[k]) for k in keys}
        elif isinstance(obj, (list | tuple)):
            return type(obj)(_flatten(item) for item in obj)
        else:
            # Non-Array leaf (int, float, etc.)
            return obj

    structure = _flatten(tree)
    return leaves, structure


def tree_unflatten(structure: Any, leaves: list[Array]) -> Any:
    """Reconstruct a pytree from structure info and list of Arrays.

    Args:
        structure: Structure info from tree_flatten
        leaves: List of Array values to place at Array positions

    Returns:
        Reconstructed pytree with the same structure as the original
    """
    leaves_iter = iter(leaves)

    def _unflatten(struct: Any) -> Any:
        if struct is None:  # Array placeholder
            try:
                return next(leaves_iter)
            except StopIteration:
                raise ValueError(
                    f"Tree unflatten error: Not enough leaves. Expected structure: {structure}, Got {len(leaves)} leaves"
                )
        elif isinstance(struct, dict):
            return {k: _unflatten(v) for k, v in struct.items()}
        elif isinstance(struct, list | tuple):
            # Use list comprehension instead of generator to avoid StopIteration -> RuntimeError conversion
            try:
                result = [_unflatten(item) for item in struct]
                return type(struct)(result)
            except StopIteration:
                raise ValueError(
                    f"Tree unflatten error: Not enough leaves for sequence. Expected structure: {structure}, Got {len(leaves)} leaves"
                )
        else:
            # Non-Array leaf
            return struct

    result = _unflatten(structure)

    # Verify we consumed all leaves
    try:
        next(leaves_iter)
        raise ValueError("Too many leaves provided for tree structure")
    except StopIteration:
        pass

    return result


def tree_map(func: Callable[[Array], Array], tree: Any) -> Any:
    """Apply a function to all Array leaves in a pytree.

    Args:
        func: Function to apply to each Array leaf
        tree: Pytree containing Arrays

    Returns:
        Pytree with the same structure but transformed Arrays
    """
    leaves, structure = tree_flatten(tree)
    transformed_leaves = [func(leaf) for leaf in leaves]
    return tree_unflatten(structure, transformed_leaves)


def _extract_arrays_from_pytree(tree: Any) -> list[Array]:
    """Extract all Arrays from a pytree structure.

    Args:
        tree: Pytree that may contain Arrays, ints, floats, etc.

    Returns:
        List of all Arrays found in the tree
    """
    leaves, _ = tree_flatten(tree)
    return leaves


def _validate_length_match(list1, list2, name1, name2):
    """Check if two lists have the same length."""
    if len(list1) != len(list2):
        raise ValueError(f"{name1} length {len(list1)} != {name2} length {len(list2)}")


def _std_basis(args: list[Array]) -> tuple[list[int], list[Array]]:
    num_total_arg_elements = 0
    max_rank = 0
    for arg in args:
        num_elements = 1
        for dim in arg.shape:
            num_elements *= dim
        num_total_arg_elements += num_elements
        rank = len(arg.shape)
        if rank > max_rank:
            max_rank = rank

    batch_ctr = 0
    sizes = list[int]()
    tangents: list[Array] = []

    for _i, arg in enumerate(args):
        num_elements = 1

        if arg.shape == ():
            from ..ops.creation import ones_like

            tangent = ones_like(arg)
            tangents.append(tangent)
            sizes.append(1)
            batch_ctr += 1

        else:
            for dim in arg.shape:
                num_elements *= dim

            batched_shape = (num_total_arg_elements,) + arg.shape

            from numpy import zeros as np_zeros

            np_tangent = np_zeros(batched_shape, dtype=arg.dtype.to_numpy()).flatten()

            offset = batch_ctr * num_elements

            for j in range(num_elements):
                idx = offset + j * num_elements + j
                np_tangent[idx] = 1.0
                batch_ctr += 1

            np_tangent = np_tangent.reshape(batched_shape)
            tangent = Array.from_numpy(np_tangent)

            from ..ops.view import broadcast_batch_dims

            tangent = broadcast_batch_dims(tangent, arg.batch_dims)

            tangents.append(tangent)
            sizes.append(num_elements)

    return sizes, tangents


def make_traced_pytree(tree: Any) -> Any:
    """Create shallow copies of arrays in a pytree and mark them as traced.

    Args:
        tree: Pytree containing Arrays to copy and mark as traced

    Returns:
        Pytree with the same structure but traced Arrays
    """

    def _make_traced_array(array: Array) -> Array:
        from ..ops.view import shallow_copy

        copied_arg = shallow_copy(array)
        copied_arg.traced = True
        return copied_arg

    return tree_map(_make_traced_array, tree)


def make_untraced_pytree(tree: Any) -> None:
    """Disable tracing for arrays in a pytree by clearing their traced flag.

    Args:
        tree: Pytree containing Arrays to disable tracing for
    """

    def _make_untraced_array(array: Array) -> Array:
        array.traced = False
        return array

    tree_map(_make_untraced_array, tree)


def make_staged_pytree(args: list[Array]) -> None:
    """Enable staged execution for arrays to optimize performance.

    Args:
        args: Arrays to enable staged execution for
    """

    def _make_staged_array(array: Array) -> Array:
        array.stage_realization = True
        return array

    tree_map(_make_staged_array, args)


def make_unstaged_pytree(args: list[Array]) -> None:
    """Disable staged execution for arrays.

    Args:
        args: Arrays to disable staged execution for
    """

    def _make_unstaged_array(array: Array) -> Array:
        array.stage_realization = False
        return array

    tree_map(_make_unstaged_array, args)


def _handle_args_consistently(args):
    """Handle both fn([x,y,z]) and fn(x,y,z) calling styles."""
    if len(args) == 1 and isinstance(args[0], list):
        return args[0], True
    return args, False


def _prepare_traced_inputs(
    actual_args, is_list_style, apply_staging=False, with_conversion=False
):
    """Prepare traced inputs for list-style or pytree-style arguments."""

    # Convert scalars to Arrays if requested
    if with_conversion:

        def convert_scalars_to_arrays(item):
            if isinstance(item, Array):
                return item
            elif isinstance(item, list | tuple):
                return type(item)(
                    convert_scalars_to_arrays(sub_item) for sub_item in item
                )
            elif isinstance(item, dict):
                return {k: convert_scalars_to_arrays(v) for k, v in item.items()}
            elif isinstance(item, (int, float, bool)):
                # Only convert basic scalar types to Nabla Arrays
                import nabla as nb

                return nb.array(item)
            else:
                # Keep everything else (functions, numpy arrays, etc.) unchanged
                return item

        actual_args = convert_scalars_to_arrays(actual_args)

    if is_list_style:
        traced_args = make_traced_pytree(actual_args)
        if apply_staging:
            make_staged_pytree(traced_args)
        return traced_args, None

    # Handle the case where actual_args might not have __len__
    if hasattr(actual_args, "__len__"):
        args_len = len(actual_args)  # type: ignore
    else:
        args_len = 1

    if args_len == 1:
        inputs_pytree = actual_args[0]
        traced_inputs_pytree = make_traced_pytree(inputs_pytree)
        traced_args = (traced_inputs_pytree,)
    else:
        inputs_pytree = actual_args
        traced_inputs_pytree = make_traced_pytree(inputs_pytree)
        traced_args = traced_inputs_pytree

    if apply_staging:
        # Apply staging to the TRACED arrays, not the original args
        arrays = _extract_arrays_from_pytree(traced_args)
        make_staged_pytree(arrays)

    return traced_args, traced_inputs_pytree


def _clean_traced_outputs(outputs, is_list_style, remove_staging=False):
    """Clean up traced outputs and handle staging flags."""
    if is_list_style:
        # For list-style, we expect a list of Arrays, but handle tuple case
        if isinstance(outputs, list):
            make_untraced_pytree(outputs)
            if remove_staging:
                make_unstaged_pytree(outputs)
        else:
            # If it's not a list (e.g., tuple from VJP), treat as pytree
            make_untraced_pytree(outputs)
            if remove_staging:
                output_arrays = _extract_arrays_from_pytree(outputs)
                make_unstaged_pytree(output_arrays)
    else:
        make_untraced_pytree(outputs)
        if remove_staging:
            output_arrays = _extract_arrays_from_pytree(outputs)
            make_unstaged_pytree(output_arrays)
    return outputs


class Trace:
    """A simple trace container that holds the computation graph."""

    def __init__(self, inputs: list[Array], outputs: list[Array] | None = None) -> None:
        self.inputs = inputs
        self.outputs = outputs if outputs is not None else []
        self.trace: list[Array] = []
        self._computed = False

        # Mark all inputs as traced for autodiff so the computation graph gets captured
        for inp in inputs:
            inp.traced = True

    @classmethod
    def trace_function(
        cls, fn: Callable[[list[Array]], list[Array]], inputs: list[Array]
    ) -> Trace:
        """
        Create a trace by executing a function with tracing enabled.

        This is the recommended way to create traces as it ensures proper
        tracing setup before function execution.
        """
        inputs = make_traced_pytree(inputs)

        # Create trace instance (this marks inputs as traced)
        trace = cls(inputs)

        # Execute function with tracing enabled
        outputs = fn(inputs)

        # Extract Arrays from outputs and store as list
        output_arrays = _extract_arrays_from_pytree(outputs)
        trace.outputs = output_arrays

        make_untraced_pytree(inputs)  # Detach inputs from the trace

        # Handle outputs properly - make them untraced
        make_untraced_pytree(output_arrays)

        return trace

    def get_traced_nodes(self) -> list[Array]:
        """Get all nodes that belong to this trace in topological order."""
        if not self._computed:
            self._compute_trace()
        return self.trace

    def _compute_trace(self) -> None:
        """Compute the topological ordering of traced nodes."""
        visited: set[Array] = set()
        self.trace = []

        for output in self.outputs:
            self._dfs_visit(output, visited)

        self._computed = True

    def _dfs_visit(self, node: Array, visited: set[Array]) -> None:
        """DFS traversal to build topological ordering."""
        if node in visited:
            return

        # Visit children first (post-order)
        for arg in node.args:
            self._dfs_visit(arg, visited)

        # Add current node after visiting children
        visited.add(node)
        self.trace.append(node)

    def __str__(self) -> str:
        """Return a JAX-like string representation of the trace."""
        if not self._computed:
            self._compute_trace()

        from ..utils.formatting import format_shape_and_dtype

        # Initialize name generator with a simple global counter
        var_names = {}
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        name_counter = 0

        def _get_next_name():
            nonlocal name_counter

            if name_counter < len(alphabet):
                # Single letters: a, b, c, ..., z
                name = alphabet[name_counter]
            else:
                # Double letters: aa, ab, ac, ..., az, ba, bb, bc, ...
                # Calculate indices for double letters
                double_index = name_counter - len(alphabet)
                first_letter = double_index // len(alphabet)
                second_letter = double_index % len(alphabet)
                name = alphabet[first_letter] + alphabet[second_letter]

            name_counter += 1
            return name

        # Assign names to inputs first
        input_vars = []
        for inp in self.inputs:
            var_name = _get_next_name()
            var_names[id(inp)] = var_name
            type_annotation = format_shape_and_dtype(inp)
            input_vars.append(f"{var_name}:{type_annotation}")

        # Single pass through trace: assign names and build equations
        equations = []
        for node in self.trace:
            node_id = id(node)

            # Skip if this is an input (already processed)
            if node_id in var_names:
                continue

            # Assign a name to this node
            var_name = _get_next_name()
            var_names[node_id] = var_name

            # Build the operation description
            # print node name or the type if no name is set

            if node.name:
                op_name = node.name
            else:
                # check if the arg is a constant scalar, then we can simply show it as the arg directly
                if (
                    isinstance(node, Array)
                    and node.shape == ()
                    and not node.batch_dims
                    and node.impl
                ):
                    # This is a constant scalar, show the raw value
                    op_name = str(node.to_numpy().item())
                else:
                    # Fallback to the type or some default name
                    op_name = "external_const"
            type_annotation = format_shape_and_dtype(node)

            if node.args:
                # Get argument variable names
                arg_vars = []
                for arg in node.args:
                    arg_id = id(arg)
                    if arg_id in var_names:
                        arg_vars.append(var_names[arg_id])
                    else:
                        # Array from external context - not part of the trace
                        arg_vars.append("external_const")

                # Format the equation with type annotation
                if len(arg_vars) == 1:
                    equation = (
                        f"    {var_name}:{type_annotation} = {op_name} {arg_vars[0]}"
                    )
                else:
                    args_joined = " ".join(arg_vars)
                    fmt_str = f"    {var_name}:{type_annotation} = {op_name}"
                    equation = f"{fmt_str} {args_joined}"
            else:
                # Node with no arguments (constants, copies of external values, etc.)
                equation = f"    {var_name}:{type_annotation} = {op_name}"

            equations.append(equation)

        # Get output variable names
        output_vars = []
        for out in self.outputs:
            out_id = id(out)
            if out_id in var_names:
                output_vars.append(var_names[out_id])
            else:
                output_vars.append("?")

        # Format the final representation
        input_sig = f"({', '.join(input_vars)})"
        output_sig = (
            f"({', '.join(output_vars)})" if len(output_vars) > 1 else output_vars[0]
        )

        result = f"{{ lambda {input_sig} ;\n"
        result += "  let\n"
        for eq in equations:
            result += f"{eq}\n"
        result += f"  in {output_sig} }}"

        return result


def _cleanup_cotangents(traced_nodes: list[Array]) -> None:
    """Clean up cotangent values from traced nodes.

    Args:
        traced_nodes: List of traced nodes to clean up
    """
    for node in traced_nodes:
        node.cotangent = None


def _compute_pullback(
    input_arrays: list[Array],
    output_arrays: list[Array],
    cotangent_arrays: list[Array],
) -> list[Array]:
    """Core reverse-mode gradient computation.

    Args:
        input_arrays: Input arrays to compute gradients for
        output_arrays: Output arrays from the computation
        cotangent_arrays: Cotangent vectors for outputs

    Returns:
        List of gradient arrays corresponding to inputs
    """
    # Build computation trace
    trace = Trace(input_arrays, output_arrays)
    traced_nodes = trace.get_traced_nodes()

    # Initialize output cotangents
    for output, cotangent in zip(output_arrays, cotangent_arrays, strict=False):
        output.cotangent = cotangent

    try:
        # Reverse-mode gradient computation
        for node in reversed(traced_nodes):
            if node.cotangent is None:
                continue

            if not node.args or node.vjp_rule is None:
                continue

            try:
                arg_cotangents = node.vjp_rule(node.args, node.cotangent, node)

                for arg, arg_cotangent in zip(node.args, arg_cotangents, strict=False):
                    if arg.cotangent is not None:
                        from ..ops.binary import add

                        arg.cotangent = add(arg.cotangent, arg_cotangent)
                    else:
                        arg.cotangent = arg_cotangent

                if node not in input_arrays:
                    node.cotangent = None

            except Exception as e:
                raise RuntimeError(
                    f"VJP rule failed for operation '{node.name}': {e}"
                ) from e

        # Collect gradients for input arrays
        gradient_arrays = []
        for inp in input_arrays:
            if inp.cotangent is not None:
                gradient_arrays.append(inp.cotangent)
            else:
                from ..ops.creation import zeros_like

                gradient_arrays.append(zeros_like(inp))

        return gradient_arrays

    finally:
        _cleanup_cotangents(traced_nodes)


def _reconstruct_gradient_structure(
    gradient_arrays: list[Array],
    inputs: Any,
) -> Any:
    """Reconstruct gradients in the same structure as inputs.

    Args:
        gradient_arrays: Flat list of gradient arrays
        inputs: Original input structure to match

    Returns:
        Gradients with the same structure as inputs
    """
    # Use the same flattening/unflattening logic as used for input extraction
    input_arrays, structure = tree_flatten(inputs)

    # Validate that we have the right number of gradients
    if len(gradient_arrays) != len(input_arrays):
        raise ValueError(
            f"Gradient arrays length {len(gradient_arrays)} != "
            f"input arrays length {len(input_arrays)}"
        )

    # Reconstruct the pytree structure with gradients
    return tree_unflatten(structure, gradient_arrays)


def pullback(
    inputs: Any,
    outputs: Any,
    cotangents: Any,
) -> Any:
    """Compute vector-Jacobian product (reverse-mode autodiff).

    Returns gradients in the exact same structure as inputs.

    Args:
        inputs: Input arrays or pytree of arrays
        outputs: Output arrays or pytree of arrays
        cotangents: Cotangent vectors or pytree of cotangents

    Returns:
        Gradients with respect to inputs, in the same structure as inputs
    """
    # Extract arrays from pytree structures
    input_arrays = _extract_arrays_from_pytree(inputs)
    output_arrays = _extract_arrays_from_pytree(outputs)
    cotangent_arrays = _extract_arrays_from_pytree(cotangents)

    _validate_length_match(
        cotangent_arrays, output_arrays, "Cotangent arrays", "output arrays"
    )

    # Core reverse-mode gradient computation
    gradient_arrays = _compute_pullback(input_arrays, output_arrays, cotangent_arrays)

    # Reconstruct gradients in input structure
    gradients_in_input_structure = _reconstruct_gradient_structure(
        gradient_arrays, inputs
    )

    return gradients_in_input_structure


def _compute_pushfwd(inputs, outputs, tangents, trace=None):
    """Compute JVP (forward-mode autodiff)."""
    _validate_length_match(tangents, inputs, "Tangents", "inputs")

    if trace is None:
        trace = Trace(inputs, outputs)
    traced_nodes = trace.get_traced_nodes()

    for input_node, tangent in zip(inputs, tangents, strict=False):
        input_node.tangent = tangent

    for node in traced_nodes:
        if node in inputs or not node.args or not node.jvp_rule:
            continue

        arg_tangents = []
        for arg in node.args:
            if arg.tangent is not None:
                arg_tangents.append(arg.tangent)
            else:
                from ..ops.creation import zeros_like

                arg_tangents.append(zeros_like(arg))

        try:
            node.tangent = node.jvp_rule(node.args, arg_tangents, node)
        except Exception as e:
            raise RuntimeError(
                f"JVP rule failed for operation '{node.name}': {e}"
            ) from e

    output_tangents = []
    for out in outputs:
        if out.tangent is not None:
            output_tangents.append(out.tangent)
        else:
            from ..ops.creation import zeros_like

            output_tangents.append(zeros_like(out))

    return output_tangents


def pushfwd(
    inputs: Any,
    outputs: Any,
    tangents: Any,
) -> Any:
    """Compute Jacobian-vector product (forward-mode autodiff).

    Returns output tangents in the same structure as outputs.

    Args:
        inputs: Input arrays or pytree of arrays
        outputs: Output arrays or pytree of arrays
        tangents: Tangent vectors or pytree of tangents

    Returns:
        Tangents with respect to outputs, in the same structure as outputs
    """
    # Extract arrays from pytree structures
    input_arrays = _extract_arrays_from_pytree(inputs)
    output_arrays = _extract_arrays_from_pytree(outputs)
    tangent_arrays = _extract_arrays_from_pytree(tangents)

    _validate_length_match(
        tangent_arrays, input_arrays, "Tangent arrays", "input arrays"
    )

    # Core forward-mode gradient computation
    output_tangents = _compute_pushfwd(input_arrays, output_arrays, tangent_arrays)

    # Reconstruct tangents in output structure
    return tree_unflatten(tree_flatten(outputs)[1], output_tangents)


def xpr(fn: Callable[..., Any], *primals) -> str:
    """Get a JAX-like string representation of the function's computation graph.

    Args:
        fn: Function to trace (should take positional arguments)
        *primals: Positional arguments to the function (can be arbitrary pytrees)

    Returns:
        JAX-like string representation of the computation graph

    Note:
        This follows the same flexible API as vjp, jvp, and vmap:
        - Accepts functions with any number of positional arguments
        - For functions requiring keyword arguments, use functools.partial or lambda
    """
    # Handle the input structure based on number of arguments (same as vjp)
    if len(primals) == 1:
        inputs_pytree = primals[0]
        is_single_arg = True
    else:
        inputs_pytree = primals
        is_single_arg = False

    any_arg_traced = any(
        getattr(arg, "traced", False)
        for arg in _extract_arrays_from_pytree(inputs_pytree)
    )

    # Make traced copies of all inputs
    traced_inputs_pytree = make_traced_pytree(inputs_pytree)

    # Extract traced args based on the structure
    traced_args = (traced_inputs_pytree,) if is_single_arg else traced_inputs_pytree

    # Execute the function with traced inputs
    outputs = fn(*traced_args)

    # Extract output arrays for trace creation
    output_arrays = _extract_arrays_from_pytree(outputs)
    if not isinstance(output_arrays, list):
        output_arrays = [output_arrays] if output_arrays is not None else []

    # Extract input arrays for trace creation
    input_arrays = _extract_arrays_from_pytree(traced_inputs_pytree)
    if not isinstance(input_arrays, list):
        input_arrays = [input_arrays] if input_arrays is not None else []

    # Ensure we have proper Array lists (not Never)
    if not input_arrays:
        input_arrays = []
    if not output_arrays:
        output_arrays = []

    # Create trace with the computation graph
    trace = Trace(input_arrays, output_arrays)  # type: ignore

    # Make everything untraced before returning
    # make_untraced_pytree(traced_inputs_pytree)
    if not any_arg_traced:
        make_untraced_pytree(outputs)

    return str(trace)
