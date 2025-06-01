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

from .array import Array


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
            return next(leaves_iter)
        elif isinstance(struct, dict):
            return {k: _unflatten(v) for k, v in struct.items()}
        elif isinstance(struct, list | tuple):
            return type(struct)(_unflatten(item) for item in struct)
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


def _apply_transformation_to_arrays(
    transformation_func: Callable[[list[Array]], Any], tree: Any
) -> tuple[Any, Callable[[Any], Any]]:
    """Apply a transformation function to Arrays in a pytree structure.

    This is a helper function that:
    1. Flattens the input tree to extract Arrays
    2. Applies the transformation to the list of Arrays
    3. Returns transformed result and a function to reverse the process

    Args:
        transformation_func: Function that takes list[Array] and returns transformed result
        tree: Input pytree containing Arrays

    Returns:
        Tuple of (transformed_result, unflatten_func) where unflatten_func
        can reconstruct the original tree structure from a list of Arrays
    """
    leaves, structure = tree_flatten(tree)

    def unflatten_func(new_leaves: list[Array]) -> Any:
        return tree_unflatten(structure, new_leaves)

    transformed_result = transformation_func(leaves)
    return transformed_result, unflatten_func


def _validate_length_match(list1, list2, name1: str, name2: str) -> None:
    """Validate that two lists have matching lengths."""
    if len(list1) != len(list2):
        raise ValueError(f"{name1} length {len(list1)} != {name2} length {len(list2)}")


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
        inputs = make_traced(inputs)

        # Create trace instance (this marks inputs as traced)
        trace = cls(inputs)

        # Execute function with tracing enabled
        outputs = fn(inputs)
        trace.outputs = outputs if isinstance(outputs, list) else [outputs]

        make_untraced(inputs)  # Detach inputs from the trace
        make_untraced(trace.outputs)

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

        # Build variable name mapping
        var_names = {}
        var_counter = 0

        # Assign names to inputs first with type annotations
        input_vars = []
        for inp in self.inputs:
            var_name = chr(ord("a") + var_counter)
            var_names[id(inp)] = var_name
            type_annotation = format_shape_and_dtype(inp)
            input_vars.append(f"{var_name}:{type_annotation}")
            var_counter += 1

        # Build the equation lines
        equations = []
        for node in self.trace:
            node_id = id(node)

            # Skip if this is an input (already named)
            if node_id in var_names:
                continue

            # Assign a variable name to this node
            var_name = chr(ord("a") + var_counter)
            var_names[node_id] = var_name
            var_counter += 1

            # Build the operation description
            if node.args:
                # Get argument variable names
                arg_vars = []
                for arg in node.args:
                    arg_id = id(arg)
                    if arg_id in var_names:
                        arg_vars.append(var_names[arg_id])
                    else:
                        # Array from external context - assign a const name
                        temp_name = f"const{len([v for v in var_names.values() if v.startswith('const')])}"
                        var_names[arg_id] = temp_name
                        arg_vars.append(temp_name)

                # Format the equation with type annotation
                op_name = node.name or "unknown"
                type_annotation = format_shape_and_dtype(node)

                if len(arg_vars) == 1:
                    equation = (
                        f"    {var_name}:{type_annotation} = {op_name} {arg_vars[0]}"
                    )
                else:
                    args_joined = " ".join(arg_vars)
                    fmt_str = f"    {var_name}:{type_annotation} = {op_name}"
                    equation = f"{fmt_str} {args_joined}"

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

    def print_trace(self) -> None:
        """Print the trace in a nice format."""
        print(self)


def _extract_arrays_from_pytree(tree: Any) -> list[Array]:
    """Extract all Arrays from a pytree structure.

    Args:
        tree: Pytree that may contain Arrays, ints, floats, etc.

    Returns:
        List of all Arrays found in the tree
    """
    leaves, _ = tree_flatten(tree)
    return leaves


def _cleanup_cotangents(traced_nodes: list[Array]) -> None:
    """Clean up cotangent values from traced nodes.

    Args:
        traced_nodes: List of traced nodes to clean up
    """
    for node in traced_nodes:
        node.cotangent = None


def _compute_reverse_gradients(
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
                from ..ops.creation import zeros

                gradient_arrays.append(zeros(inp.shape, dtype=inp.dtype))

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
    gradient_iter = iter(gradient_arrays)

    def _replace_with_gradient(value: Any) -> Any:
        if isinstance(value, Array):
            return next(gradient_iter)
        else:
            # For non-Arrays, return zeros of the same type
            if isinstance(value, (int | float)):
                return type(value)(0)
            elif isinstance(value, complex):
                return complex(0)
            else:
                return value  # Fallback for other types

    return tree_map(_replace_with_gradient, inputs)


def _adjust_gradient_structure_for_vjp(
    gradients_in_input_structure: Any,
    vjp_call_info: tuple[tuple, dict],
) -> Any:
    """Adjust gradient structure for VJP API convenience.

    Args:
        gradients_in_input_structure: Gradients in input structure
        vjp_call_info: Tuple of (original_args, original_kwargs) from VJP call

    Returns:
        Gradients adjusted for VJP API convenience
    """
    original_args, original_kwargs = vjp_call_info
    grad_args, grad_kwargs = gradients_in_input_structure

    # Return gradients in the same structure as the original VJP call
    if not original_kwargs:
        if len(original_args) == 1:
            return grad_args[0]  # Unpack single argument
        else:
            return grad_args
    else:
        return grad_args, grad_kwargs


def pullback(
    inputs: Any,
    outputs: Any,
    cotangents: Any,
    vjp_call_info: tuple[tuple, dict] | None = None,
) -> Any:
    """Compute vector-Jacobian product (reverse-mode autodiff).

    Returns gradients in the exact same structure as inputs.

    Args:
        inputs: Input arrays or pytree of arrays
        outputs: Output arrays or pytree of arrays
        cotangents: Cotangent vectors or pytree of cotangents
        vjp_call_info: Optional tuple of (original_args, original_kwargs) from VJP call
                      for special return structure handling

    Returns:
        Gradients with respect to inputs, in the same structure as inputs
        (or adjusted structure if vjp_call_info is provided)
    """
    # Extract arrays from pytree structures
    input_arrays = _extract_arrays_from_pytree(inputs)
    output_arrays = _extract_arrays_from_pytree(outputs)
    cotangent_arrays = _extract_arrays_from_pytree(cotangents)

    _validate_length_match(
        cotangent_arrays, output_arrays, "Cotangent arrays", "output arrays"
    )

    # Core reverse-mode gradient computation
    gradient_arrays = _compute_reverse_gradients(
        input_arrays, output_arrays, cotangent_arrays
    )

    # Reconstruct gradients in input structure
    gradients_in_input_structure = _reconstruct_gradient_structure(
        gradient_arrays, inputs
    )

    # Apply VJP API convenience adjustments if needed
    if vjp_call_info is not None:
        return _adjust_gradient_structure_for_vjp(
            gradients_in_input_structure, vjp_call_info
        )

    return gradients_in_input_structure


def pushfwd(
    inputs: list[Array],
    outputs: list[Array],
    tangents: list[Array],
    trace: Trace | None = None,
) -> list[Array]:
    """Compute Jacobian-vector product (forward-mode autodiff).

    Args:
        inputs: Input arrays to the computation
        outputs: Output arrays from the computation
        tangents: Tangent vectors for each input
        trace: Optional precomputed trace

    Returns:
        List of output tangents
    """
    _validate_length_match(tangents, inputs, "Tangents", "inputs")

    if trace is None:
        trace = Trace(inputs, outputs)

    traced_nodes = trace.get_traced_nodes()

    for input_node, tangent in zip(inputs, tangents, strict=False):
        input_node.tangent = tangent

    for node in traced_nodes:
        if node in inputs:
            continue

        if not node.args or node.jvp_rule is None:
            continue

        arg_tangents = []
        for arg in node.args:
            if arg.tangent is not None:
                arg_tangents.append(arg.tangent)
            else:
                from ..ops.creation import zeros

                arg_tangents.append(
                    zeros(arg.shape, dtype=arg.dtype, device=arg.device)
                )

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
            from ..ops.creation import zeros

            output_tangents.append(zeros(out.shape, dtype=out.dtype, device=out.device))

    return output_tangents


def xpr(
    fn: Callable[[list[Array]], list[Array]],
    args: list[Array],
) -> str:
    """Get a JAX-like string representation of the function's computation graph.

    Args:
        fn: Function to trace
        args: Input arrays to the function

    Returns:
        JAX-like string representation of the computation graph
    """
    trace = Trace.trace_function(fn, args)
    return str(trace)


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


def make_traced(args: list[Array]) -> list[Array]:
    """Create shallow copies of arrays and mark them as traced.

    Args:
        args: Arrays to copy and mark as traced

    Returns:
        Shallow copies of input arrays with tracing enabled
    """
    copied_args = []
    from ..ops.view import shallow_copy

    for arg in args:
        copied_arg = shallow_copy(arg)
        copied_arg.traced = True
        copied_args.append(copied_arg)
    return copied_args


def make_untraced(args: list[Array]) -> None:
    """Disable tracing for arrays by clearing their traced flag.

    Args:
        args: Arrays to disable tracing for
    """
    for arg in args:
        arg.traced = False


def make_staged(args: list[Array]) -> None:
    """Enable staged execution for arrays to optimize performance.

    Args:
        args: Arrays to enable staged execution for
    """
    for arg in args:
        arg.stage_realization = True  # Enable staged execution


def make_unstaged(args: list[Array]) -> None:
    """Disable staged execution for arrays.

    Args:
        args: Arrays to disable staged execution for
    """
    for arg in args:
        arg.stage_realization = False  # Disable staged execution


def vjp(func: Callable[..., Any], *args, **kwargs) -> tuple[Any, Callable]:
    """Compute vector-Jacobian product (reverse-mode autodiff).

    Args:
        func: Function to differentiate (can have arbitrary signature)
        *args: Positional arguments to the function (can be arbitrary pytrees)
        **kwargs: Keyword arguments to the function (can be arbitrary pytrees)

    Returns:
        Tuple of (outputs, vjp_function) where vjp_function computes gradients
    """
    # Combine args and kwargs into a single pytree for easier handling
    inputs_pytree = (args, kwargs)

    # Make traced copies of all inputs
    traced_inputs_pytree = make_traced_pytree(inputs_pytree)
    traced_args, traced_kwargs = traced_inputs_pytree

    # Execute the function with traced inputs
    outputs = func(*traced_args, **traced_kwargs)

    def vjp_fn(cotangents: Any) -> Any:
        """VJP function that computes gradients."""
        # Use the unified pullback function with pytree support and VJP call info
        gradients = pullback(
            traced_inputs_pytree, outputs, cotangents, vjp_call_info=(args, kwargs)
        )

        # Make the gradients untraced
        make_untraced_pytree(gradients)

        return gradients

    # Make outputs untraced before returning
    make_untraced_pytree(outputs)

    return outputs, vjp_fn


def jvp(
    func: Callable[[list[Array]], list[Array]],
    inputs: list[Array],
    tangents: list[Array],
) -> tuple[list[Array], list[Array]]:
    """Compute Jacobian-vector product (forward-mode autodiff).

    Args:
        func: Function to differentiate
        inputs: Input arrays to the function
        tangents: Tangent vectors for directional derivatives

    Returns:
        Tuple of (outputs, output_tangents) where output_tangents are the JVP results
    """
    _validate_length_match(tangents, inputs, "Tangents", "inputs")

    inputs = make_traced(inputs)
    outputs = func(inputs)

    if not isinstance(outputs, list):
        outputs = [outputs]

    output_tangents = pushfwd(inputs, outputs, tangents)

    make_untraced(outputs)
    make_untraced(output_tangents)

    return outputs, output_tangents


def vmap(
    func: Callable[[list[Array]], list[Array]],
    in_axes: list[int] | None = None,
    out_axes: list[int] | None = None,
) -> Callable[[list[Array]], list[Array]]:
    """Vectorize a function over specified input axes.

    Args:
        func: Function to vectorize
        in_axes: Input axes to vectorize over (default: axis 0 for all inputs)
        out_axes: Output axes to vectorize over (default: axis 0 for all outputs)

    Returns:
        Vectorized function that can handle batched inputs
    """

    def vectorized_func(inputs: list[Array]) -> list[Array]:
        adapted_in_axes = in_axes if in_axes is not None else [0] * len(inputs)

        _validate_length_match(adapted_in_axes, inputs, "adapted_in_axes", "inputs")

        batched_inputs = _prepare_vmap_inputs(inputs, adapted_in_axes)

        # Call the original function with batched inputs
        outputs = func(batched_inputs)

        # Ensure outputs are batched according to adapted_out_axes
        if not isinstance(outputs, list):
            outputs = [outputs]

        adapted_out_axes = out_axes if out_axes is not None else [0] * len(outputs)

        _validate_length_match(adapted_out_axes, outputs, "adapted_out_axes", "outputs")

        unbatched_outputs = _prepare_vmap_outputs(outputs, adapted_out_axes)

        make_untraced(unbatched_outputs)

        return unbatched_outputs

    return vectorized_func


def _prepare_vmap_inputs(
    inputs: list[Array], adapted_in_axes: list[int]
) -> list[Array]:
    """Prepare inputs for vmap by handling batching and axis transposition."""
    batched_inputs = []
    inputs = make_traced(inputs)

    for i, inp in enumerate(inputs):
        if adapted_in_axes[i] is None:
            from ..ops.view import unsqueeze

            batched_inp = unsqueeze(inp, [0])
        else:
            axis = adapted_in_axes[i]
            batched_inp = inp
            if axis != 0:
                from ..ops.view import transpose

                batched_inp = transpose(inp, axis, 0)

        from ..ops.unary import incr_batch_dim_ctr

        batched_inp = incr_batch_dim_ctr(batched_inp)
        batched_inputs.append(batched_inp)

    return batched_inputs


def _prepare_vmap_outputs(
    outputs: list[Array], adapted_out_axes: list[int]
) -> list[Array]:
    """Prepare outputs from vmap by handling unbatching and axis transposition."""
    unbatched_outputs = []

    for i, out in enumerate(outputs):
        from ..ops.unary import decr_batch_dim_ctr

        unbatched_output = decr_batch_dim_ctr(out)

        if adapted_out_axes[i] is None:
            from ..ops.view import squeeze

            unbatched_output = squeeze(unbatched_output, [0])
        else:
            axis = adapted_out_axes[i]
            if axis != 0:
                # Move axis 0 back to the original position
                from ..ops.view import transpose

                unbatched_output = transpose(unbatched_output, 0, axis)

        unbatched_outputs.append(unbatched_output)

    return unbatched_outputs


def jit(
    func: Callable[[list[Array]], list[Array]],
) -> Callable[[list[Array]], list[Array]]:
    """Just-in-time compile a function for performance optimization.

    Args:
        func: Function to JIT compile

    Returns:
        JIT-compiled function with optimized execution
    """

    def jit_func(inputs: list[Array]) -> list[Array]:
        inputs = make_traced(inputs)
        make_staged(inputs)  # Only callable after make_traced

        outputs = func(inputs)

        from .graph_execution import realize_

        realize_(outputs)

        if not isinstance(outputs, list):
            outputs = [outputs]

        make_untraced(outputs)
        make_unstaged(outputs)

        return outputs

    return jit_func
