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


def _handle_args_consistently(args):
    """Handle both fn([x,y,z]) and fn(x,y,z) calling styles."""
    if len(args) == 1 and isinstance(args[0], list):
        return args[0], True
    return args, False


def _prepare_traced_inputs(actual_args, is_list_style, apply_staging=False):
    """Prepare traced inputs for list-style or pytree-style arguments."""
    if is_list_style:
        traced_args = make_traced(actual_args)
        if apply_staging:
            make_staged(traced_args)
        return traced_args, None

    if len(actual_args) == 1:
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
        make_staged(arrays)

    return traced_args, traced_inputs_pytree


def _clean_traced_outputs(outputs, is_list_style, remove_staging=False):
    """Clean up traced outputs and handle staging flags."""
    if is_list_style:
        # For list-style, we expect a list of Arrays, but handle tuple case
        if isinstance(outputs, list):
            make_untraced(outputs)
            if remove_staging:
                make_unstaged(outputs)
        else:
            # If it's not a list (e.g., tuple from VJP), treat as pytree
            make_untraced_pytree(outputs)
            if remove_staging:
                output_arrays = _extract_arrays_from_pytree(outputs)
                make_unstaged(output_arrays)
    else:
        make_untraced_pytree(outputs)
        if remove_staging:
            output_arrays = _extract_arrays_from_pytree(outputs)
            make_unstaged(output_arrays)
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
        inputs = make_traced(inputs)

        # Create trace instance (this marks inputs as traced)
        trace = cls(inputs)

        # Execute function with tracing enabled
        outputs = fn(inputs)

        # Extract Arrays from outputs and store as list
        output_arrays = _extract_arrays_from_pytree(outputs)
        trace.outputs = output_arrays

        make_untraced(inputs)  # Detach inputs from the trace

        # Handle outputs properly - make them untraced
        make_untraced(output_arrays)

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
    # Handle args consistently
    actual_args, is_list_style = _handle_args_consistently([args])

    # Prepare traced inputs
    traced_args, _ = _prepare_traced_inputs(actual_args, is_list_style)

    # Create and return trace
    if is_list_style:
        trace = Trace.trace_function(fn, traced_args)
    else:
        # Adapting to xpr's function signature
        def wrapper(args_list):
            return fn(args_list[0])

        trace = Trace.trace_function(wrapper, [traced_args[0]])

    return str(trace)


def vjp(func: Callable[..., Any], *primals) -> tuple[Any, Callable]:
    """Compute vector-Jacobian product (reverse-mode autodiff).

    Args:
        func: Function to differentiate (should take positional arguments)
        *primals: Positional arguments to the function (can be arbitrary pytrees)

    Returns:
        Tuple of (outputs, vjp_function) where vjp_function computes gradients.

        The vjp_function always returns gradients as a tuple (matching JAX behavior):
        - Single argument: vjp_fn(cotangent) -> (gradient,)
        - Multiple arguments: vjp_fn(cotangent) -> (grad1, grad2, ...)

    Note:
        This follows JAX's vjp API exactly:
        - Only accepts positional arguments
        - Always returns gradients as tuple
        - For functions requiring keyword arguments, use functools.partial or lambda
    """
    # Handle the input structure based on number of arguments
    if len(primals) == 1:
        inputs_pytree = primals[0]
        is_single_arg = True
    else:
        inputs_pytree = primals
        is_single_arg = False

    # Make traced copies of all inputs
    traced_inputs_pytree = make_traced_pytree(inputs_pytree)

    # Extract traced args based on the structure
    traced_args = (traced_inputs_pytree,) if is_single_arg else traced_inputs_pytree

    # Execute the function with traced inputs
    outputs = func(*traced_args)

    def vjp_fn(cotangents: Any) -> tuple:
        """VJP function that computes gradients.

        Returns gradients as a tuple to match JAX's behavior:
        - Single argument: returns (gradient,)
        - Multiple arguments: returns (grad1, grad2, ...)
        """
        # Use the unified pullback function with pytree support
        gradients = pullback(traced_inputs_pytree, outputs, cotangents)

        # Make the gradients untraced
        make_untraced_pytree(gradients)

        # Always return as tuple to match JAX behavior
        if is_single_arg:
            return (gradients,)  # Wrap single gradient in tuple
        else:
            return gradients  # Already a tuple for multiple args

    # Make outputs untraced before returning
    make_untraced_pytree(outputs)

    return outputs, vjp_fn


def jvp(func: Callable[..., Any], primals, tangents) -> tuple[Any, Any]:
    """Compute Jacobian-vector product (forward-mode autodiff).

    Args:
        func: Function to differentiate (should take positional arguments)
        primals: Positional arguments to the function (can be arbitrary pytrees)
        tangents: Tangent vectors for directional derivatives (matching structure of primals)

    Returns:
        Tuple of (outputs, output_tangents) where output_tangents are the JVP results

    Note:
        This follows JAX's jvp API:
        - Only accepts positional arguments
        - For functions requiring keyword arguments, use functools.partial or lambda
    """
    # Handle inputs correctly based on structure
    is_multi_arg = isinstance(primals, tuple)

    # Validate primals and tangents match
    if is_multi_arg:
        if not isinstance(tangents, tuple) or len(primals) != len(tangents):
            raise ValueError(
                f"primals and tangents must have the same structure and length, "
                f"got {len(primals)} primals and {len(tangents) if isinstance(tangents, tuple) else 1} tangents"
            )
    elif isinstance(tangents, tuple):
        raise ValueError(
            "If primal is a single argument, tangent should also be a single argument"
        )

    # Make traced copies of all inputs
    traced_inputs_pytree = make_traced_pytree(primals)

    # Extract traced args based on structure
    traced_args = traced_inputs_pytree if is_multi_arg else (traced_inputs_pytree,)

    # Execute the function with traced inputs
    outputs = func(*traced_args)

    # Compute output tangents
    output_tangents = pushfwd(traced_inputs_pytree, outputs, tangents)

    # Make everything untraced before returning
    make_untraced_pytree(outputs)
    make_untraced_pytree(output_tangents)

    return outputs, output_tangents


def vmap(func=None, in_axes=0, out_axes=0) -> Callable[..., Any]:
    """Vectorize a function over specified input axes.
    This can be used as a function call like `vmap(func, in_axes=0)` or as a decorator `@vmap`.

    Args:
        func: Function to vectorize
        in_axes: Specification of axes to map over for inputs.
            If an integer, all inputs are mapped over that axis.
            If a tuple, should match the length of inputs with axis specifications.
        out_axes: Specification of axes for outputs.
            If an integer, all outputs are mapped over that axis.
            If a tuple, should match the structure of outputs.

    Returns:
        Vectorized function that can handle batched inputs

    Note:
        Supports both calling conventions:

        * List-style: vmapped_fn([x, y, z])
        * Unpacked-style: vmapped_fn(x, y, z)

    Example:
        As a function call::

            vmapped_func = vmap(my_func, in_axes=0, out_axes=0)

        As a decorator::

            @vmap
            def my_func(x):
                return x * 2

        As a decorator with arguments::

            @vmap(in_axes=1, out_axes=0)
            def my_func(x):
                return x * 2
    """
    # Handle being called as a decorator with arguments: @vmap(in_axes=1)
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes, out_axes=out_axes)

    # Helper function to standardize in_axes/out_axes to proper format
    def _standardize_axes(axes, length, name):
        if isinstance(axes, int) or axes is None:
            return [axes] * length
        elif isinstance(axes, tuple | list):
            if len(axes) != length:
                raise ValueError(
                    f"{name} length {len(axes)} != argument length {length}"
                )
            return list(axes)
        else:
            raise ValueError(
                f"{name} must be an integer, None, or a tuple/list, got {type(axes)}"
            )

    def vectorized_func(*args):
        # Use common argument handling logic
        actual_args, is_list_style = _handle_args_consistently(args)

        # Handle input structure
        if not actual_args:
            raise ValueError("vmap requires at least one input argument")

        # Standardize in_axes to match input structure
        adapted_in_axes = _standardize_axes(in_axes, len(actual_args), "in_axes")

        # Create traced copies of inputs
        traced_args = []
        for arg, axis in zip(actual_args, adapted_in_axes, strict=False):
            # Extract arrays and prepare for batch mapping
            arg_arrays = _extract_arrays_from_pytree(arg)
            arg_structure = tree_flatten(arg)[1]

            # Get axis for each array (replicate the axis value)
            array_axes = [axis] * len(arg_arrays)

            # Prepare arrays for batched execution
            batched_arrays = _prepare_vmap_inputs(arg_arrays, array_axes)

            # Reconstruct the original structure with batched arrays
            traced_arg = tree_unflatten(arg_structure, batched_arrays)
            traced_args.append(traced_arg)

        # Call the original function with appropriate style
        outputs = func(traced_args) if is_list_style else func(*traced_args)

        # Handle output structure - could be single output or multiple
        if not isinstance(outputs, list | tuple):
            outputs_structure = [outputs]
            is_single_output = True
        else:
            outputs_structure = outputs
            is_single_output = False

        # Standardize out_axes to match output structure
        adapted_out_axes = _standardize_axes(
            out_axes, len(outputs_structure) if not is_single_output else 1, "out_axes"
        )

        # Process each output
        unbatched_outputs = []
        for out, out_axis in zip(
            outputs_structure if not is_single_output else [outputs],
            adapted_out_axes,
            strict=False,
        ):
            # Extract arrays and prepare for unbatching
            out_arrays = _extract_arrays_from_pytree(out)
            out_structure = tree_flatten(out)[1]

            # Get axis for each array
            array_out_axes = [out_axis] * len(out_arrays)

            # Prepare arrays for unbatched results
            unbatched_arrays = _prepare_vmap_outputs(out_arrays, array_out_axes)

            # Reconstruct the original structure with unbatched arrays
            unbatched_out = tree_unflatten(out_structure, unbatched_arrays)
            unbatched_outputs.append(unbatched_out)

        # Return single output or tuple based on input structure
        return unbatched_outputs[0] if is_single_output else tuple(unbatched_outputs)

    return vectorized_func


def _validate_staging_debug(original_args, traced_args, is_list_style):
    """
    TEMPORARY DEBUG FUNCTION: Validate that all inputs are properly staged.

    This function checks if staging is correctly applied to inputs after
    the staging operation in jit. It provides detailed debug output about
    the staging status of each array.

    Args:
        original_args: Original input arguments before tracing/staging
        traced_args: Arguments after tracing and staging
        is_list_style: Whether using list-style arguments or not
    """
    print("=== STAGING VALIDATION DEBUG ===")

    # Extract arrays for checking
    if is_list_style:
        # For list-style, traced_args should be a list of Arrays
        arrays_to_check = traced_args
        print(f"List-style: checking {len(arrays_to_check)} arrays")
    else:
        # For pytree-style, need to extract arrays from the structure
        arrays_to_check = _extract_arrays_from_pytree(traced_args)
        print(f"Pytree-style: extracted {len(arrays_to_check)} arrays")

    staging_issues = []
    properly_staged = 0

    for i, array in enumerate(arrays_to_check):
        if not isinstance(array, Array):
            print(f"  Item {i}: Not an Array (type: {type(array)}) - SKIPPING")
            continue

        is_staged = getattr(array, "stage_realization", False)
        is_traced = getattr(array, "traced", False)

        print(
            f"  Array {i}: staged={is_staged}, traced={is_traced}, name='{array.name}', shape={array.shape}"
        )

        if not is_staged:
            staging_issues.append(f"Array {i} (name='{array.name}') is NOT staged")
        else:
            properly_staged += 1

    print(
        f"STAGING SUMMARY: {properly_staged}/{len(arrays_to_check)} arrays properly staged"
    )

    if staging_issues:
        print("STAGING ISSUES FOUND:")
        for issue in staging_issues:
            print(f"  ❌ {issue}")
        print("⚠️  WARNING: Some inputs may not be staged correctly!")
    else:
        print("✅ All arrays appear to be properly staged")

    print("=== END STAGING VALIDATION ===")


def jit(func: Callable[..., Any] = None) -> Callable[..., Any]:
    """Just-in-time compile a function for performance optimization.
    This can be used as a function call like `jit(func)` or as a decorator `@jit`.

    Args:
        func: Function to optimize with JIT compilation (should take positional arguments)

    Returns:
        JIT-compiled function with optimized execution

    Note:
        This follows JAX's jit API:

        * Only accepts positional arguments
        * For functions requiring keyword arguments, use functools.partial or lambda
        * Supports both list-style (legacy) and unpacked arguments style (JAX-like)

    Example:
        As a function call::

            fast_func = jit(my_func)

        As a decorator::

            @jit
            def my_func(x):
                return x * 2
    """
    # Handle being called as a decorator without arguments
    if func is None:
        return lambda f: jit(f)

    def jit_func(*args):
        # Use common argument handling logic
        actual_args, is_list_style = _handle_args_consistently(args)

        # Prepare traced inputs with staging enabled
        traced_args, _ = _prepare_traced_inputs(
            actual_args, is_list_style, apply_staging=True
        )

        # Execute the function with traced inputs and appropriate style
        outputs = func(traced_args) if is_list_style else func(*traced_args)

        # Realize only the Arrays in the outputs
        output_arrays = _extract_arrays_from_pytree(outputs)
        from .graph_execution import realize_

        realize_(output_arrays)

        # Clean up outputs and return
        return _clean_traced_outputs(outputs, is_list_style, remove_staging=True)

    return jit_func
