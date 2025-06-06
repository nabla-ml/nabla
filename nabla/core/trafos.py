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


# def _std_basis(args: list[Array]) -> tuple[list[int], list[Array]]:
#     num_total_arg_elements = 0
#     max_rank = 0
#     for arg in args:
#         num_elements = 1
#         for dim in arg.shape:
#             num_elements *= dim
#         num_total_arg_elements += num_elements
#         rank = len(arg.shape)
#         if rank > max_rank:
#             max_rank = rank

#     batch_ctr = 0
#     sizes = list[int]()
#     tangents: list[Array] = []

#     for i, arg in enumerate(args):
#         num_elements = 1

#         for dim in arg.shape:
#             num_elements *= dim

#         # For scalar outputs, preserve scalar shape instead of creating (1,) shape
#         if arg.shape == ():
#             # Scalar case: create a scalar basis vector
#             from numpy import zeros as np_zeros, array as np_array

#             # Create scalar array with batch dimensions
#             if arg.batch_dims:
#                 batched_shape = arg.batch_dims
#                 np_tangent = np_zeros(batched_shape, dtype=arg.dtype.to_numpy())
#                 np_tangent.fill(1.0)
#             else:
#                 # Pure scalar case - create a 0-dimensional numpy array
#                 np_tangent = np_array(1.0, dtype=arg.dtype.to_numpy())

#             tangent = Array.from_numpy(np_tangent)

#             # Handle batch dimensions
#             from ..ops.unary import incr_batch_dim_ctr
#             for _ in range(len(arg.batch_dims)):
#                 tangent = incr_batch_dim_ctr(tangent)

#         else:
#             # Non-scalar case: use original logic
#             batched_shape = arg.batch_dims + arg.shape
#             for _ in range(max_rank - len(batched_shape)):
#                 batched_shape = (1,) + batched_shape

#             batched_shape = arg.batch_dims + (num_total_arg_elements,) + arg.shape

#             from numpy import zeros as np_zeros

#             np_tangent = np_zeros(batched_shape, dtype=arg.dtype.to_numpy()).flatten()

#             num_els_batch_dims = 1
#             for dim in arg.batch_dims:
#                 num_els_batch_dims *= dim

#             for i in range(num_els_batch_dims):
#                 offset = (batch_ctr + num_total_arg_elements * i) * num_elements

#                 for j in range(num_elements):
#                     idx = offset + j
#                     np_tangent[idx] = 1.0
#                     offset += num_elements
#                     batch_ctr += 1

#             np_tangent = np_tangent.reshape(batched_shape)
#             tangent = Array.from_numpy(np_tangent)

#             from ..ops.unary import incr_batch_dim_ctr

#             for _ in range(len(arg.batch_dims)):
#                 tangent = incr_batch_dim_ctr(tangent)

#         tangents.append(tangent)
#         sizes.append(num_elements)
#         batch_ctr += num_elements

#     return sizes, tangents


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

    for i, arg in enumerate(args):
        num_elements = 1

        for dim in arg.shape:
            num_elements *= dim

        # batched_shape = arg.batch_dims + arg.shape
        # for _ in range(max_rank - len(batched_shape)):
        #     batched_shape = (1,) + batched_shape

        batched_shape =  (num_total_arg_elements,) + arg.shape

        from numpy import zeros as np_zeros

        np_tangent = np_zeros(batched_shape, dtype=arg.dtype.to_numpy()).flatten()

        # num_els_batch_dims = 1
        # for dim in arg.batch_dims:
        #     num_els_batch_dims *= dim

        # for i in range(num_elements):
        #     offset = batch_ctr * num_elements

        offset = batch_ctr * num_elements

        for j in range(num_elements):
            # offset = j * num_elements
            idx = offset + j * num_elements + j
            np_tangent[idx] = 1.0
            # offset += num_elements
            batch_ctr += 1

        np_tangent = np_tangent.reshape(batched_shape)
        tangent = Array.from_numpy(np_tangent)
        # tangent.batch_dims = arg.batch_dims
        # tangent.shape = tangent.shape[len(arg.batch_dims) :]

        # print(tangent)
        # print(tangent)

        from ..ops.view import broadcast_batch_dims
        tangent = broadcast_batch_dims(tangent, arg.batch_dims)

        # print(tangent)

        # from ..ops.unary import incr_batch_dim_ctr

        # for _ in range(len(arg.batch_dims)):
        #     tangent = incr_batch_dim_ctr(tangent)

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
            op_name = node.name or "unknown"
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
                from ..ops.creation import zeros_like

                arg_tangents.append(
                    zeros_like(arg)
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
        getattr(arg, "traced", False) for arg in _extract_arrays_from_pytree(inputs_pytree)
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

    # Create trace with the computation graph
    trace = Trace(input_arrays, output_arrays)

    # Make everything untraced before returning
    # make_untraced_pytree(traced_inputs_pytree)
    if not any_arg_traced:
        make_untraced_pytree(outputs)

    return str(trace)


def vjp(
    func: Callable[..., Any], *primals, has_aux: bool = False
) -> tuple[Any, Callable]:
    """Compute vector-Jacobian product (reverse-mode autodiff).

    Args:
        func: Function to differentiate (should take positional arguments)
        *primals: Positional arguments to the function (can be arbitrary pytrees)
        has_aux: Optional, bool. Indicates whether `func` returns a pair where the
            first element is considered the output of the mathematical function to be
            differentiated and the second element is auxiliary data. Default False.

    Returns:
        If has_aux is False:
            Tuple of (outputs, vjp_function) where vjp_function computes gradients.
        If has_aux is True:
            Tuple of (outputs, vjp_function, aux) where aux is the auxiliary data.

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

    any_arg_traced = any(
        getattr(arg, "traced", False) for arg in _extract_arrays_from_pytree(inputs_pytree)
    )

    # Make traced copies of all inputs
    traced_inputs_pytree = make_traced_pytree(inputs_pytree)

    # Extract traced args based on the structure
    traced_args = (traced_inputs_pytree,) if is_single_arg else traced_inputs_pytree

    # Execute the function with traced inputs
    full_outputs = func(*traced_args)

    # Handle has_aux: separate main outputs from auxiliary data
    if has_aux:
        if not isinstance(full_outputs, tuple) or len(full_outputs) != 2:
            raise ValueError(
                "Function with has_aux=True must return a tuple (output, aux)"
            )
        outputs, aux = full_outputs
    else:
        outputs = full_outputs
        aux = None

    def vjp_fn(cotangents: Any) -> tuple:
        """VJP function that computes gradients.

        Returns gradients as a tuple to match JAX's behavior:
        - Single argument: returns (gradient,)
        - Multiple arguments: returns (grad1, grad2, ...)
        """
        # Always ensure cotangents are traced for composability with other transformations
        traced_cotangents = make_traced_pytree(cotangents)

        # Use the unified pullback function with pytree support
        gradients = pullback(traced_inputs_pytree, outputs, traced_cotangents)

        # Check if original cotangents were traced - if so, keep gradients traced
        cotangent_arrays = _extract_arrays_from_pytree(cotangents)
        any_cotangent_traced = any(
            getattr(arr, "traced", False) for arr in cotangent_arrays
        )

        # Only make gradients untraced if original cotangents were not traced
        if not any_cotangent_traced and not any_arg_traced:
            make_untraced_pytree(gradients)

        # # Always return as tuple to match JAX behavior
        # if is_single_arg:
        #     return (gradients,)  # Wrap single gradient in tuple
        # else:
        #     return gradients  # Already a tuple for multiple args

        return gradients 
    
    # Make outputs untraced before returning
    # make_untraced_pytree(outputs)
    if not any_arg_traced:
        make_untraced_pytree(outputs)

    # Return based on has_aux
    if has_aux:
        return outputs, vjp_fn, aux
    else:
        return outputs, vjp_fn


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
    is_multi_arg = isinstance(primals, tuple)

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
    if not any_primal_traced and not any_tangent_traced:
        make_untraced_pytree(outputs)
        make_untraced_pytree(output_tangents)

    return outputs, output_tangents



def _apply_batching_to_tree(tree: Any, axes: Any, is_input: bool = True) -> Any:
    """Apply batching/unbatching to a pytree structure.
    
    Args:
        tree: Pytree containing Arrays
        axes: Axis specification matching tree structure
        is_input: True for input batching, False for output unbatching
    """
    def _process_array(array: Array, axis: int | None) -> Array:
        if is_input:
            # Input batching
            from nabla.ops.view import unsqueeze
            from nabla.ops.unary import incr_batch_dim_ctr

            if axis is None:
                # Broadcast: add size-1 batch dimension
                batched = unsqueeze(array, [0])
            else:
                # Move specified axis to position 0
                if axis != 0:
                    from ..ops.view import move_axis_to_front
                    batched = move_axis_to_front(array, axis)
                else:
                    batched = array
            
            return incr_batch_dim_ctr(batched)
        else:
            # Output unbatching
            from nabla.ops.view import squeeze
            from nabla.ops.unary import decr_batch_dim_ctr
            
            unbatched = decr_batch_dim_ctr(array)
            
            if axis is None:
                # Remove batch dimension
                unbatched = squeeze(unbatched, [0])
            else:
                # Move axis 0 to specified position
                if axis != 0:
                    from ..ops.view import move_axis_from_front
                    unbatched = move_axis_from_front(unbatched, axis)
            
            return unbatched
    
    def _recurse(tree_part: Any, axes_part: Any) -> Any:
        if isinstance(tree_part, Array):
            return _process_array(tree_part, axes_part)
        elif isinstance(tree_part, dict):
            return {k: _recurse(tree_part[k], axes_part[k]) for k in tree_part}
        elif isinstance(tree_part, (list, tuple)):
            result = [_recurse(t, a) for t, a in zip(tree_part, axes_part)]
            return type(tree_part)(result)
        else:
            # Non-Array leaf, return unchanged
            return tree_part
    
    return _recurse(tree, axes)

def _broadcast_axis_spec(axis_spec: Any, num_items: int) -> tuple[Any, ...]:
    """Broadcast axis specification to match number of items."""
    if isinstance(axis_spec, (int, type(None))):
        return tuple(axis_spec for _ in range(num_items))
    elif isinstance(axis_spec, (list, tuple)):
        if len(axis_spec) != num_items:
            raise ValueError(f"Axis specification length {len(axis_spec)} != number of items {num_items}")
        return tuple(axis_spec)
    else:
        raise ValueError(f"Invalid axis specification: {axis_spec}")


def vmap(func=None, in_axes=0, out_axes=0) -> Callable[..., Any]:
    """Enhanced vmap with clean pytree support.
    
    This is a simplified, clean implementation that supports all JAX vmap features:
    - Pytree inputs/outputs with matching axis specifications
    - Broadcasting (axis=None) and batching (axis=int)
    - Nested structures (tuples, lists, dicts)
    - Both list-style and unpacked argument calling conventions
    """
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes, out_axes=out_axes)
    
    def vectorized_func(*args):
        # Handle calling conventions
        actual_args, is_list_style = _handle_args_consistently(args)
        
        if not actual_args:
            raise ValueError("vmap requires at least one input argument")
        
        # Broadcast in_axes to match arguments
        structured_in_axes = _broadcast_axis_spec(in_axes, len(actual_args))
        
        # Apply input batching
        batched_args = []
        for arg, axis_spec in zip(actual_args, structured_in_axes):
            # Make traced first
            # traced_arg = tree_map(lambda a: make_traced([a])[0] if isinstance(a, Array) else a, arg)
            # Apply batching
            # batched_arg = _apply_batching_to_tree(traced_arg, axis_spec, is_input=True)
            batched_arg = _apply_batching_to_tree(arg, axis_spec, is_input=True)
            batched_args.append(batched_arg)
        
        # Execute function
        outputs = func(batched_args) if is_list_style else func(*batched_args)
        
        # Handle output structure
        if not isinstance(outputs, (list, tuple)):
            outputs_list = [outputs]
            is_single_output = True
        else:
            outputs_list = outputs
            is_single_output = False
        
        # Broadcast out_axes to match outputs
        structured_out_axes = _broadcast_axis_spec(out_axes, len(outputs_list))
        
        # Apply output unbatching
        unbatched_outputs = []
        for output, axis_spec in zip(outputs_list, structured_out_axes):
            unbatched_output = _apply_batching_to_tree(output, axis_spec, is_input=False)
            unbatched_outputs.append(unbatched_output)
        
        return unbatched_outputs[0] if is_single_output else tuple(unbatched_outputs)
    
    return vectorized_func



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
        if isinstance(argnums, int):
            selected_argnums = (argnums,)
        else:
            selected_argnums = tuple(argnums)

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

        # print("std_basis_vectors:", std_basis_vectors)
        # print("sizes:", sizes)

        # print("INTERNAL")
        # print(xpr(vmap(pullback), std_basis_vectors))

        # Compute Jacobian using vmap over pullback
        # print("\n   CALL INNER\n")
        grads = vmap(pullback)(std_basis_vectors)
        # print("inner xpr:")
        # print(xpr(vmap(pullback), std_basis_vectors))
        # print("\n   END INNER\n")

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

        # print("grads:", grads)  

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

            # print("input grad:", grads[i])

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
