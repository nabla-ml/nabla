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

from ..core.tensor import Tensor


def tree_flatten(tree: Any) -> tuple[list[Tensor], Any]:
    """Flatten a pytree into a list of Tensors and structure info.

    Args:
        tree: A pytree containing Tensors and other structures

    Returns:
        A tuple of (list of Tensor leaves, structure info for reconstruction)
    """
    leaves = []

    def _flatten(obj: Any) -> Any:
        if isinstance(obj, Tensor):
            leaves.append(obj)
            return None  # Placeholder for Tensor
        elif isinstance(obj, dict):
            keys = sorted(obj.keys())  # Deterministic ordering
            return {k: _flatten(obj[k]) for k in keys}
        elif isinstance(obj, (list | tuple)):
            return type(obj)(_flatten(item) for item in obj)
        else:
            # Non-Tensor leaf (int, float, etc.)
            return obj

    structure = _flatten(tree)
    return leaves, structure


def tree_unflatten(structure: Any, leaves: list[Tensor]) -> Any:
    """Reconstruct a pytree from structure info and list of Tensors.

    Args:
        structure: Structure info from tree_flatten
        leaves: List of Tensor values to place at Tensor positions

    Returns:
        Reconstructed pytree with the same structure as the original
    """
    leaves_iter = iter(leaves)

    def _unflatten(struct: Any) -> Any:
        if struct is None:  # Tensor placeholder
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
            # Non-Tensor leaf
            return struct

    result = _unflatten(structure)

    # Verify we consumed all leaves
    try:
        next(leaves_iter)
        raise ValueError("Too many leaves provided for tree structure")
    except StopIteration:
        pass

    return result


def tree_map(func: Callable[[Tensor], Tensor], tree: Any) -> Any:
    """Apply a function to all Tensor leaves in a pytree.

    Args:
        func: Function to apply to each Tensor leaf
        tree: Pytree containing Tensors

    Returns:
        Pytree with the same structure but transformed Tensors
    """
    leaves, structure = tree_flatten(tree)
    transformed_leaves = [func(leaf) for leaf in leaves]
    return tree_unflatten(structure, transformed_leaves)


def _map_pytree_with_axes(process_func: Callable, tree: Any, axes: Any, *other_args) -> Any:
    """
    Recursively apply a processing function to corresponding elements of a data pytree
    and an axes pytree. This is the core engine for vmap's batching and unbatching.
    """
    def _recurse(tree_part, axes_part):
        if isinstance(tree_part, Tensor):
            # Apply the processing function to the tensor leaf
            return process_func(tree_part, axes_part, *other_args)
        elif isinstance(tree_part, dict):
            # If axes is a dict, recurse with matching keys. Otherwise, broadcast the axis.
            axes_map = axes_part if isinstance(axes_part, dict) else {k: axes_part for k in tree_part}
            return {k: _recurse(v, axes_map.get(k)) for k, v in tree_part.items()}
        elif isinstance(tree_part, (list, tuple)):
            # If axes is a sequence, recurse with matching elements. Otherwise, broadcast the axis.
            axes_list = axes_part if isinstance(axes_part, (list, tuple)) else [axes_part] * len(tree_part)
            return type(tree_part)([_recurse(t, a) for t, a in zip(tree_part, axes_list)])
        else:
            # Non-Tensor leaves are returned as is
            return tree_part

    return _recurse(tree, axes)




def _map_pytree_structure(func, *trees):
    """
    Recursively apply a function to corresponding elements of pytrees.
    This is a more general version of tree_map that can handle multiple trees
    and does not rely on tree_flatten, allowing it to work with structures
    containing non-Tensor leaves like scalars.
    """
    if not trees:
        return None
    first_tree = trees[0]
    if isinstance(first_tree, Tensor):
        return func(*trees)
    if isinstance(first_tree, (list, tuple)):
        return type(first_tree)(_map_pytree_structure(func, *[t[i] for t in trees]) for i in range(len(first_tree)))
    elif isinstance(first_tree, dict):
        return {k: _map_pytree_structure(func, *[t[k] for t in trees]) for k in first_tree}
    else:
        return func(*trees)


def _convert_to_scalar_if_needed(structure_provider_pytree, result_pytree):
    """
    Maps over two pytrees, converting items in result_pytree to scalars
    if the corresponding item in structure_provider_pytree is a scalar.
    """
    def _convert(structure_provider, result):
        if isinstance(structure_provider, (int, float)) and isinstance(result, Tensor):
            if result.shape == ():
                return result.to_numpy().item()
        return result
    return _map_pytree_structure(_convert, structure_provider_pytree, result_pytree)


def _convert_scalars_to_tensors(tree: Any) -> Any:
    """Recursively convert scalar numbers in a pytree to Nabla Tensors."""
    import nabla as nb
    if isinstance(tree, (int, float)):
        return nb.tensor(tree)
    if isinstance(tree, Tensor):
        return tree
    elif isinstance(tree, dict):
        return {k: _convert_scalars_to_tensors(v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(_convert_scalars_to_tensors(x) for x in tree)
    else:
        # Return other types (like functions, strings, etc.) unchanged
        return tree


def _extract_tensors_from_pytree(tree: Any) -> list[Tensor]:
    """Extract all Tensors from a pytree structure.

    Args:
        tree: Pytree that may contain Tensors, ints, floats, etc.

    Returns:
        List of all Tensors found in the tree
    """
    leaves, _ = tree_flatten(tree)
    return leaves


def _validate_length_match(list1, list2, name1, name2):
    """Check if two lists have the same length."""
    if len(list1) != len(list2):
        raise ValueError(f"{name1} length {len(list1)} != {name2} length {len(list2)}")


def _create_jacobian_helpers(func, argnums, args):
    """Create helper functions for jacobian calculations."""
    # Normalize and validate argnums
    if argnums is None:
        selected_argnums = tuple(range(len(args)))
    else:
        selected_argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    for argnum in selected_argnums:
        if not (-len(args) <= argnum < len(args)):
            raise ValueError(f"argnum {argnum} is out of bounds for function with {len(args)} arguments")

    normalized_argnums = tuple(argnum if argnum >= 0 else len(args) + argnum for argnum in selected_argnums)
    
    # Extract arguments to be differentiated
    diff_args = tuple(args[i] for i in normalized_argnums)

    # Create a partial function that only takes the differentiated arguments
    def partial_func(*diff_args_inner):
        full_args = list(args)
        for i, arg in zip(normalized_argnums, diff_args_inner, strict=False):
            full_args[i] = arg
        return func(*full_args)

    return diff_args, partial_func


def _std_basis(args: list[Tensor]) -> tuple[list[int], list[Tensor]]:
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
    tangents: list[Tensor] = []

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
            tangent = Tensor.from_numpy(np_tangent)

            from ..ops.view import broadcast_batch_dims

            tangent = broadcast_batch_dims(tangent, arg.batch_dims)

            tangents.append(tangent)
            sizes.append(num_elements)

    return sizes, tangents


def make_traced_pytree(tree: Any) -> Any:
    """Create shallow copies of tensors in a pytree and mark them as traced.

    Args:
        tree: Pytree containing Tensors to copy and mark as traced

    Returns:
        Pytree with the same structure but traced Tensors
    """

    def _make_traced_tensor(tensor: Tensor) -> Tensor:
        from ..ops.view import shallow_copy

        copied_arg = shallow_copy(tensor)
        copied_arg.traced = True
        return copied_arg

    return tree_map(_make_traced_tensor, tree)


def make_untraced_pytree(tree: Any) -> None:
    """Disable tracing for tensors in a pytree by clearing their traced flag.

    Args:
        tree: Pytree containing Tensors to disable tracing for
    """

    def _make_untraced_tensor(tensor: Tensor) -> Tensor:
        tensor.traced = False
        return tensor

    tree_map(_make_untraced_tensor, tree)


def make_staged_pytree(args: list[Tensor]) -> None:
    """Enable staged execution for tensors to optimize performance.

    Args:
        args: Tensors to enable staged execution for
    """

    def _make_staged_tensor(tensor: Tensor) -> Tensor:
        tensor.stage_realization = True
        return tensor

    tree_map(_make_staged_tensor, args)


def make_unstaged_pytree(args: list[Tensor]) -> None:
    """Disable staged execution for tensors.

    Args:
        args: Tensors to disable staged execution for
    """

    def _make_unstaged_tensor(tensor: Tensor) -> Tensor:
        tensor.stage_realization = False
        return tensor

    tree_map(_make_unstaged_tensor, args)


def _handle_args_consistently(args):
    """Handle both fn([x,y,z]) and fn(x,y,z) calling styles."""
    if len(args) == 1 and isinstance(args[0], list):
        return args[0], True
    return args, False


def process_transform_inputs(args, convert_scalars=False, apply_staging=False):
    """
    Standardize input processing for all transformations.
    - Handles both list-style and unpacked arguments.
    - Converts scalars to Tensors if needed.
    - Creates traced copies for graph capture.
    - Applies staging for performance optimization if requested.
    """
    actual_args, is_list_style = _handle_args_consistently(args)
    
    if convert_scalars:
        actual_args = _convert_scalars_to_tensors(actual_args)

    traced_args = make_traced_pytree(actual_args)
    
    if apply_staging:
        tensors = _extract_tensors_from_pytree(traced_args)
        make_staged_pytree(tensors)

    # Return everything needed to execute the function and process outputs
    return traced_args, actual_args, is_list_style


# In the first file (autodiff_core.py)
def process_transform_outputs(outputs, original_inputs, is_list_style, untrace=True, unstage=True):
    """
    Standardize output processing for all transformations.
    - Untraces outputs to prevent graph leakage.
    - Unstages outputs to finalize computation.
    """
    if untrace:
        any_input_traced = any(
            getattr(arg, "traced", False) for arg in _extract_tensors_from_pytree(original_inputs)
        )
        if not any_input_traced:
            make_untraced_pytree(outputs)

    if unstage:
        output_tensors = _extract_tensors_from_pytree(outputs)
        make_unstaged_pytree(output_tensors)

    # The complex and buggy scalar conversion has been removed.
    # The function now simply returns the processed outputs.
    return outputs


class Trace:
    """A simple trace container that holds the computation graph."""

    def __init__(
        self, inputs: list[Tensor] | None, outputs: list[Tensor] | None = None
    ) -> None:
        if outputs is None:
            outputs = []

        if inputs is None:
            if not outputs:
                raise ValueError("Cannot infer trace inputs without outputs")
            inputs = self._discover_traced_inputs_requiring_grad(outputs)

        self.inputs = inputs
        self.outputs = outputs
        self.trace: list[Tensor] = []
        self._computed = False

        # Mark all inputs as traced for autodiff so the computation graph gets captured
        for inp in inputs:
            if not getattr(inp, "traced", False):
                inp.traced = True

    @classmethod
    def trace_function(
        cls, fn: Callable[[list[Tensor]], list[Tensor]], inputs: list[Tensor]
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

        # Extract Tensors from outputs and store as list
        output_tensors = _extract_tensors_from_pytree(outputs)
        trace.outputs = output_tensors

        make_untraced_pytree(inputs)  # Detach inputs from the trace

        # Handle outputs properly - make them untraced
        make_untraced_pytree(output_tensors)

        return trace

    def get_traced_nodes(self) -> list[Tensor]:
        """Get all nodes that belong to this trace in topological order."""
        if not self._computed:
            self._compute_trace()
        return self.trace

    def _compute_trace(self) -> None:
        """Compute the topological ordering of traced nodes."""
        visited: set[Tensor] = set()
        self.trace = []

        for output in self.outputs:
            self._dfs_visit(output, visited)

        self._computed = True

    def _dfs_visit(self, node: Tensor, visited: set[Tensor]) -> None:
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

        from ..utils.formatting import format_shape_dtype_device

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
            type_annotation = format_shape_dtype_device(inp)
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
                    isinstance(node, Tensor)
                    and node.shape == ()
                    and not node.batch_dims
                    and node.impl
                ):
                    # This is a constant scalar, show the raw value
                    op_name = str(node.to_numpy().item())
                else:
                    # Fallback to the type or some default name
                    op_name = "external_const"
            type_annotation = format_shape_dtype_device(node)

            if node.args:
                # Get argument variable names
                arg_vars = []
                for arg in node.args:
                    arg_id = id(arg)
                    if arg_id in var_names:
                        arg_vars.append(var_names[arg_id])
                    else:
                        # Tensor from external context - not part of the trace
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

    @classmethod
    def from_outputs(cls, outputs: Any) -> Trace:
        """Create a trace by discovering traced leaf inputs from outputs."""
        output_tensors = _extract_tensors_from_pytree(outputs)
        inputs = cls._discover_traced_inputs_requiring_grad(output_tensors)
        return cls(inputs, output_tensors)

    @staticmethod
    def _discover_traced_inputs_requiring_grad(output_tensors: list[Tensor]) -> list[Tensor]:
        """Find all nodes with requires_grad=True in the computation graph.
        
        This discovers all tensors that have been explicitly marked with requires_grad=True,
        regardless of whether they are leaf nodes or intermediate computations. This allows
        gradient accumulation on any node in the graph that the user has marked.
        """
        discovered: list[Tensor] = []
        discovered_ids: set[int] = set()
        visited: set[int] = set()

        def _dfs(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)

            # Recurse into children first
            for arg in getattr(node, "args", ()) or ():
                if isinstance(arg, Tensor):
                    _dfs(arg)

            # Check if this node requires gradients (explicit True check)
            requires_grad = getattr(node, "requires_grad", None)
            if requires_grad is True:
                if node_id not in discovered_ids:
                    discovered.append(node)
                    discovered_ids.add(node_id)

        for output in output_tensors:
            _dfs(output)

        return discovered


def _cleanup_cotangents(traced_nodes: list[Tensor]) -> None:
    """Clean up cotangent values from traced nodes.

    Args:
        traced_nodes: List of traced nodes to clean up
    """
    for node in traced_nodes:
        node.cotangent = None


def _compute_pullback(
    input_tensors: list[Tensor],
    output_tensors: list[Tensor],
    cotangent_tensors: list[Tensor],
) -> list[Tensor]:
    """Core reverse-mode gradient computation.

    Args:
        input_tensors: Input tensors to compute gradients for
        output_tensors: Output tensors from the computation
        cotangent_tensors: Cotangent vectors for outputs

    Returns:
        List of gradient tensors corresponding to inputs
    """
    # Build computation trace
    trace = Trace(input_tensors, output_tensors)
    traced_nodes = trace.get_traced_nodes()

    # Initialize output cotangents
    for output, cotangent in zip(output_tensors, cotangent_tensors, strict=False):
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

                if node not in input_tensors:
                    node.cotangent = None

            except Exception as e:
                raise RuntimeError(
                    f"VJP rule failed for operation '{node.name}': {e}"
                ) from e

        # Collect gradients for input tensors
        gradient_tensors = []
        for inp in input_tensors:
            if inp.cotangent is not None:
                gradient_tensors.append(inp.cotangent)
            else:
                from ..ops.creation import zeros_like

                gradient_tensors.append(zeros_like(inp))

        return gradient_tensors

    finally:
        _cleanup_cotangents(traced_nodes)


def _reconstruct_gradient_structure(
    gradient_tensors: list[Tensor],
    inputs: Any,
) -> Any:
    """Reconstruct gradients in the same structure as inputs.

    Args:
        gradient_tensors: Flat list of gradient tensors
        inputs: Original input structure to match

    Returns:
        Gradients with the same structure as inputs
    """
    # Use the same flattening/unflattening logic as used for input extraction
    input_tensors, structure = tree_flatten(inputs)

    # Validate that we have the right number of gradients
    if len(gradient_tensors) != len(input_tensors):
        raise ValueError(
            f"Gradient tensors length {len(gradient_tensors)} != "
            f"input tensors length {len(input_tensors)}"
        )

    # Reconstruct the pytree structure with gradients
    return tree_unflatten(structure, gradient_tensors)


def backward(outputs: Any, cotangents: Any, retain_graph: bool = False) -> None:
    """Accumulate gradients on traced leaf inputs for the given traced outputs.
    
    Args:
        outputs: Output tensors to backpropagate from
        cotangents: Cotangent vectors for outputs
        retain_graph: If False (default), frees the computation graph after backward pass
    """

    trace = Trace.from_outputs(outputs)
    input_tensors = trace.inputs
    output_tensors = trace.outputs

    cotangent_tensors = _extract_tensors_from_pytree(cotangents)
    if not cotangent_tensors and output_tensors:
        raise ValueError("cotangents must contain tensors")

    _validate_length_match(
        cotangent_tensors, output_tensors, "cotangents", "outputs"
    )

    gradients = _compute_pullback(input_tensors, output_tensors, cotangent_tensors)

    for inp, grad in zip(input_tensors, gradients, strict=False):
        if inp.grad is None:
            inp.grad = grad
        else:
            from ..ops.binary import add

            inp.grad = add(inp.grad, grad)

    if not retain_graph:
        traced_nodes = trace.get_traced_nodes()
        for node in traced_nodes:
            node.args = []
            node.traced = False


def pullback(
    inputs: Any,
    outputs: Any,
    cotangents: Any,
) -> Any:
    """Compute vector-Jacobian product (reverse-mode autodiff).

    Returns gradients in the exact same structure as inputs.

    Args:
        inputs: Input tensors or pytree of tensors
        outputs: Output tensors or pytree of tensors
        cotangents: Cotangent vectors or pytree of cotangents

    Returns:
        Gradients with respect to inputs, in the same structure as inputs
    """
    # Extract tensors from pytree structures
    input_tensors = _extract_tensors_from_pytree(inputs)
    output_tensors = _extract_tensors_from_pytree(outputs)
    cotangent_tensors = _extract_tensors_from_pytree(cotangents)

    _validate_length_match(
        cotangent_tensors, output_tensors, "Cotangent tensors", "output tensors"
    )

    # Core reverse-mode gradient computation
    gradient_tensors = _compute_pullback(input_tensors, output_tensors, cotangent_tensors)

    # Reconstruct gradients in input structure
    gradients_in_input_structure = _reconstruct_gradient_structure(
        gradient_tensors, inputs
    )

    return gradients_in_input_structure


def grad(
    outputs: Any,
    inputs: Any,
    grad_outputs: Any | None = None,
    create_graph: bool = False,
) -> Any:
    """Compute gradients of outputs with respect to inputs (imperative PyTorch-style API).
    
    This is similar to PyTorch's torch.autograd.grad() function. It computes and returns
    gradients without accumulating them in the .grad attribute (unlike backward()).
    
    Args:
        outputs: Output tensor(s) to differentiate. Can be a single tensor or a sequence/tuple.
        inputs: Input tensor(s) to compute gradients for. Can be a single tensor or sequence/tuple.
        grad_outputs: Gradient of outputs (cotangents). If None, defaults to ones_like for scalar outputs.
        create_graph: If True, the gradient computation will be traceable for higher-order derivatives.
    
    Returns:
        Gradients with respect to inputs. Returns a single tensor if inputs is a single tensor,
        otherwise returns a tuple of tensors matching the structure of inputs.
    
    Examples:
        >>> import nabla as nb
        >>> from nabla.transforms import autograd
        >>> 
        >>> # Simple scalar output
        >>> x = nb.tensor([1.0, 2.0, 3.0])
        >>> x.requires_grad_(True)
        >>> y = nb.sum(x ** 2)
        >>> grad_x = autograd(y, x)  # Computes dy/dx
        
        >>> # Multiple inputs
        >>> a = nb.tensor([1.0, 2.0])
        >>> b = nb.tensor([3.0, 4.0])
        >>> a.requires_grad_(True)
        >>> b.requires_grad_(True)
        >>> z = nb.sum(a * b)
        >>> grad_a, grad_b = autograd(z, [a, b])
        
        >>> # Custom grad_outputs
        >>> output = a ** 2
        >>> grad_outputs = nb.ones_like(output) * 0.5
        >>> grads = autograd(output, [a, b], grad_outputs=grad_outputs)
    """
    
    # Normalize inputs to a list and remember if it was a single tensor
    if isinstance(inputs, Tensor):
        input_list = [inputs]
        is_single_input = True
    elif isinstance(inputs, (list, tuple)):
        input_list = list(inputs)
        is_single_input = False
    else:
        raise TypeError(f"inputs must be a Tensor or sequence of Tensors, got {type(inputs)}")
    
    # Normalize outputs to a list
    if isinstance(outputs, Tensor):
        output_list = [outputs]
    elif isinstance(outputs, (list, tuple)):
        output_list = list(outputs)
    else:
        raise TypeError(f"outputs must be a Tensor or sequence of Tensors, got {type(outputs)}")
    
    # Handle grad_outputs (cotangents)
    if grad_outputs is None:
        # Default to ones_like for each output
        from ..ops.creation import ones_like
        cotangent_list = [ones_like(out) for out in output_list]
    else:
        # Normalize grad_outputs to a list
        if isinstance(grad_outputs, Tensor):
            cotangent_list = [grad_outputs]
        elif isinstance(grad_outputs, (list, tuple)):
            cotangent_list = list(grad_outputs)
        else:
            raise TypeError(f"grad_outputs must be a Tensor or sequence of Tensors, got {type(grad_outputs)}")
        
        # Validate cotangents match outputs
        if len(cotangent_list) != len(output_list):
            raise ValueError(
                f"Number of grad_outputs ({len(cotangent_list)}) must match "
                f"number of outputs ({len(output_list)})"
            )
    
    # Compute gradients using the existing pullback infrastructure
    gradient_tensors = _compute_pullback(input_list, output_list, cotangent_list)
    
    # Clean up computation graph unless create_graph=True
    if not create_graph:
        trace = Trace(input_list, output_list)
        traced_nodes = trace.get_traced_nodes()
        for node in traced_nodes:
            node.args = []
            node.traced = False
    
    # Return in the same format as inputs (single tensor or tuple)
    if is_single_input:
        return gradient_tensors[0]
    else:
        return tuple(gradient_tensors)


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
        inputs: Input tensors or pytree of tensors
        outputs: Output tensors or pytree of tensors
        tangents: Tangent vectors or pytree of tangents

    Returns:
        Tangents with respect to outputs, in the same structure as outputs
    """
    # Extract tensors from pytree structures
    input_tensors = _extract_tensors_from_pytree(inputs)
    output_tensors = _extract_tensors_from_pytree(outputs)
    tangent_tensors = _extract_tensors_from_pytree(tangents)

    _validate_length_match(
        tangent_tensors, input_tensors, "Tangent tensors", "input tensors"
    )

    # Core forward-mode gradient computation
    output_tangents = _compute_pushfwd(input_tensors, output_tensors, tangent_tensors)

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
        for arg in _extract_tensors_from_pytree(inputs_pytree)
    )

    # Make traced copies of all inputs
    traced_inputs_pytree = make_traced_pytree(inputs_pytree)

    # Extract traced args based on the structure
    traced_args = (traced_inputs_pytree,) if is_single_arg else traced_inputs_pytree

    # Execute the function with traced inputs
    outputs = fn(*traced_args)

    # Extract output tensors for trace creation
    output_tensors = _extract_tensors_from_pytree(outputs)
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors] if output_tensors is not None else []

    # Extract input tensors for trace creation
    input_tensors = _extract_tensors_from_pytree(traced_inputs_pytree)
    if not isinstance(input_tensors, list):
        input_tensors = [input_tensors] if input_tensors is not None else []

    # Ensure we have proper Tensor lists (not Never)
    if not input_tensors:
        input_tensors = []
    if not output_tensors:
        output_tensors = []

    # Create trace with the computation graph
    trace = Trace(input_tensors, output_tensors)  # type: ignore

    # Make everything untraced before returning
    # make_untraced_pytree(traced_inputs_pytree)
    if not any_arg_traced:
        make_untraced_pytree(outputs)

    return str(trace)
