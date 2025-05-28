# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
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

        # Create trace instance (this marks inputs as traced)
        trace = cls(inputs)

        try:
            # Execute function with tracing enabled
            outputs = fn(inputs)
            trace.outputs = outputs if isinstance(outputs, list) else [outputs]
            return trace
        finally:
            # Don't reset traced flags here - let the caller handle cleanup
            pass

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
        if node in visited or not node.traced:
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

        # ANSI color codes
        purple = "\033[95m"
        reset = "\033[0m"

        def format_dtype(dtype: Any) -> str:
            """Format dtype for display."""
            # Convert DType to string representation
            dtype_str = str(dtype).lower()
            if "float32" in dtype_str:
                return "f32"
            elif "float64" in dtype_str:
                return "f64"
            elif "int32" in dtype_str:
                return "i32"
            elif "int64" in dtype_str:
                return "i64"
            else:
                return dtype_str

        def format_shape_and_dtype(array: Array) -> str:
            """Format shape and dtype in JAX style."""
            dtype_str = format_dtype(array.dtype)
            if array.shape:
                shape_str = ",".join(map(str, array.shape))
                return f"{purple}{dtype_str}[{shape_str}]{reset}"
            else:
                return f"{purple}{dtype_str}[]{reset}"

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


def reset_traced_flags(arrays: list[Array]) -> None:
    """Reset autodiff traced flags for a list of arrays and their dependencies."""
    visited: set[Array] = set()

    def reset_recursive(node: Array) -> None:
        if node in visited:
            return
        visited.add(node)
        node.traced = False
        for arg in node.args:
            reset_recursive(arg)

    for array in arrays:
        reset_recursive(array)


def reset_all_traced_flags_in_graph(trace: Trace) -> None:
    """Reset traced flags for all nodes in a computation trace."""
    # Get all traced nodes and reset their flags
    all_nodes = trace.get_traced_nodes()
    for node in all_nodes:
        node.traced = False


def pullback(
    inputs: list[Array],
    outputs: list[Array],
    cotangents: list[Array],
    # trace: Trace | None = None,
) -> list[Array]:
    """Compute vector-Jacobian product (reverse-mode autodiff)."""
    if len(cotangents) != len(outputs):
        raise ValueError(
            f"Cotangents length {len(cotangents)} != outputs length {len(outputs)}"
        )

    # Use provided trace or compute new one (for backward compatibility)
    # if trace is None:
    trace = Trace(inputs, outputs)
    traced_nodes = trace.get_traced_nodes()

    # Step 1: Initialize cotangents for output nodes
    for output, cotangent in zip(outputs, cotangents, strict=False):
        output.cotangent = cotangent

    try:
        # Step 2: Traverse nodes in reverse topological order
        for node in reversed(traced_nodes):
            # Skip nodes without cotangents (shouldn't happen in well-formed graphs)
            if node.cotangent is None:
                continue

            # Skip input nodes (they don't have VJP rules to apply)
            if not node.args or node.vjp_rule is None:
                continue

            # Step 3: Apply VJP rule to get cotangents for arguments
            try:
                # print("Applying VJP rule for node:", node.name)
                arg_cotangents = node.vjp_rule(node.args, node.cotangent, node)

                # Step 4: Accumulate cotangents for each argument
                for arg, arg_cotangent in zip(node.args, arg_cotangents, strict=False):
                    if arg.cotangent is not None:
                        # Accumulate: add new cotangent to existing one
                        from ..ops.binary import add

                        arg.cotangent = add(arg.cotangent, arg_cotangent)
                    else:
                        # First cotangent for this argument
                        arg.cotangent = arg_cotangent

                # Step 5: Clean up this node's gradient immediately after processing
                # (unless it's an input node - we need those gradients at the end)
                if node not in inputs:
                    node.cotangent = None

            except Exception as e:
                raise RuntimeError(
                    f"VJP rule failed for operation '{node.name}': {e}"
                ) from e

        # Step 5: Collect gradients for input nodes
        input_gradients = []
        for inp in inputs:
            if inp.cotangent is not None:
                input_gradients.append(inp.cotangent)
            else:
                # Input has no gradient (not used in computation)
                from ..ops.creation import zeros

                input_gradients.append(zeros(inp.shape, dtype=inp.dtype))

        return input_gradients

    finally:
        # Step 6: Cleanup - only need to clean input gradients now
        # (intermediate gradients were cleaned during processing)
        for inp in inputs:
            inp.cotangent = None


def pushfwd(
    inputs: list[Array],
    outputs: list[Array],
    tangents: list[Array],
    trace: Trace | None = None,
) -> list[Array]:
    """Compute Jacobian-vector product (forward-mode autodiff)."""
    if len(tangents) != len(inputs):
        raise ValueError(
            f"Tangents length {len(tangents)} != inputs length {len(inputs)}"
        )

    # Use provided trace or compute new one (for backward compatibility)
    if trace is None:
        trace = Trace(inputs, outputs)
    traced_nodes = trace.get_traced_nodes()

    # Step 1: Initialize tangents for input nodes
    for input_node, tangent in zip(inputs, tangents, strict=False):
        input_node.tangent = tangent

    try:
        # Step 2: Traverse nodes in forward topological order
        for node in traced_nodes:
            # Skip nodes that are inputs (they already have tangents)
            if node in inputs:
                continue

            # Skip nodes without arguments (shouldn't happen in well-formed graphs)
            if not node.args or node.jvp_rule is None:
                continue

            # Step 3: Collect tangents from arguments
            arg_tangents = []
            for arg in node.args:
                if arg.tangent is not None:
                    arg_tangents.append(arg.tangent)
                else:
                    # If an argument doesn't have a tangent, use zeros
                    from ..ops.creation import zeros

                    arg_tangents.append(zeros(arg.shape, dtype=arg.dtype))

            # Step 4: Apply JVP rule to get tangent for this node
            try:
                node.tangent = node.jvp_rule(node.args, arg_tangents, node)
            except Exception as e:
                raise RuntimeError(
                    f"JVP rule failed for operation '{node.name}': {e}"
                ) from e

        # Step 5: Collect tangents for output nodes
        output_tangents = []
        for out in outputs:
            if out.tangent is not None:
                output_tangents.append(out.tangent)
            else:
                # Output has no tangent (shouldn't happen in well-formed graphs)
                from ..ops.creation import zeros

                output_tangents.append(zeros(out.shape, dtype=out.dtype))

        return output_tangents

    finally:
        # Step 6: Cleanup tangents for all nodes
        # (inputs, outputs, and any intermediate nodes)
        for node in traced_nodes:
            node.tangent = None

        # Only reset traced flags if we're not in a higher-order derivative computation
        # Check if any input still needs to be traced (indicating higher-order derivatives)
        any_input_needs_tracing = any(inp.traced for inp in inputs)
        if not any_input_needs_tracing:
            # reset all traced flags and tensor_value references
            reset_all_traced_flags_in_graph(trace)


def xpr(
    fn: Callable[[list[Array]], list[Array]],
    args: list[Array],
) -> str:
    """
    Get a JAX-like string representation of the function's computation graph.

    Args:
        fn: Function to trace
        args: List of input Arrays to the function
    Returns:
        str: JAX-like string representation of the computation graph
    """
    try:
        for arg in args:
            # Mark all input arrays as traced so computation graph gets captured
            arg.traced = True

        # Use the trace_function class method for proper tracing
        trace = Trace.trace_function(fn, args)

        # Return the string representation of the trace
        return str(trace)
    finally:
        # Clean up: reset traced flags after tracing
        reset_traced_flags(args)


def vjp(
    func: Callable[[list[Array]], list[Array]], inputs: list[Array]
) -> tuple[list[Array], Callable[[list[Array]], list[Array]]]:
    # Mark inputs for tracing
    for inp in inputs:
        inp.traced = True

    # Compute outputs once and create trace
    outputs = func(inputs)
    if not isinstance(outputs, list):
        outputs = [outputs]

    def vjp_fn(cotangents: list[Array]) -> list[Array]:
        if len(cotangents) != len(outputs):
            raise ValueError(
                f"Cotangents length {len(cotangents)} != outputs length {len(outputs)}"
            )
        return pullback(inputs, outputs, cotangents)

    return outputs, vjp_fn


def jvp(
    func: Callable[[list[Array]], list[Array]],
    inputs: list[Array],
    tangents: list[Array],
) -> tuple[list[Array], list[Array]]:
    """
    Compute Jacobian-vector product (forward-mode autodiff).

    Args:
        func: Function to differentiate
        inputs: Input arrays to the function
        tangents: Tangent vectors (directional derivatives)

    Returns:
        tuple: (outputs, output_tangents) where output_tangents are the JVP results
    """
    if len(tangents) != len(inputs):
        raise ValueError(
            f"Tangents length {len(tangents)} != inputs length {len(inputs)}"
        )

    # Mark inputs for tracing
    for inp in inputs:
        inp.traced = True

    # Compute outputs and their tangents using pushfwd
    outputs = func(inputs)
    if not isinstance(outputs, list):
        outputs = [outputs]

    # Use pushfwd to compute the output tangents
    output_tangents = pushfwd(inputs, outputs, tangents)

    return outputs, output_tangents


def vmap(
    func: Callable[[list[Array]], list[Array]],
    in_axes: list[int] | None = None,
    out_axes: list[int] | None = None,
) -> Callable[[list[Array]], list[Array]]:
    """
    Vectorize a function over its inputs.
    Args:
        func: Function to vectorize
        in_axes: Input axes to vectorize over (default: all inputs)
        out_axes: Output axes to vectorize over (default: all outputs)

    Returns:
        Callable: Vectorized function that can handle batched inputs
    """
    if in_axes is None:
        in_axes = [0] * len(func.__code__.co_varnames)

    if out_axes is None:
        out_axes = [0] * len(func.__code__.co_varnames)

    def vectorized_func(inputs: list[Array]) -> list[Array]:
        # call incr_batch_dim_ctr on all inputs, also for thos which have in axis set  to None, we usnqueeze first.
        batched_inputs = []
        for i, inp in enumerate(inputs):
            if in_axes[i] is None:
                from ..ops.view import unsqueeze

                batched_inputs.append(unsqueeze(inp, [0]))
            else:
                from ..ops.unary import incr_batch_dim_ctr

                batched_inputs.append(incr_batch_dim_ctr(inp))

        # Call the original function with batched inputs
        outputs = func(batched_inputs)

        # cleanup inputs, i.e. call decr_batch_dim_ctr on all inputs and also on teh outptus
        for i, inp in enumerate(batched_inputs):
            if in_axes[i] is None:
                from ..ops.view import squeeze

                batched_inputs[i] = squeeze(inp, [0])
            else:
                from ..ops.unary import decr_batch_dim_ctr

                batched_inputs[i] = decr_batch_dim_ctr(inp)

        # cleanup outputs, i.e. call decr_batch_dim_ctr on all outputs
        for i, out in enumerate(outputs):
            if out_axes[i] is None:
                from ..ops.view import squeeze

                outputs[i] = squeeze(out, [0])
            else:
                from ..ops.unary import decr_batch_dim_ctr

                outputs[i] = decr_batch_dim_ctr(out)

        # Ensure outputs are batched according to out_axes
        if not isinstance(outputs, list):
            outputs = [outputs]
        return [outputs[i] for i in out_axes]

    return vectorized_func
