# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
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

    def __init__(self, inputs: list[Array], outputs: list[Array]) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.trace: list[Array] = []
        self._computed = False

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

        def format_dtype(dtype) -> str:
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

        def format_shape_and_dtype(array) -> str:
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
                        # This shouldn't happen in a well-formed trace
                        arg_vars.append("?")

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
    """Reset traced flags for a list of arrays and their dependencies."""
    visited = set()

    def reset_recursive(node: Array) -> None:
        if node in visited:
            return
        visited.add(node)
        node.traced = False
        for arg in node.args:
            reset_recursive(arg)

    for array in arrays:
        reset_recursive(array)


def pullback(
    inputs: list[Array],
    outputs: list[Array],
    cotangents: list[Array],
    trace: Trace | None = None,
) -> list[Array]:
    """Compute vector-Jacobian product (reverse-mode autodiff)."""
    if len(cotangents) != len(outputs):
        raise ValueError(
            f"Cotangents length {len(cotangents)} != outputs length {len(outputs)}"
        )

    # Use provided trace or compute new one (for backward compatibility)
    if trace is None:
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


class Transformation:
    """
    A composable transformation that can wrap functions or other transformations.

    This enables JAX-style nested transformations like jvp(vjp(jvp(...))).
    Each transformation implements a 3-phase execution model:
    - pre_call: Transform inputs before calling inner transformation
    - in_call: How to call the nested transformation/function
    - post_call: Transform outputs after calling inner transformation
    """

    def __init__(self, callable_or_transform):
        """
        Initialize a transformation.

        Args:
            callable_or_transform: Either a callable function or another Transformation
        """
        self.inner = callable_or_transform
        self.trace = None  # Store trace for transformations that need it

    def pre_call(self, inputs: list[Array]) -> list[Array]:
        """
        Transform inputs before calling inner transformation.

        Base implementation: pass-through (no transformation).
        Override in subclasses for specific behavior.
        """
        return inputs

    def in_call(self, inputs: list[Array]) -> Any:
        """
        Define how to call the inner transformation/function.

        Base implementation: direct call.
        Override in subclasses for specific calling patterns.

        Returns:
            Any: Could be list[Array] or other types depending on transformation
        """
        return self.inner(inputs)

    def post_call(self, outputs: Any) -> Any:
        """
        Transform outputs after calling inner transformation.

        Base implementation: pass-through (no transformation).
        Override in subclasses for specific behavior.

        Args:
            outputs: Outputs from the inner call

        Returns:
            Any: Transformed outputs
        """
        return outputs

    def __call__(self, inputs: list[Array]) -> Any:
        """
        Execute the 3-phase transformation pipeline.

        Flow: pre_call -> in_call -> post_call
        """

        # Phase 1: Transform inputs for this transformation layer
        transformed_inputs = self.pre_call(inputs)

        # Phase 2: Call inner transformation/function
        inner_outputs = self.in_call(transformed_inputs)

        # Phase 3: Transform outputs for this transformation layer
        final_outputs = self.post_call(inner_outputs)

        return final_outputs


class VJPTransform(Transformation):
    """Transformation for Vector-Jacobian Product (reverse-mode autodiff)."""

    def __init__(self, fn: Callable):
        """
        Initialize VJP transformation.

        Args:
            fn: Function to transform
        """
        super().__init__(fn)

    def pre_call(self, inputs: list[Array]) -> list[Array]:
        """Mark inputs for tracing before calling the function."""
        # Mark all input arrays as traced
        for inp in inputs:
            inp.traced = True
        return inputs

    def in_call(self, inputs: list[Array]) -> list[Array]:
        """Call the function and capture the trace."""
        # Execute the function to get outputs
        outputs = self.inner(inputs)
        assert isinstance(outputs, list), "Function must return list of Arrays"

        # Create and store the trace
        self.trace = Trace(inputs, outputs)

        return outputs

    def post_call(self, outputs: list[Array]) -> tuple[list[Array], Callable[..., Any]]:
        """Transform outputs to return (primals, vjp_fn)."""

        if self.trace is None:
            raise RuntimeError("VJP requires a traced computation")

        def vjp_fn(cotangents: list[Array]) -> list[Array]:
            """Compute VJP given cotangents."""
            return pullback(
                self.trace.inputs, self.trace.outputs, cotangents, self.trace
            )

        return outputs, vjp_fn


class JVPTransform(Transformation):
    """Transformation for Jacobian-Vector Product (forward-mode autodiff)."""

    def __init__(self, fn: Callable, tangents: list[Array]):
        """
        Initialize JVP transformation.

        Args:
            fn: Function to transform
            tangents: Tangent vectors for inputs
        """
        super().__init__(fn)
        self.tangents = tangents

    def pre_call(self, inputs: list[Array]) -> list[Array]:
        """Mark inputs for tracing before calling the function."""
        # Mark all input arrays as traced
        for inp in inputs:
            inp.traced = True
        return inputs

    def in_call(self, inputs: list[Array]) -> list[Array]:
        """Call the function and capture the trace."""
        # Execute the function to get outputs
        outputs = self.inner(inputs)
        assert isinstance(outputs, list), "Function must return list of Arrays"

        # Create and store the trace
        self.trace = Trace(inputs, outputs)

        return outputs

    def post_call(self, outputs: list[Array]) -> tuple[list[Array], list[Array]]:
        """Transform outputs to return (primals, tangents)."""

        if self.trace is None:
            raise RuntimeError("JVP requires a traced computation")

        # Compute tangents using pushfwd
        output_tangents = pushfwd(
            self.trace.inputs, self.trace.outputs, self.tangents, self.trace
        )

        return outputs, output_tangents


# High-level JAX-style API functions
def vjp(func: Callable, inputs: list[Array]) -> tuple[list[Array], Callable[..., Any]]:
    """
    Compute vector-Jacobian product (reverse-mode autodiff) for a function.

    Args:
        func: Function to trace and compute VJP for
        inputs: List of input Arrays to the function

    Returns:
        Tuple of (outputs, vjp_fn) where:
        - outputs: List of output Arrays from the function
        - vjp_fn: Callable that computes VJP given cotangents
    """
    # Use the VJPTransform to handle the computation
    vjp_transform = VJPTransform(func)
    result = vjp_transform(inputs)
    assert isinstance(result, tuple) and len(result) == 2
    return result


def jvp(
    func: Callable, inputs: list[Array], tangents: list[Array]
) -> tuple[list[Array], list[Array]]:
    """
    Compute Jacobian-vector product (forward-mode autodiff) for a function.

    Args:
        func: Function to trace and compute JVP for
        inputs: List of input Arrays to the function
        tangents: List of tangent Arrays (same length as inputs)

    Returns:
        Tuple of (outputs, tangents) where:
        - outputs: List of output Arrays from the function
        - tangents: List of output tangent Arrays
    """
    # Use the JVPTransform to handle the computation
    jvp_transform = JVPTransform(func, tangents)
    result = jvp_transform(inputs)
    assert isinstance(result, tuple) and len(result) == 2
    return result
