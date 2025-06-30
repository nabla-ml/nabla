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

"""Core graph execution and model compilation."""

from __future__ import annotations

from collections.abc import Sequence

from max.engine.api import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType

from .array import Array
from .execution_context import global_execution_context


class GraphTracer:
    """Handles computation graph tracing and cache key generation."""

    @staticmethod
    def compute_node_hash(node: Array) -> int:
        """Compute a deterministic hash for a computation node."""
        components = [
            str(node.dtype),
            str(node.batch_dims + node.shape),
            "-",
            node.name or "input",
        ]
        node_str = "-".join(components)
        return hash(node_str)

    @staticmethod
    def get_trace(nodes: Sequence[Array]) -> tuple[list[Array], list[Array], int]:
        """
        Perform iterative DFS to get computation trace and cache key.

        Returns:
            inputs: List of leaf nodes (have impl)
            trace: Topological ordering of all nodes
            cache_key: Hash key for caching compiled models
        """
        trace: list[Array] = []
        inputs: list[Array] = []
        visited: set[Array] = set()

        for start_node in nodes:
            if start_node in visited:
                continue

            stack: list[Array] = [start_node]

            while stack:
                node = stack[-1]

                if node in visited:
                    stack.pop()
                    continue

                if node.impl is not None:
                    inputs.append(node)
                    trace.append(node)
                    visited.add(node)
                    stack.pop()
                    continue

                all_children_visited = all(arg in visited for arg in node.args)

                if not all_children_visited:
                    for arg in node.args:
                        if arg not in visited:
                            stack.append(arg)
                else:
                    visited.add(node)
                    trace.append(node)
                    stack.pop()

        cache_key = GraphTracer._compute_cache_key(trace)
        return inputs, trace, cache_key

    @staticmethod
    def _compute_cache_key(trace: list[Array]) -> int:
        """Compute a cache key from the computation trace."""
        key: int = 0
        for node in trace:
            node_hash = GraphTracer.compute_node_hash(node)
            key = key ^ (node_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
        return key % 1000000000


class ModelFactory:
    """Factory for creating MAX models from computation graphs."""

    @staticmethod
    def create_model(
        inputs: list[Array],
        trace: list[Array],
        outputs: list[Array],
        dynamic_inputs: list[Array] | None = None,
        show_graph: bool = False,
    ) -> Model:
        """Create a MAX model from the computation graph."""

        # Timing: Input type preparation
        input_types = []
        devices = []

        for input_node in inputs:
            if dynamic_inputs is not None and input_node not in dynamic_inputs:
                # If the node can be treated as a constant, skip it and add it as a constant value later in the graph
                continue

            input_types.append(
                TensorType(
                    dtype=input_node.dtype,
                    shape=input_node.batch_dims + input_node.shape,
                    device=DeviceRef.from_device(input_node.device),
                )
            )
            if input_node.device not in devices:
                devices.append(input_node.device)

        custom_ops_paths = []
        for node in trace:
            if node.kernel_impl_path and node.kernel_impl_path.exists():
                custom_ops_paths.append(node.kernel_impl_path)

        try:
            with Graph(
                "nabla_graph",
                input_types=input_types,
                custom_extensions=custom_ops_paths,
            ) as graph:
                input_symbols = graph.inputs
                j = 0
                for i, input_node in enumerate(inputs):
                    if (
                        dynamic_inputs is not None
                        and input_node not in dynamic_inputs
                        and input_node.impl is not None
                    ):
                        from max.graph.ops import constant

                        input_node.tensor_value = constant(
                            input_node.to_numpy(),
                            input_node.dtype,
                            DeviceRef.from_device(input_node.device),
                        )  # add tensor_value as constant weight to the graph
                        j += 1
                    else:
                        input_node.tensor_value = input_symbols[i - j]

                for node in trace:
                    node_name = node.name or "const"

                    if node.tensor_value is not None:
                        continue

                    arg_symbols = []
                    for j, arg in enumerate(node.get_arguments()):
                        if arg.tensor_value is None:
                            raise ValueError(
                                f"Missing tensor value for argument {j} of {node_name}"
                            )
                        arg_symbols.append(arg.tensor_value)

                    if node.maxpr is None:
                        raise ValueError(f"Node {node_name} has no maxpr function")

                    try:
                        node.maxpr(arg_symbols, node)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing node {node_name}: {type(e).__name__}: {e}"
                        ) from e

                    try:
                        ModelFactory._validate_node_output(node)
                    except ValueError as ve:
                        raise ValueError(
                            f"Validation failed for node {node_name}: {ve}"
                        ) from ve

                output_symbols = []
                for output in outputs:
                    if output.tensor_value is None:
                        raise ValueError(f"Output {output.name} has no tensor value")
                    output_symbols.append(output.tensor_value)

                try:
                    graph.output(*output_symbols)
                except Exception as e:
                    raise ValueError(f"Failed to set graph output: {e}") from e

            if show_graph:
                print(graph)

            session = InferenceSession(devices=devices)

            for node in trace:
                node.tensor_value = None

            try:
                model = session.load(graph)

                return model
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}") from e

        except Exception as e:
            import traceback

            traceback.print_exc()

            raise ValueError(
                f"Failed to build computation graph: {type(e).__name__}: {e}"
            ) from e

    @staticmethod
    def _validate_node_output(node: Array) -> None:
        """Validate that node output matches expected shape and dtype."""
        if node.tensor_value is None:
            raise ValueError(f"Node {node.name} has no tensor value after execution")

        try:
            # Handle different tensor value types with proper type checking
            tensor_value = node.tensor_value
            if hasattr(tensor_value, "shape") and hasattr(tensor_value, "dtype"):
                tensor_shape = tuple(int(dim) for dim in tensor_value.shape)  # type: ignore
                tensor_dtype = tensor_value.dtype  # type: ignore
            else:
                # For Value types that don't have shape/dtype, skip validation
                return

            if node.batch_dims + node.shape != tensor_shape:
                raise ValueError(
                    f"Shape mismatch for node {node.name}: "
                    f"expected {node.batch_dims + node.shape}, got {tensor_shape}"
                )

            if node.dtype != tensor_dtype:
                raise ValueError(
                    f"Dtype mismatch for node {node.name}: "
                    f"expected {node.dtype}, got {tensor_dtype}"
                )

        except Exception as e:
            raise ValueError(f"Validation error for node {node.name}: {e}") from e


def realize_(
    outputs: list[Array],
    dynamic_inputs: list[Array] | None = None,
    show_graph: bool = False,
) -> Model | None | tuple[Model, list[Array]]:
    """
    Realize (compute) the given output Arrays.

    This is the main entry point for executing computation graphs.
    Uses compilation caching for performance.

    Args:
        outputs: List of Arrays to realize
        dynamic_inputs: Optional list of dynamic inputs for model compilation
        return_trace_inputs: If True, return tuple of (model, trace_inputs) instead of just model

    Returns:
        If return_trace_inputs is False: Model or None
        If return_trace_inputs is True: tuple of (Model, trace_inputs) or None
    """
    if not outputs:
        return

    for output in outputs:
        if not isinstance(output, Array):
            raise TypeError(f"All outputs must be Array instances, got {type(output)}")

    # For JIT compilation with mixed realized/unrealized outputs, we need all outputs
    # to be part of the compiled model. Check if we have mixed states:
    realized_outputs = [output for output in outputs if output.impl is not None]
    unrealized_outputs = [output for output in outputs if output.impl is None]

    if realized_outputs and unrealized_outputs:
        # Mixed case - this typically happens with JIT(VMAP(...)) where some outputs
        # are already realized during VMAP processing. We need to unrealize them
        # so they can all be compiled together.
        for output in realized_outputs:
            output._impl = None  # Force unrealization
        output_list = outputs  # Process all outputs
    else:
        # Normal case - only process unrealized outputs
        output_list = unrealized_outputs

    if not output_list:
        return

    inputs, trace, cache_key = GraphTracer.get_trace(output_list)

    def create_model() -> Model:
        return ModelFactory.create_model(
            inputs, trace, output_list, dynamic_inputs, show_graph
        )

    model = global_execution_context.get_or_create(cache_key, create_model)

    if dynamic_inputs is not None:
        return model, inputs

    try:
        tensor_inputs = [input_node.impl for input_node in inputs]
        # Filter out None values and ensure we have valid tensors
        valid_tensors = [tensor for tensor in tensor_inputs if tensor is not None]
        if len(valid_tensors) != len(tensor_inputs):
            raise ValueError("Some inputs have no implementation")

        model_outputs = model.execute(*valid_tensors)

        for i, output in enumerate(output_list):
            output.impl_(model_outputs[i])  # type: ignore

    except Exception as e:
        raise ValueError(f"Error executing computation: {e}") from e


# Legacy function aliases for backward compatibility
def get_trace(nodes: Sequence[Array]) -> tuple[list[Array], list[Array], int]:
    return GraphTracer.get_trace(nodes)


def compute_node_hash(node: Array) -> int:
    return GraphTracer.compute_node_hash(node)
