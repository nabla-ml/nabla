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
from pathlib import Path

from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType

from .array import Array
from .execution_context import global_execution_context


class GraphTracer:
    """Handles computation graph tracing and cache key generation."""

    @staticmethod
    def compute_node_hash(node: Array) -> int:
        """Compute a deterministic hash for a computation node."""
        components = [node.name or "input", str(node.shape), str(node.dtype)]
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
        inputs: list[Array], trace: list[Array], outputs: list[Array]
    ) -> Model:
        """Create a MAX model from the computation graph."""
        input_types = []
        devices = []

        for input_node in inputs:
            input_types.append(
                TensorType(
                    dtype=input_node.dtype,
                    shape=input_node.shape,
                    device=DeviceRef.from_device(input_node.device),
                )
            )
            if input_node.device not in devices:
                devices.append(input_node.device)

        try:
            custom_op_package_path = Path(__file__).parent.parent / "kernels"

            with Graph(
                "nabla_graph",
                input_types=input_types,
                custom_extensions=(
                    [custom_op_package_path] if custom_op_package_path.exists() else []
                ),
            ) as graph:
                input_symbols = graph.inputs
                for i, input_node in enumerate(inputs):
                    input_node.tensor_value = input_symbols[i]

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
            tensor_shape = tuple(int(dim) for dim in node.tensor_value.shape)
            tensor_dtype = node.tensor_value.dtype

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


def realize_(outputs: list[Array]) -> None:
    """
    Realize (compute) the given output Arrays.

    This is the main entry point for executing computation graphs.
    Uses compilation caching for performance.
    """
    if not outputs:
        return

    for output in outputs:
        if not isinstance(output, Array):
            raise TypeError(f"All outputs must be Array instances, got {type(output)}")

    output_list = [output for output in outputs if output.impl is None]

    if not output_list:
        return

    inputs, trace, cache_key = GraphTracer.get_trace(output_list)

    def create_model() -> Model:
        return ModelFactory.create_model(inputs, trace, output_list)

    model = global_execution_context.get_or_create(cache_key, create_model)

    try:
        tensor_inputs = [input_node.impl for input_node in inputs]
        if any(tensor is None for tensor in tensor_inputs):
            raise ValueError("Some inputs have no implementation")

        model_outputs = model.execute(*tensor_inputs)

        for i, output in enumerate(output_list):
            output.impl = model_outputs[i]
            output._numpy_cache = None

    except Exception as e:
        raise ValueError(f"Error executing computation: {e}") from e


# Legacy function aliases for backward compatibility
def get_trace(nodes: Sequence[Array]) -> tuple[list[Array], list[Array], int]:
    return GraphTracer.get_trace(nodes)


def compute_node_hash(node: Array) -> int:
    return GraphTracer.compute_node_hash(node)
