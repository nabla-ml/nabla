"""Core graph execution and model compilation."""

from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Set, Tuple
from collections import deque

from max.engine import InferenceSession, Model
from max.driver import Device
from max.graph import DeviceRef, Graph, TensorType, Value

from .array import Array
from .execution_context import global_execution_context


class GraphTracer:
    """Handles computation graph tracing and cache key generation."""

    @staticmethod
    def compute_node_hash(node: Array) -> int:
        """Compute a deterministic hash for a computation node."""
        components = [
            str(node.shape),
            str(node.dtype),
            node.name or "unnamed",
        ]
        node_str = "-".join(components)
        return hash(node_str)

    @staticmethod
    def get_trace(nodes: Sequence[Array]) -> Tuple[List[Array], List[Array], int]:
        """
        Perform iterative DFS to get computation trace and cache key.

        Returns:
            inputs: List of leaf nodes (have impl)
            trace: Topological ordering of all nodes
            cache_key: Hash key for caching compiled models
        """
        trace: List[Array] = []
        inputs: List[Array] = []
        visited: Set[Array] = set()

        for start_node in nodes:
            if start_node in visited:
                continue

            stack: List[Array] = [start_node]

            while stack:
                node = stack[-1]  # Peek at top

                if node in visited:
                    stack.pop()
                    continue

                # If leaf node (has implementation)
                if node.impl is not None:
                    inputs.append(node)
                    trace.append(node)
                    visited.add(node)
                    stack.pop()
                    continue

                # Check if all children are visited
                all_children_visited = all(arg in visited for arg in node.args)

                if not all_children_visited:
                    # Add unvisited children to stack
                    for arg in node.args:
                        if arg not in visited:
                            stack.append(arg)
                else:
                    # All children visited, process this node
                    visited.add(node)
                    trace.append(node)
                    stack.pop()

        # Compute cache key from trace
        cache_key = GraphTracer._compute_cache_key(trace)
        return inputs, trace, cache_key

    @staticmethod
    def _compute_cache_key(trace: List[Array]) -> int:
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
        inputs: List[Array], trace: List[Array], outputs: List[Array]
    ) -> Model:
        """Create a MAX model from the computation graph."""
        # Build input types
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
            # Use custom kernels if available
            custom_op_package_path = Path(__file__).parent.parent / "mojo_kernels"

            with Graph(
                "nabla_graph",
                input_types=input_types,
                custom_extensions=(
                    [custom_op_package_path] if custom_op_package_path.exists() else []
                ),
            ) as graph:
                # Map inputs to graph symbols
                input_symbols = graph.inputs
                for i, input_node in enumerate(inputs):
                    input_node.tensor_value = input_symbols[i]

                # Process trace to build computation graph
                for node in trace:
                    if node.tensor_value is not None:
                        continue

                    # Get argument symbols
                    arg_symbols = []
                    for arg in node.get_arguments():
                        if arg.tensor_value is None:
                            raise ValueError(
                                f"Missing tensor value for argument of {node.name}"
                            )
                        arg_symbols.append(arg.tensor_value)

                    # Execute operation
                    if node.maxpr is None:
                        raise ValueError(f"Node {node.name} has no maxpr function")

                    node.maxpr(arg_symbols, node)

                    # Validate output
                    ModelFactory._validate_node_output(node)

                # Set graph outputs
                output_symbols = []
                for output in outputs:
                    if output.tensor_value is None:
                        raise ValueError(f"Output {output.name} has no tensor value")
                    output_symbols.append(output.tensor_value)

                graph.output(*output_symbols)

            # Create inference session and load model
            session = InferenceSession(devices=devices)
            return session.load(graph)

        except Exception as e:
            raise ValueError(f"Failed to build computation graph: {e}")

    @staticmethod
    def _validate_node_output(node: Array) -> None:
        """Validate that node output matches expected shape and dtype."""
        if node.tensor_value is None:
            raise ValueError(f"Node {node.name} has no tensor value after execution")

        # Convert tensor shape to tuple for comparison
        tensor_shape = tuple(int(dim) for dim in node.tensor_value.shape)

        if node.shape != tensor_shape:
            raise ValueError(
                f"Shape mismatch for node {node.name}: "
                f"expected {node.shape}, got {tensor_shape}. "
                # f"Op params: {node.op_params}"
            )

        if node.dtype != node.tensor_value.dtype:
            raise ValueError(
                f"Dtype mismatch for node {node.name}: "
                f"expected {node.dtype}, got {node.tensor_value.dtype}. "
                # f"Op params: {node.op_params}"
            )


def realize_(outputs: List[Array]) -> None:
    """
    Realize (compute) the given output Arrays.

    This is the main entry point for executing computation graphs.
    Uses compilation caching for performance.
    """
    if not outputs:
        return

    # Validate inputs
    for output in outputs:
        if not isinstance(output, Array):
            raise TypeError(f"All outputs must be Array instances, got {type(output)}")

    # Filter outputs that need computation
    output_list = [output for output in outputs if output.impl is None]

    if not output_list:
        return  # Nothing to compute

    # Get computation trace
    inputs, trace, cache_key = GraphTracer.get_trace(output_list)

    # Create model using cached compilation
    def create_model() -> Model:
        return ModelFactory.create_model(inputs, trace, output_list)

    model = global_execution_context.get_or_create(cache_key, create_model)

    # Execute the model
    try:
        tensor_inputs = [input_node.impl for input_node in inputs]
        if any(tensor is None for tensor in tensor_inputs):
            raise ValueError("Some inputs have no implementation")

        model_outputs = model.execute(*tensor_inputs)

        # Update outputs with results
        for i, output in enumerate(output_list):
            output.impl = model_outputs[i]
            output._numpy_cache = None  # Invalidate NumPy cache

    except Exception as e:
        raise ValueError(f"Error executing computation: {e}")


# Legacy function aliases for backward compatibility
def get_trace(nodes: Sequence[Array]) -> Tuple[List[Array], List[Array], int]:
    """Legacy alias for GraphTracer.get_trace."""
    return GraphTracer.get_trace(nodes)


def compute_node_hash(node: Array) -> int:
    """Legacy alias for GraphTracer.compute_node_hash."""
    return GraphTracer.compute_node_hash(node)
