from __future__ import annotations
import numpy as np
import time
from typing import (
    List,
    Final,
    ClassVar,
    Union,
    Tuple,
    Type,
    Set,
    Callable,
    Optional,
    Sequence,
    Dict,
    Protocol,
    TypedDict,
    cast,
    Any,
)
from collections import deque

from max.engine import InferenceSession, Model
from max.driver import Tensor, CPU, Accelerator, Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops, Value

# Define type aliases for better readability
MaxprCallable = Callable[[List[Value]], Value]
VJPRule = Callable[[List[Value], List[Value]], Value]
JVPRule = Callable[[List[Value], List[Value]], Value]
Shape = Tuple[int, ...]

# Execution mode flag
EAGERMODE: bool = False

# Global model cache with proper typing
global_execution_context: Dict[int, Model] = {}


def get_broadcasted_shape(
    shape1: Shape,
    shape2: Shape,
    ignore_axes: List[int] = [],
    replace_ignored_dims: List[int] = [],
) -> Shape:
    if len(replace_ignored_dims) != len(ignore_axes):
        raise ValueError(
            "replace_ignored_dims must have the same length as ignore_axes"
        )

    s1_len = len(shape1)
    s2_len = len(shape2)
    max_rank = max(s1_len, s2_len)

    # Initialize result shape. We'll fill it. Using 1s is a common default for broadcasting.
    res_shape_list = [1] * max_rank

    # Normalize ignore_axes to positive indices and store their replacement values.
    # These normalized indices refer to positions in the `max_rank` shape.
    normalized_ignored_map = {}  # Stores {normalized_idx: replacement_dim}

    for i in range(len(ignore_axes)):
        axis_spec = ignore_axes[i]
        replacement_dim = replace_ignored_dims[i]

        # Validate and normalize the axis_spec relative to max_rank
        if not (-max_rank <= axis_spec < max_rank):
            raise ValueError(
                f"ignore_axis {axis_spec} is out of bounds for max_rank {max_rank}"
            )

        normalized_idx = axis_spec if axis_spec >= 0 else max_rank + axis_spec

        # If multiple ignore_axes entries map to the same normalized_idx (e.g. 0 and -max_rank),
        # the last one in the list will win. This is typical Python dict behavior.
        normalized_ignored_map[normalized_idx] = replacement_dim
        res_shape_list[normalized_idx] = replacement_dim

    # Pad original shapes with leading 1s to align them to max_rank for broadcasting logic
    padded_shape1_list = [1] * (max_rank - s1_len) + list(shape1)
    padded_shape2_list = [1] * (max_rank - s2_len) + list(shape2)

    # Perform broadcasting for non-ignored axes
    # Iterate from the leftmost dimension of the padded shapes
    for i in range(max_rank):
        if i in normalized_ignored_map:
            # This dimension's value in res_shape_list is already set by replace_ignored_dims
            continue

        d1 = padded_shape1_list[i]
        d2 = padded_shape2_list[i]

        if d1 == d2:
            res_shape_list[i] = d1
        elif d1 == 1:
            res_shape_list[i] = d2
        elif d2 == 1:
            res_shape_list[i] = d1
        else:
            # Dimensions are different and neither is 1, broadcasting error.
            raise ValueError(
                f"Shapes {shape1} and {shape2} cannot be broadcast at dimension index {i} "
                f"(0-indexed from left of max_rank {max_rank} shape). "
                f"Padded values at this index are {d1} (from shape1) and {d2} (from shape2)."
            )

    return tuple(res_shape_list)


class Array:
    name: str
    data: Optional[Tensor]
    args: List[Array]
    visited: bool
    shape: Shape
    dtype: DType
    device: Device
    tensor_value: Optional[Value]
    maxpr: Optional[MaxprCallable]
    vjp_rule: Optional[VJPRule]
    jvp_rule: Optional[JVPRule]
    _numpy_cache: Optional[np.ndarray]

    def __init__(
        self,
        shape: Shape,
        dtype: DType = DType.float32,
        device: Device = CPU(),
        materialize: bool = False,
        name: str = "",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.name = name
        self.args = []
        self.visited = False
        self.tensor_value = None
        self.maxpr = None
        self.vjp_rule = None
        self.jvp_rule = None
        self._numpy_cache = None

        if materialize:
            self.data = Tensor(dtype, shape, device=device)
        else:
            self.data = None

    @classmethod
    def from_data(cls, data: Tensor, name: str = "") -> Array:
        if not isinstance(data, Tensor):
            raise TypeError(f"Data must be a MAX Tensor, got {type(data)}")
        if not data.shape:
            raise ValueError("Cannot create Array from empty shape Tensor")

        instance = cls(
            shape=data.shape, dtype=data.dtype, device=data.device, materialize=True
        )
        instance.data = data
        instance.name = name
        return instance

    @classmethod
    def create_buffer(cls, shape: Shape, dtype: DType, device: Device) -> Array:
        return cls(shape=shape, dtype=dtype, device=device, materialize=True)

    def copy_from(self, other: Array) -> None:
        if self.shape != other.shape or self.dtype != other.dtype:
            raise ValueError("Shape or dtype mismatch for copy")

        self.data = other.data.copy() if other.data is not None else None

    def add_argument(self, arg_node: Array) -> None:
        if not isinstance(arg_node, Array):
            raise TypeError(
                f"Argument must be an instance of Array, got {type(arg_node)}"
            )
        self.args.append(arg_node)

    def get_data(self) -> Tensor:
        realize_(self)

        if self.data is None:
            raise ValueError("Data is None after realization")
        return self.data

    def get_numpy(self) -> np.ndarray:
        if self._numpy_cache is None:
            if self.data is None:
                raise ValueError("Cannot get NumPy array from None data")
            self._numpy_cache = self.data.to_numpy()
        return self._numpy_cache

    def get_arguments(self) -> List[Array]:
        return list(self.args)

    def set_maxpr(self, fn: MaxprCallable) -> None:
        self.maxpr = fn

    def __repr__(self) -> str:
        return self.get_data().to(CPU()).to_numpy().__str__()

    def to(self, device: Device) -> Array:
        if self.data is None:
            realize_(self)
        new_data = self.data.to(device)
        return Array.from_data(new_data, name=self.name)


def arange(shape: Shape, dtype: DType, device: Device = CPU()) -> Array:
    return Array.from_data(
        Tensor.from_numpy(
            np.arange(np.prod(shape), dtype=DType.to_numpy(dtype)).reshape(shape)
        )
    ).to(device)


class Add:
    @staticmethod
    def maxpr(args: List[Value]) -> Value:
        if len(args) != 2:
            raise ValueError(f"Add operation requires 2 arguments, got {len(args)}")
        return args[0] + args[1]

    @staticmethod
    def eagerxpr(args: List[Array]) -> Array:
        if len(args) != 2:
            raise ValueError(f"Add operation requires 2 arguments, got {len(args)}")
        if args[0].shape != args[1].shape:
            raise ValueError(
                f"Shapes {args[0].shape} and {args[1].shape} are not compatible for addition."
            )
        np_result = np.add(args[0].get_numpy(), args[1].get_numpy())
        result_data = Tensor.from_numpy(np_result)
        return Array.from_data(result_data, name=f"add({args[0].name},{args[1].name})")


def add(arg0: Array, arg1: Array) -> Array:
    if arg0.dtype != arg1.dtype:
        raise ValueError(
            f"Dtypes {arg0.dtype} and {arg1.dtype} are not compatible for multiplication."
        )
    res_shape = get_broadcasted_shape(arg0.shape, arg1.shape)

    if EAGERMODE:
        res = Add.eagerxpr([arg0, arg1])
    else:
        res = Array(shape=res_shape, dtype=arg0.dtype, materialize=False, name="add")

    res.add_argument(arg0)
    res.add_argument(arg1)
    res.set_maxpr(Add.maxpr)
    return res


class Mul:
    @staticmethod
    def maxpr(args: List[Value]) -> Value:
        """MAX graph implementation of multiplication."""
        if len(args) != 2:
            raise ValueError(f"Mul operation requires 2 arguments, got {len(args)}")
        return args[0] * args[1]

    @staticmethod
    def eagerxpr(args: List[Array]) -> Array:
        if len(args) != 2:
            raise ValueError(f"Mul operation requires 2 arguments, got {len(args)}")
        if args[0].shape != args[1].shape:
            raise ValueError(
                f"Shapes {args[0].shape} and {args[1].shape} are not compatible for multiplication."
            )
        np_result = np.multiply(args[0].get_numpy(), args[1].get_numpy())
        result_data = Tensor.from_numpy(np_result)
        return Array.from_data(result_data, name=f"mul({args[0].name},{args[1].name})")


def mul(arg0: Array, arg1: Array) -> Array:
    if arg0.dtype != arg1.dtype:
        raise ValueError(
            f"Dtypes {arg0.dtype} and {arg1.dtype} are not compatible for multiplication."
        )
    res_shape = get_broadcasted_shape(arg0.shape, arg1.shape)

    if EAGERMODE:
        res = Mul.eagerxpr([arg0, arg1])
    else:
        res = Array(shape=res_shape, dtype=arg0.dtype, materialize=False, name="mul")

    res.add_argument(arg0)
    res.add_argument(arg1)
    res.set_maxpr(Mul.maxpr)
    return res


# matmul
class MatMul:
    @staticmethod
    def maxpr(args: List[Value]) -> Value:
        """MAX graph implementation of matrix multiplication."""
        if len(args) != 2:
            raise ValueError(f"MatMul operation requires 2 arguments, got {len(args)}")

        # ops.matmul takes in tensor of maxxrank 4, so we mgiht need to reshape first f needed
        if args[0].shape[-1] != args[1].shape[-2]:
            raise ValueError(
                f"Shapes {args[0].shape} and {args[1].shape} are not compatible for matrix multiplication."
            )

        # reshape arg0 if needed
        max_rank = max(args[0].rank, args[1].rank)
        reshape_res = False

        prod = 1

        if args[0].rank > 4:
            batch_dims = args[0].shape[:-2]
            prod = np.prod(batch_dims)
            reshape_res = True

        if args[1].rank > 4:
            batch_dims = args[1].shape[:-2]
            prod = max(prod, np.prod(batch_dims))
            reshape_res = True

        if reshape_res:
            arg0 = ops.reshape(args[0], (prod, args[0].shape[-2], args[0].shape[-1]))
            arg1 = ops.reshape(args[1], (prod, args[1].shape[-2], args[1].shape[-1]))

        res_shape = get_broadcasted_shape(
            arg0.shape,
            arg1.shape,
            ignore_axes=[-2, -1],
            replace_ignored_dims=[arg0.shape[-2], arg1.shape[-1]],
        )

        res = ops.matmul(arg0, arg1)
        # reshape res to the final shape
        if reshape_res:
            res = ops.reshape(res, res_shape)
        return res

    @staticmethod
    def eagerxpr(args: List[Array]) -> Array:
        if len(args) != 2:
            raise ValueError(f"MatMul operation requires 2 arguments, got {len(args)}")
        if args[0].shape[-1] != args[1].shape[-2]:
            raise ValueError(
                f"Shapes {args[0].shape} and {args[1].shape} are not compatible for matrix multiplication."
            )

        np_result = np.matmul(args[0].get_numpy(), args[1].get_numpy())
        result_data = Tensor.from_numpy(np_result)
        return Array.from_data(
            result_data, name=f"matmul({args[0].name},{args[1].name})"
        )


def matmul(arg0: Array, arg1: Array) -> Array:
    if arg0.shape[-1] != arg1.shape[-2]:
        raise ValueError(
            f"Shapes {arg0.shape} and {arg1.shape} are not compatible for matrix multiplication."
        )

    if EAGERMODE:
        res = MatMul.eagerxpr([arg0, arg1])
    else:
        res = Array(
            shape=(arg0.shape[-2], arg1.shape[-1]),
            dtype=arg0.dtype,
            materialize=False,
            name="matmul",
        )

    res.add_argument(arg0)
    res.add_argument(arg1)
    res.set_maxpr(MatMul.maxpr)
    return res


def compute_node_hash(node: Array) -> int:
    components = [
        str(node.shape),
        str(node.dtype),
        node.name or "unnamed",
        # Could add operation type for more uniqueness
    ]
    node_str = "-".join(components)
    return hash(node_str)


def get_trace(nodes: Sequence[Array]) -> Tuple[List[Array], List[Array], int]:
    trace: List[Array] = []
    inputs: List[Array] = []
    visited: Set[Array] = set()

    # Iterative DFS using a stack
    for start_node in nodes:
        if start_node in visited:
            continue

        stack: List[Array] = [start_node]
        while stack:
            node = stack[-1]  # Peek at the top node

            if node in visited:
                stack.pop()  # Already processed this node
                continue

            # If this is a leaf node (has data)
            if node.data is not None:
                inputs.append(node)
                trace.append(node)
                visited.add(node)
                stack.pop()
                continue

            # Check if all children have been visited
            all_children_visited = True
            for arg in node.args:
                if arg not in visited:
                    all_children_visited = False
                    stack.append(arg)  # Add unvisited child to stack

            # If all children have been visited, we can process this node
            if all_children_visited:
                visited.add(node)
                trace.append(node)
                stack.pop()

    # Compute the key from the trace
    key: int = 0
    for node in trace:
        node_hash = compute_node_hash(node)
        key = key ^ (node_hash + 0x9E3779B9 + (key << 6) + (key >> 2))

    key = key % 1000000000

    return inputs, trace, key


def realize_(outputs: Union[Sequence[Array], Array]) -> None:
    # Normalize to list of arrays
    if isinstance(outputs, Array):
        output_list = [outputs]
    else:
        if not all(isinstance(x, Array) for x in outputs):
            raise TypeError("All outputs must be Array instances")
        output_list = list(outputs)

    # Check if there are outputs which need to be realized
    nothing_to_compute = True
    for output in output_list:
        if output.data is None:
            nothing_to_compute = False
            break

    if nothing_to_compute:
        return

    # Retrieve the trace and inputs which are the last realized values (i.e. leaves)
    inputs, trace, key = get_trace(output_list)

    if key in global_execution_context:
        model = global_execution_context[key]
    else:
        # Build input types for the Graph
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

        # Define the MAX graph
        try:
            with Graph("max_graph", input_types=input_types) as graph:
                input_symbols = graph.inputs
                for i in range(len(input_symbols)):
                    inputs[i].tensor_value = input_symbols[i]

                for node in trace:
                    if node.tensor_value is not None:
                        continue

                    arg_symbols = []
                    for arg in node.get_arguments():
                        if arg.tensor_value is None:
                            raise ValueError(f"Error retrieving symbol for {arg.name}")
                        arg_symbols.append(arg.tensor_value)

                    if node.maxpr is None:
                        raise ValueError(f"Node {node.name} has no maxpr function")
                    node.tensor_value = node.maxpr(arg_symbols)

                output_symbols = []
                for output in output_list:
                    if output.tensor_value is None:
                        raise ValueError(f"Output {output.name} has no tensor value")
                    output_symbols.append(output.tensor_value)

                graph.output(*output_symbols)

            # Set up the MAX model and cache it
            session = InferenceSession(devices=devices)
            model = session.load(graph)
            global_execution_context[key] = model

        except Exception as e:
            raise ValueError(f"Failed to build computation graph: {e}")

    # Prepare input tensors
    tensor_inputs = []
    for input_node in inputs:
        if input_node.data is None:
            raise ValueError(f"Input {input_node.name} has no data")
        tensor_inputs.append(input_node.data)

    # Execute the model and update outputs
    try:
        model_outputs = model.execute(*tensor_inputs)
        for i, output in enumerate(output_list):
            output.data = model_outputs[i]
            output._numpy_cache = None  # Invalidate cache
    except Exception as e:
        raise ValueError(f"Error executing computation: {e}")


if __name__ == "__main__":
    # Basic tests and benchmarks
    a = arange(shape=(4, 256, 256), dtype=DType.float32)  # .to(Accelerator())
    # print("\na:")
    # print(a)

    b = arange(shape=(3, 4, 256, 256), dtype=DType.float32)  # .to(Accelerator())
    # print("\nb:")
    # print(b)

    for iter in range(1000):
        c = mul(a, b)
        res = arange(
            shape=(2, 3, 4, 256, 256), dtype=DType.float32
        )  # .to(Accelerator())

        for i in range(1000):
            res = mul(res, c)

        _ = res.get_data()  # Trigger realization

        if iter % 100 == 0:
            print(f"Iteration {iter} completed.")



# format file via: black graph.py