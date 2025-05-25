from __future__ import annotations
import numpy as np
import time
from typing import (
    List, Final, ClassVar, Union, Tuple, Type, Set, Callable, Optional,
    Sequence, Dict, Protocol, TypedDict, cast, Any
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
    
    
    def __init__(self, 
                 shape: Shape, 
                 dtype: DType = DType.float32, 
                 device: Device = CPU(),
                 materialize: bool = False, 
                 name: str = "") -> None:
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
            
        instance = cls(shape=data.shape, dtype=data.dtype, device=data.device, materialize=True) 
        instance.data = data
        instance.name = name
        return instance
    
    @classmethod
    def create_buffer(cls, shape: Shape, dtype: DType, device: Device) -> Array:
        return cls(shape=shape, dtype=dtype, device=device, materialize=True)
        
    def copy_from(self, other: Array) -> None:
        if self.shape != other.shape or self.dtype != other.dtype:
            raise ValueError("Shape or dtype mismatch for copy")
            
        # # Copy data in-place
        # if self.data is not None and other.data is not None:
        #     np.copyto(self.get_numpy(), other.get_numpy())
        # else:
        #     raise ValueError("Both arrays must have data for copy operation")
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


def add(arg0: Array, arg1: Array) -> Array:
    if arg0.shape != arg1.shape:
        raise ValueError(f"Shapes {arg0.shape} and {arg1.shape} are not compatible for addition.")

    if EAGERMODE:
        # Use numpy directly for faster computation
        try:
            np_result = np.add(arg0.get_numpy(), arg1.get_numpy())
            result_data = Tensor.from_numpy(np_result)
            res = Array.from_data(result_data, name=f"add({arg0.name},{arg1.name})")
        except (ValueError, TypeError) as e:
            # Fallback if NumPy operations fail
            result_data = Tensor.from_numpy(arg0.get_data().to_numpy() + arg1.get_data().to_numpy())
            res = Array.from_data(result_data, name=f"add({arg0.name},{arg1.name})")
    else:
        res = Array(shape=arg0.shape, dtype=arg0.dtype, materialize=False, name="add")
        
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


def mul(arg0: Array, arg1: Array) -> Array:
    if arg0.shape != arg1.shape:
        raise ValueError(f"Shapes {arg0.shape} and {arg1.shape} are not compatible for multiplication.")
    
    if EAGERMODE:
        try:
            np_result = np.multiply(arg0.get_numpy(), arg1.get_numpy())
            result_data = Tensor.from_numpy(np_result)
            res = Array.from_data(result_data, name=f"mul({arg0.name},{arg1.name})")
        except (ValueError, TypeError) as e:
            # Fallback if NumPy operations fail
            result_data = Tensor.from_numpy(arg0.get_data().to_numpy() * arg1.get_data().to_numpy())
            res = Array.from_data(result_data, name=f"mul({arg0.name},{arg1.name})")
    else:
        res = Array(shape=arg0.shape, dtype=arg0.dtype, materialize=False, name="mul")
        
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
        return ops.matmul(args[0], args[1])
    
def matmul(arg0: Array, arg1: Array) -> Array:
    if arg0.shape[-1] != arg1.shape[0]:
        raise ValueError(f"Shapes {arg0.shape} and {arg1.shape} are not compatible for matrix multiplication.")
    
    if EAGERMODE:
        try:
            np_result = np.matmul(arg0.get_numpy(), arg1.get_numpy())
            result_data = Tensor.from_numpy(np_result)
            res = Array.from_data(result_data, name=f"matmul({arg0.name},{arg1.name})")
        except (ValueError, TypeError) as e:
            # Fallback if NumPy operations fail
            result_data = Tensor.from_numpy(arg0.get_data().to_numpy() @ arg1.get_data().to_numpy())
            res = Array.from_data(result_data, name=f"matmul({arg0.name},{arg1.name})")
    else:
        res = Array(shape=(arg0.shape[-2], arg1.shape[-1]), dtype=arg0.dtype, materialize=False, name="matmul")
        
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
            input_types.append(TensorType(
                dtype=input_node.dtype,
                shape=input_node.shape,
                device=DeviceRef.from_device(input_node.device)
            ))
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
    a = arange(shape=(256, 256), dtype=DType.float32)#.to(Accelerator())
    print("\na:")
    print(a)
    
    b = arange(shape=(256, 256), dtype=DType.float32)#.to(Accelerator())
    print("\nb:")
    print(b)

    for iter in range(1000):
        c = mul(a, b)
        res = arange(shape=(256, 256), dtype=DType.float32)#.to(Accelerator())

        for i in range(1000):
            res = mul(res, c)

        # print(res)
        _ = res.get_data()  # Trigger realization
        if iter % 100 == 0:
            print(f"Iteration {iter} completed.")

