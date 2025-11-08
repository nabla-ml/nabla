import max_graph.ops as ops
from max_graph.types import (
    Device, DeviceType, TensorType, TensorValue, CPU
)
from max_graph.types import Tensor as MaxTensor
from max_graph.utils import (
    PythonBridge, Graph, DeviceRef
)
from python import Python, PythonObject
from memory import ArcPointer, UnsafePointer


struct TensorImpl(Copyable, Movable):
    """Unsafe access to internal representation of the Tensor."""
    var shape: List[Int]
    var batch_dims: List[Int]
    var dtype: DType
    var device: Device
    var name: String
    var parents: List[ArcPointer[TensorImpl]]
    var visited: Bool
    var tensor_value: Optional[TensorValue]
    var maxpr: List[fn(List[TensorValue], TensorImpl) raises -> None]
    var vjp_rule: List[fn(List[Tensor], Tensor, Tensor) raises -> List[Tensor]]
    var jvp_rule: List[fn(List[Tensor], List[TensorImpl], Tensor) raises -> Tensor]
    var traced: Bool
    var requires_grad: Bool
    var tangent: List[ArcPointer[TensorImpl]]
    var cotangent: List[ArcPointer[TensorImpl]]
    var grad: List[ArcPointer[TensorImpl]]
    var stage_realization: Bool
    var custom_kernel_path: String
    var data: List[MaxTensor]

    fn __init__(out self, shape: List[Int], dtype: DType = DType.float32) raises:
        self.shape = shape.copy()
        self.batch_dims = List[Int]()
        self.dtype = dtype
        self.device = CPU().as_device()
        self.name = String("arg")
        self.parents = []
        self.visited = False
        self.tensor_value = Optional[TensorValue](None)
        self.maxpr = []
        self.vjp_rule = []
        self.jvp_rule = []
        self.traced = False
        self.requires_grad = False
        self.tangent = []
        self.cotangent = []
        self.grad = []
        self.stage_realization = False
        self.custom_kernel_path = String("")
        self.data = [MaxTensor.zeros(self.shape, self.dtype, self.device)]

struct Tensor(ImplicitlyCopyable, Movable, Writable):
    var _storage: ArcPointer[TensorImpl]

    fn __init__(out self, storage: ArcPointer[TensorImpl]) raises:
        self._storage = storage

    fn __init__(out self, shape: List[Int], dtype: DType = DType.float32) raises:
        self._storage = ArcPointer(TensorImpl(shape, dtype))

    # Getter methods
    fn name(self) -> String:
        return self._storage[].name
    
    fn shape(self) -> List[Int]:
        return self._storage[].shape.copy()
    
    fn dtype(self) -> DType:
        return self._storage[].dtype
    
    fn device(self) -> Device:
        return self._storage[].device.copy()
    
    fn visited(self) -> Bool:
        return self._storage[].visited
    
    fn requires_grad(self) -> Bool:
        return self._storage[].requires_grad
    
    fn traced(self) -> Bool:
        return self._storage[].traced
    
    fn stage_realization(self) -> Bool:
        return self._storage[].stage_realization
    
    fn custom_kernel_path(self) -> String:
        return self._storage[].custom_kernel_path
    
    fn batch_dims(self) -> List[Int]:
        return self._storage[].batch_dims.copy()
    
    fn tensor_value(self) raises -> TensorValue:
        """Get the tensor value. Raises if not set."""
        if not self._storage[].tensor_value:
            raise Error("TensorValue is not set")
        return self._storage[].tensor_value.value()
    
    fn has_tensor_value(self) -> Bool:
        """Check if tensor_value is set."""
        return self._storage[].tensor_value.__bool__()
    
    fn data(self) raises -> MaxTensor:
        """Get the underlying MaxTensor data. Raises if not set."""
        if len(self._storage[].data) == 0:
            raise Error("Data is not set")
        return self._storage[].data[0]
    
    fn has_data(self) -> Bool:
        """Check if data is set."""
        return len(self._storage[].data) > 0
    
    fn parents(self) raises -> List[Tensor]:
        """Get parent tensors (returns proper Tensor objects, not ArcPointers)."""
        var result = List[Tensor]()
        for parent_ptr in self._storage[].parents:
            result.append(Tensor(parent_ptr))
        return result^
    
    fn has_parents(self) -> Bool:
        """Check if this tensor has parent tensors."""
        return len(self._storage[].parents) > 0

    # Setter methods
    fn set_name(mut self, value: String):
        self._storage[].name = value
    
    fn set_visited(mut self, value: Bool):
        self._storage[].visited = value
    
    fn set_requires_grad(mut self, value: Bool):
        self._storage[].requires_grad = value
    
    fn set_traced(mut self, value: Bool):
        self._storage[].traced = value
    
    fn set_stage_realization(mut self, value: Bool):
        self._storage[].stage_realization = value
    
    fn set_custom_kernel_path(mut self, value: String):
        self._storage[].custom_kernel_path = value
    
    fn set_shape(mut self, value: List[Int]):
        self._storage[].shape = value.copy()
    
    fn set_batch_dims(mut self, value: List[Int]):
        self._storage[].batch_dims = value.copy()
    
    fn set_dtype(mut self, value: DType):
        self._storage[].dtype = value
    
    fn set_device(mut self, value: Device):
        self._storage[].device = value.copy()
    
    fn set_tensor_value(mut self, value: Optional[TensorValue]):
        self._storage[].tensor_value = value
    
    # Getters/Setters for autodiff-related fields
    fn maxpr(self) raises -> fn(List[TensorValue], TensorImpl) raises -> None:
        """Get the maxpr rule. Raises if not set."""
        if len(self._storage[].maxpr) == 0:
            raise Error("maxpr rule is not set")
        return self._storage[].maxpr[0]
    
    fn has_maxpr(self) -> Bool:
        """Check if maxpr rule is set."""
        return len(self._storage[].maxpr) > 0
    
    fn set_maxpr(mut self, value: fn(List[TensorValue], TensorImpl) raises -> None):
        self._storage[].maxpr = [value]
    
    fn vjp_rule(self) raises -> fn(List[Tensor], Tensor, Tensor) raises -> List[Tensor]:
        """Get the VJP rule. Raises if not set."""
        if len(self._storage[].vjp_rule) == 0:
            raise Error("VJP rule is not set")
        return self._storage[].vjp_rule[0]
    
    fn has_vjp_rule(self) -> Bool:
        """Check if VJP rule is set."""
        return len(self._storage[].vjp_rule) > 0
    
    fn set_vjp_rule(mut self, value: fn(List[Tensor], Tensor, Tensor) raises -> List[Tensor]):
        self._storage[].vjp_rule = [value]
    
    fn jvp_rule(self) raises -> fn(List[Tensor], List[TensorImpl], Tensor) raises -> Tensor:
        """Get the JVP rule. Raises if not set."""
        if len(self._storage[].jvp_rule) == 0:
            raise Error("JVP rule is not set")
        return self._storage[].jvp_rule[0]
    
    fn has_jvp_rule(self) -> Bool:
        """Check if JVP rule is set."""
        return len(self._storage[].jvp_rule) > 0
    
    fn set_jvp_rule(mut self, value: fn(List[Tensor], List[TensorImpl], Tensor) raises -> Tensor):
        self._storage[].jvp_rule = [value]
    
    # Getters/Setters for tangent/cotangent/grad
    fn tangent(self) raises -> Tensor:
        """Get the tangent. Raises if not set."""
        if len(self._storage[].tangent) == 0:
            raise Error("Tangent is not set")
        return Tensor(self._storage[].tangent[0])
    
    fn has_tangent(self) -> Bool:
        """Check if tangent is set."""
        return len(self._storage[].tangent) > 0
    
    fn set_tangent(mut self, value: Tensor):
        self._storage[].tangent = [value._storage]
    
    fn cotangent(self) raises -> Tensor:
        """Get the cotangent. Raises if not set."""
        if len(self._storage[].cotangent) == 0:
            raise Error("Cotangent is not set")
        return Tensor(self._storage[].cotangent[0])
    
    fn has_cotangent(self) -> Bool:
        """Check if cotangent is set."""
        return len(self._storage[].cotangent) > 0
    
    fn set_cotangent(mut self, value: Tensor):
        self._storage[].cotangent = [value._storage]
    
    fn grad(self) raises -> Tensor:
        """Get the gradient. Raises if not set."""
        if len(self._storage[].grad) == 0:
            raise Error("Gradient is not set")
        return Tensor(self._storage[].grad[0])
    
    fn has_grad(self) -> Bool:
        """Check if gradient is set."""
        return len(self._storage[].grad) > 0
    
    fn set_grad(mut self, value: Tensor):
        self._storage[].grad = [value._storage]
    
    # Utility methods for graph management
    fn add_parent(mut self, parent: Tensor):
        """Add a parent tensor to the computation graph."""
        self._storage[].parents.append(parent._storage)
    
    fn set_parents(mut self, parents: List[Tensor]) raises:
        """Set all parent tensors at once."""
        self._storage[].parents.clear()
        for parent in parents:
            self._storage[].parents.append(parent._storage)
    
    fn ndim(self) -> Int:
        """Get the number of dimensions."""
        return len(self._storage[].shape)
    
    fn numel(self) -> Int:
        """Get the total number of elements."""
        var total = 1
        for dim in self._storage[].shape:
            total *= dim
        return total
    
    fn write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(self._storage[].data[0].to_numpy().__str__())
        except:
            writer.write("Error while converting Tensor to string")

    fn __add__(self, other: Self) raises -> Self:
        var res = Tensor(self._storage[].shape, self._storage[].dtype)
        res._storage[].parents = [self._storage, other._storage]
        res._storage[].name = "add"
        return res^

    fn __mul__(self, other: Self) raises -> Self:
        var res = Tensor(self._storage[].shape, self._storage[].dtype)
        res._storage[].parents = [self._storage, other._storage]
        res._storage[].name = "mul"
        return res^

fn reset_visited(mut tensor: Tensor) raises:
    """Reset visited flag for entire graph."""
    tensor.set_visited(False)
    for parent in tensor._storage[].parents:
        var parent_tensor = Tensor(parent)
        reset_visited(parent_tensor)

fn dfs(mut output: Tensor, mut trace: List[Tensor]) raises -> None:
    if not output.visited():
        output.set_visited(True)
        for parent in output._storage[].parents:
            var parent_tensor = Tensor(parent)
            dfs(parent_tensor, trace)
        trace.append(output)

fn print_trace(trace: List[Tensor]) raises -> None:
    for t in trace:
        print(t.name())

fn main() raises:
    # Test graph construction
    print("=== Testing graph ===")
    var a = Tensor([2, 3])
    var b = Tensor([2, 3])
    var c = Tensor([2, 3])
    var d = Tensor([2, 3])
    var e = Tensor([2, 3])
    var f = a + b
    var g = c * d
    var h = f + g
    var i = h + e
    var j = i * f

    a.set_name("a")
    b.set_name("b")
    c.set_name("c")

    var trace: List[Tensor] = []
    reset_visited(j)
    dfs(j, trace)
    print("Computation Trace:")
    print_trace(trace)
    
    # Test setters
    print("\n=== Testing setters ===")
    print("Before: a.name =", a.name(), ", requires_grad =", a.requires_grad())
    a.set_requires_grad(True)
    a.set_traced(True)
    print("After: a.name =", a.name(), ", requires_grad =", a.requires_grad(), ", traced =", a.traced())
    
    # Reset and traverse again
    print("\n=== After reset ===")
    reset_visited(j)
    var trace2: List[Tensor] = []
    dfs(j, trace2)
    print("Computation Trace 2:")
    print_trace(trace2)
    print("Computation Trace:")
    print_trace(trace)