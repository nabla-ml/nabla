import max_graph.ops as ops
from max_graph.types import (
    Device, DeviceType, TensorType, TensorValue, CPU
)
from max_graph.types import Tensor as MaxTensor, DTypeConverter
from max_graph.utils import (
    PythonBridge, Graph, DeviceRef, MaxModel
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
    var maxpr: List[fn(List[TensorValue]) raises -> TensorValue]
    var vjp_rule: List[fn(List[Tensor], Tensor, mut Tensor) raises -> List[Tensor]]
    var jvp_rule: List[fn(List[Tensor], List[Tensor], Tensor) raises -> Tensor]
    var traced: Bool
    var requires_grad: Bool
    var tangent: List[ArcPointer[TensorImpl]]
    var cotangent: List[ArcPointer[TensorImpl]]
    var grad: List[ArcPointer[TensorImpl]]
    var stage_realization: Bool
    var custom_kernel_path: String
    var data: List[MaxTensor]
    var metadata: Dict[String, List[Int]]

    fn __init__(out self, shape: List[Int], dtype: DType = DType.float32, stage_realization: Bool = False) raises:
        self.shape = shape.copy()
        self.batch_dims = List[Int]()
        self.dtype = dtype
        self.device = CPU().as_device()
        self.name = String("zeros")
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
        self.stage_realization = stage_realization
        self.custom_kernel_path = String("")
        if self.stage_realization:
            self.data = []
        else:
            self.data = [MaxTensor.zeros(self.shape, self.dtype, self.device)]
        self.metadata = {}

struct Tensor(ImplicitlyCopyable, Movable, Writable):
    var _storage: ArcPointer[TensorImpl]

    fn __init__(out self, storage: ArcPointer[TensorImpl]) raises:
        self._storage = storage

    fn __init__(out self, shape: List[Int], dtype: DType = DType.float32, stage_realization: Bool = False) raises:
        self._storage = ArcPointer(TensorImpl(shape, dtype, stage_realization))

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

    fn set_data(mut self, value: MaxTensor):
        self._storage[].data = [value]
    
    fn parents(self) raises -> List[Tensor]:
        """Get parent tensors (returns proper Tensor objects, not ArcPointers)."""
        var result = List[Tensor]()
        for parent_ptr in self._storage[].parents:
            result.append(Tensor(parent_ptr))
        return result^
    
    fn has_parents(self) -> Bool:
        """Check if this tensor has parent tensors."""
        return len(self._storage[].parents) > 0

    fn metadata(self) -> Dict[String, List[Int]]:
        return self._storage[].metadata.copy()

    fn __getitem__(self, key: String) raises -> List[Int]:
        if not key in self._storage[].metadata:
            raise Error("Metadata key not found: " + key)
        return self._storage[].metadata[key].copy()

    fn __setitem__(mut self, key: String, value: List[Int]):
        self._storage[].metadata[key] = value.copy()

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
    
    fn set_tensor_value(mut self, value: TensorValue):
        self._storage[].tensor_value = value

    fn remove_tensor_value(mut self):
        self._storage[].tensor_value = Optional[TensorValue](None)
    
    # Getters/Setters for autodiff-related fields
    fn maxpr(self) raises -> fn(List[TensorValue]) raises -> TensorValue:
        """Get the maxpr rule. Raises if not set."""
        if len(self._storage[].maxpr) == 0:
            raise Error("maxpr rule is not set")
        return self._storage[].maxpr[0]
    
    fn has_maxpr(self) -> Bool:
        """Check if maxpr rule is set."""
        return len(self._storage[].maxpr) > 0
    
    fn set_maxpr(mut self, value: fn(List[TensorValue]) raises -> TensorValue):
        self._storage[].maxpr = [value]
    
    fn vjp_rule(self) raises -> fn(List[Tensor], Tensor, mut Tensor) raises -> List[Tensor]:
        """Get the VJP rule. Raises if not set."""
        if len(self._storage[].vjp_rule) == 0:
            raise Error("VJP rule is not set")
        return self._storage[].vjp_rule[0]
    
    fn has_vjp_rule(self) -> Bool:
        """Check if VJP rule is set."""
        return len(self._storage[].vjp_rule) > 0
    
    fn set_vjp_rule(mut self, value: fn(List[Tensor], Tensor, mut Tensor) raises -> List[Tensor]):
        self._storage[].vjp_rule = [value]
    
    fn has_jvp_rule(self) -> Bool:
        """Check if JVP rule is set."""
        return len(self._storage[].jvp_rule) > 0
    
    fn set_jvp_rule(mut self, value: fn(List[Tensor], List[Tensor], Tensor) raises -> Tensor):
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
            if not self.has_data():
                writer.write("Unmaterialized Tensor")
                return
            else:
                writer.write(self._storage[].data[0].to_numpy().__str__())
        except:
            writer.write("Error while converting Tensor to string")

    fn __add__(self, other: Self) raises -> Self:
        return add(self, other)

    fn __mul__(self, other: Self) raises -> Self:
        return mul(self, other)

    fn __matmul__(self, other: Self) raises -> Self:
        return matmul(self, other)

fn list_to_python_tuple(lst: List[Int]) raises -> PythonObject:
    var py = PythonBridge.get_module("builtins")
    var py_list = py.list([])
    for item in lst:
        py_list.append(item)
    return py.tuple(py_list)
    
# creation ops
fn ones(shape: List[Int], dtype: DType = DType.float32) raises -> Tensor:
    var tensor = Tensor(shape, dtype, stage_realization=False)
    var np = PythonBridge.get_module("numpy")
    var np_ones = np.ones(list_to_python_tuple(shape), dtype=DTypeConverter.to_numpy(dtype))
    tensor.set_data(MaxTensor.from_numpy(np_ones))
    return tensor

fn arange(start: Int, stop: Int, step: Int = 1, dtype: DType = DType.float32) raises -> Tensor:
    var np = PythonBridge.get_module("numpy")
    var np_arange = np.arange(start, stop, step, dtype=DTypeConverter.to_numpy(dtype))
    var shape = List[Int]()
    shape.append(Int(np_arange.shape[0]))
    var tensor = Tensor(shape, dtype, stage_realization=False)
    tensor.set_data(MaxTensor.from_numpy(np_arange))
    return tensor

fn ndarange(shape: List[Int], dtype: DType = DType.float32) raises -> Tensor:
    var np = PythonBridge.get_module("numpy")
    var py = PythonBridge.get_module("builtins")
    var end = 1
    var np_shape = py.list([])
    for dim in shape:
        end *= dim
        np_shape.append(dim)
    var np_arange = np.arange(0, end, 1, dtype=DTypeConverter.to_numpy(dtype)).reshape(np_shape)
    var tensor = Tensor(shape, dtype, stage_realization=False)
    tensor.set_data(MaxTensor.from_numpy(np_arange))
    return tensor

trait Operation:
    @staticmethod
    fn name() -> String:
        pass

    @staticmethod
    fn shape(inputs: List[Tensor]) raises -> List[Int]:
        pass

    @staticmethod
    fn dtype(inputs: List[Tensor]) raises -> DType:
        pass

    @staticmethod
    fn device(inputs: List[Tensor]) raises -> Device:
        pass

    @staticmethod
    fn batch_dims(inputs: List[Tensor]) raises -> List[Int]:
        pass

    @staticmethod
    fn maxpr(inputs: List[TensorValue]) raises -> TensorValue:
        pass

    @staticmethod
    fn vjp_rule(primals: List[Tensor], cotangent: Tensor, output: Tensor) raises -> List[Tensor]:
        pass

    @staticmethod
    fn jvp_rule(primals: List[Tensor], tangents: List[Tensor], output: Tensor) raises -> Tensor:
        pass

    @staticmethod
    fn stage_realization(inputs: List[Tensor]) raises -> Bool:
        if len(inputs) != 2:
            raise Error("BinaryOp requires exactly 2 input tensors")
        return inputs[0].stage_realization() or inputs[1].stage_realization()

    @staticmethod
    fn custom_kernel_path() -> String:
        return String("")

    @staticmethod
    fn execute(inputs: List[Tensor]) raises -> Tensor:
        var res = Tensor(
            shape=Self.shape(inputs), 
            dtype=Self.dtype(inputs), 
            stage_realization=True)
        res.set_name(Self.name())
        res.set_parents(inputs)
        res.set_maxpr(Self.maxpr)
        res.set_vjp_rule(Self.vjp_rule)
        res.set_jvp_rule(Self.jvp_rule)
        res.set_custom_kernel_path(Self.custom_kernel_path())
        res.set_parents(inputs)
        return res


trait BinaryOp(Operation):
    @staticmethod
    fn batch_dims(inputs: List[Tensor]) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("BinaryOp requires exactly 2 input tensors")
        if inputs[0].batch_dims() != inputs[1].batch_dims():
            raise Error("Input tensors must have the same batch_dims for BinaryOp")
        return inputs[0].batch_dims()

    @staticmethod
    fn shape(inputs: List[Tensor]) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("BinaryOp requires exactly 2 input tensors")
        if inputs[0].shape() != inputs[1].shape():
            raise Error("Input tensors must have the same shape for BinaryOp")
        return inputs[0].shape()
    
    @staticmethod
    fn dtype(inputs: List[Tensor]) raises -> DType:
        if len(inputs) != 2:
            raise Error("BinaryOp requires exactly 2 input tensors")
        if inputs[0].dtype() != inputs[1].dtype():
            raise Error("Input tensors must have the same dtype for BinaryOp")
        return inputs[0].dtype()
    
    @staticmethod
    fn device(inputs: List[Tensor]) raises -> Device:
        if len(inputs) != 2:
            raise Error("BinaryOp requires exactly 2 input tensors")
        if inputs[0].device() != inputs[1].device():
            raise Error("Input tensors must be on the same device for BinaryOp")
        return inputs[0].device()
    
    @staticmethod
    fn maxpr(inputs: List[TensorValue]) raises -> TensorValue:
        pass

    @staticmethod
    fn vjp_rule(primals: List[Tensor], cotangent: Tensor, output: Tensor) raises -> List[Tensor]:
        pass

    @staticmethod
    fn jvp_rule(primals: List[Tensor], tangents: List[Tensor], output: Tensor) raises -> Tensor:
        pass


struct AddOp(BinaryOp):
    @staticmethod
    fn name() -> String:
        return "add"

    @staticmethod
    fn maxpr(inputs: List[TensorValue]) raises -> TensorValue:
        return ops.add(inputs[0], inputs[1])

    @staticmethod
    fn vjp_rule(primals: List[Tensor], cotangent: Tensor, output: Tensor) raises -> List[Tensor]:
        return [cotangent, cotangent]

    @staticmethod
    fn jvp_rule(primals: List[Tensor], tangents: List[Tensor], output: Tensor) raises -> Tensor:
        return tangents[0] + tangents[1]

fn add(a: Tensor, b: Tensor) raises -> Tensor:
    return AddOp.execute([a, b])

struct MulOp(BinaryOp):
    @staticmethod
    fn name() -> String:
        return "mul"

    @staticmethod
    fn maxpr(inputs: List[TensorValue]) raises -> TensorValue:
        return ops.mul(inputs[0], inputs[1])

    @staticmethod
    fn vjp_rule(primals: List[Tensor], cotangent: Tensor, output: Tensor) raises -> List[Tensor]:
        var grad_a = cotangent * primals[1]
        var grad_b = cotangent * primals[0]
        return [grad_a, grad_b]

    @staticmethod
    fn jvp_rule(primals: List[Tensor], tangents: List[Tensor], output: Tensor) raises -> Tensor:
        var term1 = tangents[0] * primals[1]
        var term2 = primals[0] * tangents[1]
        return term1 + term2

fn mul(a: Tensor, b: Tensor) raises -> Tensor:
    return MulOp.execute([a, b])

struct Matmul(Operation):
    @staticmethod
    fn name() -> String:
        return "matmul"

    @staticmethod
    fn shape(inputs: List[Tensor]) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("Matmul requires exactly 2 input tensors")
        var a_shape = inputs[0].shape()
        var b_shape = inputs[1].shape()
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise Error("Input tensors must be at least 2D for Matmul")
        if a_shape[-1] != b_shape[-2]:
            raise Error("Inner dimensions must match for Matmul")
        var result_shape = a_shape[0:-1] + b_shape[0:-2] + [b_shape[-1]]
        return result_shape^

    @staticmethod
    fn batch_dims(inputs: List[Tensor]) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("Matmul requires exactly 2 input tensors")
        if inputs[0].batch_dims() != inputs[1].batch_dims():
            raise Error("Input tensors must have the same batch_dims for Matmul")
        return inputs[0].batch_dims()

    @staticmethod
    fn dtype(inputs: List[Tensor]) raises -> DType:
        if len(inputs) != 2:
            raise Error("Matmul requires exactly 2 input tensors")
        if inputs[0].dtype() != inputs[1].dtype():
            raise Error("Input tensors must have the same dtype for Matmul")
        return inputs[0].dtype()

    @staticmethod
    fn device(inputs: List[Tensor]) raises -> Device:
        if len(inputs) != 2:
            raise Error("Matmul requires exactly 2 input tensors")
        if inputs[0].device() != inputs[1].device():
            raise Error("Input tensors must be on the same device for Matmul")
        return inputs[0].device()

    @staticmethod
    fn maxpr(inputs: List[TensorValue]) raises -> TensorValue:
        return ops.matmul(inputs[0], inputs[1])

    @staticmethod
    fn vjp_rule(primals: List[Tensor], cotangent: Tensor, output: Tensor) raises -> List[Tensor]:
        raise Error("VJP rule for Matmul not implemented yet")

    @staticmethod
    fn jvp_rule(primals: List[Tensor], tangents: List[Tensor], output: Tensor) raises -> Tensor:
        raise Error("JVP rule for Matmul not implemented yet")

fn matmul(a: Tensor, b: Tensor) raises -> Tensor:
    return Matmul.execute([a, b])



fn reset_visited(mut tensors: List[Tensor]) raises:
    """Reset visited flag for entire graph."""
    for var tensor in tensors:
        tensor.set_visited(False)
        var parents = tensor.parents()
        reset_visited(parents)


fn print_trace(trace: List[Tensor]) raises -> None:
    for t in trace:
        print(t.name())

fn get_dependencies_recursive(mut outputs: List[Tensor], mut trace: List[Tensor], mut inputs: List[Tensor]) raises -> None:
    for var output in outputs:
        if not output.visited():
            if output.has_parents():
                output.set_visited(True)
                var parents = output.parents()
                get_dependencies_recursive(parents, trace, inputs)
                trace.append(output)
            else:
                inputs.append(output)               

fn get_unmaterialized_recursive(mut outputs: List[Tensor], mut trace: List[Tensor], mut inputs: List[Tensor]) raises -> None:
    for var output in outputs:
        if not output.visited():
            if output.has_data():
                inputs.append(output)
            else:
                output.set_visited(True)
                var parents = output.parents()
                get_unmaterialized_recursive(parents, trace, inputs)
                trace.append(output)

fn get_dependency_trace(outputs: List[Tensor]) raises -> Tuple[List[Tensor], List[Tensor]]:
    var trace: List[Tensor] = []
    var inputs: List[Tensor] = []
    var _outputs = outputs.copy()
    reset_visited(_outputs)
    get_dependencies_recursive(_outputs, trace, inputs)
    reset_visited(_outputs)
    return (trace^, inputs^)

fn get_unmaterialized_trace(outputs: List[Tensor]) raises -> Tuple[List[Tensor], List[Tensor]]:
    var trace: List[Tensor] = []
    var inputs: List[Tensor] = []
    var _outputs = outputs.copy()
    reset_visited(_outputs)
    get_unmaterialized_recursive(_outputs, trace, inputs)
    reset_visited(_outputs)
    return (trace^, inputs^)

fn realize(mut outputs: List[Tensor]) raises:
    """Realize all tensors in the graph by assigning dummy data."""
    var unmaterialized_tuple = get_unmaterialized_trace(outputs)
    var unmaterialized_trace = unmaterialized_tuple[0].copy()
    var materialized_inputs = unmaterialized_tuple[1].copy()
    var input_types = List[TensorType]()
    for tensor in materialized_inputs:
        input_types.append(TensorType(tensor.dtype(), tensor.shape(), tensor.device()))

    # initialize MAX graph
    var graph = Graph("realization_graph", input_types)

    # retreive inputs as TensorValues
    var graph_inputs = graph.inputs()

    # set twein tensor values for materialized inputs
    for i in range(len(input_types)):
        materialized_inputs[i].set_tensor_value(graph_inputs[i])

    # iterate through the unmaterialized trace and create tensor values wrt. the maxpr of each operation
    for var tensor in unmaterialized_trace:
        var parent_tvs = List[TensorValue]()
        if tensor.has_parents():
            var parents = tensor.parents()
            for parent in parents:
                parent_tvs.append(parent.tensor_value())
        
        if tensor.has_maxpr():
            var maxpr_fn = tensor.maxpr()
            tensor.set_tensor_value(maxpr_fn(parent_tvs))
            if not tensor.has_tensor_value():
                raise Error("maxpr did not set tensor_value for tensor: " + tensor.name())
        else:
            raise Error("No maxpr defined for tensor: " + tensor.name())

    var outputs_tvs = List[TensorValue]()
    for output in outputs:
        outputs_tvs.append(output.tensor_value())

    graph.output(outputs_tvs)

    var model = graph.compile()
    var input_data = List[MaxTensor]()
    for tensor in materialized_inputs:
        input_data.append(tensor.data())

    # execute the model to realize all output tensors
    var output_data = model.execute(input_data)
    for i in range(len(outputs)):
        outputs[i].set_data(output_data[i])

    # cleanups: reset tensor values in the graph
    for var tensor in materialized_inputs:
        tensor.remove_tensor_value()
    for var tensor in unmaterialized_trace:
        tensor.remove_tensor_value()
    for var output in outputs:
        output.remove_tensor_value()

struct Callable:
    var func: fn(args: List[Tensor]) raises -> List[Tensor]
    var name: String
    var compiled_model: Dict[UInt64, ArcPointer[MaxModel]]
    var compiled: Bool


    fn __init__(out self, func: fn(args: List[Tensor]) raises -> List[Tensor], name: String, compiled: Bool = False) raises:
        self.func = func
        self.name = name
        self.compiled_model = {}
        self.compiled = compiled

    fn compile(mut self) raises:
        self.compiled = True

    fn __call__(mut self, args: List[Tensor]) raises -> List[Tensor]:
        if not self.compiled:
            return self.func(args)
        else:
            # check if indeed all args are materialized
            for arg in args:
                if not arg.has_data():
                    raise Error("All input tensors must be materialized for compiled execution")

            var unmaterialized_outputs = self.func(args)
            # get a hash value based on String(shape) + String(dtype) + String(name) of each tensor in teh trace
            var trace_tuple = get_unmaterialized_trace(unmaterialized_outputs)
            var trace = trace_tuple[0].copy() 
            var trace_inputs = trace_tuple[1].copy()

            var key: UInt64 = 0
            for tensor in trace_inputs:
                var tensor_hash: UInt64 = hash(tensor.name()) + hash(tensor.shape().__str__()) + hash(String(tensor.dtype()))
                key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
            for tensor in trace:
                var tensor_hash: UInt64 = hash(tensor.name()) + hash(tensor.shape().__str__()) + hash(String(tensor.dtype()))
                key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
            key = key % 1000000007
            
            # check for available compiled model
            if not key in self.compiled_model:
                # compile new model
                var input_types = List[TensorType]()
                for tensor in trace_inputs:
                    input_types.append(TensorType(tensor.dtype(), tensor.shape(), tensor.device()))

                var graph = Graph(self.name + "_compiled_model", input_types)
                var graph_inputs = graph.inputs()
                for i in range(len(input_types)):
                    trace_inputs[i].set_tensor_value(graph_inputs[i])
                for var tensor in trace:
                    var parent_tvs = List[TensorValue]()
                    if tensor.has_parents():
                        var parents = tensor.parents()
                        for parent in parents:
                            parent_tvs.append(parent.tensor_value())
                    if tensor.has_maxpr():
                        var maxpr_fn = tensor.maxpr()
                        tensor.set_tensor_value(maxpr_fn(parent_tvs))
                        if not tensor.has_tensor_value():
                            raise Error("maxpr did not set tensor_value for tensor: " + tensor.name())
                    else:
                        raise Error("No maxpr defined for tensor: " + tensor.name())
                var outputs_tvs = List[TensorValue]()
                for output in unmaterialized_outputs:
                    outputs_tvs.append(output.tensor_value())
                graph.output(outputs_tvs)
                self.compiled_model[key] = ArcPointer(graph.compile())
            
            # execute compiled model
            var max_inputs = List[MaxTensor]()
            for arg in args:
                max_inputs.append(arg.data())

            max_outputs = self.compiled_model[key][].execute(max_inputs)
            for i in range(len(unmaterialized_outputs)):
                unmaterialized_outputs[i].set_data(max_outputs[i])

            # cleanups: reset tensor values in the graph
            for var tensor in trace_inputs:
                tensor.remove_tensor_value()
            for var tensor in trace:
                tensor.remove_tensor_value()
            for var output in unmaterialized_outputs:
                output.remove_tensor_value()

            return unmaterialized_outputs^

fn main() raises:

    fn foo(args: List[Tensor]) raises -> List[Tensor]:
        var x = args[0]   # (2, 3)
        var w1 = args[1]  # (3, 5)
        var b1 = args[2]  # (2, 5)
        var w2 = args[3]  # (5, 4)
        var b2 = args[4]  # (2, 4)

        var hidden = x @ w1          # (2, 5)
        var hidden_bias = hidden + b1  # (2, 5)
        var logits = hidden_bias @ w2  # (2, 4)
        var output = logits + b2       # (2, 4)
        return [output]

    var foo_jitted = Callable(foo, "foo_jitted", True)

    for it in range(100):
        print("Iteration: " + String(it))
        var x = ndarange([2, 3])
        var w1 = ones([3, 5])
        var b1 = ones([2, 5])
        var w2 = ones([5, 4])
        var b2 = ones([2, 4])

        var res = foo_jitted([x, w1, b1, w2, b2])
        print(res[0])
