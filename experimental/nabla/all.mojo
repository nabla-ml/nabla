from .max_bindings import (
    MaxTensor,
    MaxTensorValue,
    MaxTensorType,
    MaxDType,
    MaxDevice,
    MaxGraph,
    MaxInferenceSession,
    MaxModel,
    graph_ops,
    _list_to_py_tuple,
    _list_to_py_list,
    _py_tuple_to_mojo_int_list,
    _py_list_to_mojo_int_list,
)
from python import Python, PythonObject
from memory import ArcPointer, UnsafePointer
from time import perf_counter_ns
from builtin._location import __call_location

# helper function for debugging
@always_inline
fn err_loc() -> String:
    return "\n" + String(__call_location())

struct TensorImpl(Copyable, Movable):
    """Unsafe access to internal representation of the Tensor."""

    var shape: List[Int]
    var batch_dims: List[Int]
    var dtype: DType
    var device: MaxDevice
    var name: String
    var parents: List[ArcPointer[TensorImpl]]
    var visited: Bool
    var tensor_value: Optional[MaxTensorValue]
    var maxpr: List[
        fn (List[MaxTensorValue], Dict[String, List[Int]]) raises -> MaxTensorValue
    ]
    var vjp_rule: List[
        fn (List[Tensor], Tensor, Dict[String, List[Int]]) raises -> List[Tensor]
    ]
    var jvp_rule: List[
        fn (List[Tensor], List[Tensor], Dict[String, List[Int]]) raises -> Tensor
    ]
    var traced: Bool
    var requires_grad: Bool
    var tangent: List[ArcPointer[TensorImpl]]
    var cotangent: List[ArcPointer[TensorImpl]]
    var grad: List[ArcPointer[TensorImpl]]
    var stage_realization: Bool
    var custom_kernel_path: String
    var data: List[MaxTensor]
    var kwargs: Dict[String, List[Int]]

    fn __init__(
        out self,
        shape: List[Int],
        dtype: DType = DType.float32,
        stage_realization: Bool = False,
    ) raises:
        self.shape = shape.copy()
        self.batch_dims = List[Int]()
        self.dtype = dtype
        self.device = MaxDevice.cpu()
        self.name = String("zeros")
        self.parents = []
        self.visited = False
        self.tensor_value = Optional[MaxTensorValue](None)
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
            var np_zeros = Python.import_module("numpy").zeros(
                _list_to_py_tuple(shape),
                dtype=MaxDType.from_dtype(dtype).to_numpy_dtype(),
            )
            self.data = [MaxTensor.from_numpy(np_zeros)]
        self.kwargs = {}


struct Tensor(ImplicitlyCopyable, Movable, Writable):
    var _storage: ArcPointer[TensorImpl]

    fn __init__(out self, storage: ArcPointer[TensorImpl]) raises:
        self._storage = storage

    fn __init__(
        out self,
        shape: List[Int],
        dtype: DType = DType.float32,
        stage_realization: Bool = False,
    ) raises:
        self._storage = ArcPointer(TensorImpl(shape, dtype, stage_realization))

    # Getter methods
    fn name(self) -> String:
        return self._storage[].name

    fn shape(self) -> List[Int]:
        return self._storage[].shape.copy()

    fn dtype(self) -> DType:
        return self._storage[].dtype

    fn device(self) -> MaxDevice:
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

    fn tensor_value(self) raises -> MaxTensorValue:
        """Get the tensor value. Raises if not set."""
        if not self._storage[].tensor_value:
            raise Error("\nMaxTensorValue is not set" + err_loc())
        return self._storage[].tensor_value.value()

    fn has_tensor_value(self) -> Bool:
        """Check if tensor_value is set."""
        return self._storage[].tensor_value.__bool__()

    fn data(self) raises -> MaxTensor:
        """Get the underlying MaxTensor data. Raises if not set."""
        if len(self._storage[].data) == 0:
            raise Error("\nData is not set" + err_loc())
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

    fn kwargs(self) -> Dict[String, List[Int]]:
        return self._storage[].kwargs.copy()

    fn __getitem__(self, key: String) raises -> List[Int]:
        if not key in self._storage[].kwargs:
            raise Error("\nMetadata key not found: " + key + err_loc())
        return self._storage[].kwargs[key].copy()

    fn __setitem__(mut self, key: String, value: List[Int]):
        self._storage[].kwargs[key] = value.copy()

    fn to_numpy(self) raises -> PythonObject:
        if not self.has_data():
            raise Error("\nTensor data is not materialized" + err_loc())
        return self.data().to_numpy()

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

    fn set_device(mut self, value: MaxDevice):
        self._storage[].device = value.copy()

    fn set_tensor_value(mut self, value: MaxTensorValue):
        self._storage[].tensor_value = value

    fn remove_tensor_value(mut self):
        self._storage[].tensor_value = Optional[MaxTensorValue](None)

    fn to_numpy_ptr[dtype: DType](self) raises -> UnsafePointer[Scalar[dtype]]:
        """Get an unsafe pointer to the underlying data copy on the Host."""
        if not self.has_data():
            raise Error("\nTensor data is not materialized" + err_loc())
        return self.to_numpy().__array_interface__["data"][0].unsafe_get_as_pointer[dtype]()

    # Getters/Setters for autodiff-related fields
    fn maxpr(
        self,
    ) raises -> fn (
        List[MaxTensorValue], Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        """Get the maxpr rule. Raises if not set."""
        if len(self._storage[].maxpr) == 0:
            raise Error("\nmaxpr rule is not set" + err_loc())
        return self._storage[].maxpr[0]

    fn has_maxpr(self) -> Bool:
        """Check if maxpr rule is set."""
        return len(self._storage[].maxpr) > 0

    fn set_maxpr(
        mut self,
        value: fn (
            List[MaxTensorValue], Dict[String, List[Int]]
        ) raises -> MaxTensorValue,
    ):
        self._storage[].maxpr = [value]

    fn vjp_rule(
        self,
    ) raises -> fn (List[Tensor], Tensor, Dict[String, List[Int]]) raises -> List[
        Tensor
    ]:
        """Get the VJP rule. Raises if not set."""
        if len(self._storage[].vjp_rule) == 0:
            raise Error("\nVJP rule is not set" + err_loc())
        return self._storage[].vjp_rule[0]

    fn has_vjp_rule(self) -> Bool:
        """Check if VJP rule is set."""
        return len(self._storage[].vjp_rule) > 0

    fn set_vjp_rule(
        mut self,
        value: fn (List[Tensor], Tensor, Dict[String, List[Int]]) raises -> List[
            Tensor
        ],
    ):
        self._storage[].vjp_rule = [value]

    fn has_jvp_rule(self) -> Bool:
        """Check if JVP rule is set."""
        return len(self._storage[].jvp_rule) > 0

    fn set_jvp_rule(
        mut self,
        value: fn (
            List[Tensor], List[Tensor], Dict[String, List[Int]]
        ) raises -> Tensor,
    ):
        self._storage[].jvp_rule = [value]

    # Getters/Setters for tangent/cotangent/grad
    fn tangent(self) raises -> Tensor:
        """Get the tangent. Raises if not set."""
        if len(self._storage[].tangent) == 0:
            raise Error("\nTangent is not set" + err_loc())
        return Tensor(self._storage[].tangent[0])

    fn has_tangent(self) -> Bool:
        """Check if tangent is set."""
        return len(self._storage[].tangent) > 0

    fn set_tangent(mut self, value: Tensor):
        self._storage[].tangent = [value._storage]

    fn cotangent(self) raises -> Tensor:
        """Get the cotangent. Raises if not set."""
        if len(self._storage[].cotangent) == 0:
            raise Error("\nCotangent is not set" + err_loc())
        return Tensor(self._storage[].cotangent[0])

    fn has_cotangent(self) -> Bool:
        """Check if cotangent is set."""
        return len(self._storage[].cotangent) > 0

    fn set_cotangent(mut self, value: Tensor):
        self._storage[].cotangent = [value._storage]

    fn grad(self) raises -> Tensor:
        """Get the gradient. Raises if not set."""
        if len(self._storage[].grad) == 0:
            raise Error("\nGradient is not set" + err_loc())
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
            else:
                writer.write(self._storage[].data[0].to_numpy())
        except:
            writer.write("Error while converting Tensor to string")

    @always_inline
    fn __add__(self, other: Self) raises -> Self:
        return add(self, other)

    @always_inline
    fn __sub__(self, other: Self) raises -> Self:
        return sub(self, other)

    @always_inline
    fn __mul__(self, other: Self) raises -> Self:
        return mul(self, other)

    @always_inline
    fn __truediv__(self, other: Self) raises -> Self:
        return div(self, other)

    @always_inline
    fn __matmul__(self, other: Self) raises -> Self:
        return matmul(self, other)

    @always_inline
    fn __neg__(self) raises -> Self:
        return neg(self)


# creation ops
@always_inline
fn full(
    value: Float32, shape: List[Int], dtype: DType = DType.float32
) raises -> Tensor:
    """Create a tensor full of a scalar value."""
    try:
        var tensor = Tensor(shape, dtype, stage_realization=True)
        var np_full = Python.import_module("numpy").full(
            _list_to_py_tuple(shape),
            value,
            dtype=MaxDType.from_dtype(dtype).to_numpy_dtype(),
        )
        tensor.set_data(MaxTensor.from_numpy(np_full))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in full operation: " + String(err) + err_loc())


@always_inline
fn zeros(shape: List[Int], dtype: DType = DType.float32) raises -> Tensor:
    """Create a tensor full of zeros using the full operation."""
    return full(0.0, shape, dtype)


@always_inline
fn ones(shape: List[Int], dtype: DType = DType.float32) raises -> Tensor:
    """Create a tensor full of ones using the full operation."""
    return full(1.0, shape, dtype)


@always_inline
fn arange(
    start: Int, stop: Int, step: Int = 1, dtype: DType = DType.float32
) raises -> Tensor:
    try:
        var np_arange = Python.import_module("numpy").arange(
            start, stop, step, dtype=MaxDType.from_dtype(dtype).to_numpy_dtype()
        )
        var shape = [Int(np_arange.shape[0])]
        var tensor = Tensor(shape, dtype, stage_realization=True)
        tensor.set_data(MaxTensor.from_numpy(np_arange))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in arange operation: " + String(err) + err_loc())


@always_inline
fn ndarange(shape: List[Int], dtype: DType = DType.float32) raises -> Tensor:
    try:
        var end = 1
        for dim in shape:
            end *= dim
        var np_shape = _list_to_py_list(shape)
        var np_arange = (
            Python.import_module("numpy")
            .arange(0, end, 1, dtype=MaxDType.from_dtype(dtype).to_numpy_dtype())
            .reshape(np_shape)
        )
        var tensor = Tensor(shape, dtype, stage_realization=True)
        tensor.set_data(MaxTensor.from_numpy(np_arange))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in ndarange operation: " + String(err) + err_loc())


@always_inline
fn randn(
    shape: List[Int],
    mean: Float32 = 0.0,
    std: Float32 = 1.0,
    dtype: DType = DType.float32,
) raises -> Tensor:
    """Create a tensor with random values from a normal distribution.

    Args:
        shape: The shape of the tensor.
        mean: Mean of the normal distribution (default: 0.0).
        std: Standard deviation of the normal distribution (default: 1.0).
        dtype: Data type of the tensor (default: DType.float32).
    """
    try:
        var np_randn = (
            Python.import_module("numpy")
            .random.normal(mean, std, _list_to_py_tuple(shape))
            .astype(MaxDType.from_dtype(dtype).to_numpy_dtype())
        )
        var tensor = Tensor(shape, dtype, stage_realization=True)
        tensor.set_data(MaxTensor.from_numpy(np_randn))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in randn operation: " + String(err) + err_loc())


@always_inline
fn randu(
    shape: List[Int],
    low: Float32 = 0.0,
    high: Float32 = 1.0,
    dtype: DType = DType.float32,
) raises -> Tensor:
    """Create a tensor with random values from a uniform distribution.

    Args:
        shape: The shape of the tensor.
        low: Lower bound of the uniform distribution (inclusive, default: 0.0).
        high: Upper bound of the uniform distribution (exclusive, default: 1.0).
        dtype: Data type of the tensor (default: DType.float32).
    """
    try:
        var np_randu = (
            Python.import_module("numpy")
            .random.uniform(low, high, _list_to_py_tuple(shape))
            .astype(MaxDType.from_dtype(dtype).to_numpy_dtype())
        )
        var tensor = Tensor(shape, dtype, stage_realization=True)
        tensor.set_data(MaxTensor.from_numpy(np_randu))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in randu operation: " + String(err) + err_loc())


fn is_broadcastable(shape_a: List[Int], shape_b: List[Int]) raises -> Bool:
    """Check if two shapes are broadcastable and return True if they are."""
    var ndim_a = len(shape_a)
    var ndim_b = len(shape_b)
    var max_ndim = max(ndim_a, ndim_b)

    for i in range(max_ndim):
        var dim_a = shape_a[ndim_a - max_ndim + i] if i >= (max_ndim - ndim_a) else 1
        var dim_b = shape_b[ndim_b - max_ndim + i] if i >= (max_ndim - ndim_b) else 1

        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False

    return True


fn broadcast_shapes(shape_a: List[Int], shape_b: List[Int]) raises -> List[Int]:
    """Compute the broadcasted shape of two input shapes."""
    if not is_broadcastable(shape_a, shape_b):
        var msg = "Shapes " + shape_a.__str__() + " and " + shape_b.__str__() + " are not broadcastable"
        raise Error(msg + err_loc())

    var ndim_a = len(shape_a)
    var ndim_b = len(shape_b)
    var max_ndim = max(ndim_a, ndim_b)

    var result_shape = List[Int]()
    for i in range(max_ndim):
        var dim_a = shape_a[ndim_a - max_ndim + i] if i >= (max_ndim - ndim_a) else 1
        var dim_b = shape_b[ndim_b - max_ndim + i] if i >= (max_ndim - ndim_b) else 1
        result_shape.append(max(dim_a, dim_b))

    return result_shape^


trait Operation:
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        pass

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        pass

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        pass

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        pass

    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        pass

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        pass

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        pass

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        pass

    @staticmethod
    fn stage_realization(inputs: List[Tensor]) raises -> Bool:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())
        return inputs[0].stage_realization() or inputs[1].stage_realization()

    @staticmethod
    fn custom_kernel_path() -> String:
        return String("")

    @staticmethod
    fn execute(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> Tensor:
        try:
            var res = Tensor(
                shape=Self.shape(inputs, kwargs),
                dtype=Self.dtype(inputs, kwargs),
                stage_realization=True,
            )
            res.set_name(Self.name(kwargs))
            res.set_parents(inputs)
            res.set_maxpr(Self.maxpr)
            res.set_vjp_rule(Self.vjp_rule)
            res.set_jvp_rule(Self.jvp_rule)
            res.set_custom_kernel_path(Self.custom_kernel_path())
            res.set_parents(inputs)
            # Store kwargs on the tensor so they can be used during realization
            # BUT: don't copy "is_arg" metadata to child tensors
            for key in kwargs.keys():
                if key != "is_arg":
                    res[key] = kwargs[key].copy()
            return res
        except err:
            raise Error(
                "\nError during execution of operation '"
                + Self.name(kwargs)
                + "': "
                + String(err)
            )


trait BinaryOp(Operation):
    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())
        if inputs[0].batch_dims() != inputs[1].batch_dims():
            raise Error(
                "\nInput tensors must have the same batch_dims for BinaryOp, but got "
                + inputs[0].batch_dims().__str__()
                + " and "
                + inputs[1].batch_dims().__str__()
                + err_loc()
            )
        return inputs[0].batch_dims()

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())

        var shape_a = inputs[0].shape()
        var shape_b = inputs[1].shape()

        # Use the helper function to compute broadcasted shape
        return broadcast_shapes(shape_a, shape_b)

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())
        if inputs[0].dtype() != inputs[1].dtype():
            raise Error(
                "\nInput tensors must have the same dtype for BinaryOp, but got "
                + String(inputs[0].dtype())
                + " and "
                + String(inputs[1].dtype())
                + err_loc()
            )
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())
        if inputs[0].device() != inputs[1].device():
            raise Error(
                "\nInput tensors must be on the same device for BinaryOp, but got "
                + inputs[0].device().label()
                + " and "
                + inputs[1].device().label()
                + err_loc()
            )
        return inputs[0].device()

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        pass

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        pass

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        pass


struct AddOp(BinaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "add"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().add(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        return [cotangent, cotangent]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        return tangents[0] + tangents[1]

@always_inline
fn add(a: Tensor, b: Tensor) raises -> Tensor:
    try:
        return AddOp.execute([a, b], {})
    except err:
        raise Error("\nError in Add operation: " + String(err) + err_loc())


struct MulOp(BinaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "mul"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().mul(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        var grad_a = cotangent * primals[1]
        var grad_b = cotangent * primals[0]
        return [grad_a, grad_b]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        var term1 = tangents[0] * primals[1]
        var term2 = primals[0] * tangents[1]
        return term1 + term2

@always_inline
fn mul(a: Tensor, b: Tensor) raises -> Tensor:
    try:
        return MulOp.execute([a, b], {})
    except err:
        raise Error("\nError in Mul operation: " + String(err) + err_loc())


struct Matmul(Operation):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "matmul"

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("\nMatmul requires exactly 2 input tensors" + err_loc())
        var a_shape = inputs[0].shape()
        var b_shape = inputs[1].shape()
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise Error("\nInput tensors must be at least 2D for Matmul" + err_loc())
        if a_shape[-1] != b_shape[-2]:
            raise Error(
                "\nInner dimensions must match for Matmul, but got shapes "
                + a_shape.__str__()
                + " and "
                + b_shape.__str__()
                + err_loc()
            )
        var result_shape = a_shape[0:-1] + b_shape[0:-2] + [b_shape[-1]]
        return result_shape^

    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("\nMatmul requires exactly 2 input tensors" + err_loc())
        if inputs[0].batch_dims() != inputs[1].batch_dims():
            raise Error("\nInput tensors must have the same batch_dims for Matmul" + err_loc())
        return inputs[0].batch_dims()

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 2:
            raise Error("\nMatmul requires exactly 2 input tensors" + err_loc())
        if inputs[0].dtype() != inputs[1].dtype():
            raise Error("\nInput tensors must have the same dtype for Matmul" + err_loc())
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 2:
            raise Error("\nMatmul requires exactly 2 input tensors" + err_loc())
        if inputs[0].device() != inputs[1].device():
            raise Error("\nInput tensors must be on the same device for Matmul" + err_loc())
        return inputs[0].device()

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().matmul(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        raise Error("\nVJP rule for Matmul not implemented yet" + err_loc())

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        raise Error("\nJVP rule for Matmul not implemented yet" + err_loc())

@always_inline
fn matmul(a: Tensor, b: Tensor) raises -> Tensor:
    var err_loc = err_loc()
    try:
        return Matmul.execute([a, b], {})
    except err:
        raise Error("\nError in Matmul operation: " + String(err) + err_loc)


struct SubOp(BinaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "sub"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().sub(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        return [cotangent, -cotangent]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        return sub(tangents[0], tangents[1])

@always_inline
fn sub(a: Tensor, b: Tensor) raises -> Tensor:
    try:
        return SubOp.execute([a, b], {})
    except err:
        raise Error("\nError in Sub operation: " + String(err))


struct DivOp(BinaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "div"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().div(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        return [
            cotangent / primals[1],
            -(cotangent * primals[0]) / (primals[1] * primals[1]),
        ]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        return tangents[0] / primals[1] - (primals[0] * tangents[1]) / (
            primals[1] * primals[1]
        )

@always_inline
fn div(a: Tensor, b: Tensor) raises -> Tensor:
    try:
        return DivOp.execute([a, b], {})
    except err:
        raise Error("\nError in Div operation: " + String(err))


trait UnaryOp(Operation):
    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nUnaryOp requires exactly 1 input tensor" + err_loc())
        return inputs[0].batch_dims()

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nUnaryOp requires exactly 1 input tensor" + err_loc())
        return inputs[0].shape()

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 1:
            raise Error("\nUnaryOp requires exactly 1 input tensor" + err_loc())
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 1:
            raise Error("\nUnaryOp requires exactly 1 input tensor" + err_loc())
        return inputs[0].device()


struct ReluOp(UnaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "relu"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(graph_ops().relu(inputs[0].to_python()))
        

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        raise Error("\nVJP rule for Relu not implemented yet" + err_loc())

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        raise Error("\nJVP rule for Relu not implemented yet" + err_loc())

@always_inline
fn relu(x: Tensor) raises -> Tensor:
    try:
        return ReluOp.execute([x], {})
    except err:
        raise Error("\nError in ReLU operation: " + String(err) + err_loc())


struct NegOp(UnaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "neg"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(graph_ops().neg(inputs[0].to_python()))

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        return [-cotangent]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        return -tangents[0]

@always_inline
fn neg(a: Tensor) raises -> Tensor:
    try:
        return NegOp.execute([a], {})
    except err:
        raise Error("\nError in Neg operation: " + String(err))


struct ReshapeOp(Operation):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "reshape"

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nReshape requires exactly 1 input tensor" + err_loc())
        if not "target_shape" in kwargs:
            var available_keys = List[String]()
            for key in kwargs.keys():
                available_keys.append(key)
            raise Error(
                "\nReshape requires 'target_shape' in kwargs, but it was not found. Available keys: "
                + available_keys.__str__()
                + err_loc()
            )
        return kwargs["target_shape"].copy()

    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nReshape requires exactly 1 input tensor" + err_loc())
        return inputs[0].batch_dims()

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 1:
            raise Error("\nReshape requires exactly 1 input tensor" + err_loc())
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 1:
            raise Error("\nReshape requires exactly 1 input tensor" + err_loc())
        return inputs[0].device()

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        if not "target_shape" in kwargs:
            var available_keys = List[String]()
            for key in kwargs.keys():
                available_keys.append(key)
            raise Error(
                "\nReshape.maxpr requires 'target_shape' in kwargs, but it was not found. Available keys: "
                + available_keys.__str__()
                + err_loc()
            )
        var target_shape = kwargs["target_shape"].copy()
        return MaxTensorValue(
            graph_ops().reshape(inputs[0].to_python(), _list_to_py_tuple(target_shape))
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        # Gradient of reshape is just reshaping back to original shape
        var original_shape = primals[0].shape()
        var grad_kwargs = Dict[String, List[Int]]()
        grad_kwargs["target_shape"] = original_shape.copy()
        var grad = ReshapeOp.execute([cotangent], grad_kwargs)
        return [grad]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        # Forward-mode AD for reshape is just reshaping the tangent
        return ReshapeOp.execute([tangents[0]], kwargs)

@always_inline
fn reshape(x: Tensor, target_shape: List[Int]) raises -> Tensor:
    try:
        var kwargs = Dict[String, List[Int]]()
        kwargs["target_shape"] = target_shape.copy()
        return ReshapeOp.execute([x], kwargs)
    except err:
        raise Error("\nError in Reshape operation: " + String(err))


struct BroadcastOp(Operation):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "broadcast"

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nBroadcast requires exactly 1 input tensor" + err_loc())
        if not "target_shape" in kwargs:
            raise Error("\nBroadcast requires 'target_shape' in kwargs" + err_loc())

        var input_shape = inputs[0].shape()
        var target_shape = kwargs["target_shape"].copy()

        # Check if input can be broadcast to target
        if not is_broadcastable(input_shape, target_shape):
            raise Error(
                "\nInput shape "
                + input_shape.__str__()
                + " cannot be broadcast to target shape "
                + target_shape.__str__()
                + err_loc()
            )

        return target_shape.copy()

    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nBroadcast requires exactly 1 input tensor" + err_loc())
        return inputs[0].batch_dims()

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 1:
            raise Error("\nBroadcast requires exactly 1 input tensor" + err_loc())
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 1:
            raise Error("\nBroadcast requires exactly 1 input tensor" + err_loc())
        return inputs[0].device()

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        if not "target_shape" in kwargs:
            raise Error("\nBroadcast requires 'target_shape' in kwargs" + err_loc())
        var target_shape = kwargs["target_shape"].copy()
        # return bindings_broadcast_to(inputs[0], target_shape)

        return MaxTensorValue(
            graph_ops().broadcast_to(
                inputs[0].to_python(), _list_to_py_tuple(target_shape)
            )
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        # Gradient of broadcast is sum reduction over the broadcast dimensions
        # For now, just return the cotangent (simplified)
        raise Error("\nVJP rule for Broadcast not implemented yet" + err_loc())

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        # Forward-mode AD for broadcast is just broadcasting the tangent
        return BroadcastOp.execute([tangents[0]], kwargs)

@always_inline
fn broadcast(x: Tensor, target_shape: List[Int]) raises -> Tensor:
    try:
        var kwargs = Dict[String, List[Int]]()
        kwargs["target_shape"] = target_shape.copy()
        return BroadcastOp.execute([x], kwargs)
    except err:
        raise Error("\nError in Broadcast operation: " + String(err))


fn reset_visited(mut tensors: List[Tensor]) raises:
    """Reset visited flag for entire graph."""
    for var tensor in tensors:
        tensor.set_visited(False)
        var parents = tensor.parents()
        reset_visited(parents)


fn print_trace(trace: List[Tensor]) raises -> None:
    for t in trace:
        print(t.name())


fn get_dependencies_recursive(
    mut outputs: List[Tensor], mut trace: List[Tensor], mut inputs: List[Tensor]
) raises -> None:
    for var output in outputs:
        if not output.visited():
            if output.has_parents():
                output.set_visited(True)
                var parents = output.parents()
                get_dependencies_recursive(parents, trace, inputs)
                trace.append(output)
            else:
                inputs.append(output)


fn get_unmaterialized_recursive(
    mut outputs: List[Tensor], mut trace: List[Tensor], mut inputs: List[Tensor]
) raises -> None:
    for var output in outputs:
        if not output.visited():
            if output.has_data():
                inputs.append(output)
            else:
                output.set_visited(True)
                var parents = output.parents()
                get_unmaterialized_recursive(parents, trace, inputs)
                trace.append(output)


fn get_unmaterialized_recursive_with_constants(
    mut outputs: List[Tensor],
    mut trace: List[Tensor],
    mut inputs: List[Tensor],
    mut constants: List[Tensor],
) raises -> None:
    """DFS that separates materialized inputs into args (marked with 'is_arg') and constants.
    """
    for var output in outputs:
        if not output.visited():
            if output.has_data():
                # Check if this tensor is marked as an arg
                if "is_arg" in output.kwargs():
                    inputs.append(output)
                else:
                    constants.append(output)
            else:
                output.set_visited(True)
                var parents = output.parents()
                get_unmaterialized_recursive_with_constants(
                    parents, trace, inputs, constants
                )
                trace.append(output)


fn get_dependency_trace(
    outputs: List[Tensor],
) raises -> Tuple[List[Tensor], List[Tensor]]:
    var trace: List[Tensor] = []
    var inputs: List[Tensor] = []
    var _outputs = outputs.copy()
    reset_visited(_outputs)
    get_dependencies_recursive(_outputs, trace, inputs)
    reset_visited(_outputs)
    return (trace^, inputs^)


fn get_unmaterialized_trace(
    outputs: List[Tensor],
) raises -> Tuple[List[Tensor], List[Tensor]]:
    var trace: List[Tensor] = []
    var inputs: List[Tensor] = []
    var _outputs = outputs.copy()
    reset_visited(_outputs)
    get_unmaterialized_recursive(_outputs, trace, inputs)
    reset_visited(_outputs)
    return (trace^, inputs^)


fn get_unmaterialized_trace_with_constants(
    outputs: List[Tensor],
) raises -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Get trace separating inputs (marked with 'is_arg') from constants (created inside function).
    """
    var trace: List[Tensor] = []
    var inputs: List[Tensor] = []
    var constants: List[Tensor] = []
    var _outputs = outputs.copy()
    reset_visited(_outputs)
    get_unmaterialized_recursive_with_constants(_outputs, trace, inputs, constants)
    reset_visited(_outputs)
    return (trace^, inputs^, constants^)


fn realize(mut unmaterialized_outputs: List[Tensor], ctx: ArcPointer[Dict[UInt64, ArcPointer[MaxModel]]]) raises:
    """Realize all tensors' data."""
    # get trace with constants separated
    var trace_tuple = get_unmaterialized_trace_with_constants(
        unmaterialized_outputs
    )
    var trace = trace_tuple[0].copy()
    var constants = trace_tuple[2].copy()

    var key: UInt64 = 0
    for var constant in constants:
        var tensor_hash: UInt64 = (
            hash(constant.name())
            + hash(constant.shape().__str__())
            + hash(String(constant.dtype()))
        )
        key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
    for tensor in trace:
        var tensor_hash: UInt64 = (
            hash(tensor.name())
            + hash(tensor.shape().__str__())
            + hash(String(tensor.dtype()))
        )
        key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
    key = key % 1000000007

    # check for available compiled model
    if not key in ctx[]:
        # compile new model using inputs (args + constants) as graph inputs
        var input_types = List[MaxTensorType]()
        # Then add constants
        for constant in constants:
            input_types.append(
                MaxTensorType(
                    MaxDType.from_dtype(constant.dtype()),
                    constant.shape(),
                    constant.device(),
                )
            )

        var graph = MaxGraph("compiled_model", input_types)

        with graph:
            var graph_inputs = graph.inputs()

            # Set tensor values for all inputs (args first, then constants)
            var idx = 0
            for i in range(len(constants)):
                constants[i].set_tensor_value(graph_inputs[idx])
                idx += 1

            # Now build the computation graph
            try:
                for var tensor in trace:
                    var parent_tvs = List[MaxTensorValue]()
                    if tensor.has_parents():
                        var parents = tensor.parents()
                        for i in range(len(parents)):
                            parent_tvs.append(parents[i].tensor_value())
                    if tensor.has_maxpr():
                        var maxpr_fn = tensor.maxpr()
                        tensor.set_tensor_value(
                            maxpr_fn(parent_tvs, tensor.kwargs())
                        )
                        if not tensor.has_tensor_value():
                            raise Error(
                                "maxpr did not set tensor_value for tensor: "
                                + tensor.name()
                            )
                    else:
                        raise Error(
                            "No maxpr defined for tensor: " + tensor.name()
                        )
                var outputs_tvs = List[MaxTensorValue]()
                for output in unmaterialized_outputs:
                    outputs_tvs.append(output.tensor_value())
                graph.output(outputs_tvs)
            except err:
                raise Error(
                    "Failed to compile function '"
                    + "': "
                    + String(err)
                )

        # Create MAX inference session and load the compiled graph
        var session = MaxInferenceSession([MaxDevice.cpu()])
        ctx[][key] = ArcPointer(session.load(graph))

        # Clean up tensor values from constants after compilation
        for var constant in constants:
            constant.remove_tensor_value()

    # execute compiled model with args + constants
    var max_inputs = List[MaxTensor]()
    # Add the constants data
    for i in range(len(constants)):
        max_inputs.append(constants[i].data())

    max_outputs = ctx[][key][].execute(max_inputs)
    for i in range(len(unmaterialized_outputs)):
        unmaterialized_outputs[i].set_data(max_outputs[i])

    # cleanups: reset tensor values in the graph
    for var tensor in trace:
        tensor.remove_tensor_value()
    for var output in unmaterialized_outputs:
        output.remove_tensor_value()


struct Callable(Copyable, Movable):
    var func: fn (args: List[Tensor]) raises -> List[Tensor]
    var name: String
    var compiled_model: Dict[UInt64, ArcPointer[MaxModel]]
    var constants: Dict[UInt64, List[Tensor]]
    var compiled: Bool

    fn __init__(
        out self,
        func: fn (args: List[Tensor]) raises -> List[Tensor],
        name: String,
        compiled: Bool = False,
    ) raises:
        self.func = func
        self.name = name
        self.compiled_model = {}
        self.constants = {}
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
                    raise Error(
                        "All input tensors must be materialized for compiled execution" + err_loc()
                    )

            # Mark all args with "is_arg" metadata
            # NOTE: args uses ArcPointer, so modifying through the list modifies the shared storage
            var mut_args = List[Tensor]()
            for i in range(len(args)):
                # Create a wrapper that marks this specific arg
                var arg = args[i]
                arg["is_arg"] = List[Int]()
                mut_args.append(arg)

            var unmaterialized_outputs = self.func(mut_args)
            # get trace with constants separated
            var trace_tuple = get_unmaterialized_trace_with_constants(
                unmaterialized_outputs
            )
            var trace = trace_tuple[0].copy()
            # var inputs = trace_tuple[1].copy()
            var constants = trace_tuple[2].copy()

            var key: UInt64 = 0
            # Hash based on the original args
            for arg in args:
                var tensor_hash: UInt64 = (
                    hash(arg.name())
                    + hash(arg.shape().__str__())
                    + hash(String(arg.dtype()))
                )
                key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
            for var constant in constants:
                var tensor_hash: UInt64 = (
                    hash(constant.name())
                    + hash(constant.shape().__str__())
                    + hash(String(constant.dtype()))
                )
                key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
            for tensor in trace:
                var tensor_hash: UInt64 = (
                    hash(tensor.name())
                    + hash(tensor.shape().__str__())
                    + hash(String(tensor.dtype()))
                )
                key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
            key = key % 1000000007

            # check for available compiled model
            if not key in self.compiled_model:
                # Store constants for this compiled model
                self.constants[key] = constants.copy()

                # compile new model using inputs (args + constants) as graph inputs
                var input_types = List[MaxTensorType]()
                # First add args
                for arg in args:
                    input_types.append(
                        MaxTensorType(
                            MaxDType.from_dtype(arg.dtype()), arg.shape(), arg.device()
                        )
                    )
                # Then add constants
                for constant in constants:
                    input_types.append(
                        MaxTensorType(
                            MaxDType.from_dtype(constant.dtype()),
                            constant.shape(),
                            constant.device(),
                        )
                    )

                var graph = MaxGraph(self.name + "_compiled_model", input_types)

                with graph:
                    var graph_inputs = graph.inputs()

                    # Set tensor values for all inputs (args first, then constants)
                    var idx = 0
                    var _args = args.copy()
                    for i in range(len(args)):
                        _args[i].set_tensor_value(graph_inputs[idx])
                        idx += 1
                    for i in range(len(constants)):
                        constants[i].set_tensor_value(graph_inputs[idx])
                        idx += 1

                    # Now build the computation graph
                    try:
                        for var tensor in trace:
                            var parent_tvs = List[MaxTensorValue]()
                            if tensor.has_parents():
                                var parents = tensor.parents()
                                for i in range(len(parents)):
                                    parent_tvs.append(parents[i].tensor_value())
                            if tensor.has_maxpr():
                                var maxpr_fn = tensor.maxpr()
                                tensor.set_tensor_value(
                                    maxpr_fn(parent_tvs, tensor.kwargs())
                                )
                                if not tensor.has_tensor_value():
                                    raise Error(
                                        "maxpr did not set tensor_value for tensor: "
                                        + tensor.name()
                                    )
                            else:
                                raise Error(
                                    "No maxpr defined for tensor: " + tensor.name()
                                )
                        var outputs_tvs = List[MaxTensorValue]()
                        for output in unmaterialized_outputs:
                            outputs_tvs.append(output.tensor_value())
                        graph.output(outputs_tvs)
                    except err:
                        raise Error(
                            "Failed to compile function '"
                            + self.name
                            + "': "
                            + String(err)
                        )

                # Create MAX inference session and load the compiled graph
                var session = MaxInferenceSession([MaxDevice.cpu()])
                self.compiled_model[key] = ArcPointer(session.load(graph))

                # Clean up tensor values from inputs and constants after compilation
                for var arg in args:
                    arg.remove_tensor_value()
                for var constant in constants:
                    constant.remove_tensor_value()

            # execute compiled model with args + constants
            var max_inputs = List[MaxTensor]()
            # First add the args data
            for arg in args:
                max_inputs.append(arg.data())
            # Then add the constants data
            if key in self.constants:
                for i in range(len(self.constants[key])):
                    max_inputs.append(self.constants[key][i].data())

            max_outputs = self.compiled_model[key][].execute(max_inputs)
            for i in range(len(unmaterialized_outputs)):
                unmaterialized_outputs[i].set_data(max_outputs[i])

            # cleanups: reset tensor values in the graph
            for var tensor in trace:
                tensor.remove_tensor_value()
            for var output in unmaterialized_outputs:
                output.remove_tensor_value()

            return unmaterialized_outputs^

fn jit(func: fn (args: List[Tensor]) raises -> List[Tensor]) raises -> Callable:
    return Callable(func, "func", True)

trait Module(Copyable, Movable):
    fn __call__(self, args: List[Tensor]) raises -> List[Tensor]:
        ...

    fn params(self) raises -> List[Tensor]:
        ...

    fn ctx(self) raises -> ArcPointer[Dict[UInt64, ArcPointer[MaxModel]]]:
        ...

    fn execute(self, args: List[Tensor]) raises -> List[Tensor]:
        var all_args_have_data = True
        for arg in args:
            if not arg.has_data():
                all_args_have_data = False

        var outputs = self.__call__(args)

        if all_args_have_data:
            realize(outputs, self.ctx())

        return outputs^


struct Linear(Module):
    var weight: Tensor
    var bias: Tensor
    var _ctx: ArcPointer[Dict[UInt64, ArcPointer[MaxModel]]]

    fn __init__(out self, in_dim: Int, out_dim: Int) raises:
        self.weight = randn([in_dim, out_dim])
        self.bias = randn([out_dim])
        self._ctx = ArcPointer(Dict[UInt64, ArcPointer[MaxModel]]())

    fn __call__(self, args: List[Tensor]) raises -> List[Tensor]:
        return [relu(args[0] @ self.weight + self.bias)]

    fn params(self) raises -> List[Tensor]:
        return [self.weight, self.bias]

    fn ctx(self) raises -> ArcPointer[Dict[UInt64, ArcPointer[MaxModel]]]:
        return self._ctx


struct MLP(Module):
    var layers: List[Linear]
    var _ctx: ArcPointer[Dict[UInt64, ArcPointer[MaxModel]]]

    fn __init__(out self, layer_dims: List[Int]) raises:
        self.layers = []
        for i in range(1, len(layer_dims)):
            self.layers.append(Linear(layer_dims[i-1], layer_dims[i]))
        self._ctx = ArcPointer(Dict[UInt64, ArcPointer[MaxModel]]())

    fn __call__(self, args: List[Tensor]) raises -> List[Tensor]:
        var hidden = args.copy()
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden^

    fn params(self) raises -> List[Tensor]:
        var layer_params = List[Tensor]()
        for layer in self.layers:
            layer_params.extend(layer.params())
        return layer_params^

    fn ctx(self) raises -> ArcPointer[Dict[UInt64, ArcPointer[MaxModel]]]:
        return self._ctx
