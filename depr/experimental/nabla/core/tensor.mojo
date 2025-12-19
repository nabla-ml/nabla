from ..utils.max_bindings import (
    MaxTensor,
    MaxTensorValue,
    MaxDType,
    MaxDevice,
    _list_to_py_tuple,
    _list_to_py_list,
)
from ..utils.debug import err_loc
from python import Python, PythonObject
from memory import ArcPointer, UnsafePointer


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
        return (
            self.to_numpy()
            .__array_interface__["data"][0]
            .unsafe_get_as_pointer[dtype]()
        )

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

    # Operator overloads
    fn __add__(self, other: Tensor) raises -> Tensor:
        """Add two tensors using the + operator."""
        from ..ops.math import add

        return add(self, other)

    fn __sub__(self, other: Tensor) raises -> Tensor:
        """Subtract two tensors using the - operator."""
        from ..ops.math import sub

        return sub(self, other)

    fn __mul__(self, other: Tensor) raises -> Tensor:
        """Multiply two tensors element-wise using the * operator."""
        from ..ops.math import mul

        return mul(self, other)

    fn __truediv__(self, other: Tensor) raises -> Tensor:
        """Divide two tensors element-wise using the / operator."""
        from ..ops.math import div

        return div(self, other)

    fn __matmul__(self, other: Tensor) raises -> Tensor:
        """Matrix multiplication using the @ operator."""
        from ..ops.math import matmul

        return matmul(self, other)

    fn __neg__(self) raises -> Tensor:
        """Negate a tensor using the - unary operator."""
        from ..ops.math import neg

        return neg(self)
