"""
Simplified MAX Engine bindings for Mojo.
Direct wrappers around max.graph, max.driver, max.dtype Python APIs.
"""

from python import Python, PythonObject


# ============================================================================
# UTILITIES
# ============================================================================


fn _list_to_py_tuple(lst: List[Int]) raises -> PythonObject:
    """Convert Mojo List[Int] to Python tuple."""
    var builtins = Python.import_module("builtins")
    var py_list = builtins.list()
    for i in range(len(lst)):
        _ = py_list.append(lst[i])
    return builtins.tuple(py_list)


fn _list_to_py_list(lst: List[Int]) raises -> PythonObject:
    """Convert Mojo List[Int] to Python list."""
    var builtins = Python.import_module("builtins")
    var py_list = builtins.list()
    for i in range(len(lst)):
        _ = py_list.append(lst[i])
    return py_list


fn _py_tuple_to_mojo_int_list(py_tuple: PythonObject) raises -> List[Int]:
    """Convert Python tuple to Mojo List[Int]."""
    var result = List[Int]()
    for item in py_tuple:
        result.append(Int(item))
    return result^


fn _py_list_to_mojo_int_list(py_list: PythonObject) raises -> List[Int]:
    """Convert Python list to Mojo List[Int]."""
    var result = List[Int]()
    for item in py_list:
        result.append(Int(item))
    return result^


# ============================================================================
# DTYPE
# ============================================================================


struct MaxDType(ImplicitlyCopyable, Movable):
    """Wrapper for max.dtype.DType."""

    var _py_dtype: PythonObject

    fn __init__(out self, py_dtype: PythonObject):
        """Wrap existing MAX DType."""
        self._py_dtype = py_dtype

    fn to_python(self) -> PythonObject:
        """Get underlying Python DType."""
        return self._py_dtype

    @staticmethod
    fn float16() raises -> MaxDType:
        """Half precision float."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.float16)

    @staticmethod
    fn float32() raises -> MaxDType:
        """Single precision float."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.float32)

    @staticmethod
    fn float64() raises -> MaxDType:
        """Double precision float."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.float64)

    @staticmethod
    fn bfloat16() raises -> MaxDType:
        """Brain float 16."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.bfloat16)

    @staticmethod
    fn int8() raises -> MaxDType:
        """8-bit signed integer."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.int8)

    @staticmethod
    fn int16() raises -> MaxDType:
        """16-bit signed integer."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.int16)

    @staticmethod
    fn int32() raises -> MaxDType:
        """32-bit signed integer."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.int32)

    @staticmethod
    fn int64() raises -> MaxDType:
        """64-bit signed integer."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.int64)

    @staticmethod
    fn uint8() raises -> MaxDType:
        """8-bit unsigned integer."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.uint8)

    @staticmethod
    fn uint16() raises -> MaxDType:
        """16-bit unsigned integer."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.uint16)

    @staticmethod
    fn uint32() raises -> MaxDType:
        """32-bit unsigned integer."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.uint32)

    @staticmethod
    fn uint64() raises -> MaxDType:
        """64-bit unsigned integer."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.uint64)

    @staticmethod
    fn bool_type() raises -> MaxDType:
        """Boolean dtype."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType.bool)

    @staticmethod
    fn from_numpy_dtype(np_dtype: PythonObject) raises -> MaxDType:
        """Convert from numpy dtype."""
        var dtype_mod = Python.import_module("max.dtype")
        return MaxDType(dtype_mod.DType(np_dtype))

    fn to_numpy_dtype(self) raises -> PythonObject:
        """Convert to numpy dtype."""
        var np = Python.import_module("numpy")
        # Get the string representation of the MAX dtype and convert to numpy
        var dtype_str = String(self._py_dtype.__str__())
        # MAX dtypes like "DType.float32" need to be converted to "float32"
        if "." in dtype_str:
            var parts = dtype_str.split(".")
            dtype_str = String(parts[len(parts) - 1])
        return np.dtype(dtype_str)

    @staticmethod
    fn from_dtype(mojo_dtype: DType) raises -> MaxDType:
        """Convert from Mojo's built-in DType to MaxDType.

        Args:
            mojo_dtype: Mojo's native DType value.

        Returns:
            Corresponding MaxDType wrapper.
        """
        if mojo_dtype == DType.float16:
            return MaxDType.float16()
        elif mojo_dtype == DType.float32:
            return MaxDType.float32()
        elif mojo_dtype == DType.float64:
            return MaxDType.float64()
        elif mojo_dtype == DType.bfloat16:
            return MaxDType.bfloat16()
        elif mojo_dtype == DType.int8:
            return MaxDType.int8()
        elif mojo_dtype == DType.int16:
            return MaxDType.int16()
        elif mojo_dtype == DType.int32:
            return MaxDType.int32()
        elif mojo_dtype == DType.int64:
            return MaxDType.int64()
        elif mojo_dtype == DType.uint8:
            return MaxDType.uint8()
        elif mojo_dtype == DType.uint16:
            return MaxDType.uint16()
        elif mojo_dtype == DType.uint32:
            return MaxDType.uint32()
        elif mojo_dtype == DType.uint64:
            return MaxDType.uint64()
        elif mojo_dtype == DType.bool:
            return MaxDType.bool_type()
        else:
            raise Error("Unsupported Mojo DType")


# ============================================================================
# DEVICE
# ============================================================================


struct MaxDevice(ImplicitlyCopyable, Movable):
    """Wrapper for max.driver device (CPU or Accelerator)."""

    var _py_device: PythonObject

    fn __init__(out self, py_device: PythonObject):
        """Wrap existing MAX device."""
        self._py_device = py_device

    fn to_python(self) -> PythonObject:
        """Get underlying Python device."""
        return self._py_device

    fn id(self) raises -> Int:
        """Get device ID (0 for CPU, 0+ for GPUs)."""
        return Int(self._py_device.id)

    fn label(self) raises -> String:
        """Get device label ('cpu' or 'gpu')."""
        return String(self._py_device.label)

    fn api(self) raises -> String:
        """Get API ('cpu', 'cuda', 'hip')."""
        return String(self._py_device.api)

    fn architecture_name(self) raises -> String:
        """Get architecture ('sm_80', 'gfx90a', etc.) - raises for CPU."""
        return String(self._py_device.architecture_name)

    fn is_host(self) raises -> Bool:
        """Check if CPU device."""
        return Bool(self._py_device.is_host)

    fn is_compatible(self) raises -> Bool:
        """Check if compatible with MAX."""
        return Bool(self._py_device.is_compatible)

    fn stats(self) raises -> PythonObject:
        """Get device utilization stats (dict)."""
        return self._py_device.stats

    fn synchronize(self) raises:
        """Wait for all operations to complete."""
        _ = self._py_device.synchronize()

    fn can_access(self, other: MaxDevice) raises -> Bool:
        """Check peer memory access capability."""
        return Bool(self._py_device.can_access(other.to_python()))

    fn default_stream(self) raises -> MaxDeviceStream:
        """Get default execution stream."""
        return MaxDeviceStream(self._py_device.default_stream)

    @staticmethod
    fn cpu(id: Int = -1) raises -> MaxDevice:
        """Create CPU device."""
        var driver = Python.import_module("max.driver")
        return MaxDevice(driver.CPU(id=id))

    @staticmethod
    fn accelerator(id: Int = -1, device_memory_limit: Int = -1) raises -> MaxDevice:
        """Create GPU device."""
        var driver = Python.import_module("max.driver")
        return MaxDevice(
            driver.Accelerator(id=id, device_memory_limit=device_memory_limit)
        )

    fn __eq__(self, other: MaxDevice) raises -> Bool:
        """Equality check."""
        return Bool(self._py_device == other.to_python())

    fn __ne__(self, other: MaxDevice) raises -> Bool:
        """Inequality check."""
        return Bool(self._py_device != other.to_python())


struct MaxDeviceStream(Movable):
    """Wrapper for max.driver.DeviceStream."""

    var _py_stream: PythonObject

    fn __init__(out self, py_stream: PythonObject):
        """Wrap existing stream."""
        self._py_stream = py_stream

    fn __init__(out self, device: MaxDevice) raises:
        """Create new stream on device."""
        self._py_stream = device.to_python().create_stream()

    fn to_python(self) -> PythonObject:
        """Get underlying stream."""
        return self._py_stream

    fn device(self) raises -> MaxDevice:
        """Get associated device."""
        return MaxDevice(self._py_stream.device)

    fn wait_for_stream(self, other: MaxDeviceStream) raises:
        """Wait for another stream."""
        _ = self._py_stream.wait_for_stream(other.to_python())


# ============================================================================
# TENSORS
# ============================================================================


struct MaxTensor(ImplicitlyCopyable, Movable):
    """Wrapper for max.driver.Tensor."""

    var _py_tensor: PythonObject

    fn __init__(out self, py_tensor: PythonObject):
        """Wrap existing MAX tensor."""
        self._py_tensor = py_tensor

    fn __init__(
        out self,
        shape: List[Int],
        dtype: MaxDType,
        device: MaxDevice,
        pinned: Bool = False,
    ) raises:
        """Create new tensor."""
        var driver = Python.import_module("max.driver")
        var py_shape = _list_to_py_list(shape)
        self._py_tensor = driver.Tensor(
            dtype.to_python(), py_shape, device.to_python(), pinned
        )

    fn to_python(self) -> PythonObject:
        """Get underlying Python tensor."""
        return self._py_tensor

    fn shape(self) raises -> List[Int]:
        """Get shape as Mojo List."""
        return _py_tuple_to_mojo_int_list(self._py_tensor.shape)

    fn dtype(self) raises -> MaxDType:
        """Get dtype."""
        return MaxDType(self._py_tensor.dtype)

    fn device(self) raises -> MaxDevice:
        """Get device tensor resides on."""
        return MaxDevice(self._py_tensor.device)

    fn rank(self) raises -> Int:
        """Get tensor rank (number of dimensions)."""
        return Int(self._py_tensor.rank)

    fn num_elements(self) raises -> Int:
        """Get total number of elements."""
        return Int(self._py_tensor.num_elements)

    fn element_size(self) raises -> Int:
        """Get size of element type in bytes."""
        return Int(self._py_tensor.element_size)

    fn is_contiguous(self) raises -> Bool:
        """Check if tensor is contiguous in memory."""
        return Bool(self._py_tensor.is_contiguous)

    fn is_host(self) raises -> Bool:
        """Check if tensor is on CPU."""
        return Bool(self._py_tensor.is_host)

    fn pinned(self) raises -> Bool:
        """Check if memory is pinned."""
        return Bool(self._py_tensor.pinned)

    fn to_numpy(self) raises -> PythonObject:
        """Convert to numpy array (must be on host)."""
        return self._py_tensor.to_numpy()

    fn item(self) raises -> PythonObject:
        """Get scalar value (for rank-0 tensors only)."""
        return self._py_tensor.item()

    fn copy(self, device: MaxDevice) raises -> MaxTensor:
        """Deep copy to device."""
        var driver = Python.import_module("max.driver")
        var tensor_class = driver.Tensor
        var copy_fn = self._py_tensor.__getattribute__("copy")
        return MaxTensor(copy_fn(device.to_python()))

    fn to(self, device: MaxDevice) raises -> MaxTensor:
        """Ensure tensor is on device (copy if needed)."""
        return MaxTensor(self._py_tensor.to(device.to_python()))

    fn contiguous(self) raises -> MaxTensor:
        """Create contiguous copy."""
        return MaxTensor(self._py_tensor.contiguous())

    fn inplace_copy_from(self, src: MaxTensor) raises:
        """Copy contents from another tensor."""
        _ = self._py_tensor.inplace_copy_from(src.to_python())

    fn view(self, dtype: MaxDType, shape: List[Int]) raises -> MaxTensor:
        """View with different dtype/shape."""
        var py_shape = _list_to_py_list(shape)
        return MaxTensor(self._py_tensor.view(dtype.to_python(), py_shape))

    @staticmethod
    fn from_numpy(np_array: PythonObject) raises -> MaxTensor:
        """Create from numpy (data not copied unless non-contiguous)."""
        var driver = Python.import_module("max.driver")
        return MaxTensor(driver.Tensor.from_numpy(np_array))

    @staticmethod
    fn from_dlpack(array: PythonObject) raises -> MaxTensor:
        """Create from DLPack object."""
        var driver = Python.import_module("max.driver")
        return MaxTensor(driver.Tensor.from_dlpack(array))

    @staticmethod
    fn zeros(shape: List[Int], dtype: MaxDType, device: MaxDevice) raises -> MaxTensor:
        """Create zeros tensor."""
        var driver = Python.import_module("max.driver")
        var py_shape = _list_to_py_list(shape)
        return MaxTensor(
            driver.Tensor.zeros(py_shape, dtype.to_python(), device.to_python())
        )

    @staticmethod
    fn scalar(
        value: PythonObject, dtype: MaxDType, device: MaxDevice
    ) raises -> MaxTensor:
        """Create scalar (rank-0) tensor."""
        var driver = Python.import_module("max.driver")
        return MaxTensor(
            driver.Tensor.scalar(value, dtype.to_python(), device.to_python())
        )


struct MaxTensorValue(ImplicitlyCopyable, Movable):
    """Wrapper for max.graph.TensorValue (symbolic tensor in graph)."""

    var _py_value: PythonObject

    fn __init__(out self, py_value: PythonObject):
        """Wrap existing MAX TensorValue."""
        self._py_value = py_value

    fn to_python(self) -> PythonObject:
        """Get underlying Python value."""
        return self._py_value

    fn shape(self) raises -> PythonObject:
        """Get symbolic shape."""
        return self._py_value.shape

    fn dtype(self) raises -> PythonObject:
        """Get dtype."""
        return self._py_value.dtype


struct MaxTensorType(ImplicitlyCopyable, Movable):
    """Wrapper for max.graph.TensorType (type specification for graph inputs)."""

    var _py_type: PythonObject

    fn __init__(out self, dtype: MaxDType, shape: List[Int], device: MaxDevice) raises:
        """Create type spec."""
        var max_graph = Python.import_module("max.graph")
        var py_shape = _list_to_py_tuple(shape)
        self._py_type = max_graph.TensorType(
            dtype.to_python(), shape=py_shape, device=device.to_python()
        )

    fn to_python(self) -> PythonObject:
        """Get underlying Python type."""
        return self._py_type


# ============================================================================
# GRAPH & MODEL
# ============================================================================


struct MaxGraph(ImplicitlyCopyable, Movable):
    """Wrapper for max.graph.Graph (computation graph builder)."""

    var _py_graph: PythonObject
    var name: String

    fn __init__(out self, name: String, input_types: List[MaxTensorType]) raises:
        """Create graph."""
        var max_graph = Python.import_module("max.graph")
        var builtins = Python.import_module("builtins")

        # Convert input types to Python list
        var py_input_types = builtins.list()
        for i in range(len(input_types)):
            _ = py_input_types.append(input_types[i].to_python())

        self.name = name
        self._py_graph = max_graph.Graph(name, input_types=py_input_types)

    fn to_python(self) -> PythonObject:
        """Get underlying Python graph."""
        return self._py_graph

    fn inputs(self) raises -> List[MaxTensorValue]:
        """Get input tensors."""
        var py_inputs = self._py_graph.inputs
        var result = List[MaxTensorValue]()
        for inp in py_inputs:
            result.append(MaxTensorValue(inp))
        return result^

    fn output(self, outputs: List[MaxTensorValue]) raises:
        """Set graph outputs.

        Args:
            outputs: List of output tensors to return from the graph.
        """
        # Unpack the list and call graph.output() with positional args
        if len(outputs) == 1:
            _ = self._py_graph.output(outputs[0].to_python())
        elif len(outputs) == 2:
            _ = self._py_graph.output(outputs[0].to_python(), outputs[1].to_python())
        elif len(outputs) == 3:
            _ = self._py_graph.output(
                outputs[0].to_python(), outputs[1].to_python(), outputs[2].to_python()
            )
        elif len(outputs) == 4:
            _ = self._py_graph.output(
                outputs[0].to_python(),
                outputs[1].to_python(),
                outputs[2].to_python(),
                outputs[3].to_python(),
            )
        else:
            raise Error("output() supports up to 4 outputs currently")

    fn __enter__(self) raises:
        """Enter graph context - enables with statement pattern."""
        _ = self._py_graph.__enter__()

    fn __exit__(self) raises:
        """Exit graph context - enables with statement pattern."""
        _ = self._py_graph.__exit__(None, None, None)

    fn enter_context(self) raises:
        """Enter graph context (explicit alternative to __enter__)."""
        self.__enter__()

    fn exit_context(self) raises:
        """Exit graph context (explicit alternative to __exit__)."""
        self.__exit__()


# helper to get the max.graph.ops module
fn graph_ops() raises -> PythonObject:
    """Get the max.graph.ops module."""
    return Python.import_module("max.graph").ops


# ============================================================================
# INFERENCE SESSION
# ============================================================================


struct MaxInferenceSession(Movable):
    """Wrapper for max.engine.InferenceSession."""

    var _py_session: PythonObject

    fn __init__(out self, devices: List[MaxDevice], num_threads: Int = -1) raises:
        """Create inference session.

        Args:
            devices: List of devices on which to run inference.
            num_threads: Number of threads to use (-1 for default).
        """
        var max_engine = Python.import_module("max.engine")
        var builtins = Python.import_module("builtins")

        # Convert devices to Python list
        var py_devices = builtins.list()
        for i in range(len(devices)):
            _ = py_devices.append(devices[i].to_python())

        # Create session with or without num_threads
        if num_threads > 0:
            self._py_session = max_engine.InferenceSession(
                devices=py_devices, num_threads=num_threads
            )
        else:
            self._py_session = max_engine.InferenceSession(devices=py_devices)

    fn to_python(self) -> PythonObject:
        """Get underlying Python session."""
        return self._py_session

    fn load(self, model: MaxGraph) raises -> MaxModel:
        """Load and compile a graph into an executable model.

        Args:
            model: The graph to compile.

        Returns:
            Compiled model ready for execution.
        """
        var py_model = self._py_session.load(model.to_python())
        return MaxModel(py_model)

    fn load_from_path(self, model_path: String) raises -> MaxModel:
        """Load a model from a file path.

        Args:
            model_path: Path to the model file.

        Returns:
            Compiled model ready for execution.
        """
        var pathlib = Python.import_module("pathlib")
        var path = pathlib.Path(model_path)
        var py_model = self._py_session.load(path)
        return MaxModel(py_model)

    fn devices(self) raises -> List[MaxDevice]:
        """Get list of available devices in this session."""
        var py_devices = self._py_session.devices
        var result = List[MaxDevice]()
        for dev in py_devices:
            result.append(MaxDevice(dev))
        return result^


# ============================================================================
# MODEL
# ============================================================================


struct MaxModel(Movable):
    """Wrapper for compiled max.engine.Model."""

    var _py_model: PythonObject

    fn __init__(out self, py_model: PythonObject):
        """Wrap existing MAX model."""
        self._py_model = py_model

    fn execute(self, inputs: List[MaxTensor]) raises -> List[MaxTensor]:
        """Run inference with list of tensors."""
        # Convert inputs to Python list
        var py_inputs = Python.evaluate("list")()
        for i in range(len(inputs)):
            _ = py_inputs.append(inputs[i].to_python())

        # Use a Python lambda to unpack the list
        var unpack_call = Python.evaluate("lambda f, args: f(*args)")
        var py_outputs = unpack_call(self._py_model.execute, py_inputs)

        # Convert outputs back
        var result = List[MaxTensor]()
        for output in py_outputs:
            result.append(MaxTensor(output))
        return result^

    fn input_metadata(self) raises -> PythonObject:
        """Get input specs."""
        return self._py_model.input_metadata

    fn output_metadata(self) raises -> PythonObject:
        """Get output specs."""
        return self._py_model.output_metadata


# ============================================================================
# MODULE-LEVEL DEVICE QUERY FUNCTIONS
# ============================================================================


fn accelerator_count() raises -> Int:
    """Get number of available accelerators."""
    var driver = Python.import_module("max.driver")
    return Int(driver.accelerator_count())


fn accelerator_api() raises -> String:
    """Get accelerator API ('cuda', 'hip')."""
    var driver = Python.import_module("max.driver")
    return String(driver.accelerator_api())


fn accelerator_architecture_name() raises -> String:
    """Get accelerator architecture."""
    var driver = Python.import_module("max.driver")
    return String(driver.accelerator_architecture_name())


fn scan_available_devices() raises -> PythonObject:
    """Get list of available DeviceSpecs."""
    var driver = Python.import_module("max.driver")
    return driver.scan_available_devices()


fn devices_exist(device_specs: PythonObject) raises -> Bool:
    """Check if devices exist."""
    var driver = Python.import_module("max.driver")
    return Bool(driver.devices_exist(device_specs))


fn load_devices(device_specs: PythonObject) raises -> PythonObject:
    """Initialize devices from specs."""
    var driver = Python.import_module("max.driver")
    return driver.load_devices(device_specs)
