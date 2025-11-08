
from python import Python, PythonObject
from memory import ArcPointer
from .utils import PythonBridge, Graph


# ============================================================================
# 1. TENSOR - Wrapper for MAX Tensor type
# ============================================================================

struct Tensor(ImplicitlyCopyable, Movable):
    """Wrapper for MAX driver Tensor type."""
    var tensor: PythonObject
    
    fn __init__(out self, tensor: PythonObject):
        """Create a Tensor from a Python MAX Tensor object.
        
        Args:
            tensor: Python MAX Tensor object.
        """
        self.tensor = tensor
    
    @staticmethod
    fn from_numpy(np_array: PythonObject) raises -> Tensor:
        """Create a Tensor from a numpy array.
        
        Args:
            np_array: Numpy array to convert.
        
        Returns:
            Tensor wrapping the MAX Tensor.
        """
        var driver = PythonBridge.get_module("max.driver")
        var max_tensor = driver.Tensor.from_numpy(np_array)
        return Tensor(max_tensor)
    
    fn to_numpy(self) raises -> PythonObject:
        """Convert Tensor back to numpy array.
        
        Returns:
            Numpy array containing the tensor data.
        """
        return self.tensor.to_numpy()
    
    fn to_python(self) -> PythonObject:
        """Get the underlying Python Tensor object.
        
        Returns:
            The wrapped Python MAX Tensor.
        """
        return self.tensor


# ============================================================================
# 2. DEVICE MANAGEMENT - Abstract device handling
# ============================================================================

struct DeviceType(Copyable, Movable, ImplicitlyCopyable):
    """Device type enum."""
    var value: String
    
    fn __init__(out self, value: String):
        self.value = value
    
    @staticmethod
    fn CPU() -> DeviceType:
        return DeviceType("CPU")
    
    @staticmethod
    fn GPU() -> DeviceType:
        return DeviceType("GPU")
    
    fn __eq__(self, other: DeviceType) -> Bool:
        return self.value == other.value
    
    fn __ne__(self, other: DeviceType) -> Bool:
        return self.value != other.value


struct Device:
    """Abstract device representation."""
    var device_type: DeviceType
    var device_obj: PythonObject
    
    fn __init__(out self, device_type: DeviceType) raises:
        self.device_type = device_type
        
        if device_type == DeviceType.CPU():
            var driver = PythonBridge.get_module("max.driver")
            self.device_obj = driver.CPU()
        elif device_type == DeviceType.GPU():
            var driver = PythonBridge.get_module("max.driver")
            self.device_obj = driver.GPU()
        else:
            raise Error("Unsupported device type: " + device_type.value)
    
    fn to_python(self) -> PythonObject:
        return self.device_obj


# ============================================================================
# 3. DTYPE CONVERSION - Centralized dtype handling
# ============================================================================

struct DTypeConverter:
    """Converts Mojo DType to MAX DType."""
    
    @staticmethod
    fn to_python(dtype: DType) raises -> PythonObject:
        """Convert Mojo DType to Python MAX DType."""
        var max_dtype = PythonBridge.get_module("max.dtype").DType
        
        if dtype == DType.float32:
            return max_dtype.float32
        elif dtype == DType.float64:
            return max_dtype.float64
        elif dtype == DType.int32:
            return max_dtype.int32
        elif dtype == DType.int64:
            return max_dtype.int64
        elif dtype == DType.int8:
            return max_dtype.int8
        elif dtype == DType.uint8:
            return max_dtype.uint8
        elif dtype == DType.bool:
            return max_dtype.bool
        else:
            raise Error("Unsupported DType")


# ============================================================================
# 4. TENSOR TYPE - Symbolic tensor type definitions
# ============================================================================

struct TensorType(Copyable, Movable):
    """Symbolic tensor type for graph inputs."""
    var tensor_type: PythonObject

    fn __init__(out self, dtype: DType, shape: List[Int], device: Device) raises:
        """Create a tensor type specification.
        
        Args:
            dtype: Data type (float32, int32, etc.).
            shape: Tensor shape as List[Int].
            device: Device where tensor will reside.
        """
        var max_TensorType = PythonBridge.get_module("max.graph").TensorType
        var python_dtype = DTypeConverter.to_python(dtype)
        var python_shape = PythonBridge.shape_to_python(shape)
        
        self.tensor_type = max_TensorType(
            python_dtype, 
            python_shape, 
            device=device.to_python()
        )

    fn to_python(self) -> PythonObject:
        """Get underlying Python TensorType object."""
        return self.tensor_type


# ============================================================================
# 5. TENSOR VALUE - Runtime tensor with graph context
# ============================================================================

struct TensorValue(ImplicitlyCopyable, Movable):
    """Runtime tensor value with graph context."""
    var ctx: ArcPointer[Graph]
    var tensor_value: PythonObject

    fn __init__(out self, ctx: ArcPointer[Graph], tensor: PythonObject) raises:
        """Create a tensor value bound to a graph context.
        
        Args:
            ctx: Graph context this tensor belongs to.
            tensor: Underlying Python TensorValue object.
        """
        self.ctx = ctx
        self.tensor_value = tensor
    
    fn get_graph(self) -> ArcPointer[Graph]:
        """Get the graph context."""
        return self.ctx
    
    fn to_python(self) -> PythonObject:
        """Get underlying Python tensor object."""
        return self.tensor_value
