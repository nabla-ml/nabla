
from python import Python, PythonObject
from memory import ArcPointer
from .utils import PythonBridge, Graph
from .driver import CPU, Accelerator, Device as DriverDevice
from . import ops


# ============================================================================
# 1. TENSOR - Wrapper for MAX Tensor type
# ============================================================================

struct Tensor(ImplicitlyCopyable, Movable):
    """Wrapper for MAX driver Tensor type.
    
    Device-resident tensor representation. Allocates memory onto a given 
    device with the provided shape and dtype.
    """
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
        
        The underlying data is not copied unless the array is noncontiguous.
        
        Args:
            np_array: Numpy array to convert.
        
        Returns:
            Tensor wrapping the MAX Tensor.
        """
        var driver = PythonBridge.get_module("max.driver")
        var max_tensor = driver.Tensor.from_numpy(np_array)
        return Tensor(max_tensor)
    
    @staticmethod
    fn zeros(shape: List[Int], dtype: DType, device: DriverDevice) raises -> Tensor:
        """Create a tensor filled with zeros.
        
        Args:
            shape: List of positive, non-zero integers denoting the tensor shape.
            dtype: Mojo DType (will be converted to MAX DType).
            device: Device to allocate tensor onto.
        
        Returns:
            Zero-filled tensor.
        """
        var driver = PythonBridge.get_module("max.driver")
        var python_dtype = DTypeConverter.to_python(dtype)
        var python_shape = PythonBridge.shape_to_python(shape)
        return Tensor(driver.Tensor.zeros(python_shape, python_dtype, device.to_python()))
    
    @staticmethod
    fn scalar(value: PythonObject, dtype: DType, device: DriverDevice) raises -> Tensor:
        """Create a scalar (rank-0) tensor.
        
        Args:
            value: Scalar value.
            dtype: Mojo DType (will be converted to MAX DType).
            device: Device to allocate tensor onto.
        
        Returns:
            Scalar tensor.
        """
        var driver = PythonBridge.get_module("max.driver")
        var python_dtype = DTypeConverter.to_python(dtype)
        return Tensor(driver.Tensor.scalar(value, python_dtype, device.to_python()))
    
    @staticmethod
    fn create(shape: List[Int], dtype: DType, device: DriverDevice, pinned: Bool = False) raises -> Tensor:
        """Create a new tensor with given dtype, shape, and device.
        
        Args:
            shape: List of positive, non-zero integers denoting the tensor shape.
            dtype: Mojo DType (will be converted to MAX DType).
            device: Device to allocate tensor onto.
            pinned: If True, memory is page-locked (pinned). Defaults to False.
        
        Returns:
            New tensor.
        """
        var driver = PythonBridge.get_module("max.driver")
        var python_dtype = DTypeConverter.to_python(dtype)
        var python_shape = PythonBridge.shape_to_python(shape)
        return Tensor(driver.Tensor(python_dtype, python_shape, device.to_python(), pinned))
    
    fn to_numpy(self) raises -> PythonObject:
        """Convert Tensor back to numpy array.
        
        If the tensor is not on the host, an exception is raised.
        
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
    
    fn copy(self, device: DriverDevice) raises -> Tensor:
        """Creates a copy on the specified device using the .to() method.
        
        Args:
            device: The device to create the copy on.
        
        Returns:
            A new tensor that is a copy of this tensor on the specified device.
        """
        var copied = self.tensor.to(device.to_python())
        return Tensor(copied)
    
    fn copy_to_cpu(self) raises -> Tensor:
        """Creates a copy on CPU.
        
        Returns:
            A new CPU-resident tensor.
        """
        var cpu_dev = CPU()
        var copied = self.tensor.to(cpu_dev.to_python())
        return Tensor(copied)
    
    fn copy_to_accelerator(self, accel: Accelerator) raises -> Tensor:
        """Creates a copy on an accelerator device.
        
        Args:
            accel: The accelerator to copy to.
        
        Returns:
            A new accelerator-resident tensor.
        """
        var copied = self.tensor.to(accel.to_python())
        return Tensor(copied)
    
    fn shape(self) raises -> PythonObject:
        """Shape of tensor.
        
        Returns:
            Tuple representing tensor shape.
        """
        return self.tensor.shape
    
    fn dtype(self) raises -> PythonObject:
        """DType of constituent elements in tensor.
        
        Returns:
            MAX DType object.
        """
        return self.tensor.dtype
    
    fn is_host(self) raises -> Bool:
        """Whether or not tensor is host-resident.
        
        Returns:
            False for accelerator tensors, True for CPU tensors.
        """
        return Bool(self.tensor.is_host)
    
    fn is_contiguous(self) raises -> Bool:
        """Whether or not tensor is contiguously allocated in memory.
        
        Returns:
            False if the tensor is a non-contiguous slice.
        """
        return Bool(self.tensor.is_contiguous)
    
    fn contiguous(self) raises -> Tensor:
        """Creates a contiguous copy of the parent tensor.
        
        Returns:
            A contiguous tensor.
        """
        return Tensor(self.tensor.contiguous())
    
    fn synchronize(self) raises:
        """Synchronize the tensor's device stream."""
        _ = self.tensor.stream.synchronize()
    
    fn device(self) raises -> DriverDevice:
        """Device on which tensor is resident.
        
        Returns:
            The device this tensor is on.
        """
        return DriverDevice(self.tensor.device)
    
    fn stream(self) raises -> PythonObject:
        """Stream to which tensor is bound.
        
        Returns:
            Python DeviceStream object.
        """
        return self.tensor.stream
    
    fn rank(self) raises -> Int:
        """Tensor rank.
        
        Returns:
            Number of dimensions.
        """
        return Int(self.tensor.rank)
    
    fn num_elements(self) raises -> Int:
        """Returns the number of elements in this tensor.
        
        Rank-0 tensors have 1 element by convention.
        
        Returns:
            Total number of elements.
        """
        return Int(self.tensor.num_elements)
    
    fn element_size(self) raises -> Int:
        """Return the size of the element type in bytes.
        
        Returns:
            Size in bytes of each element.
        """
        return Int(self.tensor.element_size)
    
    fn pinned(self) raises -> Bool:
        """Whether or not the underlying memory is pinned (page-locked).
        
        Returns:
            True if memory is pinned, False otherwise.
        """
        return Bool(self.tensor.pinned)
    
    fn item(self) raises -> PythonObject:
        """Returns the scalar value at a given location.
        
        Currently implemented only for zero-rank tensors. The return type 
        is converted to a Python built-in type.
        
        Returns:
            Scalar value as Python object.
        """
        return self.tensor.item()
    
    fn inplace_copy_from(self, src: Tensor) raises:
        """Copy the contents of another tensor into this one.
        
        These tensors may be on different devices. Requires that both 
        tensors are contiguous and have same size.
        
        Args:
            src: Source tensor to copy from.
        """
        _ = self.tensor.inplace_copy_from(src.to_python())
    
    fn view(self, dtype: DType) raises -> Tensor:
        """Return a new tensor with the given type that shares the underlying memory.
        
        Shape will be deduced automatically.
        
        Args:
            dtype: New Mojo data type.
        
        Returns:
            View of this tensor with new dtype.
        """
        var python_dtype = DTypeConverter.to_python(dtype)
        return Tensor(self.tensor.view(python_dtype))
    
    fn view_with_shape(self, dtype: DType, shape: List[Int]) raises -> Tensor:
        """Return a new tensor with the given type and shape that shares 
        the underlying memory.
        
        Args:
            dtype: New Mojo data type.
            shape: New shape as List[Int].
        
        Returns:
            View of this tensor with new dtype/shape.
        """
        var python_dtype = DTypeConverter.to_python(dtype)
        var python_shape = PythonBridge.shape_to_python(shape)
        return Tensor(self.tensor.view(python_dtype, python_shape))


# ============================================================================
# 2. DEVICE MANAGEMENT - Legacy compatibility wrappers
# ============================================================================

struct DeviceType(Copyable, Movable, ImplicitlyCopyable):
    """Device type enum - legacy compatibility wrapper.
    
    Note: Prefer using CPU and Accelerator structs from driver module directly.
    """
    var value: String
    
    fn __init__(out self, value: String):
        self.value = value
    
    @staticmethod
    fn CPU() -> DeviceType:
        """Create CPU device type."""
        return DeviceType("CPU")
    
    @staticmethod
    fn Accelerator() -> DeviceType:
        """Create Accelerator device type (replaces GPU)."""
        return DeviceType("Accelerator")
    
    # Deprecated: GPU is now Accelerator
    @staticmethod
    fn GPU() -> DeviceType:
        """[DEPRECATED] Use Accelerator() instead."""
        return DeviceType("Accelerator")
    
    fn __eq__(self, other: DeviceType) -> Bool:
        return self.value == other.value
    
    fn __ne__(self, other: DeviceType) -> Bool:
        return self.value != other.value


struct Device:
    """Device wrapper - legacy compatibility.
    
    Note: Prefer using CPU and Accelerator structs from driver module directly.
    """
    var device_type: DeviceType
    var device_obj: PythonObject
    
    fn __init__(out self, device_type: DeviceType) raises:
        """Create a device from DeviceType.
        
        Args:
            device_type: The type of device (CPU or Accelerator).
        """
        self.device_type = device_type
        
        if device_type == DeviceType.CPU():
            var cpu_dev = CPU()
            self.device_obj = cpu_dev.to_python()
        elif device_type == DeviceType.Accelerator() or device_type == DeviceType.GPU():
            var accel_dev = Accelerator()
            self.device_obj = accel_dev.to_python()
        else:
            raise Error("Unsupported device type: " + device_type.value)
    
    fn to_python(self) -> PythonObject:
        """Get underlying Python device object.
        
        Returns:
            Python MAX Device object.
        """
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
    """Symbolic tensor value within a Graph.
    
    Represents a value semantic tensor within a Graph. Provides methods to 
    manipulate and query tensor attributes such as shape, dtype, device, and more.
    """
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
    
    # Properties
    # ==========
    
    fn shape(self) raises -> PythonObject:
        """Returns the shape of the TensorValue.
        
        Returns:
            Shape object from MAX graph API.
        """
        return self.tensor_value.shape
    
    fn dtype(self) raises -> PythonObject:
        """Returns the tensor data type.
        
        Returns:
            MAX DType object.
        """
        return self.tensor_value.dtype
    
    fn rank(self) raises -> Int:
        """Returns the rank (number of dimensions) of the tensor.
        
        Returns:
            Number of dimensions.
        """
        return Int(self.tensor_value.rank)
    
    fn device(self) raises -> PythonObject:
        """Returns the device of the TensorValue.
        
        Returns:
            DeviceRef object.
        """
        return self.tensor_value.device
    
    fn type(self) raises -> PythonObject:
        """Returns the type of the TensorValue as a TensorType.
        
        Returns:
            TensorType object.
        """
        return self.tensor_value.type
    
    # Shape manipulation
    # ==================
    
    fn reshape(self, shape: List[Int]) raises -> TensorValue:
        """Creates a new tensor with the same data but reshaped.
        
        Args:
            shape: The new shape as List[Int].
        
        Returns:
            A new TensorValue with the reshaped dimensions.
        """
        return ops.reshape(self, shape)
    
    fn flatten(self, start_dim: Int = 0, end_dim: Int = -1) raises -> TensorValue:
        """Flattens the specified dims of a symbolic tensor.
        
        The number and order of elements is unchanged. All dimensions from 
        start_dim to end_dim (inclusive) are merged into a single output dim.
        
        Args:
            start_dim: The starting dimension to flatten. Defaults to 0.
            end_dim: The ending dimension to flatten. Defaults to -1.
        
        Returns:
            A new TensorValue with the flattened dimensions.
        """
        return ops.flatten(self, start_dim, end_dim)
    
    fn broadcast_to(self, shape: List[Int]) raises -> TensorValue:
        """Broadcasts the tensor to a new shape.
        
        Args:
            shape: List of integers for the target shape.
        
        Returns:
            A new TensorValue with the broadcasted shape.
        """
        return ops.broadcast_to(self, shape)
    
    fn rebind(self, shape: List[Int], message: String = "") raises -> TensorValue:
        """Rebinds the tensor to a new shape with error handling.
        
        Args:
            shape: The new shape as List[Int].
            message: Optional message for logging or debugging.
        
        Returns:
            A new TensorValue with the updated shape.
        """
        return ops.rebind(self, shape, message)
    
    # Transpose and permutation
    # =========================
    
    fn transpose(self, dim_1: Int, dim_2: Int) raises -> TensorValue:
        """Swaps two dimensions of the tensor.
        
        Args:
            dim_1: The first dimension to swap.
            dim_2: The second dimension to swap.
        
        Returns:
            A new TensorValue with swapped dimensions.
        """
        return ops.transpose(self, dim_1, dim_2)
    
    fn T(self) raises -> TensorValue:
        """Returns the transposed tensor.
        
        T is shorthand notation for transposing. Equivalent to transpose().
        
        Returns:
            A new TensorValue with swapped dimensions.
        """
        # T property typically transposes last two dimensions
        return ops.transpose(self, -2, -1)
    
    fn permute(self, dims: List[Int]) raises -> TensorValue:
        """Permutes the tensor's dimensions based on provided indices.
        
        Args:
            dims: List of integers specifying the new order of dimensions.
        
        Returns:
            A new TensorValue with permuted dimensions.
        """
        return ops.permute(self, dims)
    
    # Type conversion
    # ===============
    
    fn cast(self, dtype: DType) raises -> TensorValue:
        """Casts a symbolic tensor to a different data type.
        
        Args:
            dtype: The target Mojo data type.
        
        Returns:
            A new TensorValue with the casted data type.
        """
        return ops.cast(self, dtype)
    
    # Reduction operations
    # ====================
    
    fn argmax(self, axis: Int = -1) raises -> TensorValue:
        """Reduces the tensor using an argmax operation along axis.
        
        When the result is ambiguous (multiple maxima), selects one arbitrarily.
        
        Args:
            axis: The axis along which to compute the reduction. 
                  If negative, indexes from the last dimension.
        
        Returns:
            A TensorValue of dtype DType.int64 with the same rank as input,
            and the same shape except along axis, which will have size 1.
        """
        return ops.argmax(self, axis)
    
    fn max(self, axis: Int = -1) raises -> TensorValue:
        """Reduces the tensor using a max operation along axis.
        
        Args:
            axis: The axis along which to compute the reduction.
        
        Returns:
            A TensorValue with the same rank as input and the same shape 
            except along axis, which will have size 1.
        """
        return ops.max(self, axis)
    
    fn min(self, axis: Int = -1) raises -> TensorValue:
        """Reduces the tensor using a min operation along axis.
        
        Args:
            axis: The axis along which to compute the reduction.
        
        Returns:
            A TensorValue with the same rank as input and the same shape 
            except along axis, which will have size 1.
        """
        return ops.min(self, axis)
    
    fn mean(self, axis: Int = -1) raises -> TensorValue:
        """Reduces the tensor using a mean operation along axis.
        
        Args:
            axis: The axis along which to compute the reduction.
        
        Returns:
            A TensorValue with the same rank as input and the same shape 
            except along axis, which will have size 1.
        """
        return ops.mean(self, axis)
    
    fn var(self, axis: Int = -1) raises -> TensorValue:
        """Reduces the tensor using a variance operation along axis.
        
        The variance is computed as the mean of squared deviations from the 
        mean (population variance, without Bessel's correction).
        
        Args:
            axis: The axis along which to compute the reduction.
        
        Returns:
            A TensorValue with the same rank as input and the same shape 
            except along axis, which will have size 1.
        """
        # variance = E[(x - mean)^2] = E[x^2] - E[x]^2
        var mean_val = self.mean(axis)
        var squared = ops.mul(self, self)
        var mean_squared = squared.mean(axis)
        return ops.sub(mean_squared, ops.mul(mean_val, mean_val))
    
    fn stdev(self, axis: Int = -1) raises -> TensorValue:
        """Reduces the tensor using a standard deviation operation along axis.
        
        The standard deviation is computed as the square root of the 
        population variance along the specified axis.
        
        Args:
            axis: The axis along which to compute the reduction.
        
        Returns:
            A TensorValue with the same rank as input and the same shape 
            except along axis, which will have size 1.
        """
        # stdev = sqrt(var), compute using variance
        return ops.sqrt(self.var(axis))
    
    # Device transfer
    # ===============
    
    fn to(self, device: Device) raises -> TensorValue:
        """Transfers the tensor to a specified device without mutation.
        
        Args:
            device: A Device object specifying the target device.
        
        Returns:
            A new TensorValue on the specified device.
        """
        return ops.transfer_to(self, device)
    
    # Debugging
    # =========
    
    fn print(self, label: String = "debug_tensor") raises:
        """Prints detailed information about the tensor.
        
        Args:
            label: A string label for the printed output.
        """
        print(label + ":")
        print("  shape:", self.shape())
        print("  dtype:", self.dtype())
        print("  rank:", self.rank())
        print("  device:", self.device())
