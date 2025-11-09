"""
MAX Driver Module - Mojo Wrapper
=================================

Provides Mojo wrappers for MAX driver module APIs for interacting with hardware,
such as allocating tensors on accelerators and moving tensors between CPU and accelerators.

This module provides a clean interface to avoid dealing with opaque PythonObjects in Mojo,
wrapping the Python MAX driver API with type-safe Mojo structs.
"""

from python import Python, PythonObject
from .utils import PythonBridge
from .types import DeviceType

# ============================================================================
# DEVICE BASE CLASS
# ============================================================================

struct Device(Copyable, Movable):
    """Base device representation wrapping max.driver.Device.
    
    Provides access to device properties, synchronization, and peer access checks.
    """
    var _device: PythonObject
    
    fn __init__(out self, device_obj: PythonObject):
        """Create a Device wrapper from a Python MAX Device object.
        
        Args:
            device_obj: Python MAX Device object.
        """
        self._device = device_obj
    
    fn __init__(out self, device_type: DeviceType) raises:
        """Create a device from DeviceType.
        
        Args:
            device_type: The type of device (CPU or Accelerator).
        """
        
        if device_type == DeviceType.CPU():
            var cpu_dev = CPU()
            self._device = cpu_dev.to_python()
        elif device_type == DeviceType.Accelerator() or device_type == DeviceType.GPU():
            var accel_dev = Accelerator()
            self._device = accel_dev.to_python()
        else:
            raise Error("Unsupported device type: " + device_type.value)
    
    fn to_python(self) -> PythonObject:
        """Get the underlying Python Device object.
        
        Returns:
            The wrapped Python MAX Device.
        """
        return self._device
    
    fn id(self) raises -> Int:
        """Returns a zero-based device id.
        
        For CPU devices this is always 0. For accelerators this is the id 
        relative to this host.
        
        Returns:
            The device ID.
        """
        return Int(self._device.id)

    fn __eq__(self, other: Device) raises -> Bool:
        """Equality comparison between two Device instances.
        
        Args:
            other: The other Device to compare with.
        
        Returns:
            True if both devices are the same, False otherwise.
        """
        return Bool(self._device == other.to_python())

    fn __ne__(self, other: Device) raises -> Bool:
        """Inequality comparison between two Device instances.
        
        Args:
            other: The other Device to compare with.
        
        Returns:
            True if both devices are different, False otherwise.
        """
        return Bool(self._device != other.to_python())
    
    fn label(self) raises -> String:
        """Returns device label.
        
        Returns:
            "cpu" for host devices, "gpu" for accelerators.
        """
        return String(self._device.label)
    
    fn api(self) raises -> String:
        """Returns the API used to program the device.
        
        Returns:
            "cpu" for host devices, "cuda" for NVIDIA GPUs, "hip" for AMD GPUs.
        """
        return String(self._device.api)
    
    fn architecture_name(self) raises -> String:
        """Returns the architecture name of the device.
        
        Examples: "gfx90a", "gfx942" for AMD GPUs, "sm_80", "sm_86" for NVIDIA GPUs.
        CPU devices raise an exception.
        
        Returns:
            The architecture name string.
        """
        return String(self._device.architecture_name)
    
    fn is_host(self) raises -> Bool:
        """Whether this device is the CPU (host) device.
        
        Returns:
            True if CPU device, False otherwise.
        """
        return Bool(self._device.is_host)
    
    fn is_compatible(self) raises -> Bool:
        """Returns whether this device is compatible with MAX.
        
        Returns:
            True if device is compatible with MAX, False otherwise.
        """
        return Bool(self._device.is_compatible)
    
    fn synchronize(self) raises:
        """Ensures all operations on this device complete before returning.
        
        Raises:
            Error: If any enqueued operations had an internal error.
        """
        _ = self._device.synchronize()
    
    fn can_access(self, other: Device) raises -> Bool:
        """Checks if this device can directly access memory of another device.
        
        Args:
            other: The other device to check peer access against.
        
        Returns:
            True if peer access is possible, False otherwise.
        """
        return Bool(self._device.can_access(other.to_python()))
    
    fn default_stream(self) raises -> DeviceStream:
        """Returns the default stream for this device.
        
        The default stream is initialized when the device object is created.
        
        Returns:
            The default execution stream for this device.
        """
        return DeviceStream(self._device.default_stream)
    
    fn stats(self) raises -> PythonObject:
        """Returns utilization data for the device.
        
        Returns:
            A dictionary containing device utilization statistics.
        """
        return self._device.stats


# ============================================================================
# CPU DEVICE
# ============================================================================

struct CPU(Copyable, Movable):
    """CPU device wrapper for max.driver.CPU."""
    var _device: PythonObject
    
    fn __init__(out self, id: Int = -1) raises:
        """Creates a CPU device.
        
        Args:
            id: The device ID to use. Defaults to -1.
        """
        var driver = PythonBridge.get_module("max.driver")
        self._device = driver.CPU(id=id)
    
    fn to_python(self) -> PythonObject:
        """Get the underlying Python CPU device object.
        
        Returns:
            The wrapped Python MAX CPU device.
        """
        return self._device
    
    fn as_device(self) -> Device:
        """Convert to base Device type.
        
        Returns:
            Device wrapper around this CPU device.
        """
        return Device(self._device)
    
    fn id(self) raises -> Int:
        """Device id is always 0 for CPU devices.
        
        Returns:
            0
        """
        return Int(self._device.id)
    
    fn synchronize(self) raises:
        """Ensures all operations on this device complete before returning."""
        _ = self._device.synchronize()


# ============================================================================
# ACCELERATOR DEVICE (GPU/Hardware Accelerator)
# ============================================================================

struct Accelerator(Copyable, Movable):
    """Accelerator device wrapper for max.driver.Accelerator.
    
    Provides access to GPU or other hardware accelerators in the system.
    
    Repeated instantiations with a previously-used device-id will still refer 
    to the first such instance that was created. This is especially important 
    when providing a different memory limit: only the value provided in the 
    first such instantiation is effective.
    """
    var _device: PythonObject
    
    fn __init__(out self, id: Int = -1, device_memory_limit: Int = -1) raises:
        """Creates an accelerator device with the specified ID and memory limit.
        
        Args:
            id: The device ID to use. Defaults to -1, which selects
                the first available accelerator.
            device_memory_limit: The maximum amount of memory in bytes that 
                can be allocated on the device. Defaults to 99% of free memory.
        """
        var driver = PythonBridge.get_module("max.driver")
        self._device = driver.Accelerator(id=id, device_memory_limit=device_memory_limit)
    
    fn to_python(self) -> PythonObject:
        """Get the underlying Python Accelerator device object.
        
        Returns:
            The wrapped Python MAX Accelerator device.
        """
        return self._device
    
    fn as_device(self) -> Device:
        """Convert to base Device type.
        
        Returns:
            Device wrapper around this Accelerator device.
        """
        return Device(self._device)
    
    fn id(self) raises -> Int:
        """Get the accelerator device ID.
        
        Returns:
            The device ID.
        """
        return Int(self._device.id)
    
    fn api(self) raises -> String:
        """Returns the API used to program the accelerator.
        
        Returns:
            "cuda" for NVIDIA GPUs, "hip" for AMD GPUs.
        """
        return String(self._device.api)
    
    fn architecture_name(self) raises -> String:
        """Returns the architecture name of the accelerator.
        
        Examples: "gfx90a", "gfx942" for AMD GPUs, "sm_80", "sm_86" for NVIDIA GPUs.
        
        Returns:
            The architecture name string.
        """
        return String(self._device.architecture_name)
    
    fn synchronize(self) raises:
        """Ensures all operations on this device complete before returning."""
        _ = self._device.synchronize()
    
    fn can_access(self, other: Accelerator) raises -> Bool:
        """Checks if this accelerator can directly access memory of another accelerator.
        
        Args:
            other: The other accelerator to check peer access against.
        
        Returns:
            True if peer access is possible, False otherwise.
        """
        return Bool(self._device.can_access(other.to_python()))
    
    fn default_stream(self) raises -> DeviceStream:
        """Returns the default stream for this device.
        
        Returns:
            The default execution stream for this device.
        """
        return DeviceStream(self._device.default_stream)


# ============================================================================
# DEVICE STREAM
# ============================================================================

struct DeviceStream(Copyable, Movable):
    """Device stream wrapper for max.driver.DeviceStream.
    
    Provides access to a stream of execution on a device. A stream represents 
    a sequence of operations that will be executed in order. Multiple streams 
    on the same device can execute concurrently.
    """
    var _stream: PythonObject
    
    fn __init__(out self, stream_obj: PythonObject):
        """Create a DeviceStream wrapper from a Python MAX DeviceStream object.
        
        Args:
            stream_obj: Python MAX DeviceStream object.
        """
        self._stream = stream_obj
    
    @staticmethod
    fn create(device: Device) raises -> DeviceStream:
        """Creates a new stream of execution associated with the device.
        
        Args:
            device: The device to create the stream on.
        
        Returns:
            A new stream of execution.
        """
        var driver = PythonBridge.get_module("max.driver")
        var stream_obj = driver.DeviceStream(device.to_python())
        return DeviceStream(stream_obj)
    
    fn to_python(self) -> PythonObject:
        """Get the underlying Python DeviceStream object.
        
        Returns:
            The wrapped Python MAX DeviceStream.
        """
        return self._stream
    
    fn device(self) raises -> Device:
        """The device this stream is executing on.
        
        Returns:
            The Device this stream belongs to.
        """
        return Device(self._stream.device)
    
    fn synchronize(self) raises:
        """Ensures all operations on this stream complete before returning.
        
        Raises:
            Error: If any enqueued operations had an internal error.
        """
        _ = self._stream.synchronize()
    
    fn wait_for_stream(self, other_stream: DeviceStream) raises:
        """Ensures all operations on the other stream complete before future 
        work submitted to this stream is scheduled.
        
        Args:
            other_stream: The stream to wait for.
        """
        _ = self._stream.wait_for(other_stream.to_python())
    
    fn wait_for_device(self, device: Device) raises:
        """Ensures all operations on device's default stream complete before 
        future work submitted to this stream is scheduled.
        
        Args:
            device: The device whose default stream to wait for.
        """
        _ = self._stream.wait_for(device.to_python())


# ============================================================================
# DEVICE SPECIFICATION
# ============================================================================

struct DeviceSpec(Copyable, Movable):
    """Device specification containing device ID and type.
    
    Provides a way to specify device parameters like ID and type (CPU/GPU) 
    for creating Device instances.
    """
    var id: Int
    var device_type: String  # "cpu" or "gpu"
    
    fn __init__(out self, id: Int, device_type: String = "cpu"):
        """Create a device specification.
        
        Args:
            id: Provided id for this device.
            device_type: Type of specified device ("cpu" or "gpu").
        """
        self.id = id
        self.device_type = device_type
    
    @staticmethod
    fn cpu(id: Int = -1) -> DeviceSpec:
        """Creates a CPU device specification.
        
        Args:
            id: Device ID (default: -1).
        
        Returns:
            CPU device specification.
        """
        return DeviceSpec(id, "cpu")
    
    @staticmethod
    fn accelerator(id: Int = 0) -> DeviceSpec:
        """Creates an accelerator (GPU) device specification.
        
        Args:
            id: Device ID (default: 0).
        
        Returns:
            Accelerator device specification.
        """
        return DeviceSpec(id, "gpu")
    
    fn to_python(self) raises -> PythonObject:
        """Convert to Python DeviceSpec object.
        
        Returns:
            Python MAX DeviceSpec object.
        """
        var driver = PythonBridge.get_module("max.driver")
        return driver.DeviceSpec(id=self.id, device_type=self.device_type)


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

fn accelerator_count() raises -> Int:
    """Returns number of accelerator devices available.
    
    Returns:
        Number of available accelerators.
    """
    var driver = PythonBridge.get_module("max.driver")
    return Int(driver.accelerator_count())


fn accelerator_api() raises -> String:
    """Returns the API used to program the accelerator.
    
    Returns:
        API name string ("cuda", "hip", etc.).
    """
    var driver = PythonBridge.get_module("max.driver")
    return String(driver.accelerator_api())


fn accelerator_architecture_name() raises -> String:
    """Returns the architecture name of the accelerator device.
    
    Returns:
        Architecture name string (e.g., "gfx90a", "sm_80").
    """
    var driver = PythonBridge.get_module("max.driver")
    return String(driver.accelerator_architecture_name())


fn scan_available_devices() raises -> PythonObject:
    """Returns all accelerators if available, else return cpu.
    
    Returns:
        List of DeviceSpec objects.
    """
    var driver = PythonBridge.get_module("max.driver")
    return driver.scan_available_devices()


fn devices_exist(device_specs: PythonObject) raises -> Bool:
    """Identify if devices exist.
    
    Args:
        device_specs: List of DeviceSpec objects (Python list).
    
    Returns:
        True if all devices exist, False otherwise.
    """
    var driver = PythonBridge.get_module("max.driver")
    return Bool(driver.devices_exist(device_specs))


fn load_devices(device_specs: PythonObject) raises -> PythonObject:
    """Initialize and return a list of devices, given a list of device specs.
    
    Args:
        device_specs: List of DeviceSpec objects (Python list).
    
    Returns:
        List of Device objects.
    """
    var driver = PythonBridge.get_module("max.driver")
    return driver.load_devices(device_specs)
