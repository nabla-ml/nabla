"""
Shared test utilities for distributed sharding tests.

Provides auto-detection of accelerators and portable mesh creation
that works on CPU (simulation) and GPU (real distributed) modes.
"""

from max.graph import DeviceRef
from max import driver
import numpy as np

from nabla import DeviceMesh


def get_accelerator_count() -> int:
    """Get number of available accelerators."""
    try:
        return driver.accelerator_count()
    except Exception:
        return 0


def create_mesh(name: str, shape: tuple, axis_names: tuple, 
                devices: list = None, force_cpu: bool = False) -> DeviceMesh:
    """Create mesh with auto-detected device refs.
    
    - If accelerators available (and not force_cpu): uses GPU refs (distributed mode)
    - Otherwise: uses CPU refs (simulation mode)
    
    Args:
        name: Mesh name
        shape: Mesh shape, e.g., (2,) or (2, 2)
        axis_names: Axis names, e.g., ("x",) or ("dp", "tp")
        devices: Optional explicit device IDs (default: 0..N-1)
        force_cpu: Force CPU mode even if GPUs available
    
    Returns:
        DeviceMesh configured for the available hardware
    """
    total_devices = int(np.prod(shape))
    accel_count = get_accelerator_count()
    
    if devices is None:
        devices = list(range(total_devices))
    
    if not force_cpu and accel_count >= total_devices:
        # Real GPU mode
        device_refs = [DeviceRef.GPU(i) for i in range(total_devices)]
        print(f"  [{name}] GPU MODE: {total_devices} accelerators")
    else:
        # CPU simulation mode
        device_refs = [DeviceRef.CPU() for _ in range(total_devices)]
        if accel_count > 0 and accel_count < total_devices:
            print(f"  [{name}] CPU MODE: need {total_devices}, have {accel_count} accelerators")
        else:
            print(f"  [{name}] CPU MODE: simulating {total_devices} devices")
    
    return DeviceMesh(name, shape, axis_names, devices=devices, device_refs=device_refs)


def is_distributed_mode(required_gpus: int = 2) -> bool:
    """Check if we have enough accelerators for distributed mode."""
    return get_accelerator_count() >= required_gpus


def get_mode_string() -> str:
    """Get a string describing the current mode."""
    count = get_accelerator_count()
    if count >= 2:
        return f"GPU ({count} accelerators)"
    elif count == 1:
        return "GPU (1 accelerator - limited)"
    else:
        return "CPU (simulated)"
