# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Base operation classes for a clean OOP design."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from max.dtype import DType
from max.graph import Value

from ..core.array import Array


class Operation(ABC):
    """Abstract base class for all operations."""

    def __init__(self, name: str):
        self.name = name
        # print(f"[DEBUG] Created operation: {name}")

    @abstractmethod
    def forward(self, *args: Array) -> Array:
        """Forward pass - creates the result Array."""
        pass

    @abstractmethod
    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute the output shape given input shapes."""
        pass

    @abstractmethod
    def maxpr(self, args: list[Value], output: Array) -> None:
        """MAX graph computation."""
        pass

    @abstractmethod
    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager computation using NumPy."""
        pass

    @abstractmethod
    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """Vector-Jacobian product rule for reverse-mode autodiff."""
        pass

    @abstractmethod
    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """Jacobian-vector product rule for forward-mode autodiff."""
        pass

    def custom_kernel_path(self) -> Optional[Path]:
        """Optional: path to custom kernel implementation."""
        return None


class UnaryOperation(Operation):
    """Base class for unary operations."""

    def forward(self, *args: Array) -> Array:
        """Forward pass for unary operations."""
        if len(args) != 1:
            raise ValueError(f"Unary operation requires 1 argument, got {len(args)}")
        arg = args[0]

        output_shape = self.compute_output_shape(arg.shape)
        output_batch_dims = self.compute_output_batch_dims(arg.batch_dims)
        output_dtype = self.compute_output_dtype(arg)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,
            device=arg.device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(arg)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([arg], res)

        return res

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Default: output shape same as input shape."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Unary operation requires 1 input shape, got {len(input_shapes)}"
            )
        return input_shapes[0]

    def compute_output_dtype(self, arg: Array) -> DType:
        """Default: output dtype same as input dtype."""
        return arg.dtype

    def compute_output_batch_dims(self, *input_batch_dims: int) -> int:
        """Default: output batch dims same as input batch dims."""
        if len(input_batch_dims) != 1:
            raise ValueError(
                f"Unary operation requires 1 input batch dims, got {len(input_batch_dims)}"
            )
        return input_batch_dims[0]

def move_to_best_device(*args: Array) -> tuple[Array, ...]:
    """Move all arrays to the best available device."""
    if len(args) <= 1:
        return args

    # We can use the following max api to access information abou the array.device:Device
    # class max.driver.Device

    # api
    # property api

    # Returns the API used to program the device.

    # Possible values are:

    # cpu for host devices.
    # cuda for NVIDIA GPUs.
    # hip for AMD GPUs.
    # from max import driver

    # device = driver.CPU()
    # device.api

    # can_access
    # can_access

    # Checks if this device can directly access memory of another device.

    # from max import driver

    # gpu0 = driver.Accelerator(id=0)
    # gpu1 = driver.Accelerator(id=1)

    # if gpu0.can_access(gpu1):
    #     print("GPU0 can directly access GPU1 memory.")

    # Parameters:

    # other (Device ) – The other device to check peer access against.

    # Returns:

    # True if peer access is possible, False otherwise.

    # Return type:

    # bool

    # cpu
    # cpu = <nanobind.nb_func object>

    # default_stream
    # property default_stream

    # Returns the default stream for this device.

    # The default stream is initialized when the device object is created.

    # Returns:

    # The default execution stream for this device.

    # Return type:

    # DeviceStream

    # id
    # property id

    # Returns a zero-based device id. For a CPU device this is always 0. For GPU accelerators this is the id of the device relative to this host. Along with the label, an id can uniquely identify a device, e.g. gpu:0, gpu:1.

    # from max import driver

    # device = driver.Accelerator()
    # device_id = device.id

    # Returns:

    # The device ID.

    # Return type:

    # int

    # is_compatible
    # property is_compatible

    # Returns whether this device is compatible with MAX.

    # Returns:

    # True if the device is compatible with MAX, False otherwise.

    # Return type:

    # bool

    # is_host
    # property is_host

    # Whether this device is the CPU (host) device.

    # from max import driver

    # device = driver.CPU()
    # device.is_host

    # label
    # property label

    # Returns device label.

    # Possible values are:

    # cpu for host devices.
    # gpu for accelerators.
    # from max import driver

    # device = driver.CPU()
    # device.label

    # stats
    # property stats

    # Returns utilization data for the device.

    # from max import driver

    # device = driver.CPU()
    # stats = device.stats

    # Returns:

    # A dictionary containing device utilization statistics.

    # Return type:

    # dict

    # synchronize
    # synchronize

    # Ensures all operations on this device complete before returning.

    # Raises:

    # ValueError – If any enqueued operations had an internal error.

    # Task: create a small dictionary to track the devices, and how much data lives on each device by checking np.prod(array.shape). then go through all available devices in the dict and find the one where the most data lives, then move all other arrays to that device. this ensures that we move as little data as possible. however, there is one rule. if any array lives on an accelerator, we always move away from the cpu to the accelerator, even if it has less data. this is because the accelerator is always faster than the cpu, so we want to use it if possible.
    import numpy as np
    
    # Track devices and data amounts
    device_data = {}
    accelerator_devices = set()
    
    for arg in args:
        device = arg.device
        data_size = np.prod(arg.shape)
        device_data[device] = device_data.get(device, 0) + data_size
        
        # Check if this device is an accelerator (non-host device)
        if not device.is_host:
            accelerator_devices.add(device)
    
    # Determine best device according to the rules:
    # 1. If any accelerator has data, choose the best accelerator considering peer access
    # 2. Otherwise, choose the device (CPU) with most data
    if accelerator_devices:
        # For multi-accelerator scenarios, consider peer access costs
        if len(accelerator_devices) > 1:
            # Calculate effective data amount considering peer access
            accelerator_scores = {}
            for candidate_device in accelerator_devices:
                # Base score is the data already on this device
                base_score = device_data[candidate_device]
                
                # Add bonus for data that can be directly accessed from other accelerators
                peer_accessible_data = 0
                for other_device in accelerator_devices:
                    if other_device != candidate_device and candidate_device.can_access(other_device):
                        peer_accessible_data += device_data[other_device]
                
                # Weight peer-accessible data less than local data (avoid unnecessary moves)
                accelerator_scores[candidate_device] = base_score + (peer_accessible_data * 0.1)
            
            best_device = max(accelerator_scores, key=accelerator_scores.get)
        else:
            # Single accelerator case - simple selection
            best_device = max(accelerator_devices, key=lambda d: device_data[d])
    else:
        # Find device with most data (will be CPU in this case)
        best_device = max(device_data, key=device_data.get)
    
    # Move all arrays to the best device
    result_args = []
    for arg in args:
        if arg.device != best_device:
            result_args.append(arg.to(best_device))
        else:
            result_args.append(arg)
    
    return tuple(result_args)

class BinaryOperation(Operation):
    """Base class for binary operations."""

    def forward(self, *args: Array) -> Array:
        """Forward pass for binary operations."""
        if len(args) != 2:
            raise ValueError(f"Binary operation requires 2 arguments, got {len(args)}")
        
        # Move arrays to best device
        args = move_to_best_device(*args)
        arg1, arg2 = args[0], args[1]

        from ..ops.view import broadcast_batch_dims, broadcast_to

        self._validate_inputs(arg1, arg2)

        output_shape = self.compute_output_shape(arg1.shape, arg2.shape)
        output_batch_dims = self.compute_output_batch_dims(
            arg1.batch_dims, arg2.batch_dims
        )
        output_dtype = self.compute_output_dtype(arg1, arg2)
        if arg1.traced:
            arg1 = broadcast_to(arg1, output_shape)
            arg1 = broadcast_batch_dims(arg1, output_batch_dims)
        if arg2.traced:
            arg2 = broadcast_to(arg2, output_shape)
            arg2 = broadcast_batch_dims(arg2, output_batch_dims)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,
            device=arg1.device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(arg1, arg2)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([arg1, arg2], res)

        return res

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute broadcasted output shape."""
        if len(input_shapes) != 2:
            raise ValueError(
                f"Binary operation requires 2 input shapes, got {len(input_shapes)}"
            )
        shape1, shape2 = input_shapes[0], input_shapes[1]

        from ..utils.shape_utils import get_broadcasted_shape

        return get_broadcasted_shape(shape1, shape2)

    def compute_output_dtype(self, arg1: Array, arg2: Array) -> DType:
        """Default: output dtype same as first input dtype."""
        return arg1.dtype

    def _validate_inputs(self, arg1: Array, arg2: Array) -> None:
        """Validate binary operation inputs."""
        if not isinstance(arg1, Array) or not isinstance(arg2, Array):
            raise TypeError("Both arguments must be Array instances")
        if arg1.dtype != arg2.dtype:
            raise ValueError(f"Dtypes {arg1.dtype} and {arg2.dtype} are incompatible")
        if arg1.device != arg2.device:
            raise ValueError(
                f"Devices {arg1.device} and {arg2.device} are incompatible"
            )

    def compute_output_batch_dims(self, *input_batch_dims: tuple) -> tuple:
        """Default: output batch dims same as input batch dims."""
        if len(input_batch_dims) != 2:
            raise ValueError(
                f"Binary operation requires 2 input batch dims, got {len(input_batch_dims)}"
            )
        shape1, shape2 = input_batch_dims[0], input_batch_dims[1]

        from ..utils.shape_utils import get_broadcasted_shape

        return get_broadcasted_shape(shape1, shape2)


class ReductionOperation(UnaryOperation):
    """Base class for reduction operations."""

    def __init__(
        self,
        name: str,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(name)
        self.axes = axes
        self.keep_dims = keep_dims

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for reduction."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Reduction operation requires 1 input shape, got {len(input_shapes)}"
            )
        input_shape = input_shapes[0]
        return self._compute_reduction_shape(input_shape, self.axes)

    def compute_output_batch_dims(self, *input_batch_dims: tuple) -> tuple:
        """Compute output batch dims for reduction."""
        if len(input_batch_dims) != 1:
            raise ValueError(
                f"Reduction operation requires 1 input batch dims, got {len(input_batch_dims)}"
            )
        # For regular reductions, batch_dims are not affected - they pass through unchanged
        # Only SumBatchDimsOp overrides this to actually reduce batch dimensions
        return input_batch_dims[0]

    @staticmethod
    def _compute_reduction_shape(
        input_shape: tuple,
        axes: int | list[int] | tuple[int, ...] | None,
    ) -> tuple:
        """Compute the output shape for a reduction operation.

        Always preserves dimensions (sets reduced axes to size 1).
        Dimension removal should be handled separately by squeeze operations.
        """
        if axes is None:
            # Reduce all axes - return shape with all dimensions set to 1
            return (1,) * len(input_shape)

        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, tuple):
            axes = list(axes)

        normalized_axes = []
        for axis in axes:
            if axis < 0:
                axis += len(input_shape)
            if axis < 0 or axis >= len(input_shape):
                raise ValueError(
                    f"Axis {axis} is out of bounds for shape {input_shape}"
                )
            normalized_axes.append(axis)

        output_shape = []
        for i, dim in enumerate(input_shape):
            if i in normalized_axes:
                # Always preserve dimensions - set reduced axes to size 1
                output_shape.append(1)
            else:
                output_shape.append(dim)

        return tuple(output_shape)


class ViewOperation(UnaryOperation):
    """Base class for view operations (reshape, transpose, etc.)."""

    def __init__(self, name: str):
        super().__init__(name)

    def compute_output_batch_dims(self, *input_batch_dims: tuple) -> tuple:
        """Default: output batch dims same as input batch dims."""
        if len(input_batch_dims) != 1:
            raise ValueError(
                f"View operation requires 1 input batch dims, got {len(input_batch_dims)}"
            )
        return input_batch_dims[0]
