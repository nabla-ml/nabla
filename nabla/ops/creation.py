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

"""Array creation and initialization operations."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import DeviceRef, TensorType, Value, ops

from ..core.array import Array, Shape
from .operation import Operation
from .view import broadcast_batch_dims, broadcast_to

# Public API
__all__ = [
    "array",
    "arange",
    "arange_like",
    "randn",
    "randn_like",
    "rand",
    "rand_like",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_uniform",
    "lecun_normal",
]

# Constants
_DEFAULT_CPU = CPU()
_DEFAULT_SEED = 0
_DEFAULT_DTYPE = DType.float32


def _validate_shape(shape: Shape) -> None:
    """Validate shape parameter."""
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}")


def _validate_numeric(value: float | int, name: str) -> None:
    """Validate numeric parameter."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")


def _create_filled_array(
    shape: Shape,
    fill_value: float,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
) -> Array:
    """Create array filled with constant value using broadcasting."""
    _validate_shape(shape)
    _validate_shape(batch_dims)

    # Create scalar with fill value
    scalar = Array.from_numpy(np.array(fill_value, dtype=DType.to_numpy(dtype))).to(
        device
    )

    # Broadcast to desired shape
    array = broadcast_to(scalar, shape)

    if batch_dims:
        array = broadcast_batch_dims(array, batch_dims)

    return array


class RandomOp(Operation):
    """Base class for random number generators."""

    def __init__(self, shape: Shape, device: Device, seed: int, op_name: str):
        super().__init__(f"rng_{op_name}[shape={shape}]")
        self.shape = shape
        self.device = device
        self.seed = seed

        # Validate common parameters
        _validate_shape(shape)
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be int, got {type(seed)}")

    def forward(self, *args: Array) -> Array:
        """Forward pass for creation operations."""
        if args:
            raise ValueError(
                f"Creation operation requires 0 arguments, got {len(args)}"
            )

        res = Array(
            shape=self.shape,
            dtype=_DEFAULT_DTYPE,
            device=self.device,
            materialize=False,
            name=self.name,
        )

        res.set_maxpr(self.maxpr)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([], res)

        return res

    def compute_output_shape(self, *input_shapes) -> tuple:
        return self.shape

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        raise NotImplementedError("VJP for random creation operations is not defined.")

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        raise NotImplementedError("JVP for random creation operations is not defined.")


class RandNOp(RandomOp):
    """Normal distribution random number generator."""

    def __init__(
        self,
        shape: Shape,
        mean: float = 0.0,
        std: float = 1.0,
        device: Device = _DEFAULT_CPU,
        seed: int = _DEFAULT_SEED,
    ):
        super().__init__(shape, device, seed, "normal")
        self.mean = mean
        self.std = std

        _validate_numeric(mean, "Mean")
        _validate_numeric(std, "Std")
        if std <= 0:
            raise ValueError(f"Std must be positive, got {std}")

    def maxpr(self, args: list[Value], output: Array) -> None:
        ops.random.set_seed(self.seed)
        output.tensor_value = ops.random.normal(
            TensorType(
                output.dtype, output.shape, DeviceRef.from_device(output.device)
            ),
            mean=self.mean,
            std=self.std,
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np.random.seed(self.seed)
        np_result = np.random.normal(
            loc=self.mean, scale=self.std, size=output.shape
        ).astype(DType.to_numpy(output.dtype))
        output.impl = Tensor.from_numpy(np_result).to(output.device)


class RandUniformOp(RandomOp):
    """Uniform distribution random number generator."""

    def __init__(
        self,
        shape: Shape,
        lower: float = 0.0,
        upper: float = 1.0,
        device: Device = _DEFAULT_CPU,
        seed: int = _DEFAULT_SEED,
    ):
        super().__init__(shape, device, seed, "uniform")
        self.lower = lower
        self.upper = upper

        _validate_numeric(lower, "Lower bound")
        _validate_numeric(upper, "Upper bound")
        if upper <= lower:
            raise ValueError(
                f"Upper bound must be greater than lower bound, got {lower} and {upper}"
            )

    def maxpr(self, args: list[Value], output: Array) -> None:
        ops.random.set_seed(self.seed)
        output.tensor_value = ops.random.uniform(
            TensorType(
                output.dtype, output.shape, DeviceRef.from_device(output.device)
            ),
            range=(self.lower, self.upper),
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np.random.seed(self.seed)
        np_result = np.random.uniform(
            low=self.lower, high=self.upper, size=output.shape
        ).astype(DType.to_numpy(output.dtype))
        output.impl = Tensor.from_numpy(np_result).to(output.device)


def array(
    data: list | np.ndarray | float | int,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
) -> Array:
    """Create an array from Python list, numpy array, or scalar value."""
    if isinstance(data, list):
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    elif isinstance(data, np.ndarray):
        np_data = data.astype(DType.to_numpy(dtype))
    elif isinstance(data, (int, float)):
        # Handle scalar values
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    else:
        raise TypeError(f"Data must be a list, numpy array, or scalar, got {type(data)}")

    array = Array.from_numpy(np_data).to(device)
    return broadcast_batch_dims(array, batch_dims) if batch_dims else array


def arange(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
) -> Array:
    """Create an array with values from 0 to prod(shape)-1 reshaped to given shape."""
    _validate_shape(shape)

    total_size = np.prod(shape) if shape else 1
    np_data = np.arange(total_size, dtype=DType.to_numpy(dtype)).reshape(shape)
    array = Array.from_numpy(np_data).to(device)
    return broadcast_batch_dims(array, batch_dims) if batch_dims else array


def arange_like(template: Array) -> Array:
    """Create an array with values from 0 to prod(template.shape)-1 reshaped to template's shape."""
    return arange(template.shape, template.dtype, template.device, template.batch_dims)


def randn(
    shape: Shape,
    mean: float = 0.0,
    std: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
) -> Array:
    """Create array with normally distributed random values."""
    array = RandNOp(shape, mean, std, device, seed).forward()
    return broadcast_batch_dims(array, batch_dims) if batch_dims else array


def randn_like(
    template: Array, mean: float = 0.0, std: float = 1.0, seed: int = _DEFAULT_SEED
) -> Array:
    """Create an array with normally distributed random values like the template."""
    return randn(template.shape, mean, std, template.device, seed, template.batch_dims)


def rand(
    shape: Shape,
    lower: float = 0.0,
    upper: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
) -> Array:
    """Create array with uniformly distributed random values."""
    array = RandUniformOp(shape, lower, upper, device, seed).forward()
    return broadcast_batch_dims(array, batch_dims) if batch_dims else array


def rand_like(
    template: Array, lower: float = 0.0, upper: float = 1.0, seed: int = _DEFAULT_SEED
) -> Array:
    """Create an array with uniformly distributed random values like the template."""
    return rand(
        template.shape, lower, upper, template.device, seed, template.batch_dims
    )


def zeros(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
) -> Array:
    """Create an array filled with zeros."""
    return _create_filled_array(shape, 0.0, dtype, device, batch_dims)


def ones(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
) -> Array:
    """Create an array filled with ones."""
    return _create_filled_array(shape, 1.0, dtype, device, batch_dims)


def zeros_like(template: Array) -> Array:
    """Create an array of zeros with the same shape, dtype, and device as template."""
    return zeros(template.shape, template.dtype, template.device, template.batch_dims)


def ones_like(template: Array) -> Array:
    """Create an array of ones with the same shape, dtype, and device as template."""
    return ones(template.shape, template.dtype, template.device, template.batch_dims)


# Neural Network Initialization Methods


def xavier_uniform(
    shape: Shape,
    gain: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
) -> Array:
    """Xavier/Glorot uniform initialization for sigmoid/tanh activations.

    Samples from uniform distribution U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"Xavier initialization requires at least 2D shape, got {shape}"
        )

    fan_in, fan_out = shape[-2], shape[-1]
    std = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rand(shape, -std, std, device, seed, batch_dims)


def xavier_normal(
    shape: Shape,
    gain: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
) -> Array:
    """Xavier/Glorot normal initialization for sigmoid/tanh activations.

    Samples from normal distribution N(0, std²) where std = gain * sqrt(2 / (fan_in + fan_out))
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"Xavier initialization requires at least 2D shape, got {shape}"
        )

    fan_in, fan_out = shape[-2], shape[-1]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return randn(shape, 0.0, std, device, seed, batch_dims)


def he_uniform(
    shape: Shape,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
) -> Array:
    """He uniform initialization for ReLU activations.

    Samples from uniform distribution U(-a, a) where a = sqrt(6 / fan_in)
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(f"He initialization requires at least 2D shape, got {shape}")

    fan_in = shape[-2]
    bound = np.sqrt(6.0 / fan_in)
    return rand(shape, -bound, bound, device, seed, batch_dims)


def he_normal(
    shape: Shape,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
) -> Array:
    """He normal initialization for ReLU activations.

    Samples from normal distribution N(0, std²) where std = sqrt(2 / fan_in)
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(f"He initialization requires at least 2D shape, got {shape}")

    fan_in = shape[-2]
    std = np.sqrt(2.0 / fan_in)
    return randn(shape, 0.0, std, device, seed, batch_dims)


def lecun_uniform(
    shape: Shape,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
) -> Array:
    """LeCun uniform initialization for SELU activations.

    Samples from uniform distribution U(-a, a) where a = sqrt(3 / fan_in)
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"LeCun initialization requires at least 2D shape, got {shape}"
        )

    fan_in = shape[-2]
    bound = np.sqrt(3.0 / fan_in)
    return rand(shape, -bound, bound, device, seed, batch_dims)


def lecun_normal(
    shape: Shape,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
) -> Array:
    """LeCun normal initialization for SELU activations.

    Samples from normal distribution N(0, std²) where std = sqrt(1 / fan_in)
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"LeCun initialization requires at least 2D shape, got {shape}"
        )

    fan_in = shape[-2]
    std = np.sqrt(1.0 / fan_in)
    return randn(shape, 0.0, std, device, seed, batch_dims)
