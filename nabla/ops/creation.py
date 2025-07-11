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
from max.graph import DeviceRef, TensorType, TensorValue, ops

from ..core.array import Array, Shape
from .operation import Operation
from .view import broadcast_batch_dims, broadcast_to

# Public API
__all__ = [
    "array",
    "arange",
    "ndarange",
    "ndarange_like",
    "randn",
    "randn_like",
    "rand",
    "rand_like",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "full_like",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_uniform",
    "lecun_normal",
    "glorot_uniform",
    "triu",
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
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric, got {type(value)}")


def _create_filled_array(
    shape: Shape,
    fill_value: float,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Create array filled with constant value using broadcasting."""
    _validate_shape(shape)
    _validate_shape(batch_dims)

    # WORKAROUND: Handle scalar boolean tensors (MAX tensor bug)
    # Workaround for MAX boolean tensor bug: ANY boolean tensor creation fails in MAX
    # when creating the scalar seed value, so we need special handling for all boolean cases
    if dtype == DType.bool:
        # Create boolean array by starting with float and converting
        try:
            # Try creating (1,) boolean array first
            scalar_1d = Array.from_numpy(
                np.array([fill_value], dtype=DType.to_numpy(dtype))
            ).to(device)
            scalar_1d.traced = traced

            if not shape:
                # For scalar boolean, reshape (1,) to ()
                from .view import reshape

                array = reshape(scalar_1d, ())
            else:
                # For non-scalar boolean, broadcast (1,) to target shape
                array = broadcast_to(scalar_1d, shape)
        except Exception:
            # Fallback: create as float and convert to bool
            scalar_float = Array.from_numpy(
                np.array([fill_value], dtype=np.float32)
            ).to(device)
            scalar_float.traced = traced

            if not shape:
                # Convert scalar float to scalar bool
                array = scalar_float.astype(dtype)
            else:
                # Broadcast float to shape, then convert to bool
                float_array = broadcast_to(scalar_float, shape)
                array = float_array.astype(dtype)
    else:
        # Original implementation for non-boolean types
        scalar = Array.from_numpy(np.array(fill_value, dtype=DType.to_numpy(dtype))).to(
            device
        )
        scalar.traced = traced

        if not shape:
            array = scalar
        else:
            array = broadcast_to(scalar, shape)

    if batch_dims:
        array = broadcast_batch_dims(array, batch_dims)

    return array


class RandomOp(Operation):
    """Base class for random number generators."""

    def __init__(
        self, shape: Shape, dtype: DType, device: Device, seed: int, op_name: str
    ):
        super().__init__(f"rng_{op_name}[shape={shape}]")
        self.shape = shape
        self.dtype = dtype
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
            dtype=self.dtype,
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
        dtype: DType = _DEFAULT_DTYPE,
        mean: float = 0.0,
        std: float = 1.0,
        device: Device = _DEFAULT_CPU,
        seed: int = _DEFAULT_SEED,
    ):
        super().__init__(shape, dtype, device, seed, "normal")
        self.mean = mean
        self.std = std

        _validate_numeric(mean, "Mean")
        _validate_numeric(std, "Std")
        if std <= 0:
            raise ValueError(f"Std must be positive, got {std}")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
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
        output.impl_(Tensor.from_numpy(np_result).to(output.device))


class RandUniformOp(RandomOp):
    """Uniform distribution random number generator."""

    def __init__(
        self,
        shape: Shape,
        dtype: DType = _DEFAULT_DTYPE,
        lower: float = 0.0,
        upper: float = 1.0,
        device: Device = _DEFAULT_CPU,
        seed: int = _DEFAULT_SEED,
    ):
        super().__init__(shape, dtype, device, seed, "uniform")
        self.lower = lower
        self.upper = upper

        _validate_numeric(lower, "Lower bound")
        _validate_numeric(upper, "Upper bound")
        if upper <= lower:
            raise ValueError(
                f"Upper bound must be greater than lower bound, got {lower} and {upper}"
            )

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
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
        output.impl_(Tensor.from_numpy(np_result).to(output.device))


def array(
    data: list | np.ndarray | float | int,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Create an array from Python list, numpy array, or scalar value."""
    if isinstance(data, list):
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    elif isinstance(data, np.ndarray):
        np_data = data.astype(DType.to_numpy(dtype))
    elif isinstance(data, int | float):
        # Handle scalar values
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    elif isinstance(data, (np.bool_, bool)):
        # Handle numpy boolean and Python boolean scalars
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    else:
        raise TypeError(
            f"Data must be a list, numpy array, or scalar, got {type(data)}"
        )

    # Special handling for boolean scalar tensors (MAX bug workaround)
    if np_data.shape == () and dtype == DType.bool:
        # For scalar boolean, create as float and convert
        float_array = Array.from_numpy(np_data.astype(np.float32)).to(device)
        float_array.traced = traced
        array = float_array.astype(DType.bool)

    else:
        array = Array.from_numpy(np_data).to(device)
        array.traced = traced

    return broadcast_batch_dims(array, batch_dims) if batch_dims else array


class ArangeOp(Operation):
    """Operation to create a 1D array with evenly spaced values."""

    def __init__(
        self,
        start: float | int,
        stop: float | int,
        step: float | int,
        dtype: DType,
        device: Device,
    ):
        super().__init__(f"arange[start={start},stop={stop},step={step}]")
        self.start = start
        self.stop = stop
        self.step = step
        self.dtype = dtype
        self.device = device

        # Pre-compute the output shape using numpy's robust implementation
        # This handles all edge cases like float steps, negative steps, etc.
        self._np_arange_for_shape = np.arange(
            start, stop, step, dtype=DType.to_numpy(dtype)
        )
        self.shape = self._np_arange_for_shape.shape

    def forward(self, *args: Array) -> Array:
        """Forward pass for the arange creation operation."""
        if args:
            raise ValueError(
                f"Creation operation 'arange' requires 0 arguments, got {len(args)}"
            )

        res = Array(
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            materialize=False,
            name=self.name,
        )

        res.set_maxpr(self.maxpr)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule

        if not res.stage_realization:
            self.eagerxpr([], res)

        return res

    def compute_output_shape(self, *input_shapes) -> tuple:
        return self.shape

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        """Graph-mode execution using max.ops.arange."""
        # This assumes an equivalent ops.arange exists in the MAX graph library.
        # This is a common and expected operation for a backend.
        output.tensor_value = ops.range(
            start=self.start,
            stop=self.stop,
            step=self.step,
            dtype=output.dtype,
            device=DeviceRef.from_device(output.device),
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager-mode execution using numpy."""
        # We can reuse the numpy array we created for the shape calculation
        output.impl_(Tensor.from_numpy(self._np_arange_for_shape).to(output.device))

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        # The arange operation does not depend on any Array inputs,
        # so its gradient is not defined in this context.
        raise NotImplementedError("VJP for 'arange' creation operation is not defined.")

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        # The arange operation does not depend on any Array inputs,
        # so its gradient is not defined in this context.
        raise NotImplementedError("JVP for 'arange' creation operation is not defined.")


def arange(
    start: int | float,
    stop: int | float | None = None,
    step: int | float | None = None,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    traced: bool = False,
    batch_dims: Shape = (),
) -> Array:
    """
    Return evenly spaced values within a given interval.

    This function follows the JAX/NumPy `arange` API.

    Args:
        start: Start of interval. The interval includes this value.
        stop: End of interval. The interval does not include this value. If None,
            the range is `[0, start)`.
        step: Spacing between values. The default step size is 1.
        dtype: The data type of the output array.
        device: The device to place the array on.
        traced: Whether the operation should be traced in the graph.

    Returns:
        A 1D array of evenly spaced values.
    """
    # Handle the case where only one positional argument is provided, e.g., arange(5)
    if stop is None:
        stop = start
        start = 0

    if step is None:
        step = 1

    _validate_numeric(start, "start")
    _validate_numeric(stop, "stop")
    _validate_numeric(step, "step")

    if step == 0:
        raise ValueError("arange: step cannot be zero.")

    op = ArangeOp(start=start, stop=stop, step=step, dtype=dtype, device=device)
    array = op.forward()
    array.traced = traced
    if batch_dims:
        array = broadcast_batch_dims(array, batch_dims)
    return array


def ndarange(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Create an array with values from 0 to prod(shape)-1 reshaped to given shape."""
    return arange(
        0, int(np.prod(shape)), 1, dtype=dtype, device=device, traced=traced
    ).reshape(shape)


def ndarange_like(template: Array) -> Array:
    """Create an array with values from 0 to prod(template.shape)-1 reshaped to template's shape."""
    return ndarange(
        template.shape,
        template.dtype,
        template.device,
        template.batch_dims,
        template.traced,
    )


def randn(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    mean: float = 0.0,
    std: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Create array with normally distributed random values."""
    array = RandNOp(shape, dtype, mean, std, device, seed).forward()
    array.traced = traced
    return broadcast_batch_dims(array, batch_dims) if batch_dims else array


def randn_like(
    template: Array, mean: float = 0.0, std: float = 1.0, seed: int = _DEFAULT_SEED
) -> Array:
    """Create an array with normally distributed random values like the template."""
    res = randn(
        template.shape,
        template.dtype,
        mean,
        std,
        template.device,
        seed,
        template.batch_dims,
        traced=template.traced,
    )
    return res


def rand(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    lower: float = 0.0,
    upper: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Create array with uniformly distributed random values."""
    array = RandUniformOp(shape, dtype, lower, upper, device, seed).forward()
    array.traced = traced
    return broadcast_batch_dims(array, batch_dims) if batch_dims else array


def rand_like(
    template: Array, lower: float = 0.0, upper: float = 1.0, seed: int = _DEFAULT_SEED
) -> Array:
    """Create an array with uniformly distributed random values like the template."""
    res = rand(
        template.shape,
        template.dtype,
        lower,
        upper,
        template.device,
        seed,
        template.batch_dims,
        traced=template.traced,
    )
    return res


def zeros(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Create an array filled with zeros."""
    return _create_filled_array(shape, 0.0, dtype, device, batch_dims, traced=traced)


def ones(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Create an array filled with ones."""
    return _create_filled_array(shape, 1.0, dtype, device, batch_dims, traced=traced)


def zeros_like(template: Array) -> Array:
    """Create an array of zeros with the same shape, dtype, and device as template."""
    return zeros(
        template.shape,
        template.dtype,
        template.device,
        template.batch_dims,
        traced=template.traced,
    )


def ones_like(template: Array) -> Array:
    """Create an array of ones with the same shape, dtype, and device as template."""
    return ones(
        template.shape,
        template.dtype,
        template.device,
        template.batch_dims,
        traced=template.traced,
    )


def full_like(template: Array, fill_value: float) -> Array:
    """Create an array filled with a specific value, with the same shape, dtype, and device as template."""
    return _create_filled_array(
        template.shape,
        fill_value,
        template.dtype,
        template.device,
        template.batch_dims,
        template.traced,
    )


# Neural Network Initialization Methods


def xavier_uniform(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    gain: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
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
    return rand(shape, dtype, -std, std, device, seed, batch_dims, traced=traced)


def xavier_normal(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    gain: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
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
    return randn(shape, dtype, 0.0, std, device, seed, batch_dims, traced=traced)


def he_uniform(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """He uniform initialization for ReLU activations.

    Samples from uniform distribution U(-a, a) where a = sqrt(6 / fan_in)
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(f"He initialization requires at least 2D shape, got {shape}")

    fan_in = shape[-2]
    bound = np.sqrt(6.0 / fan_in)
    return rand(shape, dtype, -bound, bound, device, seed, batch_dims, traced=traced)


def he_normal(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """He normal initialization for ReLU activations.

    Samples from normal distribution N(0, std²) where std = sqrt(2 / fan_in)
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(f"He initialization requires at least 2D shape, got {shape}")

    fan_in = shape[-2]
    std = np.sqrt(2.0 / fan_in)
    return randn(shape, dtype, 0.0, std, device, seed, batch_dims, traced=traced)


def lecun_uniform(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
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
    return rand(shape, dtype, -bound, bound, device, seed, batch_dims, traced=traced)


def lecun_normal(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
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
    return randn(shape, dtype, 0.0, std, device, seed, batch_dims, traced=traced)


def glorot_uniform(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    gain: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Glorot/Xavier uniform initialization for sigmoid/tanh activations.

    Samples from uniform distribution U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"Glorot initialization requires at least 2D shape, got {shape}"
        )

    fan_in, fan_out = shape[-2], shape[-1]
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rand(shape, dtype, -bound, bound, device, seed, batch_dims, traced=traced)


def triu(x, k=0):
    """
    Return the upper triangular part of an array.

    Args:
        x: Input array (batch, seq_len, seq_len)
        k: Diagonal offset (0 = main diagonal, > 0 = above, < 0 = below)

    Returns:
        Upper triangular part of the input array
    """
    from .special import where

    mask = ndarange((x.shape[-1],)) < ndarange((x.shape[-1],))[:, None] + k
    return where(mask, x, array(0, dtype=x.dtype, device=x.device))
