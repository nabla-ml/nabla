# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
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

# Default device singleton to avoid function calls in defaults
_DEFAULT_CPU = CPU()


def array(
    data: list | np.ndarray,
    dtype: DType = DType.float32,
    device: Device = _DEFAULT_CPU,
) -> Array:
    """Create an array from Python list or numpy array."""
    if isinstance(data, list):
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    elif isinstance(data, np.ndarray):
        np_data = data.astype(DType.to_numpy(dtype))
    else:
        raise TypeError(f"Data must be a list or numpy array, got {type(data)}")

    tensor = Tensor.from_numpy(np_data).to(device)
    return Array.from_impl(tensor)


def arange(
    shape: Shape, dtype: DType = DType.float32, device: Device = _DEFAULT_CPU
) -> Array:
    """Create an array with values from 0 to prod(shape)-1 reshaped to given shape."""
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}")

    total_size = np.prod(shape) if shape else 1
    np_data = np.arange(total_size, dtype=DType.to_numpy(dtype)).reshape(shape)
    tensor = Tensor.from_numpy(np_data).to(device)

    return Array.from_impl(tensor)


class RandNOp(Operation):
    """Normal distribution random number generator."""

    def __init__(
        self,
        shape: Shape,
        mean: float = 0.0,
        std: float = 1.0,
        device: Device = _DEFAULT_CPU,
        seed: int = 0,
    ):
        super().__init__(f"rng_normal[shape={shape}]")
        self.shape = shape
        self.mean = mean
        self.std = std
        self.device = device
        self.seed = seed

        # Validate parameters
        if not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape)}")
        if not isinstance(mean, int | float):
            raise TypeError(f"Mean must be numeric, got {type(mean)}")
        if not isinstance(std, int | float):
            raise TypeError(f"Std must be numeric, got {type(std)}")
        if std <= 0:
            raise ValueError(f"Std must be positive, got {std}")
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be int, got {type(seed)}")

    def forward(self, *args: Array) -> Array:
        """Forward pass for creation operations (no arguments) with compatible signature."""
        if len(args) != 0:
            raise ValueError(
                f"Creation operation requires 0 arguments, got {len(args)}"
            )

        res = Array(
            shape=self.shape,
            dtype=DType.float32,  # Creation ops should define their dtype
            device=self.device,
            materialize=False,
            name=self.name,
        )

        res.set_maxpr(self.maxpr)
        # Creation ops don't have primals, so VJP/JVP rules are simplified
        # They are set to produce empty/zero grads respectively
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule

        from .base import EAGERMODE

        if EAGERMODE:
            self.eagerxpr([], res)  # Pass empty list for args

        return res

    def compute_output_shape(
        self, *input_shapes
    ) -> tuple:  # Added *input_shapes for consistency, though not used
        """Compute the output shape."""
        return self.shape

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

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """VJP for random creation - no gradients to propagate."""
        return []

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """JVP for random creation - zero tangent."""
        from .creation import zeros

        return zeros(output.shape, output.dtype, output.device)


def randn(
    shape: Shape,
    mean: float = 0.0,
    std: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = 0,
) -> Array:
    """Create array with normally distributed random values."""
    op = RandNOp(shape, mean, std, device, seed)
    return op.forward()


def zeros(
    shape: Shape, dtype: DType = DType.float32, device: Device = _DEFAULT_CPU
) -> Array:
    """Create an array filled with zeros."""
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}")

    np_data = np.zeros(shape, dtype=DType.to_numpy(dtype))
    tensor = Tensor.from_numpy(np_data).to(device)

    return Array.from_impl(tensor)


def ones(
    shape: Shape, dtype: DType = DType.float32, device: Device = _DEFAULT_CPU
) -> Array:
    """Create an array filled with ones."""
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}")

    np_data = np.ones(shape, dtype=DType.to_numpy(dtype))
    tensor = Tensor.from_numpy(np_data).to(device)

    return Array.from_impl(tensor)


def zeros_like(template: Array) -> Array:
    """Create an array of zeros with the same shape, dtype, and device as template."""
    return zeros(template.shape, template.dtype, template.device)


def ones_like(template: Array) -> Array:
    """Create an array of ones with the same shape, dtype, and device as template."""
    return ones(template.shape, template.dtype, template.device)
