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

"""Core Array class with improved organization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import Value

Shape = tuple[int, ...]
MaxprCallable = Callable[[list[Value], "Array"], None]
VJPRule = Callable[[list["Array"], "Array", "Array"], list["Array"]]
JVPRule = Callable[[list["Array"], list["Array"], "Array"], "Array"]

_DEFAULT_CPU = CPU()


class Array:
    """Core tensor-like array class with automatic differentiation support."""

    def __init__(
        self,
        shape: Shape,
        dtype: DType = DType.float32,
        device: Device = _DEFAULT_CPU,
        materialize: bool = False,
        name: str = "",
        batch_dims: Shape = (),
    ) -> None:
        self.shape = shape
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.device = device
        self.name = name
        self.args: list[Array] = []
        self.visited: bool = False
        self.tensor_value: Optional[Value] = None
        self.maxpr: Optional[MaxprCallable] = None
        self.vjp_rule: Optional[VJPRule] = None
        self.jvp_rule: Optional[JVPRule] = None
        self.traced: bool = False
        self._numpy_cache: Optional[np.ndarray] = None
        self.tangent: Optional[Array] = None
        self.cotangent: Optional[Array] = None
        self.stage_realization: bool = False
        self.kernel_impl_path: Optional[Path] = None

        if materialize:
            self.impl = Tensor(dtype, batch_dims + shape, device=device)
        else:
            self.impl = None

    @classmethod
    def from_impl(cls, impl: Tensor, name: str = "") -> Array:
        """Create Array from existing Tensor implementation."""
        if not isinstance(impl, Tensor):
            raise TypeError(f"Data must be a MAX Tensor, got {type(impl)}")
        if impl.shape is None:
            raise ValueError("Cannot create Array from None shape Tensor")

        instance = cls(
            shape=impl.shape, dtype=impl.dtype, device=impl.device, materialize=True
        )
        instance.impl = impl if impl else None
        instance.name = name
        return instance

    def copy_from(self, other: Array) -> None:
        """Copy data from another Array."""
        if self.shape != other.shape or self.dtype != other.dtype:
            raise ValueError("Shape or dtype mismatch for copy")
        self.impl = other.impl.copy()

    def add_arguments(self, *arg_nodes: Array) -> None:
        """Add an arguments to this Array's computation graph if traced."""
        for arg in arg_nodes:
            if not isinstance(arg, Array):
                raise TypeError(f"Argument must be an Array, got {type(arg)}")
            if arg.traced:
                self.traced = True
            if arg.stage_realization:
                self.stage_realization = True

        if self.traced or self.stage_realization:
            for arg in arg_nodes:
                self.args.append(arg)

    def realize(self) -> None:
        """Force computation of this Array."""
        if self.impl is not None:
            return

        from .graph_execution import realize_

        realize_([self])
        if self.impl is None:
            raise ValueError("Data is None after realization")

    def to_numpy(self) -> np.ndarray:
        """Get NumPy representation with caching."""
        self.realize()  # Ensure the Array is realized before converting
        if self._numpy_cache is None:
            if self.impl is None:
                raise ValueError("Cannot get NumPy array from None impl")
            self._numpy_cache = self.impl.to_numpy()
        return self._numpy_cache

    @classmethod
    def from_numpy(cls, np_array: np.ndarray) -> Array:
        """Create a new Array from a NumPy array."""
        if not isinstance(np_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(np_array)}")

        array = cls(
            shape=np_array.shape,
            dtype=DType.from_numpy(np_array.dtype),
            device=_DEFAULT_CPU,
            name=np_array.name if hasattr(np_array, "name") else "",
        )
        array.impl = Tensor.from_numpy(np_array)
        array.device = array.impl.device
        array._numpy_cache = np_array
        return array

    def get_arguments(self) -> list[Array]:
        """Get list of argument Arrays."""
        return list(self.args)

    def set_maxpr(self, fn: MaxprCallable) -> None:
        """Set the MAX PR function for this operation."""
        self.maxpr = fn

    def __repr__(self) -> str:
        """String representation of the Array."""
        # self.realize()
        from ..utils.formatting import format_shape_and_dtype

        return str(self.impl.to(CPU()).to_numpy()) + ":" + format_shape_and_dtype(self)

    def to(self, device: Device) -> Array:
        """Move Array to specified device."""
        # if self.impl is None:
        #     self.realize()
        new_impl = self.impl.to(device)
        return Array.from_impl(new_impl, name=self.name)

    # Operator overloading methods
    def __add__(self, other) -> Array:
        """Addition operator."""
        from ..ops.binary import add

        return add(self, other)

    def __mul__(self, other) -> Array:
        """Multiplication operator."""
        from ..ops.binary import mul

        return mul(self, other)

    def __sub__(self, other) -> Array:
        """Subtraction operator."""
        from ..ops.binary import sub

        return sub(self, other)

    def __pow__(self, power) -> Array:
        """Power operator."""
        from ..ops.binary import pow as power_op

        return power_op(self, power)

    def __truediv__(self, other) -> Array:
        """Division operator."""
        from ..ops.binary import div

        return div(self, other)

    def __matmul__(self, other: Array) -> Array:
        """Matrix multiplication operator (@)."""
        from ..ops.linalg import matmul

        return matmul(self, other)

    def __neg__(self) -> Array:
        """Negation operator."""
        from ..ops.unary import negate

        return negate(self)
