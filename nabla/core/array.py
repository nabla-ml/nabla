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
from typing import Optional, Union

import numpy as np
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, Value

Shape = tuple[int, ...]
MaxprCallable = Callable[[list[TensorValue], "Array"], None]
VJPRule = Callable[[list["Array"], "Array", "Array"], list["Array"]]
JVPRule = Callable[[list["Array"], list["Array"], "Array"], "Array"]

_DEFAULT_CPU = CPU()


class Array:
    """Core tensor-like array class with automatic differentiation support."""

    # Class-level type annotations for better Pylance support
    shape: Shape
    batch_dims: Shape
    dtype: DType
    device: Device
    name: str
    args: list[Array]
    visited: bool
    tensor_value: Optional[Union[Value, TensorValue, TensorValueLike]]
    maxpr: Optional[MaxprCallable]
    vjp_rule: Optional[VJPRule]
    jvp_rule: Optional[JVPRule]
    traced: bool
    _numpy_cache: Optional[np.ndarray]
    tangent: Optional[Array]
    cotangent: Optional[Array]
    stage_realization: bool
    kernel_impl_path: Optional[Path]
    custom_kernel_path: Optional[Path]
    impl: Optional[Tensor]

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
        self.tensor_value: Optional[Union[Value, TensorValue, TensorValueLike]] = None
        self.maxpr: Optional[MaxprCallable] = None
        self.vjp_rule: Optional[VJPRule] = None
        self.jvp_rule: Optional[JVPRule] = None
        self.traced: bool = False
        self._numpy_cache: Optional[np.ndarray] = None
        self.tangent: Optional[Array] = None
        self.cotangent: Optional[Array] = None
        self.stage_realization: bool = False
        self.kernel_impl_path: Optional[Path] = None
        self.custom_kernel_path: Optional[Path] = None

        # Debug print for newly created arrays
        # print(f"[DEBUG] Created array: name='{name}', shape={shape}, dtype={dtype}")

        if materialize:
            self.impl = Tensor(dtype, batch_dims + shape, device=device)
        else:
            self.impl = None

    @property
    def size(self) -> int:
        """Return the total number of elements in the array."""
        if not self.shape:
            return 1  # Scalar array
        size = 1
        for dim in self.shape:
            size *= dim
        return size

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
        if other.impl is not None:
            self.impl = other.impl.copy()
        else:
            self.impl = None

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
            name=getattr(np_array, "name", ""),
        )

        # WORKAROUND: Handle scalar boolean arrays to avoid MAX library bug
        # The MAX library's tensor.view(DType.bool) fails for scalar tensors
        if np_array.dtype == bool and np_array.shape == ():
            # For scalar boolean, convert to float32 to avoid the bug
            float_array = np_array.astype(np.float32)
            array.impl = Tensor.from_numpy(float_array)
            # Update the dtype to reflect what we actually stored
            array.dtype = DType.float32
        else:
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

        if self.impl is not None:
            return (
                str(self.impl.to(CPU()).to_numpy()) + ":" + format_shape_and_dtype(self)
            )
        else:
            return (
                f"Array(shape={self.shape}, dtype={self.dtype}, unrealized):"
                + format_shape_and_dtype(self)
            )

    def to(self, device: Device) -> Array:
        """Move Array to specified device."""
        if self.impl:
            new_impl = self.impl.to(device)
            return Array.from_impl(new_impl, name=self.name)
        else:
            from ..ops.unary import transfer_to

            return transfer_to(self, device)

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

    def __floordiv__(self, other) -> Array:
        """Floor division operator (//)."""
        from ..ops.binary import floordiv

        return floordiv(self, other)

    def __matmul__(self, other) -> Array:
        """Matrix multiplication operator (@)."""
        from ..ops.linalg import matmul

        return matmul(self, other)

    def __neg__(self) -> Array:
        """Negation operator."""
        from ..ops.unary import negate

        return negate(self)

    def __mod__(self, other) -> Array:
        """Modulo operator (%)."""
        from ..ops.binary import mod

        return mod(self, other)

    # Comparison operators
    def __lt__(self, other) -> Array:
        """Less than operator (<)."""
        from ..ops.binary import greater_equal
        from ..ops.unary import logical_not

        # a < b is equivalent to not (a >= b)
        return logical_not(greater_equal(self, other))

    def __le__(self, other) -> Array:
        """Less than or equal operator (<=)."""
        from ..ops.binary import greater_equal

        # a <= b is equivalent to b >= a
        return greater_equal(other, self)

    def __gt__(self, other) -> Array:
        """Greater than operator (>)."""
        from ..ops.binary import greater_equal
        from ..ops.unary import logical_not

        # a > b is equivalent to not (b >= a)
        return logical_not(greater_equal(other, self))

    def __ge__(self, other) -> Array:
        """Greater than or equal operator (>=)."""
        from ..ops.binary import greater_equal

        return greater_equal(
            self, other
        )  # Hash and equality for making Arrays usable as dictionary keys

    def __hash__(self) -> int:
        """Make Arrays hashable based on object identity.

        This allows Arrays to be used as dictionary keys in optimizers.
        Two Arrays are considered equal only if they are the same object.
        """
        return id(self)

    # Reverse operators for when Array is on the right-hand side
    def __radd__(self, other) -> Array:
        """Reverse addition operator (other + self)."""
        from ..ops.binary import add

        return add(other, self)

    def __rmul__(self, other) -> Array:
        """Reverse multiplication operator (other * self)."""
        from ..ops.binary import mul

        return mul(other, self)

    def __rsub__(self, other) -> Array:
        """Reverse subtraction operator (other - self)."""
        from ..ops.binary import sub

        return sub(other, self)

    def __rtruediv__(self, other) -> Array:
        """Reverse division operator (other / self)."""
        from ..ops.binary import div

        return div(other, self)

    def __rpow__(self, other) -> Array:
        """Reverse power operator (other ** self)."""
        from ..ops.binary import pow as power_op

        return power_op(other, self)

    def __getitem__(self, key) -> Array:
        """Array slicing using standard Python syntax.

        Supports both basic indexing (slices, integers) and advanced indexing (Array indices).

        Examples::

            arr[1:3]        # Slice first dimension
            arr[:, 2:5]     # Slice second dimension
            arr[1:3, 2:5]   # Slice multiple dimensions
            arr[-2:]        # Negative indices
            arr[..., :2]    # Ellipsis (all dimensions up to last)

            # Advanced indexing with Array indices:
            indices = nb.array([0, 2, 1])
            arr[indices]    # Gather elements along first axis
            arr[indices, :] # Gather rows
        """

        # Check if this is advanced indexing with Array indices
        if isinstance(key, Array):
            # Single Array index - use gather along axis 0
            from ..ops.indexing import gather

            return gather(self, key, axis=0)
        elif isinstance(key, tuple) and any(isinstance(k, Array) for k in key):
            # Mixed indexing with Array indices in tuple
            return self._handle_mixed_advanced_indexing(key)

        # Handle single slice, integer, or ellipsis (original logic)
        if isinstance(key, slice | int | type(...)):
            key = (key,)
        elif not isinstance(key, tuple):
            raise TypeError(
                f"Array indices must be integers, slices, ellipsis, Arrays, or tuples, got {type(key)}"
            )

        # Handle ellipsis expansion
        if ... in key:
            ellipsis_idx = key.index(...)
            # Count non-ellipsis elements
            non_ellipsis_count = len([k for k in key if k is not ...])
            # Calculate how many slice(None) to insert
            missing_dims = len(self.shape) - non_ellipsis_count
            if missing_dims < 0:
                missing_dims = 0  # Don't allow negative

            # Build expanded key
            expanded_key = (
                key[:ellipsis_idx]
                + (slice(None),) * missing_dims
                + key[ellipsis_idx + 1 :]
            )
            key = expanded_key

        # Special case: if we have indices but the array is scalar, that's an error
        if (
            len(self.shape) == 0
            and len(key) > 0
            and not (len(key) == 1 and key[0] is ...)
        ):
            raise IndexError(f"Too many indices for array: expected 0, got {len(key)}")

        # Convert integers to slices and build slice list
        # Track which dimensions should be squeezed (removed) due to integer indexing
        slices = []
        squeeze_axes = []
        for i, k in enumerate(key):
            if i >= len(self.shape):
                raise IndexError(
                    f"Too many indices for array: expected {len(self.shape)}, got {len(key)}"
                )

            if isinstance(k, int):
                # Convert integer index to slice
                if k < 0:
                    # Handle negative indexing
                    k = self.shape[i] + k
                slices.append(slice(k, k + 1))
                squeeze_axes.append(i)  # Mark this dimension for squeezing
            elif isinstance(k, slice):
                slices.append(k)
            else:
                raise TypeError(
                    f"Array index {i} must be an integer or slice, got {type(k)}"
                )

        # Create ArraySliceOp with squeeze information
        from ..ops.view import ArraySliceOp

        op = ArraySliceOp(slices, squeeze_axes)
        return op.forward(self)

    def astype(self, dtype: DType) -> Array:
        """Convert array to a different data type.

        Args:
            dtype: Target data type

        Returns:
            New Array with the specified data type
        """
        if self.dtype == dtype:
            return self  # No conversion needed

        # Use nabla's cast operation
        from ..ops.unary import cast

        return cast(self, dtype)

    def sum(self, axes=None, keep_dims=False) -> Array:
        """Sum array elements over given axes.

        Args:
            axes: Axis or axes along which to sum. Can be int, list of ints, or None (sum all)
            keep_dims: If True, reduced axes are left as dimensions with size 1

        Returns:
            Array with the sum along the specified axes

        Examples::

            arr.sum()           # Sum all elements
            arr.sum(axis=0)     # Sum along first axis
            arr.sum(axis=[0,1]) # Sum along first two axes
        """
        from ..ops.reduce import sum as array_sum

        return array_sum(self, axes=axes, keep_dims=keep_dims)

    def reshape(self, shape: Shape) -> Array:
        """Change the shape of an array without changing its data.

        Args:
            shape: New shape for the array

        Returns:
            Array with the new shape

        Examples::

            arr.reshape((2, 3))     # Reshape to 2x3
            arr.reshape((-1,))      # Flatten to 1D (note: -1 not yet supported)
        """
        from ..ops.view import reshape

        return reshape(arg=self, shape=shape)

    def permute(self, axes: tuple[int, ...]) -> Array:
        """Permute the dimensions of the array.

        Args:
            axes: List of integers specifying the new order of dimensions

        Returns:
            Array with dimensions permuted according to the specified axes

        Examples::

            arr.permute([1, 0]) # If arr.shape is (2, 3), this will return an array with shape (3, 2)
        """
        from ..ops.view import permute

        return permute(self, axes)

    def at(self, key, value):
        """Update array at specified indices/slices, returning new array."""
        from ..ops.binary import add, sub
        from ..ops.view import pad

        # Convert value to Array if needed
        if not isinstance(value, Array):
            # Match the dtype of the original array
            value_np = np.array(value, dtype=self.dtype.to_numpy())
            value = Array.from_numpy(value_np)
        else:
            # If value is already an Array, ensure it matches our dtype
            if value.dtype != self.dtype:
                value_np = value.to_numpy().astype(self.dtype.to_numpy())
                value = Array.from_numpy(value_np)

        # Handle single slice, integer, or ellipsis
        if isinstance(key, slice | int | type(...)):
            key = (key,)
        elif not isinstance(key, tuple):
            raise TypeError(
                f"Array indices must be integers, slices, ellipsis, or tuples, got {type(key)}"
            )

        # Handle ellipsis expansion (same logic as __getitem__)
        if ... in key:
            ellipsis_idx = key.index(...)
            # Count non-ellipsis elements
            non_ellipsis_count = len([k for k in key if k is not ...])
            # Calculate how many slice(None) to insert
            missing_dims = len(self.shape) - non_ellipsis_count
            if missing_dims < 0:
                missing_dims = 0  # Don't allow negative

            # Build expanded key
            expanded_key = (
                key[:ellipsis_idx]
                + (slice(None),) * missing_dims
                + key[ellipsis_idx + 1 :]
            )
            key = expanded_key

        # Convert integers to slices for pad operation, handling negative indices
        slices = []
        for i, k in enumerate(key):
            if isinstance(k, int):
                # Handle negative indexing before converting to slice
                if k < 0:
                    k = self.shape[i] + k
                slices.append(slice(k, k + 1))
            elif isinstance(k, slice):
                slices.append(k)
            else:
                raise TypeError(f"Unsupported key type: {type(k)}")

        # 1. Slice out the part being replaced (using converted slices for consistency)
        sliced_part = self[tuple(slices)]

        # 2. Ensure value has the same shape as sliced_part
        if value.shape != sliced_part.shape:
            # Try to reshape/broadcast value to match sliced shape
            value_np = value.to_numpy()
            try:
                if value_np.size == np.prod(sliced_part.shape):
                    # Reshape if same number of elements
                    value = Array.from_numpy(value_np.reshape(sliced_part.shape))
                else:
                    # Try broadcasting
                    value = Array.from_numpy(
                        np.broadcast_to(value_np, sliced_part.shape)
                    )
            except:
                raise ValueError(
                    f"Cannot broadcast value shape {value.shape} to sliced shape {sliced_part.shape}"
                )

        # 3. Calculate the difference
        diff = sub(value, sliced_part)

        # 4. Pad the difference to full array shape (using converted slices)
        padded_diff = pad(diff, slices, self.shape)

        # 5. Add to original array
        result = add(self, padded_diff)

        return result

    # Comparison operators
    def __eq__(self, other) -> bool:
        """Object identity comparison for hashability.

        This returns True only if both Arrays are the same object.
        For element-wise comparison, use nb.equal(a, b) explicitly.
        """
        return isinstance(other, Array) and self is other

    def __ne__(self, other) -> bool:
        """Object identity inequality comparison for hashability.

        This returns True if the Arrays are different objects.
        For element-wise comparison, use nb.not_equal(a, b) explicitly.
        """
        return not self.__eq__(other)

    def set(self, key, value) -> Array:
        """Set values at specified indices/slices, returning a new array.

        This is a functional operation that returns a new Array with the specified
        values updated, leaving the original Array unchanged.

        Args:
            key: Index specification (int, slice, tuple of indices/slices, ellipsis)
            value: Value(s) to set at the specified location

        Returns:
            New Array with updated values

        Examples:
            new_arr = arr.set(1, 99.0)              # Set single element
            new_arr = arr.set((1, 2), 99.0)         # Set element at (1,2)
            new_arr = arr.set(slice(1, 3), 99.0)    # Set slice
            new_arr = arr.set(..., 99.0)            # Set with ellipsis
        """
        return self.at(key, value)

    def _handle_mixed_advanced_indexing(self, key: tuple) -> Array:
        """Handle mixed indexing with Array indices and slices/integers.

        Args:
            key: Tuple containing mix of Array indices, slices, and integers

        Returns:
            Array result of advanced indexing
        """
        from ..ops.indexing import gather

        # For now, implement a simplified version that handles the most common case:
        # Array index in first position, followed by slices/integers
        # More complex cases can be added later

        # Find the first Array index
        array_index_pos = None
        for i, k in enumerate(key):
            if isinstance(k, Array):
                if array_index_pos is None:
                    array_index_pos = i
                else:
                    # Multiple Array indices - more complex case
                    raise NotImplementedError(
                        "Multiple Array indices not yet supported. "
                        "Use gather/scatter operations directly for complex indexing."
                    )

        if array_index_pos is None:
            # No Array indices found - shouldn't reach here
            raise ValueError("Expected Array index in mixed indexing")

        array_index = key[array_index_pos]

        if array_index_pos == 0:
            # Array index in first position: arr[indices, slice1, slice2, ...]
            remaining_key = key[1:]

            # First apply gather along axis 0
            gathered = gather(self, array_index, axis=0)

            # Then apply remaining indexing if any
            if remaining_key:
                # The remaining key should be applied starting from the first dimension
                # after the array-indexed dimension. Since we array-indexed dimension 0,
                # the remaining key applies to dimensions 1, 2, 3, ... of the original shape
                # which are dimensions 1, 2, 3, ... of the gathered result.
                # So we need to prepend a slice(None) to cover the new first dimension from gather
                full_key = (slice(None),) + remaining_key
                return gathered[full_key]
            else:
                return gathered
        else:
            # Array index not in first position - more complex
            # For now, we'll convert to a sequence of operations
            # This is a simplified implementation
            raise NotImplementedError(
                f"Array index at position {array_index_pos} not yet supported. "
                "Use gather operation directly or put Array index first."
            )

    def __setitem__(self, key, value) -> None:
        """Array assignment using standard Python syntax.

        Supports both basic assignment (slices, integers) and advanced assignment (Array indices).

        Examples::

            arr[1:3] = value        # Assign to slice
            arr[:, 2:5] = value     # Assign to slice in second dimension

            # Advanced indexing with Array indices:
            indices = nb.array([0, 2, 1])
            arr[indices] = value    # Scatter values to specified indices
        """
        # Convert value to Array if needed
        if not isinstance(value, Array):
            from ..ops.creation import array

            value = array(value)

        # Check if this is advanced indexing with Array indices
        if isinstance(key, Array):
            # Single Array index - use scatter along axis 0
            self._setitem_with_array_index(key, value, axis=0)
        elif isinstance(key, tuple) and any(isinstance(k, Array) for k in key):
            # Mixed indexing with Array indices
            self._setitem_mixed_advanced_indexing(key, value)
        else:
            # Basic indexing - not implemented for now
            raise NotImplementedError(
                "Basic slice assignment not yet implemented. "
                "Use Array indices for scatter operations."
            )

    def _setitem_with_array_index(
        self, indices: Array, values: Array, axis: int = 0
    ) -> None:
        """Helper method for setitem with Array indices.

        Args:
            indices: Array of indices where to place values
            values: Array of values to place
            axis: Axis along which to scatter
        """
        from ..ops.indexing import scatter

        # Create new array by scattering values into a copy of self
        # Note: This creates a new array rather than in-place modification
        # In-place modification would require mutable arrays
        new_array = scatter(
            target_shape=self.shape, indices=indices, values=values, axis=axis
        )

        # Update self's implementation to point to new data
        # This simulates in-place modification
        self.impl = new_array.impl
        self._numpy_cache = None  # Invalidate cache

    def _setitem_mixed_advanced_indexing(self, key: tuple, value: Array) -> None:
        """Helper method for mixed advanced indexing assignment.

        Args:
            key: Tuple containing mix of Array indices, slices, and integers
            value: Array to assign
        """
        # For now, implement a simplified version
        # Find the first Array index
        array_index_pos = None
        for i, k in enumerate(key):
            if isinstance(k, Array):
                if array_index_pos is None:
                    array_index_pos = i
                else:
                    raise NotImplementedError(
                        "Multiple Array indices not yet supported"
                    )

        if array_index_pos != 0:
            raise NotImplementedError(
                "Array index must be in first position for assignment"
            )

        array_index = key[0]
        remaining_key = key[1:]

        if remaining_key:
            # Need to handle partial assignment like arr[indices, :, slice] = value
            raise NotImplementedError(
                "Mixed Array index with slices in assignment not yet supported"
            )
        else:
            # Simple case: arr[indices] = value
            self._setitem_with_array_index(array_index, value, axis=0)
