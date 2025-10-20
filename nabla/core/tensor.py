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

"""Core Tensor class with improved organization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Union

import numpy as np
from max.driver import CPU, Device
from max.driver import Tensor as MAXTensor
from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, Value

Shape = tuple[int, ...]
MaxprCallable = Callable[[list[TensorValue], "Tensor"], None]
VJPRule = Callable[[list["Tensor"], "Tensor", "Tensor"], list["Tensor"]]
JVPRule = Callable[[list["Tensor"], list["Tensor"], "Tensor"], "Tensor"]

_DEFAULT_CPU = CPU()


class Tensor:
    """Core tensor-like tensor class with automatic differentiation support."""

    # Class-level type annotations for better Pylance support
    shape: Shape
    batch_dims: Shape
    dtype: DType
    device: Device
    name: str
    args: list[Tensor]
    visited: bool
    tensor_value: Union[Value, TensorValue, TensorValueLike] | None
    maxpr: MaxprCallable | None
    vjp_rule: VJPRule | None
    jvp_rule: JVPRule | None
    traced: bool
    requires_grad: bool | None
    tangent: Tensor | None
    cotangent: Tensor | None
    grad: Tensor | None
    stage_realization: bool
    kernel_impl_path: Path | None
    custom_kernel_path: Path | None
    _impl: Union[np.ndarray, MAXTensor] | None

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
        self.logical_device = device
        self.name = name
        self.args: list[Tensor] = []
        self.visited: bool = False
        self.tensor_value: Union[Value, TensorValue, TensorValueLike] | None = None
        self.maxpr: MaxprCallable | None = None
        self.vjp_rule: VJPRule | None = None
        self.jvp_rule: JVPRule | None = None
        self.traced: bool = False
        self.tangent: Tensor | None = None
        self.cotangent: Tensor | None = None
        self.requires_grad: bool | None = None
        self.grad: Tensor | None = None
        self.stage_realization: bool = False
        self.kernel_impl_path: Path | None = None
        self.custom_kernel_path: Path | None = None

        from ..ops.operation import Operation

        self.creator_op: Operation | None = None

        if materialize:
            self._impl = MAXTensor(dtype, batch_dims + shape, device=device)
        else:
            self._impl = None

    @property
    def device(self) -> Device:
        """Get the logical device of this Tensor. This can differ from the logical device and will show the actual device the buffer lives on."""
        if self._impl is None:
            return self.logical_device
        if isinstance(self._impl, MAXTensor):
            return self._impl.device
        elif isinstance(self._impl, np.ndarray):
            return _DEFAULT_CPU
        else:
            raise TypeError(f"Unsupported implementation type: {type(self._impl)}")

    @property
    def impl(self) -> MAXTensor | None:
        """Get the max.Tensor representation of this Tensor. If the underlying _impl field is a Numpy tensor, convert it to a Tensor."""
        if isinstance(self._impl, MAXTensor):
            return self._impl
        elif isinstance(self._impl, np.ndarray):
            # Convert numpy tensor to Tensor
            val = MAXTensor.from_numpy(self._impl)
            if val.device != self.logical_device:
                val = val.to(self.logical_device)
            return val
        else:
            return None

    def impl_(self, value: Union[np.ndarray, MAXTensor] | None) -> None:
        """Set the implementation of this Tensor to a Numpy tensor or Tensor."""
        self._impl = value

    @property
    def size(self) -> int:
        """Return the total number of elements in the tensor."""
        if not self.shape:
            return 1  # Scalar tensor
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    @classmethod
    def from_impl(cls, impl: MAXTensor, name: str = "") -> Tensor:
        """Create Tensor from existing max.Tensor implementation."""
        if not isinstance(impl, MAXTensor):
            raise TypeError(f"Data must be a MAX Tensor, got {type(impl)}")
        if impl.shape is None:
            raise ValueError("Cannot create Tensor from None shape Tensor")

        instance = cls(
            shape=impl.shape, dtype=impl.dtype, device=impl.device, materialize=True
        )
        instance._impl = impl if impl else None
        instance.name = name
        return instance

    def copy_from(self, other: Tensor) -> None:
        """Copy data from another Tensor."""
        if self.shape != other.shape or self.dtype != other.dtype:
            raise ValueError("Shape or dtype mismatch for copy")
        if other._impl is not None:
            self._impl = other._impl.copy()
        else:
            self._impl = None

    def add_arguments(self, *arg_nodes: Tensor) -> None:
        """Add an arguments to this Tensor's computation graph if traced."""
        for arg in arg_nodes:
            if not isinstance(arg, Tensor):
                raise TypeError(f"Argument must be an Tensor, got {type(arg)}")
            if arg.traced:
                self.traced = True
            if arg.stage_realization:
                self.stage_realization = True

        if self.traced or self.stage_realization:
            for arg in arg_nodes:
                self.args.append(arg)

    def realize(self) -> None:
        """Force computation of this Tensor."""
        if self._impl is not None:
            return

        from .graph_execution import realize_

        realize_([self])
        if self._impl is None:
            raise ValueError("Data is None after realization")

    def to_numpy(self) -> np.ndarray:
        """Get NumPy representation."""
        self.realize()  # Ensure the Tensor is realized before converting
        if self._impl is None:
            raise ValueError("Cannot get NumPy tensor from None impl")
        if isinstance(self._impl, np.ndarray):
            return self._impl
        if not isinstance(self._impl, MAXTensor):
            raise TypeError(
                f"Cannot convert Tensor with impl type {type(self._impl)} to NumPy tensor"
            )
        return self._impl.to_numpy()

    @classmethod
    def from_numpy(cls, np_tensor: np.ndarray) -> Tensor:
        """Create a new Tensor from a NumPy tensor."""
        if not isinstance(np_tensor, np.ndarray):
            raise TypeError(f"Expected numpy.ndtensor, got {type(np_tensor)}")

        tensor = cls(
            shape=np_tensor.shape,
            dtype=DType.from_numpy(np_tensor.dtype),
            device=_DEFAULT_CPU,
            name=getattr(np_tensor, "name", ""),
        )

        # # WORKAROUND: Handle scalar boolean tensors to avoid MAX library bug
        # # The MAX library's tensor.view(DType.bool) fails for scalar tensors
        # if np_tensor.dtype == bool and np_tensor.shape == ():
        #     # For scalar boolean, convert to float32 to avoid the bug
        #     float_tensor = np_tensor.astype(np.float32)
        #     tensor._impl = float_tensor#MAXTensor.from_numpy(float_tensor)
        #     # Update the dtype to reflect what we actually stored
        #     tensor.dtype = DType.float32
        # else:
        tensor._impl = np_tensor  # MAXTensor.from_numpy(np_tensor)

        # tensor.logical_device = _DEFAULT_CPU
        return tensor

    def get_arguments(self) -> list[Tensor]:
        """Get list of argument Tensors."""
        return list(self.args)

    def set_maxpr(self, fn: MaxprCallable) -> None:
        """Set the MAX PR function for this operation."""
        self.maxpr = fn

    def __repr__(self) -> str:
        """String representation of the Tensor."""
        # self.realize()
        from ..utils.formatting import format_shape_dtype_device

        if self.impl is not None:
            return (
                str(self.impl.to(CPU()).to_numpy()) + ":" + format_shape_dtype_device(self)
            )
        else:
            return (
                f"Tensor(shape={self.shape}, dtype={self.dtype}, logical_device={self.logical_device}, unrealized):"
            )

    def to(self, device: Device) -> Tensor:
        """Move Tensor to specified device."""
        from ..ops.unary import transfer_to

        return transfer_to(self, device)

    def backward(self, grad: Tensor | None = None, retain_graph: bool = False) -> None:
        """Compute gradients flowing into traced leaf inputs that influence this Tensor.
        
        Args:
            grad: Optional cotangent tensor; defaults to ones for scalar outputs
            retain_graph: If False (default), frees the computation graph after backward pass
        """

        if grad is None:
            if self.shape != () or self.batch_dims:
                raise ValueError(
                    "grad argument required for non-scalar outputs in backward()"
                )
            from ..ops.creation import ones_like

            grad = ones_like(self)
        elif not isinstance(grad, Tensor):
            raise TypeError("grad must be a Nabla Tensor")

        from ..transforms.utils import backward as backward_transform

        backward_transform(self, grad, retain_graph=retain_graph)

    def requires_grad_(self, val: bool = True) -> Tensor:
        """Opt into or out of gradient tracking for imperative workflows.
        
        This is an in-place operation that returns self for method chaining.
        Similar to PyTorch's requires_grad_() method.
        """
        self.requires_grad = bool(val)
        self.traced = bool(val)
        return self

    # Operator overloading methods
    def __add__(self, other) -> Tensor:
        """Addition operator."""
        from ..ops.binary import add

        return add(self, other)

    def __mul__(self, other) -> Tensor:
        """Multiplication operator."""
        from ..ops.binary import mul

        return mul(self, other)

    def __sub__(self, other) -> Tensor:
        """Subtraction operator."""
        from ..ops.binary import sub

        return sub(self, other)

    def __pow__(self, power) -> Tensor:
        """Power operator."""
        from ..ops.binary import pow as power_op

        return power_op(self, power)

    def __truediv__(self, other) -> Tensor:
        """Division operator."""
        from ..ops.binary import div

        return div(self, other)

    def __floordiv__(self, other) -> Tensor:
        """Floor division operator (//)."""
        from ..ops.binary import floordiv

        return floordiv(self, other)

    def __matmul__(self, other) -> Tensor:
        """Matrix multiplication operator (@)."""
        from ..ops.linalg import matmul

        return matmul(self, other)

    def __neg__(self) -> Tensor:
        """Negation operator."""
        from ..ops.unary import negate

        return negate(self)

    def __mod__(self, other) -> Tensor:
        """Modulo operator (%)."""
        from ..ops.binary import mod

        return mod(self, other)

    # Comparison operators
    def __lt__(self, other) -> Tensor:
        """Less than operator (<)."""
        from ..ops.binary import greater_equal
        from ..ops.unary import logical_not

        # a < b is equivalent to not (a >= b)
        return logical_not(greater_equal(self, other))

    def __le__(self, other) -> Tensor:
        """Less than or equal operator (<=)."""
        from ..ops.binary import greater_equal

        # a <= b is equivalent to b >= a
        return greater_equal(other, self)

    def __gt__(self, other) -> Tensor:
        """Greater than operator (>)."""
        from ..ops.binary import greater_equal
        from ..ops.unary import logical_not

        # a > b is equivalent to not (b >= a)
        return logical_not(greater_equal(other, self))

    def __ge__(self, other) -> Tensor:
        """Greater than or equal operator (>=)."""
        from ..ops.binary import greater_equal

        return greater_equal(
            self, other
        )  # Hash and equality for making Tensors usable as dictionary keys

    def __hash__(self) -> int:
        """Make Tensors hashable based on object identity.

        This allows Tensors to be used as dictionary keys in optimizers.
        Two Tensors are considered equal only if they are the same object.
        """
        return id(self)

    # Reverse operators for when Tensor is on the right-hand side
    def __radd__(self, other) -> Tensor:
        """Reverse addition operator (other + self)."""
        from ..ops.binary import add

        return add(other, self)

    def __rmul__(self, other) -> Tensor:
        """Reverse multiplication operator (other * self)."""
        from ..ops.binary import mul

        return mul(other, self)

    def __rsub__(self, other) -> Tensor:
        """Reverse subtraction operator (other - self)."""
        from ..ops.binary import sub

        return sub(other, self)

    def __rtruediv__(self, other) -> Tensor:
        """Reverse division operator (other / self)."""
        from ..ops.binary import div

        return div(other, self)

    def __rpow__(self, other) -> Tensor:
        """Reverse power operator (other ** self)."""
        from ..ops.binary import pow as power_op

        return power_op(other, self)

    def __getitem__(self, key) -> Tensor:
        """Tensor slicing using standard Python syntax.

        Supports both basic indexing (slices, integers) and advanced indexing (Tensor indices).

        Examples::

            arr[1:3]        # Slice first dimension
            arr[:, 2:5]     # Slice second dimension
            arr[1:3, 2:5]   # Slice multiple dimensions
            arr[-2:]        # Negative indices
            arr[..., :2]    # Ellipsis (all dimensions up to last)

            # Advanced indexing with Tensor indices:
            indices = nb.tensor([0, 2, 1])
            arr[indices]    # Gather elements along first axis
            arr[indices, :] # Gather rows
        """

        # Check if this is advanced indexing with Tensor indices
        if isinstance(key, Tensor):
            # Single Tensor index - use gather along axis 0
            from ..ops.indexing import gather

            return gather(self, key, axis=0)
        elif isinstance(key, tuple) and any(isinstance(k, Tensor) for k in key):
            # Mixed indexing with Tensor indices in tuple
            return self._handle_mixed_advanced_indexing(key)

        # Handle single slice, integer, or ellipsis (original logic)
        if isinstance(key, slice | int | type(...)):
            key = (key,)
        elif not isinstance(key, tuple):
            raise TypeError(
                f"Tensor indices must be integers, slices, ellipsis, Tensors, or tuples, got {type(key)}"
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

        # Special case: if we have indices but the tensor is scalar, that's an error
        if (
            len(self.shape) == 0
            and len(key) > 0
            and not (len(key) == 1 and key[0] is ...)
        ):
            raise IndexError(f"Too many indices for tensor: expected 0, got {len(key)}")

        # Convert integers to slices and build slice list
        # Track which dimensions should be squeezed (removed) due to integer indexing
        slices = []
        squeeze_axes = []
        for i, k in enumerate(key):
            if i >= len(self.shape):
                raise IndexError(
                    f"Too many indices for tensor: expected {len(self.shape)}, got {len(key)}"
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
                    f"Tensor index {i} must be an integer or slice, got {type(k)}"
                )

        # Create TensorSliceOp with squeeze information
        from ..ops.view import TensorSliceOp

        op = TensorSliceOp(slices, squeeze_axes)
        return op.forward(self)

    def astype(self, dtype: DType) -> Tensor:
        """Convert tensor to a different data type.

        Args:
            dtype: Target data type

        Returns:
            New Tensor with the specified data type
        """
        if self.dtype == dtype:
            return self  # No conversion needed

        # Use nabla's cast operation
        from ..ops.unary import cast

        return cast(self, dtype)

    def sum(self, axes=None, keep_dims=False) -> Tensor:
        """Sum tensor elements over given axes.

        Args:
            axes: Axis or axes along which to sum. Can be int, list of ints, or None (sum all)
            keep_dims: If True, reduced axes are left as dimensions with size 1

        Returns:
            Tensor with the sum along the specified axes

        Examples::

            arr.sum()           # Sum all elements
            arr.sum(axis=0)     # Sum along first axis
            arr.sum(axis=[0,1]) # Sum along first two axes
        """
        from ..ops.reduce import sum as tensor_sum

        return tensor_sum(self, axes=axes, keep_dims=keep_dims)

    def reshape(self, shape: Shape) -> Tensor:
        """Change the shape of an tensor without changing its data.

        Args:
            shape: New shape for the tensor

        Returns:
            Tensor with the new shape

        Examples::

            arr.reshape((2, 3))     # Reshape to 2x3
            arr.reshape((-1,))      # Flatten to 1D (note: -1 not yet supported)
        """
        from ..ops.view import reshape

        return reshape(arg=self, shape=shape)

    def permute(self, axes: tuple[int, ...]) -> Tensor:
        """Permute the dimensions of the tensor.

        Args:
            axes: List of integers specifying the new order of dimensions

        Returns:
            Tensor with dimensions permuted according to the specified axes

        Examples::

            arr.permute([1, 0]) # If arr.shape is (2, 3), this will return an tensor with shape (3, 2)
        """
        from ..ops.view import permute

        return permute(self, axes)

    def transpose(self, axes: tuple[int, ...]) -> Tensor:
        """Permute the dimensions of the tensor.

        Args:
            axes: List of integers specifying the new order of dimensions

        Returns:
            Tensor with dimensions permuted according to the specified axes

        Examples::

            arr.permute([1, 0]) # If arr.shape is (2, 3), this will return an tensor with shape (3, 2)
        """
        from ..ops.view import permute

        return permute(self, axes)

    def at(self, key, value):
        """Update tensor at specified indices/slices, returning new tensor."""
        from ..ops.binary import add, sub
        from ..ops.view import pad

        # Convert value to Tensor if needed
        if not isinstance(value, Tensor):
            # Match the dtype of the original tensor
            value_np = np.array(value, dtype=self.dtype.to_numpy())
            value = Tensor.from_numpy(value_np)
        else:
            # If value is already an Tensor, ensure it matches our dtype
            if value.dtype != self.dtype:
                value_np = value.to_numpy().astype(self.dtype.to_numpy())
                value = Tensor.from_numpy(value_np)

        # Handle single slice, integer, or ellipsis
        if isinstance(key, slice | int | type(...)):
            key = (key,)
        elif not isinstance(key, tuple):
            raise TypeError(
                f"Tensor indices must be integers, slices, ellipsis, or tuples, got {type(key)}"
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
                    value = Tensor.from_numpy(value_np.reshape(sliced_part.shape))
                else:
                    # Try broadcasting
                    value = Tensor.from_numpy(
                        np.broadcast_to(value_np, sliced_part.shape)
                    )
            except:
                raise ValueError(
                    f"Cannot broadcast value shape {value.shape} to sliced shape {sliced_part.shape}"
                )

        # 3. Calculate the difference
        diff = sub(value, sliced_part)

        # 4. Pad the difference to full tensor shape (using converted slices)
        padded_diff = pad(diff, slices, self.shape)

        # 5. Add to original tensor
        result = add(self, padded_diff)

        return result

    # Comparison operators
    def __eq__(self, other) -> bool:
        """Object identity comparison for hashability.

        This returns True only if both Tensors are the same object.
        For element-wise comparison, use nb.equal(a, b) explicitly.
        """
        return isinstance(other, Tensor) and self is other

    def __ne__(self, other) -> bool:
        """Object identity inequality comparison for hashability.

        This returns True if the Tensors are different objects.
        For element-wise comparison, use nb.not_equal(a, b) explicitly.
        """
        return not self.__eq__(other)

    def set(self, key, value) -> Tensor:
        """Set values at specified indices/slices, returning a new tensor.

        This is a functional operation that returns a new Tensor with the specified
        values updated, leaving the original Tensor unchanged.

        Args:
            key: Index specification (int, slice, tuple of indices/slices, ellipsis)
            value: Value(s) to set at the specified location

        Returns:
            New Tensor with updated values

        Examples:
            new_arr = arr.set(1, 99.0)              # Set single element
            new_arr = arr.set((1, 2), 99.0)         # Set element at (1,2)
            new_arr = arr.set(slice(1, 3), 99.0)    # Set slice
            new_arr = arr.set(..., 99.0)            # Set with ellipsis
        """
        return self.at(key, value)

    def _handle_mixed_advanced_indexing(self, key: tuple) -> Tensor:
        """Handle mixed indexing with Tensor indices and slices/integers.

        Args:
            key: Tuple containing mix of Tensor indices, slices, and integers

        Returns:
            Tensor result of advanced indexing
        """
        from ..ops.indexing import gather

        # For now, implement a simplified version that handles the most common case:
        # Tensor index in first position, followed by slices/integers
        # More complex cases can be added later

        # Find the first Tensor index
        tensor_index_pos = None
        for i, k in enumerate(key):
            if isinstance(k, Tensor):
                if tensor_index_pos is None:
                    tensor_index_pos = i
                else:
                    # Multiple Tensor indices - more complex case
                    raise NotImplementedError(
                        "Multiple Tensor indices not yet supported. "
                        "Use gather/scatter operations directly for complex indexing."
                    )

        if tensor_index_pos is None:
            # No Tensor indices found - shouldn't reach here
            raise ValueError("Expected Tensor index in mixed indexing")

        tensor_index = key[tensor_index_pos]

        if tensor_index_pos == 0:
            # Tensor index in first position: arr[indices, slice1, slice2, ...]
            remaining_key = key[1:]

            # First apply gather along axis 0
            gathered = gather(self, tensor_index, axis=0)

            # Then apply remaining indexing if any
            if remaining_key:
                # The remaining key should be applied starting from the first dimension
                # after the tensor-indexed dimension. Since we tensor-indexed dimension 0,
                # the remaining key applies to dimensions 1, 2, 3, ... of the original shape
                # which are dimensions 1, 2, 3, ... of the gathered result.
                # So we need to prepend a slice(None) to cover the new first dimension from gather
                full_key = (slice(None),) + remaining_key
                return gathered[full_key]
            else:
                return gathered
        else:
            # Tensor index not in first position - more complex
            # For now, we'll convert to a sequence of operations
            # This is a simplified implementation
            raise NotImplementedError(
                f"Tensor index at position {tensor_index_pos} not yet supported. "
                "Use gather operation directly or put Tensor index first."
            )

    def __setitem__(self, key, value) -> None:
        """Tensor assignment using standard Python syntax.

        Supports both basic assignment (slices, integers) and advanced assignment (Tensor indices).

        Examples::

            arr[1:3] = value        # Assign to slice
            arr[:, 2:5] = value     # Assign to slice in second dimension

            # Advanced indexing with Tensor indices:
            indices = nb.tensor([0, 2, 1])
            arr[indices] = value    # Scatter values to specified indices
        """
        # Convert value to Tensor if needed
        if not isinstance(value, Tensor):
            from ..ops.creation import tensor

            value = tensor(value)

        # Check if this is advanced indexing with Tensor indices
        if isinstance(key, Tensor):
            # Single Tensor index - use scatter along axis 0
            self._setitem_with_tensor_index(key, value, axis=0)
        elif isinstance(key, tuple) and any(isinstance(k, Tensor) for k in key):
            # Mixed indexing with Tensor indices
            self._setitem_mixed_advanced_indexing(key, value)
        else:
            # Basic indexing - not implemented for now
            raise NotImplementedError(
                "Basic slice assignment not yet implemented. "
                "Use Tensor indices for scatter operations."
            )

    def _setitem_with_tensor_index(
        self, indices: Tensor, values: Tensor, axis: int = 0
    ) -> None:
        """Helper method for setitem with Tensor indices.

        Args:
            indices: Tensor of indices where to place values
            values: Tensor of values to place
            axis: Axis along which to scatter
        """
        from ..ops.indexing import scatter

        # Create new tensor by scattering values into a copy of self
        # Note: This creates a new tensor rather than in-place modification
        # In-place modification would require mutable tensors
        new_tensor = scatter(
            target_shape=self.shape, indices=indices, values=values, axis=axis
        )

        # Update self's implementation to point to new data
        # This simulates in-place modification
        self._impl = new_tensor._impl

    def _setitem_mixed_advanced_indexing(self, key: tuple, value: Tensor) -> None:
        """Helper method for mixed advanced indexing assignment.

        Args:
            key: Tuple containing mix of Tensor indices, slices, and integers
            value: Tensor to assign
        """
        # For now, implement a simplified version
        # Find the first Tensor index
        tensor_index_pos = None
        for i, k in enumerate(key):
            if isinstance(k, Tensor):
                if tensor_index_pos is None:
                    tensor_index_pos = i
                else:
                    raise NotImplementedError(
                        "Multiple Tensor indices not yet supported"
                    )

        if tensor_index_pos != 0:
            raise NotImplementedError(
                "Tensor index must be in first position for assignment"
            )

        tensor_index = key[0]
        remaining_key = key[1:]

        if remaining_key:
            # Need to handle partial assignment like arr[indices, :, slice] = value
            raise NotImplementedError(
                "Mixed Tensor index with slices in assignment not yet supported"
            )
        else:
            # Simple case: arr[indices] = value
            self._setitem_with_tensor_index(tensor_index, value, axis=0)
