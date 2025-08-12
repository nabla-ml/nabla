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

"""
Tensor indexing operations: gather and scatter.

This module provides two fundamental tensor indexing operations:
- `gather`: Selects elements from an input tensor using an index tensor along
            a specified axis, similar to `numpy.take`.
- `scatter`: Updates elements in a target tensor at specified indices with
             given values, effectively the inverse of the `gather` operation.

These operations are designed to integrate with the MAX graph and support
automatic differentiation through custom VJP and JVP rules.
"""

__all__ = ["gather", "scatter"]


import numpy as np
from max.dtype import DType
from max.graph import TensorValue, ops

from ..core.array import Array
from .operation import Operation


class GatherOp(Operation):
    """Gather operation."""

    def __init__(self, axis: int = -1):
        """
        Initializes the GatherOp.

        Parameters
        ----------
        axis : int, optional
            The axis along which to gather values. A negative value counts
            from the last dimension. Defaults to -1.
        """
        super().__init__("gather")
        self.axis = axis

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """
        Computes the output shape for the gather operation.

        The output shape is determined by replacing the indexed dimension of the
        input shape with the shape of the indices tensor.

        Parameters
        ----------
        input_shapes : tuple
            A tuple containing the input tensor shape and the indices tensor shape,
            in the format `(input_shape, indices_shape)`.

        Returns
        -------
        tuple
            The computed shape of the output tensor.
        """
        input_shape, indices_shape = input_shapes

        # Normalize negative axis
        axis = self.axis
        if axis < 0:
            axis += len(input_shape)

        # The output shape is formed by the parts of the input shape before and
        # after the gather axis, with the indices shape inserted at the axis.
        output_shape = input_shape[:axis] + indices_shape + input_shape[axis + 1 :]

        return output_shape

    def compute_output_batch_dims(self, *input_batch_dims: tuple) -> tuple:
        """
        Computes the output batch dimensions for the gather operation.

        The output batch dimensions are the result of broadcasting the batch
        dimensions of the input tensor and the indices tensor.

        Parameters
        ----------
        input_batch_dims : tuple
            A tuple containing the batch dimensions of the input tensor and
            the indices tensor.

        Returns
        -------
        tuple
            The broadcasted batch dimensions.
        """
        if len(input_batch_dims) != 2:
            raise ValueError(
                f"Gather operation requires 2 input batch dims, got {len(input_batch_dims)}"
            )

        input_batch_dims_val, indices_batch_dims_val = (
            input_batch_dims[0],
            input_batch_dims[1],
        )

        from ..utils.shape_utils import get_broadcasted_shape
        return get_broadcasted_shape(input_batch_dims_val, indices_batch_dims_val)

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        """
        Defines the MAX graph implementation using `max.graph.ops.gather`.

        Parameters
        ----------
        args : list[TensorValue]
            A list containing the input tensor and the indices tensor, as
            `[input_tensor, indices_tensor]`.
        output : Array
            The output array where the result will be stored.
        """
        input_tensor, indices_tensor = args

        # Ensure indices are integers for MAX graph operations
        if indices_tensor.type.dtype.name != "int64":
            indices_tensor = ops.cast(indices_tensor, ops.DType.int64)

        result = ops.gather(input_tensor, indices_tensor, axis=self.axis)
        output.tensor_value = result

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """
        Defines the eager mode execution using `numpy.take`.

        Parameters
        ----------
        args : list[Array]
            A list containing the input array and the indices array, as
            `[input_array, indices_array]`.
        output : Array
            The output array where the result will be stored.
        """
        values_np = args[0].to_numpy()
        indices_np = args[1].to_numpy()
        output.impl_(np.take(values_np, indices_np, axis=self.axis))

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """
        Defines the vector-Jacobian product (VJP) rule for the gather operation.

        The gradient with respect to the input is computed by scattering the
        cotangent back to the original shape. The gradient for the indices is zero.

        Parameters
        ----------
        primals : list[Array]
            The inputs to the forward pass, as `[input_array, indices_array]`.
        cotangent : Array
            The gradient of the loss with respect to the output of this operation.
        output : Array
            The output of the forward pass.

        Returns
        -------
        list[Array]
            A list containing the gradients with respect to the inputs,
            as `[input_grad, indices_grad]`.
        """
        input_array, indices_array = primals
        target_shape = input_array.shape

        # Scatter the incoming gradient to the appropriate locations.
        input_grad = scatter(target_shape, indices_array, cotangent, axis=self.axis)

        # No gradient flows through the indices.
        from ..ops.creation import zeros
        indices_grad = zeros(indices_array.shape, dtype=input_array.dtype)

        return [input_grad, indices_grad]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """
        Defines the Jacobian-vector product (JVP) rule for the gather operation.

        The forward-mode derivative is computed by gathering from the input tangent
        at the same indices.

        Parameters
        ----------
        primals : list[Array]
            The inputs to the forward pass, as `[input_array, indices_array]`.
        tangents : list[Array]
            The tangents of the inputs, as `[input_tangent, indices_tangent]`.
        output : Array
            The output of the forward pass.

        Returns
        -------
        Array
            The tangent of the output.
        """
        input_tangent, _ = tangents
        # Indices are discrete, so their tangents are ignored.
        # Apply the same gather operation to the input tangent.
        return gather(input_tangent, indices=primals[1], axis=self.axis)

    def compute_output_dtype(self, input_array: Array, indices: Array) -> DType:
        """
        Computes the output dtype, which is the same as the input array's dtype.
        """
        return input_array.dtype

    def forward(self, *args: Array) -> Array:
        """
        Executes the forward pass for the gather operation.

        Parameters
        ----------
        *args : Array
            Variable arguments, expected to be `(input_array, indices_array)`.

        Returns
        -------
        Array
            The resulting array after the gather operation.
        """
        if len(args) != 2:
            raise ValueError(f"Gather operation requires 2 arguments, got {len(args)}")

        from .operation import move_to_best_device
        args = move_to_best_device(*args)
        input_array, indices = args

        if not isinstance(input_array, Array) or not isinstance(indices, Array):
            raise TypeError("Both arguments must be Array instances")

        output_shape = self.compute_output_shape(input_array.shape, indices.shape)
        output_batch_dims = self.compute_output_batch_dims(
            input_array.batch_dims, indices.batch_dims
        )
        output_dtype = self.compute_output_dtype(input_array, indices)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,
            device=input_array.logical_device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(input_array, indices)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([input_array, indices], res)

        res.creator_op = self
        return res


def gather(input_array: Array, indices: Array, axis: int = -1) -> Array:
    """
    Selects elements from an input array using indices along a specified axis.

    This function is analogous to `numpy.take_along_axis`. It selects elements
    from `input_array` at the positions specified by `indices`.

    Parameters
    ----------
    input_array : Array
        The source array from which to gather values.
    indices : Array
        The array of indices to gather. Must be an integer-typed array.
    axis : int, optional
        The axis along which to gather. A negative value counts from the last
        dimension. Defaults to -1.

    Returns
    -------
    Array
        A new array containing the elements of `input_array` at the given
        `indices`.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([[10, 20, 30], [40, 50, 60]])
    >>> indices = nb.array([[0, 2], [1, 0]])
    >>> # Gather along axis 1
    >>> nb.gather(x, indices, axis=1)
    Array([[10, 30],
           [50, 40]], dtype=int32)

    >>> # Gather along axis 0
    >>> indices = nb.array([[0, 1, 0]])
    >>> nb.gather(x, indices, axis=0)
    Array([[10, 50, 30]], dtype=int32)
    """
    if axis >= 0:
        axis = axis - len(input_array.shape)

    op = GatherOp(axis)
    return op.forward(input_array, indices)


class ScatterOp(Operation):
    """Scatter operation."""

    def __init__(self, target_shape: tuple, axis: int = -1):
        """
        Initializes the ScatterOp.

        Parameters
        ----------
        target_shape : tuple
            The shape of the output tensor into which values are scattered.
        axis : int, optional
            The axis along which to scatter values. A negative value counts from
            the last dimension. Defaults to -1.
        """
        super().__init__("scatter")
        self.target_shape = target_shape
        self.axis = axis

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """
        Computes the output shape for the scatter operation.

        The output shape is fixed by the `target_shape` provided during
        initialization.

        Parameters
        ----------
        input_shapes : tuple
            A tuple containing the indices and values shapes, ignored in this
            implementation.

        Returns
        -------
        tuple
            The `target_shape` of the output tensor.
        """
        shape_list = []
        for dim in self.target_shape:
            if hasattr(dim, "to_numpy"):
                shape_list.append(int(dim.to_numpy().item()))
            else:
                shape_list.append(int(dim))
        return tuple(shape_list)

    def compute_output_batch_dims(self, *input_batch_dims: tuple) -> tuple:
        """
        Computes the output batch dimensions for the scatter operation.

        The output batch dimensions are the result of broadcasting the batch
        dimensions of the indices and values tensors.

        Parameters
        ----------
        input_batch_dims : tuple
            A tuple containing the batch dimensions of the indices tensor and
            the values tensor.

        Returns
        -------
        tuple
            The broadcasted batch dimensions.
        """
        if len(input_batch_dims) != 2:
            raise ValueError(
                f"Scatter operation requires 2 input batch dims, got {len(input_batch_dims)}"
            )

        indices_batch_dims, values_batch_dims = input_batch_dims[0], input_batch_dims[1]

        from ..utils.shape_utils import get_broadcasted_shape
        return get_broadcasted_shape(indices_batch_dims, values_batch_dims)

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        """
        Defines the MAX graph implementation using `max.graph.ops.scatter_nd`.

        Parameters
        ----------
        args : list[TensorValue]
            A list containing the indices tensor and the values tensor, as
            `[indices_tensor, values_tensor]`.
        output : Array
            The output array where the result will be stored.
        """
        indices_tensor, values_tensor = args
        from max.graph import DeviceRef

        if indices_tensor.type.dtype.name not in ["int32", "int64"]:
            indices_tensor = ops.cast(indices_tensor, ops.DType.int64)

        zero_scalar = ops.constant(
            0.0, dtype=output.dtype, device=DeviceRef.from_device(output.logical_device)
        )
        target_shape = list(output.batch_dims) + list(output.shape)
        zero_tensor = ops.broadcast_to(zero_scalar, target_shape)

        axis = self.axis
        if axis < 0:
            axis += len(output.shape)

        axis_with_batch = axis + len(output.batch_dims)

        if axis_with_batch == 0:
            indices_nd = ops.unsqueeze(indices_tensor, -1)
        else:
            values_shape = values_tensor.type.shape
            indices_shape = indices_tensor.type.shape
            if len(indices_shape) < len(values_shape):
                for _ in range(len(values_shape) - len(indices_shape)):
                    indices_tensor = ops.unsqueeze(indices_tensor, -1)
                indices_tensor = ops.broadcast_to(indices_tensor, values_shape)
            result = ops.scatter(
                zero_tensor, values_tensor, indices_tensor, axis=axis_with_batch
            )
            output.tensor_value = result
            return

        result = ops.scatter_nd(zero_tensor, values_tensor, indices_nd)
        output.tensor_value = result

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """
        Defines the eager mode execution using NumPy indexing.

        Parameters
        ----------
        args : list[Array]
            A list containing the indices array and the values array, as
            `[indices_array, values_array]`.
        output : Array
            The output array where the result will be stored.
        """
        indices_array, values_array = args
        indices_np = indices_array.to_numpy()
        values_np = values_array.to_numpy()

        if indices_np.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"Indices array must be of integer type, got {indices_np.dtype}"
            )

        full_shape = list(output.batch_dims) + list(output.shape)
        result_np = np.zeros(full_shape, dtype=values_np.dtype)

        if indices_array.batch_dims or values_array.batch_dims:
            batch_size = indices_np.shape[0]
            for i in range(batch_size):
                indices_batch_i = indices_np[i]
                values_batch_i = values_np[i]
                self._scatter_single(
                    result_np[i], indices_batch_i, values_batch_i, self.axis
                )
        else:
            self._scatter_single(result_np, indices_np, values_np, self.axis)

        output.impl_(result_np)

    def _scatter_single(
        self,
        target_array: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        axis: int,
    ) -> None:
        """
        Helper method to scatter values into a target array along an axis.

        Parameters
        ----------
        target_array : np.ndarray
            The array to scatter values into.
        indices : np.ndarray
            The indices where values should be placed.
        values : np.ndarray
            The values to scatter.
        axis : int
            The axis along which to perform the scatter.
        """
        if axis < 0:
            axis += target_array.ndim

        idx = [slice(None)] * target_array.ndim
        idx[axis] = indices
        target_array[tuple(idx)] = values

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """
        Defines the vector-Jacobian product (VJP) rule for the scatter operation.

        The gradient for the values is computed by gathering from the cotangent at
        the same indices. The gradient for the indices is zero.

        Parameters
        ----------
        primals : list[Array]
            The inputs to the forward pass, as `[indices_array, values_array]`.
        cotangent : Array
            The gradient of the loss with respect to the output of this operation.
        output : Array
            The output of the forward pass.

        Returns
        -------
        list[Array]
            A list containing the gradients with respect to the inputs,
            as `[indices_grad, values_grad]`.
        """
        indices_array, values_array = primals
        from ..ops.creation import zeros

        indices_grad = zeros(indices_array.shape, dtype=values_array.dtype)
        values_grad = gather(cotangent, indices_array, axis=self.axis)

        return [indices_grad, values_grad]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """
        Defines the Jacobian-vector product (JVP) rule for the scatter operation.

        The forward-mode derivative is computed by scattering the values tangent
        using the original indices.

        Parameters
        ----------
        primals : list[Array]
            The inputs to the forward pass, as `[indices_array, values_array]`.
        tangents : list[Array]
            The tangents of the inputs, as `[indices_tangent, values_tangent]`.
        output : Array
            The output of the forward pass.

        Returns
        -------
        Array
            The tangent of the output.
        """
        _, values_tangent = tangents
        # Indices are discrete, so their tangents are ignored.
        return scatter(self.target_shape, primals[0], values_tangent, axis=self.axis)

    def compute_output_dtype(self, indices: Array, values: Array) -> DType:
        """
        Computes the output dtype, which is the same as the values array's dtype.
        """
        return values.dtype

    def forward(self, *args: Array) -> Array:
        """
        Executes the forward pass for the scatter operation.

        Parameters
        ----------
        *args : Array
            Variable arguments, expected to be `(indices_array, values_array)`.

        Returns
        -------
        Array
            The resulting array after the scatter operation.
        """
        if len(args) != 2:
            raise ValueError(f"Scatter operation requires 2 arguments, got {len(args)}")

        from .operation import move_to_best_device
        args = move_to_best_device(*args)
        indices, values = args

        if not isinstance(indices, Array) or not isinstance(values, Array):
            raise TypeError("Both arguments must be Array instances")

        output_shape = self.compute_output_shape(indices.shape, values.shape)
        output_batch_dims = self.compute_output_batch_dims(
            indices.batch_dims, values.batch_dims
        )
        output_dtype = self.compute_output_dtype(indices, values)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,
            device=values.logical_device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(indices, values)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([indices, values], res)

        return res


def scatter(
    target_shape: tuple, indices: Array, values: Array, axis: int = -1
) -> Array:
    """
    Updates an array of zeros with given values at specified indices.

    This function creates an array of shape `target_shape` filled with zeros
    and then places the `values` at the locations specified by `indices` along
    the given `axis`. This operation is the inverse of `gather`.

    Parameters
    ----------
    target_shape : tuple
        The shape of the output array.
    indices : Array
        An integer array specifying the indices to update.
    values : Array
        The array of values to scatter into the new array.
    axis : int, optional
        The axis along which to scatter. A negative value counts from the last
        dimension. Defaults to -1.

    Returns
    -------
    Array
        A new array of shape `target_shape` with `values` scattered at the
        specified `indices`.

    Examples
    --------
    >>> import nabla as nb
    >>> target_shape = (3, 4)
    >>> indices = nb.array([0, 2, 1])
    >>> values = nb.array([10, 20, 30])
    >>> # Scatter values into a 1D target
    >>> nb.scatter((4,), nb.array([0, 3, 1]), nb.array([1, 2, 3]))
    Array([1, 3, 0, 2], dtype=int32)

    >>> # Scatter rows into a 2D target along axis 0
    >>> values_2d = nb.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
    >>> nb.scatter(target_shape, indices, values_2d, axis=0)
    Array([[1, 1, 1, 1],
           [3, 3, 3, 3],
           [2, 2, 2, 2]], dtype=int32)
    """
    if axis >= 0:
        axis = axis - len(target_shape)
    op = ScatterOp(target_shape, axis)
    return op.forward(indices, values)