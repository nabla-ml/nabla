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

from collections import Optional

from nabla.compiler.tensor import Tensor, TensorShape

from ..type import StaticDim


def repeat_interleave(
    input: Symbol,
    repeats: Int,
    dim: Optional[Int] = None,
) -> Symbol:
    """Repeats elements of a tensor along the given dimension.

    Modeled after `torch.repeat_interleave`, with the constraint that
     Tensor-valued `repeats` are not yet supported.

    For example, given `repeats=2` and the following input:

    ```python
    input = max.tensor.Tensor[DType.float32](
        max.tensor.TensorShape(2, 2),
        1.0, 2.0,
        3.0, 4.0,
    )
    ```

    `repeat_interleave` with `dim=0`:

    ```python
    output = max.tensor.Tensor[DType.float32](
        max.tensor.TensorShape(4, 2),
        1.0, 2.0,
        1.0, 2.0,
        3.0, 4.0,
        3.0, 4.0,
    )
    ```

    `repeat_interleave` with `dim=1`:

    ```python
    output = max.tensor.Tensor[DType.float32](
        max.tensor.TensorShape(2, 4),
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
    )
    ```

    `repeat_interleave` with `dim=None` (the default):

    ```python
    output = max.tensor.Tensor[DType.float32](
        max.tensor.TensorShape(8),
        1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0,
    )
    ```

    Args:
        input: The input tensor.
        repeats: The number of repetitions for each element.
        dim: The dimension along which to repeat values. By default (or if `dim`
             is `None`), flatten the input array.

    Returns:
        A symbolic tensor with the elements interleaved.
    """
    g = input.graph()

    if dim is None:
        input = input.reshape(-1)
        dim = 0

    # To implement `repeat_interleave`, we need to unsqueeze at dim+1 so that
    # we can tile each element of the given dim. (Tiling without unsqueezing
    # would not give us interleaved elements.) We can then squeeze the input
    # back to the expected shape. For example: input=1x8x1025x128xbf16,
    # repeats=3, and dim=1 would be:
    #    - input:     1x8x1025x128
    #    - unsqueeze: 1x8x1x1025x128   (use dim+1)
    #    - tile:      1x8x3x1025x128   (use dim+1)
    #    - reshape:   1x24x1025x128
    # The one exception is when `dim` is `None`, in which case the input array
    # is flattened and all elements are tiled.

    repeat_dim = dim.value()

    unsqueezed_input = unsqueeze(input, repeat_dim + 1)

    tiles = List[Int64]()
    for i in range(unsqueezed_input.tensor_type().rank()):
        if i == repeat_dim + 1:
            tiles.append(repeats)
        else:
            tiles.append(1)
    tiled_input = tile(unsqueezed_input, tiles)

    old_input_shape = shape_of(input)

    scale = List[Int64]()
    for i in range(input.tensor_type().rank()):
        if i == repeat_dim:
            scale.append(repeats)
        else:
            scale.append(1)

    result_shape = g.vector[DType.int64](scale) * old_input_shape

    reshape_dims = input.tensor_type().dims
    if reshape_dims[repeat_dim] != Dim.dynamic():
        repeat_dim_value = (
            input.tensor_type().dim(repeat_dim).value[StaticDim].dim
        )
        reshape_dims[repeat_dim] = Int(repeat_dim_value) * repeats

    return reshape(tiled_input, result_shape, reshape_dims)
