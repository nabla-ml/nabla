# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Broadcasting and shape manipulation utilities."""

Shape = tuple[int, ...]


def get_broadcasted_shape(
    shape1: Shape,
    shape2: Shape,
    ignore_axes: list[int] | None = None,
    replace_ignored_dims: list[int] | None = None,
) -> Shape:
    """
    Compute the broadcasted shape of two input shapes.

    Args:
        shape1: First input shape
        shape2: Second input shape
        ignore_axes: Axes to ignore during broadcasting
        replace_ignored_dims: Replacement dimensions for ignored axes

    Returns:
        The broadcasted output shape

    Raises:
        ValueError: If shapes cannot be broadcast together
    """
    if ignore_axes is None:
        ignore_axes = []
    if replace_ignored_dims is None:
        replace_ignored_dims = []

    if len(replace_ignored_dims) != len(ignore_axes):
        raise ValueError(
            "replace_ignored_dims must have the same length as ignore_axes"
        )

    s1_len = len(shape1)
    s2_len = len(shape2)
    max_rank = max(s1_len, s2_len)

    # Initialize result shape with 1s (common default for broadcasting)
    res_shape_list = [1] * max_rank

    # Normalize ignore_axes to positive indices and store replacement values
    normalized_ignored_map = {}
    for i, axis_spec in enumerate(ignore_axes):
        replacement_dim = replace_ignored_dims[i]

        # Validate and normalize the axis_spec relative to max_rank
        if not (-max_rank <= axis_spec < max_rank):
            raise ValueError(
                f"ignore_axis {axis_spec} is out of bounds for max_rank {max_rank}"
            )

        normalized_idx = axis_spec if axis_spec >= 0 else max_rank + axis_spec
        normalized_ignored_map[normalized_idx] = replacement_dim
        res_shape_list[normalized_idx] = replacement_dim

    # Pad original shapes with leading 1s to align them to max_rank
    padded_shape1_list = [1] * (max_rank - s1_len) + list(shape1)
    padded_shape2_list = [1] * (max_rank - s2_len) + list(shape2)

    # Perform broadcasting for non-ignored axes
    for i in range(max_rank):
        if i in normalized_ignored_map:
            # This dimension's value is already set by replace_ignored_dims
            continue

        d1 = padded_shape1_list[i]
        d2 = padded_shape2_list[i]

        if d1 == d2:
            res_shape_list[i] = d1
        elif d1 == 1:
            res_shape_list[i] = d2
        elif d2 == 1:
            res_shape_list[i] = d1
        else:
            # Dimensions are different and neither is 1, broadcasting error
            raise ValueError(
                f"Shapes {shape1} and {shape2} cannot be broadcast at dimension index {i} "
                f"(0-indexed from left of max_rank {max_rank} shape). "
                f"Padded values at this index are {d1} (from shape1) and {d2} (from shape2)."
            )

    return tuple(res_shape_list)
