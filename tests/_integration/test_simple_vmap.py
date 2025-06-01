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

import numpy as np
import pytest

import nabla as nb


def simple_add(args):
    return [args[0] + args[1]]


def test_simple_vmap_basic():
    """Test basic vmap functionality with broadcasting."""
    # Test simple vmap
    a = nb.arange((3, 4), nb.DType.float32)  # shape (3, 4)
    b = nb.arange((4,), nb.DType.float32)  # shape (4,)

    # This should vectorize over the first axis of a, and broadcast b
    vmapped_add = nb.vmap(simple_add, [0, None])
    result = vmapped_add([a, b])

    # Assertions
    assert result[0].shape == (3, 4), f"Expected shape (3, 4), got {result[0].shape}"

    # Verify result values - each row of a should be added to b
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    expected = a_np + b_np[np.newaxis, :]  # Broadcasting b to match a

    assert np.allclose(result[0].to_numpy(), expected, rtol=1e-6), (
        "vmap result doesn't match expected broadcast addition"
    )


def test_simple_vmap_different_shapes():
    """Test vmap with different input shapes."""
    # Test with different shapes
    a = nb.ones((2, 3), nb.DType.float32)
    b = nb.array([1.0, 2.0, 3.0], nb.DType.float32)

    vmapped_add = nb.vmap(simple_add, [0, None])
    result = vmapped_add([a, b])

    assert result[0].shape == (2, 3), f"Expected shape (2, 3), got {result[0].shape}"

    # Each row should be [2, 3, 4] since ones + [1, 2, 3] = [2, 3, 4]
    expected = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    assert np.allclose(result[0].to_numpy(), expected, rtol=1e-6), (
        "vmap result values don't match expected"
    )


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_simple_vmap_parametrized(batch_size):
    """Test vmap with different batch sizes."""
    a = nb.ones((batch_size, 2), nb.DType.float32)
    b = nb.array([10.0, 20.0], nb.DType.float32)

    vmapped_add = nb.vmap(simple_add, [0, None])
    result = vmapped_add([a, b])

    assert result[0].shape == (batch_size, 2), (
        f"Expected shape ({batch_size}, 2), got {result[0].shape}"
    )

    # Each row should be [11, 21] since ones + [10, 20] = [11, 21]
    expected = np.tile([11.0, 21.0], (batch_size, 1)).astype(np.float32)
    assert np.allclose(result[0].to_numpy(), expected, rtol=1e-6), (
        f"vmap result values don't match expected for batch_size={batch_size}"
    )


if __name__ == "__main__":
    # Run tests manually if executed directly
    test_simple_vmap_basic()
    test_simple_vmap_different_shapes()

    for batch_size in [1, 2, 5]:
        test_simple_vmap_parametrized(batch_size)

    print("All simple vmap tests passed!")
