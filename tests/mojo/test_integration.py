# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import sys
from pathlib import Path

import numpy as np

import nabla as nb
from nabla.core.sharding import DeviceMesh

if __package__:
    from .op import add_one_custom
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from op import add_one_custom


def test_add_one_custom_unsharded():
    """Test the custom Mojo kernel on an unsharded tensor."""
    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = nb.Tensor.constant(x_np)

    y = add_one_custom(x)
    y_val = y.realize().to_numpy()

    expected = x_np + 1.0
    np.testing.assert_allclose(y_val, expected)
    print("Unsharded test passed!")


def test_add_one_custom_sharded():
    """Test the custom Mojo kernel on a sharded tensor."""
    # Setup a simple 2-device mesh (simulation if no GPUs)
    mesh = DeviceMesh("mesh", (2,), ("x",))

    x_np = np.arange(8, dtype=np.float32)
    # Shard along the first axis
    x = nb.Tensor.constant(x_np).shard(mesh, [nb.DimSpec(axes=["x"])])

    y = add_one_custom(x)
    y_val = y.realize().to_numpy()

    expected = x_np + 1.0
    np.testing.assert_allclose(y_val, expected)
    assert y.is_sharded
    print("Sharded test passed!")


if __name__ == "__main__":
    test_add_one_custom_unsharded()
    test_add_one_custom_sharded()
