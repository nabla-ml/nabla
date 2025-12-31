# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Custom Operations Tests
# ===----------------------------------------------------------------------=== #

"""Tests for custom Mojo kernel operations."""

import unittest

import numpy as np

import nabla


class TestAddOneCustomOp(unittest.TestCase):
    """Test the add_one_custom operation."""

    def test_add_one_custom_basic(self):
        """Test basic add_one_custom functionality."""
        import asyncio

        from .custom import add_one_custom

        async def run_test():
            x = nabla.Tensor.ones((3, 4))
            result = add_one_custom(x)
            result_np = (await result).to_numpy()

            expected = np.ones((3, 4)) + 1
            np.testing.assert_allclose(result_np, expected)

        asyncio.run(run_test())

    def test_add_one_custom_different_shape(self):
        """Test with different tensor shape."""
        import asyncio

        from .custom import add_one_custom

        async def run_test():
            x = nabla.reshape(nabla.Tensor.arange(12), (3, 4))
            result = add_one_custom(x)
            result_np = (await result).to_numpy()

            expected = np.arange(12).reshape((3, 4)) + 1
            np.testing.assert_allclose(result_np, expected)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
