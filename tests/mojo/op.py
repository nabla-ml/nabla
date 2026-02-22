# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nabla.ops import UnaryOperation, call_custom_kernel

if TYPE_CHECKING:
    from nabla.core import Tensor


class AddOneCustomOp(UnaryOperation):
    @property
    def name(self) -> str:
        return "my_kernel"

    def kernel(self, args: list[Any], kwargs: dict[str, Any]) -> list[Any]:
        """Invokes the custom Mojo kernel."""
        # Path relative to this file: ./kernels/
        kernel_dir = Path(__file__).parent / "kernels"
        x = args[0]

        result = call_custom_kernel("my_kernel", kernel_dir, x, x.type)
        return [result]


def add_one_custom(x: "Tensor") -> "Tensor":
    """Custom op that adds one to each element using a Mojo kernel."""
    return AddOneCustomOp()([x], {})[0]
