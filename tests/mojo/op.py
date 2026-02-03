# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from pathlib import Path
from typing import Any, TYPE_CHECKING
from max.graph import TensorValue
from nabla.ops import UnaryOperation, call_custom_kernel

if TYPE_CHECKING:
    from nabla.core import Tensor

class AddOneCustomOp(UnaryOperation):
    @property
    def name(self) -> str:
        return "add_one_custom"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Invokes the custom Mojo kernel."""
        # Path relative to this file: ./kernels/
        kernel_dir = Path(__file__).parent / "kernels"

        return call_custom_kernel(
            func_name="add_one_custom",
            kernel_path=kernel_dir,
            values=x,
            out_types=x.type,
        )

def add_one_custom(x: "Tensor") -> "Tensor":
    """Custom op that adds one to each element using a Mojo kernel."""
    return AddOneCustomOp()(x)
