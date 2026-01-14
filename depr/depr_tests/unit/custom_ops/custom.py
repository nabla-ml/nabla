# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Custom Operations Module
# ===----------------------------------------------------------------------=== #

"""Custom operations using Mojo kernels."""

from pathlib import Path
from typing import Any

from max.graph import DeviceRef, TensorValue, ops

from nabla.ops.operation import Operation


class AddOneCustomOp(Operation):
    """Custom operation that adds 1 to input using Mojo kernel."""

    @property
    def name(self) -> str:
        return "add_one_custom"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        from nabla.ops.custom_mojo import call_custom_kernel

        # Use absolute path relative to this file
        kernel_dir = Path(__file__).parent / "custom_kernels"
        
        result = call_custom_kernel(
            func_name="add_one_custom",
            kernel_path=str(kernel_dir),
            values=args[0],
            out_types=args[0].type,
        )
        return result


add_one_custom = AddOneCustomOp()


__all__ = ["AddOneCustomOp", "add_one_custom"]
