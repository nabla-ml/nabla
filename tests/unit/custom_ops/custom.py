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
        from nabla.core.compute_graph import GRAPH

        kernel_dir = Path(__file__).parent / "custom_kernels"
        # Use load_paths which supports source directories according to docs
        GRAPH.graph._kernel_library.load_paths(GRAPH.graph._context, [kernel_dir])

        result = ops.custom(
            name="add_one_custom",
            device=DeviceRef.CPU(),
            values=[args[0]],
            out_types=[args[0].type],
        )
        return result[0]


add_one_custom = AddOneCustomOp()


__all__ = ["AddOneCustomOp", "add_one_custom"]
