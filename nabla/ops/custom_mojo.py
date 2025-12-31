# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Mojo Custom Op Helper
# ===----------------------------------------------------------------------=== #

"""Helper utilities for invoking Mojo custom kernels."""

import sys
from pathlib import Path
from typing import Any, List, Union

from max.graph import DeviceRef, TensorValue, ops

from nabla.core.compute_graph import GRAPH


def call_custom_kernel(
    func_name: str,
    kernel_path: Union[str, Path, List[Union[str, Path]]],
    values: Union[TensorValue, List[TensorValue]],
    out_types: Union[Any, List[Any]],
    device: None | DeviceRef = None,
    **kwargs: Any,
) -> Union[TensorValue, List[TensorValue]]:
    """
    Helper to invoke a custom Mojo kernel, handling library loading automatically.

    Args:
        func_name: The name of the registered Mojo kernel (e.g. @register("name")).
        kernel_path: Path(s) to the kernel source file or directory.
                     If a relative path is string, it is resolved relative to the CALLER's file.
        values: Input TensorValue(s). Can be a single value or a list.
        out_types: Expected output type(s). Can be a single type or a list.
        device: Device to run on. Defaults to CPU.
        **kwargs: Additional arguments passed to ops.custom.

    Returns:
        Result TensorValue(s). Returns a single value if out_types was a single value,
        otherwise returns a list.
    """
    if device is None:
        device = DeviceRef.CPU()

    if not isinstance(kernel_path, list):
        kernel_path = [kernel_path]

    # Normalize inputs to list
    if isinstance(values, TensorValue):
        values_list = [values]
    else:
        values_list = values

    # Normalize out_types to list and track if we need to unwrap
    unwrap_result = False
    if not isinstance(out_types, list):
        out_types_list = [out_types]
        unwrap_result = True
    else:
        out_types_list = out_types

    resolved_paths: List[Path] = []
    for p in kernel_path:
        path_obj = Path(p)
        if not path_obj.is_absolute():
            # "Magic" resolution relative to caller
            try:
                # Frame 0: this function
                # Frame 1: caller (e.g. user's maxpr method)
                frame = sys._getframe(1)
                caller_file = frame.f_code.co_filename
                if caller_file:
                    resolved_path = Path(caller_file).parent / path_obj
                    path_obj = resolved_path
            except Exception:
                # Fallback to CWD if stack inspection fails
                path_obj = path_obj.absolute()
        
        resolved_paths.append(path_obj)

    # 1. Load the kernels into the graph context
    # This is safe to call multiple times (idempotent-ish in MAX)
    GRAPH.graph._kernel_library.load_paths(GRAPH.graph._context, resolved_paths)

    # 2. Invoke the custom op
    results = ops.custom(
        name=func_name,
        device=device,
        values=values_list,
        out_types=out_types_list,
        **kwargs
    )

    if unwrap_result:
        return results[0]
    return results
