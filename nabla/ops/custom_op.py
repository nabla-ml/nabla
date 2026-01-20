# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from pathlib import Path
from typing import Any, Union

from max.graph import DeviceRef, TensorValue, ops

from nabla.core import GRAPH


def call_custom_kernel(
    func_name: str,
    kernel_path: Union[str, Path, list[Union[str, Path]]],
    values: Union[TensorValue, list[TensorValue]],
    out_types: Union[Any, list[Any]],
    device: None | DeviceRef = None,
    **kwargs: Any,
) -> Union[TensorValue, list[TensorValue]]:
    """Helper to invoke a custom Mojo kernel, handling library loading automatically.

    Args:
        func_name: The name of the registered Mojo kernel (e.g. @register("name")).
        kernel_path: Path(s) to the kernel source file or directory.
        values: Input TensorValue(s).
        out_types: Expected output type(s).
        device: Device to run on (default: CPU).
        **kwargs: Additional arguments passed to ops.custom.

    Returns:
        Result TensorValue(s).
    """
    if device is None:
        device = DeviceRef.CPU()

    if not isinstance(kernel_path, list):
        kernel_path = [kernel_path]

    if isinstance(values, TensorValue):
        values_list = [values]
    else:
        values_list = values
    unwrap_result = False
    if not isinstance(out_types, list):
        out_types_list = [out_types]
        unwrap_result = True
    else:
        out_types_list = out_types

    resolved_paths: list[Path] = []
    for p in kernel_path:
        path_obj = Path(p).resolve()
        resolved_paths.append(path_obj)
    GRAPH.graph._kernel_library.load_paths(GRAPH.graph._context, resolved_paths)

    results = ops.custom(
        name=func_name,
        device=device,
        values=values_list,
        out_types=out_types_list,
        **kwargs,
    )

    if results:
        op_instance = results[0].to_mlir().owner
        GRAPH.graph._kernel_library.verify_custom_op(op_instance)

    if unwrap_result:
        return results[0]
    return results
