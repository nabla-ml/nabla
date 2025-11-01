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

from collections.abc import Callable
from typing import Any

from ..core.tensor import Tensor
from ..utils.max_interop import accelerator, accelerator_count, cpu
from .utils import (
    _extract_tensors_from_pytree,
    process_transform_inputs,
    process_transform_outputs,
    tree_flatten,
    tree_unflatten,
)

_DEFAULT_DEVICE = cpu() if accelerator_count() == 0 else accelerator(0)
# Cache CPU device for comparison
_CPU_DEVICE = cpu()


def _build_fast_input_extractors(actual_args, is_list_style):
    """Build fast input extractors to minimize overhead in subsequent calls."""
    # Cache input structure for ultra-fast conversion

    def analyze_structure(item):
        if isinstance(item, Tensor):
            return "tensor"
        elif isinstance(item, int | float):
            return "scalar"
        elif isinstance(item, list | tuple):
            return ("container", type(item), [analyze_structure(sub) for sub in item])
        else:
            return "other"

    structure = analyze_structure(actual_args)
    return {"is_list_style": is_list_style, "structure": structure}


# UPDATED helper: early automatic device placement with controls
_DEF_ACC_AVAILABLE = accelerator_count() > 0

def _move_args_to_default_device(obj, *, convert_scalars: bool, only_from_cpu: bool):
    if not _DEF_ACC_AVAILABLE:
        return obj

    def move(item):
        if isinstance(item, Tensor):
            # Only move if on CPU (default host) when only_from_cpu=True
            if only_from_cpu:
                if item.logical_device == _CPU_DEVICE and _DEFAULT_DEVICE != _CPU_DEVICE:
                    return item.to(_DEFAULT_DEVICE)
            else:  # legacy behavior (not used presently)
                if item.logical_device != _DEFAULT_DEVICE:
                    return item.to(_DEFAULT_DEVICE)
            return item
        if convert_scalars and isinstance(item, (int, float, bool)):
            import nabla as nb
            arr = nb.tensor(item)
            if _DEFAULT_DEVICE != _CPU_DEVICE:
                arr = arr.to(_DEFAULT_DEVICE)
            return arr
        if isinstance(item, (list, tuple)):
            return type(item)(move(sub) for sub in item)
        if isinstance(item, dict):
            return {k: move(v) for k, v in item.items()}
        return item

    return move(obj)


def _fast_extract_tensors(actual_args, is_list_style, extractors):
    """Ultra-fast tensor extraction using cached structure analysis."""
    if isinstance(extractors, dict) and "structure" in extractors:
        # Use cached structure for minimal overhead
        return _ultra_fast_extract_with_cache(actual_args, extractors["structure"])
    else:
        # Fallback to current method
        return _fast_extract_tensors_fallback(actual_args, is_list_style)


def _ultra_fast_extract_with_cache(args, structure):
    """Extract tensors using pre-analyzed structure - minimal overhead."""
    import nabla as nb

    def extract_with_structure(item, struct):
        if struct == "tensor":
            return [item.impl]
        elif struct == "scalar":
            return [nb.tensor(item).impl]
        elif isinstance(struct, tuple) and struct[0] == "container":
            _, container_type, substruct_list = struct
            extracted = []
            for sub_item, sub_struct in zip(item, substruct_list, strict=False):
                extracted.extend(extract_with_structure(sub_item, sub_struct))
            return extracted
        elif isinstance(item, dict):
            # Handle dictionaries by extracting tensors from all values
            extracted = []
            for key in sorted(item.keys()):  # Deterministic ordering
                if isinstance(item[key], Tensor):
                    extracted.append(item[key].impl)
                elif isinstance(item[key], dict) or isinstance(
                    item[key], (list, tuple)
                ):
                    extracted.extend(extract_with_structure(item[key], struct))
                elif isinstance(item[key], (int, float)):
                    extracted.append(nb.tensor(item[key]).impl)
            return extracted
        elif isinstance(item, (list, tuple)):
            # Handle lists and tuples
            extracted = []
            for sub_item in item:
                extracted.extend(extract_with_structure(sub_item, struct))
            return extracted
        elif isinstance(item, Tensor):
            return [item.impl]
        elif isinstance(item, (int, float)):
            return [nb.tensor(item).impl]
        else:
            # Try to convert to tensor as fallback, but handle dict error
            try:
                return [nb.tensor(item).impl]
            except TypeError:
                # If conversion fails, it might be a complex structure - use tree_flatten
                from .utils import tree_flatten

                flat_tensors, _ = tree_flatten(item)
                return [arr.impl for arr in flat_tensors]

    return extract_with_structure(args, structure)


def _fast_extract_tensors_fallback(actual_args, is_list_style):
    """Fallback fast tensor extraction method."""

    # Convert to Tensors first, then extract tensors - matches compilation path
    def quick_convert_to_tensor(item):
        if isinstance(item, Tensor):
            return item
        elif isinstance(item, int | float):
            # Fast scalar to Tensor conversion
            import nabla as nb

            return nb.tensor(item)
        elif isinstance(item, dict):
            # Handle dictionaries by recursively converting values
            return {k: quick_convert_to_tensor(v) for k, v in item.items()}
        elif isinstance(item, list | tuple):
            return type(item)(quick_convert_to_tensor(sub_item) for sub_item in item)
        else:
            import nabla as nb

            # Try to convert, but handle cases where conversion might fail
            try:
                return nb.tensor(item)
            except TypeError:
                # If it's a complex structure that can't be converted, return as is
                # tree_flatten will handle extracting Tensors from it
                return item

    # Convert to Tensors first
    converted_args = quick_convert_to_tensor(actual_args)
    # Then flatten to match the compilation path
    flat_tensors = tree_flatten(converted_args)[0]
    # Finally extract impl tensors
    return [arr.impl for arr in flat_tensors]


def jit(
    func: Callable[..., Any] | None = None,
    static: bool = True,
    show_graph: bool = False,
    auto_device: bool = True,
) -> Callable[..., Any]:
    """Just-in-time compile a function for performance optimization.
    This can be used as a function call like `jit(func)` or as a decorator `@jit`.

    Args:
        func: Function to optimize with JIT compilation (should take positional arguments)
        static: If True, compile once and reuse a cached model (fast path). If False, behaves like dynamic JIT (see `djit`).
        show_graph: If True, prints the compiled graph representation when first realized.
        auto_device: If True (default) and an accelerator is available, automatically moves CPU-resident input Tensors
            to the default accelerator device before tracing/execution. In static mode, Python scalars are also
            eagerly converted to device Tensors (since they would be converted during tracing anyway). In dynamic
            mode (`static=False` / `djit`), scalars are left as Python scalars (original behavior) but CPU Tensors
            are still moved. Set to False to disable all automatic device movement/conversion.

    Returns:
        JIT-compiled function with optimized execution

    Note:
        This follows JAX's jit API:

        * Only accepts positional arguments
        * For functions requiring keyword arguments, use functools.partial or lambda
        * Supports both list-style (legacy) and unpacked arguments style (JAX-like)
        * Device auto-movement is a Nabla extension controlled by `auto_device`.

    Example:
        As a function call::
        ```python
        fast_func = jit(my_func)
        ```

        As a decorator::
        ```python
        @jit
        def my_func(x):
            return x * 2
        ```
    """
    # Handle being called as a decorator without arguments
    if func is None:
        return lambda f: jit(f, static=static, show_graph=show_graph, auto_device=auto_device)

    # Store the compiled model as a closure variable
    if static:
        cached_model = None
        output_structure = None
        param_to_model_index = None
        _fast_conversion_cache = None
        _fast_input_extractors = None

    def jit_func(*args):
        nonlocal \
            cached_model, \
            output_structure, \
            param_to_model_index, \
            _fast_conversion_cache, \
            _fast_input_extractors

        # Phase 1: Input Processing (Common for all paths)
        # Note: _handle_args_consistently is now inside process_transform_inputs
        traced_args, actual_args, is_list_style = process_transform_inputs(
            args, 
            convert_scalars=static,  # Scalars are converted only in static mode
            apply_staging=True
        )

        # Early automatic device placement (if enabled)
        if auto_device and _DEF_ACC_AVAILABLE:
            actual_args = _move_args_to_default_device(
                actual_args,
                convert_scalars=static,
                only_from_cpu=True,
            )

        if static:
            # Fast path optimization for subsequent runs
            if cached_model is not None:
                if _fast_conversion_cache is None:
                    _fast_input_extractors = _build_fast_input_extractors(
                        actual_args, is_list_style
                    )
                    _fast_conversion_cache = True

                function_param_tensors = _fast_extract_tensors(
                    actual_args, is_list_style, _fast_input_extractors
                )
                
                ordered_tensor_inputs = [
                    function_param_tensors[func_idx]
                    for func_idx, _ in param_to_model_index
                ]

                model_outputs = cached_model.execute(*ordered_tensor_inputs)
                output_tensors = [Tensor.from_impl(out) for out in model_outputs]
                
                outputs = tree_unflatten(output_structure, output_tensors)
                return process_transform_outputs(outputs, actual_args, is_list_style, untrace=False, unstage=False)

            # COMPILATION PATH (first run)
            flat_input_tensors = tree_flatten(traced_args)[0]

            outputs = func(traced_args) if is_list_style else func(*traced_args)

            flat_output_tensors, output_structure_local = tree_flatten(outputs)
            output_structure = output_structure_local
            
            from ..core.graph_execution import realize_
            cached_model, trace_inputs = realize_(
                flat_output_tensors, flat_input_tensors, show_graph=show_graph
            )

            # Create mapping from function parameter index to model input index
            param_to_model_index = []
            for model_input_idx, trace_input in enumerate(trace_inputs):
                if trace_input in flat_input_tensors:
                    func_param_idx = flat_input_tensors.index(trace_input)
                    param_to_model_index.append((func_param_idx, model_input_idx))

            # Execute with the newly compiled model
            current_flat_tensors = tree_flatten(actual_args)[0]
            function_param_tensors = [arr.impl for arr in current_flat_tensors]
            
            ordered_tensor_inputs = [
                function_param_tensors[func_idx]
                for func_idx, _ in param_to_model_index
            ]
            
            model_outputs = cached_model.execute(*ordered_tensor_inputs)
            output_tensors = [Tensor.from_impl(out) for out in model_outputs]
            
            outputs = tree_unflatten(output_structure, output_tensors)
            
            return process_transform_outputs(outputs, actual_args, is_list_style)

        else:  # Dynamic JIT path
            outputs = func(traced_args) if is_list_style else func(*traced_args)
            
            output_tensors = _extract_tensors_from_pytree(outputs)
            from ..core.graph_execution import realize_
            realize_(output_tensors, show_graph=show_graph)
            
            return process_transform_outputs(outputs, actual_args, is_list_style)

    return jit_func


def djit(
    func: Callable[..., Any] | None = None, show_graph: bool = False, auto_device: bool = True
) -> Callable[..., Any]:
    """Dynamic JIT compile a function for performance optimization.
    This can be used as a function call like `djit(func)` or as a decorator `@djit`.

    Args:
        func: Function to optimize with JIT compilation (should take positional arguments)
        show_graph: If True, prints the compiled graph representation when realized.
        auto_device: If True (default) and an accelerator is available, automatically moves CPU-resident input Tensors
            to the default accelerator device before tracing/execution. Unlike static `jit`, dynamic mode does not
            eagerly convert Python scalars to Tensors during the early device pass (to preserve prior semantics).
            Disable by setting to False.

    Returns:
        JIT-compiled function with optimized execution

    Note:
        This follows JAX's jit API:

        * Only accepts positional arguments
        * For functions requiring keyword arguments, use functools.partial or lambda
        * Supports both list-style (legacy) and unpacked arguments style (JAX-like)
        * Device auto-movement is a Nabla extension controlled by `auto_device`.
    """
    return jit(func, static=False, show_graph=show_graph, auto_device=auto_device)


def compile(
    func: Callable[..., Any],
    static: bool = False,
    show_graph: bool = False,
    auto_device: bool = True,
) -> Callable[..., Any]:
    """Just-in-time compile a function for performance optimization.
    This can be used as a function call like `jit(func)` or as a decorator `@jit`.

    Args:
        func: Function to optimize with JIT compilation (should take positional arguments)
        static: If True, compile once and reuse a cached model (fast path). If False, behaves like dynamic JIT (see `djit`). Unlike in 'jit', static defaults to False here.
        show_graph: If True, prints the compiled graph representation when first realized.
        auto_device: If True (default) and an accelerator is available, automatically moves CPU-resident input Tensors
            to the default accelerator device before tracing/execution. In static mode, Python scalars are also
            eagerly converted to device Tensors (since they would be converted during tracing anyway). In dynamic
            mode (`static=False` / `djit`), scalars are left as Python scalars (original behavior) but CPU Tensors
            are still moved. Set to False to disable all automatic device movement/conversion.

    Returns:
        JIT-compiled function with optimized execution

    Note:
        This follows JAX's jit API:

        * Only accepts positional arguments
        * For functions requiring keyword arguments, use functools.partial or lambda
        * Device auto-movement is a Nabla extension controlled by `auto_device`.

    Example:
        As a function call::
        ```python
        fast_func = compile(my_func)
        ```
    """
    return jit(
        func,
        static=static,
        show_graph=show_graph,
        auto_device=auto_device,
    )