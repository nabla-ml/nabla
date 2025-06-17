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
from typing import Any, Optional

from ..core.array import Array
from .utils import (
    _clean_traced_outputs,
    _extract_arrays_from_pytree,
    _handle_args_consistently,
    _prepare_traced_inputs,
    make_untraced_pytree,
    tree_flatten,
    tree_unflatten,
)


def _build_fast_input_extractors(actual_args, is_list_style):
    """Build fast input extractors to minimize overhead in subsequent calls."""
    # Cache input structure for ultra-fast conversion

    def analyze_structure(item):
        if isinstance(item, Array):
            return "array"
        elif isinstance(item, int | float):
            return "scalar"
        elif isinstance(item, list | tuple):
            return ("container", type(item), [analyze_structure(sub) for sub in item])
        else:
            return "other"

    structure = analyze_structure(actual_args)
    return {"is_list_style": is_list_style, "structure": structure}


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
        if struct == "array":
            return [item.impl]
        elif struct == "scalar":
            return [nb.array(item).impl]
        elif isinstance(struct, tuple) and struct[0] == "container":
            _, container_type, substruct_list = struct
            extracted = []
            for sub_item, sub_struct in zip(item, substruct_list, strict=False):
                extracted.extend(extract_with_structure(sub_item, sub_struct))
            return extracted
        else:
            return [nb.array(item).impl]

    return extract_with_structure(args, structure)


def _fast_extract_tensors_fallback(actual_args, is_list_style):
    """Fallback fast tensor extraction method."""

    # Convert to Arrays first, then extract tensors - matches compilation path
    def quick_convert_to_array(item):
        if isinstance(item, Array):
            return item
        elif isinstance(item, int | float):
            # Fast scalar to Array conversion
            import nabla as nb

            return nb.array(item)
        elif isinstance(item, list | tuple):
            return type(item)(quick_convert_to_array(sub_item) for sub_item in item)
        else:
            import nabla as nb

            return nb.array(item)

    # Convert to Arrays first
    converted_args = quick_convert_to_array(actual_args)
    # Then flatten to match the compilation path
    flat_arrays = tree_flatten(converted_args)[0]
    # Finally extract impl tensors
    return [arr.impl for arr in flat_arrays]


def jit(
    func: Optional[Callable[..., Any]] = None,
    static: bool = True,
    show_graph: bool = False,
) -> Callable[..., Any]:
    """Just-in-time compile a function for performance optimization.
    This can be used as a function call like `jit(func)` or as a decorator `@jit`.

    Args:
        func: Function to optimize with JIT compilation (should take positional arguments)

    Returns:
        JIT-compiled function with optimized execution

    Note:
        This follows JAX's jit API:

        * Only accepts positional arguments
        * For functions requiring keyword arguments, use functools.partial or lambda
        * Supports both list-style (legacy) and unpacked arguments style (JAX-like)

    Example:
        As a function call::

            fast_func = jit(my_func)

        As a decorator::

            @jit
            def my_func(x):
                return x * 2
    """
    # Handle being called as a decorator without arguments
    if func is None:
        return lambda f: jit(f, static=static, show_graph=show_graph)

    # Store the compiled model as a closure variable
    if static:
        cached_model = None
        output_structure = None
        param_to_model_index = None
        # Pre-allocate fast path variables
        _fast_conversion_cache = None
        _fast_input_extractors = None

    def jit_func(*args):
        nonlocal \
            cached_model, \
            output_structure, \
            param_to_model_index, \
            _fast_conversion_cache, \
            _fast_input_extractors

        # Common argument processing - needed for both static and non-static paths
        any_arg_traced = any(
            getattr(arg, "traced", False) for arg in _extract_arrays_from_pytree(args)
        )
        actual_args, is_list_style = _handle_args_consistently(args)

        if static:
            # Fast path optimization: skip most overhead for compiled models
            if cached_model is not None:
                # OPTIMIZED FAST PATH - minimal Python overhead
                if _fast_conversion_cache is None:
                    # First fast execution - build conversion cache
                    _fast_input_extractors = _build_fast_input_extractors(
                        actual_args, is_list_style
                    )
                    _fast_conversion_cache = True

                    # Extract tensors for this run
                    function_param_tensors = _fast_extract_tensors(
                        actual_args, is_list_style, _fast_input_extractors
                    )
                else:
                    # Ultra-fast path: direct extraction without full tracing
                    function_param_tensors = _fast_extract_tensors(
                        actual_args, is_list_style, _fast_input_extractors
                    )

                # Pre-computed reordering (this was the biggest bottleneck!)
                if param_to_model_index is None:
                    raise ValueError(
                        "param_to_model_index should not be None in fast path"
                    )
                ordered_tensor_inputs = [
                    function_param_tensors[func_idx]
                    for func_idx, _ in param_to_model_index  # type: ignore
                ]

                if cached_model is None:
                    raise ValueError("cached_model should not be None in fast path")
                model_outputs = cached_model.execute(*ordered_tensor_inputs)

                # Fast output conversion - avoid full tree operations
                output_arrays = [Array.from_impl(out) for out in model_outputs]  # type: ignore
                if output_structure is None:
                    # Single output case - return the first (and only) output array
                    outputs = (
                        output_arrays[0] if len(output_arrays) == 1 else output_arrays
                    )
                else:
                    outputs = tree_unflatten(output_structure, output_arrays)

                return outputs

            # COMPILATION PATH (first run)
            # For static JIT, use conversion to turn scalars into Arrays
            traced_args, _ = _prepare_traced_inputs(
                actual_args, is_list_style, apply_staging=True, with_conversion=True
            )
            flat_input_arrays = tree_flatten(traced_args)[0]

            # Check if we need to compile the model
            if cached_model is None:
                # Execute the function with traced inputs and appropriate style
                outputs = func(traced_args) if is_list_style else func(*traced_args)

                # Realize only the Arrays in the outputs
                flat_output_arrays, output_structure_local = tree_flatten(outputs)
                output_structure = output_structure_local  # Assign to nonlocal variable
                print(f"DEBUG: Setting output_structure to {output_structure}")
                from ..core.graph_execution import realize_

                result = realize_(
                    flat_output_arrays, flat_input_arrays, show_graph=show_graph
                )
                if isinstance(result, tuple):
                    cached_model, trace_inputs = result
                else:
                    raise ValueError(
                        "Expected tuple result from realize_ with dynamic_inputs"
                    )

                # Create mapping: function parameter index -> model input index
                param_to_model_index = []
                model_input_idx = 0
                for trace_input in trace_inputs:
                    if trace_input in flat_input_arrays:
                        func_param_idx = flat_input_arrays.index(trace_input)
                        param_to_model_index.append((func_param_idx, model_input_idx))
                        model_input_idx += 1

                # Don't return here - fall through to execute the model on first run too

            # Use the cached model for execution (both first run and subsequent runs)
            # Convert current args using the same conversion approach
            current_traced_args, _ = _prepare_traced_inputs(
                actual_args, is_list_style, apply_staging=False, with_conversion=True
            )
            current_flat_arrays = tree_flatten(current_traced_args)[0]

            # Reorder inputs to match the model's expected order
            function_param_tensors = [
                input_array.impl for input_array in current_flat_arrays
            ]

            # Reorder according to the mapping we stored during compilation
            if param_to_model_index is None:
                raise ValueError(
                    "param_to_model_index should not be None at execution time"
                )

            ordered_tensor_inputs = [None] * len(param_to_model_index)
            for func_idx, model_idx in param_to_model_index:
                ordered_tensor_inputs[model_idx] = function_param_tensors[func_idx]

            # Filter out None values and ensure we have valid tensors
            valid_inputs = [inp for inp in ordered_tensor_inputs if inp is not None]
            if cached_model is None:
                raise ValueError("cached_model should not be None at execution time")
            model_outputs = cached_model.execute(*valid_inputs)

            output_arrays = [Array.from_impl(out) for out in model_outputs]  # type: ignore

            # Convert model outputs back to the original structure
            print(f"DEBUG: About to use output_structure: {output_structure}")
            if output_structure is None:
                # Single output case - return the first (and only) output array
                outputs = output_arrays[0] if len(output_arrays) == 1 else output_arrays
            else:
                outputs = tree_unflatten(output_structure, output_arrays)

            return outputs

        else:
            # Regular JIT - use existing logic
            # Prepare traced inputs with staging enabled
            traced_args, _ = _prepare_traced_inputs(
                actual_args, is_list_style, apply_staging=True
            )

            # Execute the function with traced inputs and appropriate style
            outputs = func(traced_args) if is_list_style else func(*traced_args)

            # Realize only the Arrays in the outputs
            output_arrays = _extract_arrays_from_pytree(outputs)
            from ..core.graph_execution import realize_

            realize_(output_arrays, show_graph=show_graph)

            # make output_arrays untraced, but only if all the inputs were originally untraced
            if not any_arg_traced:
                make_untraced_pytree(outputs)

            return _clean_traced_outputs(outputs, is_list_style, remove_staging=True)

    return jit_func


def djit(
    func: Optional[Callable[..., Any]] = None, show_graph: bool = False
) -> Callable[..., Any]:
    """Dynamic JIT compile a function for performance optimization.
    This can be used as a function call like `djit(func)` or as a decorator `@djit`.

    Args:
        func: Function to optimize with JIT compilation (should take positional arguments)

    Returns:
        JIT-compiled function with optimized execution

    Note:
        This follows JAX's jit API:

        * Only accepts positional arguments
        * For functions requiring keyword arguments, use functools.partial or lambda
        * Supports both list-style (legacy) and unpacked arguments style (JAX-like)
    """
    return jit(func, static=False, show_graph=show_graph)
