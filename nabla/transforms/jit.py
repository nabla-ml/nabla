from collections.abc import Callable
from typing import Any

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


def jit(
    func: Callable[..., Any] = None, static: bool = False, show_graph: bool = False
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

    def jit_func(*args):
        nonlocal cached_model, output_structure, param_to_model_index

        any_arg_traced = any(
            getattr(arg, "traced", False) for arg in _extract_arrays_from_pytree(args)
        )
        # Use common argument handling logic
        actual_args, is_list_style = _handle_args_consistently(args)

        if static:
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
                flat_output_arrays, output_structure = tree_flatten(outputs)
                from ..core.graph_execution import realize_

                cached_model, trace_inputs = realize_(
                    flat_output_arrays, flat_input_arrays, show_graph=show_graph
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
            ordered_tensor_inputs = [None] * len(param_to_model_index)
            for func_idx, model_idx in param_to_model_index:
                ordered_tensor_inputs[model_idx] = function_param_tensors[func_idx]

            model_outputs = cached_model.execute(*ordered_tensor_inputs)
            output_arrays = [Array.from_impl(out) for out in model_outputs]

            # Convert model outputs back to the original structure
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


# create alias sjit for jit with static = True
def sjit(
    func: Callable[..., Any] = None, show_graph: bool = False
) -> Callable[..., Any]:
    """
    Just-in-time compile a function for performance optimization with static=True.
    This is a simplified alias for jit with static=True, which is useful for
    functions that do not require dynamic shapes or inputs.
    Args:
        func: Function to optimize with JIT compilation (should take positional arguments)
    Returns:
        JIT-compiled function with optimized execution
    Note:
        This follows JAX's jit API with static=True:
        - Only accepts positional arguments
        - For functions requiring keyword arguments, use functools.partial or lambda
        - Supports both list-style (legacy) and unpacked arguments style (JAX-like)
    Example:
        As a function call::

            fast_func = sjit(my_func)

        As a decorator::

            @sjit
            def my_func(x):
                return x * 2
    """
    return jit(func, static=True, show_graph=show_graph)
