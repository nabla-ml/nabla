"""
Enhanced vmap implementation brainstorming

This file explores different approaches for improving the current vmap implementation
to better handle pytree structures like JAX does.

Key issues with current implementation:
1. Limited support for nested pytree structures with different axis specifications
2. Inconsistent handling of in_axes/out_axes when they should match pytree structure
3. Complex logic for flattening/unflattening trees multiple times
4. Not following JAX's precise API for in_axes/out_axes pytree matching

JAX vmap behavior we want to match:
- in_axes must be a pytree prefix of the input arguments
- out_axes must match the structure of outputs
- Supports complex nested structures like:
  - in_axes=((0, (1, 2)),) for nested tuples
  - in_axes={'a': None, 'b': 0} for dictionaries
  - Mixed None and int values for broadcasting vs batching

Two main approaches to consider:
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nabla.core.array import Array
from nabla.core.trafos import (
    _extract_arrays_from_pytree,
    _handle_args_consistently,
    make_traced_pytree,
    make_untraced_pytree,
    tree_flatten,
    tree_map,
    tree_unflatten,
)

# ===== APPROACH 1: Direct Pytree Structure Matching =====
"""
This approach directly works with pytree structures without excessive flattening.
It validates that in_axes matches the input structure and processes trees recursively.

Pros:
- More intuitive and matches JAX behavior closely
- Avoids multiple flatten/unflatten cycles
- Cleaner error messages for structure mismatches
- More efficient for deeply nested structures

Cons:
- More complex recursive logic
- Need careful validation of structure matching
"""


def _validate_axes_structure(axes: Any, tree: Any, name: str) -> None:
    """Validate that axes structure matches the tree structure."""

    def _check_structure(axes_part: Any, tree_part: Any, path: str = "") -> None:
        if isinstance(tree_part, dict):
            if not isinstance(axes_part, dict):
                raise ValueError(
                    f"{name} at {path} must be a dict, got {type(axes_part)}"
                )
            for key in tree_part:
                if key not in axes_part:
                    raise ValueError(f"{name} missing key '{key}' at {path}")
                _check_structure(axes_part[key], tree_part[key], f"{path}.{key}")
        elif isinstance(tree_part, list | tuple):
            if not isinstance(axes_part, list | tuple) or len(axes_part) != len(
                tree_part
            ):
                raise ValueError(
                    f"{name} at {path} must be {type(tree_part)} of length {len(tree_part)}"
                )
            for i, (a, t) in enumerate(zip(axes_part, tree_part, strict=False)):
                _check_structure(a, t, f"{path}[{i}]")
        elif isinstance(tree_part, Array):
            if not isinstance(axes_part, int | type(None)):
                raise ValueError(
                    f"{name} at {path} must be int or None for Array, got {type(axes_part)}"
                )
        # else: non-Array leaf, axes_part can be anything

    _check_structure(axes, tree, "")


def _apply_vmap_to_tree(
    tree: Any, axes: Any, batch_fn: Callable[[Array, int | None], Array]
) -> Any:
    """Apply batching function to arrays in a pytree based on axes structure."""

    def _apply_recursive(tree_part: Any, axes_part: Any) -> Any:
        if isinstance(tree_part, dict):
            return {k: _apply_recursive(tree_part[k], axes_part[k]) for k in tree_part}
        elif isinstance(tree_part, list | tuple):
            result = [
                _apply_recursive(t, a)
                for t, a in zip(tree_part, axes_part, strict=False)
            ]
            return type(tree_part)(result)
        elif isinstance(tree_part, Array):
            return batch_fn(tree_part, axes_part)
        else:
            # Non-Array leaf, return unchanged
            return tree_part

    return _apply_recursive(tree, axes)


def vmap_approach1(func=None, in_axes=0, out_axes=0) -> Callable[..., Any]:
    """Enhanced vmap using direct pytree structure matching."""
    if func is None:
        return lambda f: vmap_approach1(f, in_axes=in_axes, out_axes=out_axes)

    def _prepare_input_batch(array: Array, axis: int | None) -> Array:
        """Prepare a single array for batching."""
        from nabla.ops.unary import incr_batch_dim_ctr
        from nabla.ops.view import transpose, unsqueeze

        if axis is None:
            # Broadcast case: add a size-1 batch dimension
            batched = unsqueeze(array, [0])
        else:
            # Move the specified axis to position 0
            batched = transpose(array, axis, 0) if axis != 0 else array

        # Increment batch dimension counter for tracking
        return incr_batch_dim_ctr(batched)

    def _prepare_output_unbatch(array: Array, axis: int | None) -> Array:
        """Prepare a single output array for unbatching."""
        from nabla.ops.unary import decr_batch_dim_ctr
        from nabla.ops.view import squeeze, transpose

        # Decrement batch dimension counter
        unbatched = decr_batch_dim_ctr(array)

        if axis is None:
            # Remove the batch dimension completely
            unbatched = squeeze(unbatched, [0])
        else:
            # Move axis 0 to the specified position
            if axis != 0:
                unbatched = transpose(unbatched, 0, axis)

        return unbatched

    def vectorized_func(*args):
        # Handle both list-style and unpacked arguments
        actual_args, is_list_style = _handle_args_consistently(args)

        if not actual_args:
            raise ValueError("vmap requires at least one input argument")

        # For single argument case, wrap in_axes in a tuple to indicate it's for one argument
        if len(actual_args) == 1:
            # The in_axes specification is for the structure of the single argument
            structured_in_axes = (in_axes,)
        else:
            if not isinstance(in_axes, list | tuple) or len(in_axes) != len(
                actual_args
            ):
                # Broadcast single axis spec to all arguments
                if isinstance(in_axes, int | type(None)):
                    structured_in_axes = tuple(in_axes for _ in actual_args)
                else:
                    raise ValueError(
                        f"in_axes must be a sequence of length {len(actual_args)} or a single axis spec"
                    )
            else:
                structured_in_axes = in_axes

        # Validate that in_axes structure matches input arguments structure
        for i, (arg, axis_spec) in enumerate(
            zip(actual_args, structured_in_axes, strict=False)
        ):
            try:
                _validate_axes_structure(axis_spec, arg, f"in_axes[{i}]")
            except ValueError as e:
                raise ValueError(
                    f"in_axes structure mismatch for argument {i}: {e}"
                ) from e

        # Apply batching to inputs using pytree structure
        traced_batched_args = []
        for arg, axis_spec in zip(actual_args, structured_in_axes, strict=False):
            # Make arrays traced first
            traced_arg = tree_map(lambda a: make_traced_pytree([a])[0], arg)
            # Apply batching according to axis specification
            batched_arg = _apply_vmap_to_tree(
                traced_arg, axis_spec, _prepare_input_batch
            )
            traced_batched_args.append(batched_arg)

        # Execute function with batched inputs
        if is_list_style:
            outputs = func(traced_batched_args)
        else:
            outputs = func(*traced_batched_args)

        # Handle output structure and out_axes
        if not isinstance(outputs, list | tuple):
            # Single output
            if not isinstance(out_axes, list | tuple):
                structured_out_axes = out_axes
            else:
                if len(out_axes) != 1:
                    raise ValueError("out_axes length must match number of outputs")
                structured_out_axes = out_axes[0]
            outputs_to_process = [outputs]
            axes_to_process = [structured_out_axes]
            is_single_output = True
        else:
            # Multiple outputs
            if not isinstance(out_axes, list | tuple) or len(out_axes) != len(outputs):
                # Broadcast single axis spec to all outputs
                if isinstance(out_axes, int | type(None)):
                    structured_out_axes = tuple(out_axes for _ in outputs)
                else:
                    raise ValueError(
                        f"out_axes must be a sequence of length {len(outputs)} or a single axis spec"
                    )
            else:
                structured_out_axes = out_axes
            outputs_to_process = outputs
            axes_to_process = structured_out_axes
            is_single_output = False

        # Apply unbatching to outputs
        unbatched_outputs = []
        for output, axis_spec in zip(outputs_to_process, axes_to_process, strict=False):
            try:
                _validate_axes_structure(axis_spec, output, "out_axes")
            except ValueError as e:
                raise ValueError(f"out_axes structure mismatch: {e}") from e

            unbatched_output = _apply_vmap_to_tree(
                output, axis_spec, _prepare_output_unbatch
            )
            # Make untraced
            tree_map(lambda a: make_untraced_pytree([a]), unbatched_output)
            unbatched_outputs.append(unbatched_output)

        return unbatched_outputs[0] if is_single_output else tuple(unbatched_outputs)

    return vectorized_func


# ===== APPROACH 2: Enhanced Flatten-Based with Proper Structure Validation =====
"""
This approach improves the current flatten-based method by adding proper structure validation
and more sophisticated axis handling while still using the flatten/unflatten pattern.

Pros:
- Builds on existing working code
- Easier to debug and reason about
- Less risk of breaking existing functionality
- Can reuse existing helper functions

Cons:
- Still requires multiple flatten/unflatten cycles
- Less intuitive than direct pytree matching
- May be less efficient for very nested structures
"""


def _standardize_axes_enhanced(axes: Any, trees: list[Any], name: str) -> list[Any]:
    """Enhanced version of axes standardization with proper pytree support."""
    if isinstance(axes, int | type(None)):
        # Broadcast single axis to all trees
        return [axes for _ in trees]
    elif isinstance(axes, list | tuple):
        if len(axes) != len(trees):
            raise ValueError(
                f"{name} length {len(axes)} != number of arguments {len(trees)}"
            )

        # Validate each axis specification against its corresponding tree
        standardized = []
        for i, (axis_spec, tree) in enumerate(zip(axes, trees, strict=False)):
            try:
                _validate_axes_structure(axis_spec, tree, f"{name}[{i}]")
                standardized.append(axis_spec)
            except ValueError as e:
                raise ValueError(f"Invalid {name}[{i}]: {e}") from e

        return standardized
    else:
        raise ValueError(f"{name} must be int, None, or sequence, got {type(axes)}")


def _extract_axis_info_from_pytree(
    tree: Any, axis_spec: Any
) -> tuple[list[int | None], Any]:
    """Extract flat list of axis specifications matching flattened arrays."""
    tree_arrays, tree_structure = tree_flatten(tree)

    def _extract_axes_recursive(spec: Any, struct: Any) -> list[int | None]:
        if struct is None:  # Array placeholder
            return [spec]
        elif isinstance(struct, dict):
            axes_list = []
            for key in sorted(struct.keys()):
                axes_list.extend(_extract_axes_recursive(spec[key], struct[key]))
            return axes_list
        elif isinstance(struct, list | tuple):
            axes_list = []
            for i, item in enumerate(struct):
                axes_list.extend(_extract_axes_recursive(spec[i], item))
            return axes_list
        else:
            # Non-Array leaf, no axes info needed
            return []

    flat_axes = _extract_axes_recursive(axis_spec, tree_structure)
    return flat_axes, tree_structure


def vmap_approach2(func=None, in_axes=0, out_axes=0) -> Callable[..., Any]:
    """Enhanced vmap using improved flatten-based approach."""
    if func is None:
        return lambda f: vmap_approach2(f, in_axes=in_axes, out_axes=out_axes)

    def vectorized_func(*args):
        # Handle both list-style and unpacked arguments
        actual_args, is_list_style = _handle_args_consistently(args)

        if not actual_args:
            raise ValueError("vmap requires at least one input argument")

        # Standardize and validate in_axes
        standardized_in_axes = _standardize_axes_enhanced(
            in_axes, actual_args, "in_axes"
        )

        # Process each input argument
        batched_args = []
        for arg, axis_spec in zip(actual_args, standardized_in_axes, strict=False):
            # Extract flat arrays and axis info
            flat_axes, tree_structure = _extract_axis_info_from_pytree(arg, axis_spec)
            arg_arrays = _extract_arrays_from_pytree(arg)

            # Validate axis specifications
            if len(flat_axes) != len(arg_arrays):
                raise ValueError(
                    f"Axis specification length {len(flat_axes)} doesn't match number of arrays {len(arg_arrays)}"
                )

            # Apply batching to each array
            from nabla.core.trafos import _prepare_vmap_inputs

            batched_arrays = _prepare_vmap_inputs(arg_arrays, flat_axes)

            # Reconstruct tree structure with batched arrays
            batched_arg = tree_unflatten(tree_structure, batched_arrays)
            batched_args.append(batched_arg)

        # Execute function
        outputs = func(batched_args) if is_list_style else func(*batched_args)

        # Handle outputs
        if not isinstance(outputs, list | tuple):
            outputs_list = [outputs]
            is_single_output = True
        else:
            outputs_list = outputs
            is_single_output = False

        # Standardize out_axes
        standardized_out_axes = _standardize_axes_enhanced(
            out_axes, outputs_list, "out_axes"
        )

        # Process each output
        unbatched_outputs = []
        for output, axis_spec in zip(outputs_list, standardized_out_axes, strict=False):
            # Extract flat arrays and axis info for output
            flat_out_axes, out_tree_structure = _extract_axis_info_from_pytree(
                output, axis_spec
            )
            output_arrays = _extract_arrays_from_pytree(output)

            # Validate axis specifications
            if len(flat_out_axes) != len(output_arrays):
                raise ValueError(
                    f"Output axis specification length {len(flat_out_axes)} doesn't match number of arrays {len(output_arrays)}"
                )

            # Apply unbatching to each array
            from nabla.core.trafos import _prepare_vmap_outputs

            unbatched_arrays = _prepare_vmap_outputs(output_arrays, flat_out_axes)

            # Reconstruct tree structure with unbatched arrays
            unbatched_output = tree_unflatten(out_tree_structure, unbatched_arrays)
            unbatched_outputs.append(unbatched_output)

        return unbatched_outputs[0] if is_single_output else tuple(unbatched_outputs)

    return vectorized_func


# ===== APPROACH 3: Hybrid Approach =====
"""
This approach combines the best of both worlds:
- Uses direct pytree processing for simple cases (int/None axes)
- Falls back to enhanced flatten-based processing for complex pytree structures
- Provides maximum compatibility and performance

Pros:
- Best performance for common cases
- Full support for complex pytree structures
- Maintains backward compatibility
- Easier incremental adoption

Cons:
- More complex implementation
- Two code paths to maintain
"""


def _is_simple_axis_spec(axis_spec: Any) -> bool:
    """Check if axis specification is simple (int, None, or flat sequence)."""
    if isinstance(axis_spec, int | type(None)):
        return True
    if isinstance(axis_spec, list | tuple):
        return all(isinstance(x, int | type(None)) for x in axis_spec)
    return False


def vmap_approach3(func=None, in_axes=0, out_axes=0) -> Callable[..., Any]:
    """Hybrid vmap implementation combining direct pytree and flatten-based approaches."""
    if func is None:
        return lambda f: vmap_approach3(f, in_axes=in_axes, out_axes=out_axes)

    def vectorized_func(*args):
        actual_args, is_list_style = _handle_args_consistently(args)

        if not actual_args:
            raise ValueError("vmap requires at least one input argument")

        # Determine if we can use the fast path (simple axis specifications)
        all_args_simple = all(
            _is_simple_axis_spec(
                in_axes
                if isinstance(in_axes, int | type(None))
                else in_axes[i]
                if isinstance(in_axes, list | tuple) and i < len(in_axes)
                else in_axes
            )
            for i in range(len(actual_args))
        )

        outputs_will_be_simple = _is_simple_axis_spec(out_axes)

        if all_args_simple and outputs_will_be_simple:
            # Use optimized path for simple cases
            return _vmap_simple_path(
                func, actual_args, is_list_style, in_axes, out_axes
            )
        else:
            # Use full pytree path for complex cases
            return _vmap_pytree_path(
                func, actual_args, is_list_style, in_axes, out_axes
            )

    return vectorized_func


def _vmap_simple_path(func, actual_args, is_list_style, in_axes, out_axes):
    """Optimized path for simple axis specifications."""
    # This would be similar to the current implementation but cleaner
    from nabla.core.trafos import (
        _prepare_vmap_inputs,
        _prepare_vmap_outputs,
    )

    # Standardize axes the simple way
    if isinstance(in_axes, int | type(None)):
        adapted_in_axes = [in_axes] * len(actual_args)
    else:
        adapted_in_axes = list(in_axes)

    # Process inputs (simplified for flat arrays)
    batched_args = []
    for arg, axis in zip(actual_args, adapted_in_axes, strict=False):
        arg_arrays = _extract_arrays_from_pytree(arg)
        arg_structure = tree_flatten(arg)[1]

        # All arrays get the same axis treatment
        array_axes = [axis] * len(arg_arrays)
        batched_arrays = _prepare_vmap_inputs(arg_arrays, array_axes)

        batched_arg = tree_unflatten(arg_structure, batched_arrays)
        batched_args.append(batched_arg)

    # Execute function
    outputs = func(batched_args) if is_list_style else func(*batched_args)

    # Process outputs (similar simplification)
    if not isinstance(outputs, list | tuple):
        outputs_list = [outputs]
        is_single_output = True
    else:
        outputs_list = outputs
        is_single_output = False

    if isinstance(out_axes, int | type(None)):
        adapted_out_axes = [out_axes] * len(outputs_list)
    else:
        adapted_out_axes = list(out_axes)

    unbatched_outputs = []
    for output, axis in zip(outputs_list, adapted_out_axes, strict=False):
        output_arrays = _extract_arrays_from_pytree(output)
        output_structure = tree_flatten(output)[1]

        array_out_axes = [axis] * len(output_arrays)
        unbatched_arrays = _prepare_vmap_outputs(output_arrays, array_out_axes)

        unbatched_output = tree_unflatten(output_structure, unbatched_arrays)
        unbatched_outputs.append(unbatched_output)

    return unbatched_outputs[0] if is_single_output else tuple(unbatched_outputs)


def _vmap_pytree_path(func, actual_args, is_list_style, in_axes, out_axes):
    """Full pytree path for complex axis specifications."""
    # This would use approach 1 or 2 from above
    return vmap_approach1(func, in_axes, out_axes)(
        *([actual_args] if is_list_style else actual_args)
    )


# ===== TESTING AND VALIDATION EXAMPLES =====
"""
Examples of how the enhanced vmap should handle various cases:
"""


def test_examples():
    """Examples of enhanced vmap usage matching JAX behavior."""
    import nabla as nb

    # Example 1: Simple case (should work with all approaches)
    def simple_func(x):
        return x * 2

    x = nb.randn((5, 3))
    vmap_simple = vmap_approach1(simple_func, in_axes=0, out_axes=0)
    vmap_simple(x)

    # Example 2: Dictionary inputs (requires enhanced approaches)
    def dict_func(inputs):
        return {"output": inputs["a"] + inputs["b"]}

    inputs = {
        "a": nb.randn((5, 3)),
        "b": nb.randn((3,)),  # broadcasted
    }
    vmap_dict = vmap_approach1(
        dict_func, in_axes={"a": 0, "b": None}, out_axes={"output": 0}
    )
    vmap_dict(inputs)

    # Example 3: Nested tuple inputs
    def nested_func(inputs):
        x, (y, z) = inputs
        return nb.matmul(x, nb.matmul(y, z))

    a, b, c, d = 2, 3, 4, 5
    k = 6  # batch size
    inputs = (
        nb.ones((k, a, b)),  # x: batched on axis 0
        (
            nb.ones((b, k, c)),  # y: batched on axis 1
            nb.ones((c, k, d)),  # z: batched on axis 1 (changed from axis 2)
        ),
    )
    vmap_nested = vmap_approach1(nested_func, in_axes=(0, (1, 1)), out_axes=0)
    vmap_nested(inputs)

    # Example 4: Mixed None and integer axes
    def broadcast_func(x, y):
        return x + y

    x_batched = nb.randn((5, 3))
    y_scalar = nb.randn((3,))
    vmap_broadcast = vmap_approach1(broadcast_func, in_axes=(0, None), out_axes=0)
    vmap_broadcast(x_batched, y_scalar)


# ===== RECOMMENDATIONS =====
"""
Recommendation: Start with Approach 1 (Direct Pytree Matching)

Reasons:
1. Most closely matches JAX behavior and expectations
2. Cleaner, more intuitive implementation
3. Better performance for nested structures
4. Easier to extend and maintain
5. More precise error messages

Implementation strategy:
1. Implement Approach 1 first in vmapped_enhanced.py
2. Add comprehensive tests covering all JAX vmap examples
3. Validate against current implementation for simple cases
4. Once stable, integrate into main trafos.py file
5. Consider adding Approach 3 (hybrid) later for performance optimization

Key validation points:
- All current vmap tests should still pass
- JAX compatibility examples should work correctly
- Performance should be comparable or better
- Error messages should be clear and helpful
"""

if __name__ == "__main__":
    # Test basic functionality
    test_examples()
    print("Enhanced vmap examples completed!")
