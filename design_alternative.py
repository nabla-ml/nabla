"""
Alternative design approach for cleaner separation of concerns.
This is just a design sketch, not meant to be executed.
"""

def pullback_core(
    inputs: Any,
    outputs: Any, 
    cotangents: Any,
) -> Any:
    """Core pullback computation - always returns gradients in input structure."""
    # Extract arrays from pytree structures
    input_arrays = _extract_arrays_from_pytree(inputs)
    output_arrays = _extract_arrays_from_pytree(outputs)
    cotangent_arrays = _extract_arrays_from_pytree(cotangents)
    
    # ... same core VJP computation as before ...
    
    # Always return gradients in the same structure as inputs
    return tree_map(_replace_with_gradient, inputs)


def pullback(inputs: Any, outputs: Any, cotangents: Any) -> Any:
    """Public pullback API - simple wrapper around core function."""
    return pullback_core(inputs, outputs, cotangents)


def vjp(func: Callable[..., Any], *args, **kwargs) -> tuple[Any, Callable]:
    """VJP with clean structure handling logic."""
    inputs_pytree = (args, kwargs)
    traced_inputs_pytree = make_traced_pytree(inputs_pytree)
    traced_args, traced_kwargs = traced_inputs_pytree
    outputs = func(*traced_args, **traced_kwargs)

    def vjp_fn(cotangents: Any) -> Any:
        """VJP function with its own structure logic."""
        # Get gradients in (args, kwargs) structure
        grad_args, grad_kwargs = pullback_core(traced_inputs_pytree, outputs, cotangents)
        
        make_untraced_pytree(grad_args)
        make_untraced_pytree(grad_kwargs)
        
        # Apply VJP-specific return structure logic here
        if not kwargs:
            if len(args) == 1:
                return grad_args[0]  # Single argument case
            else:
                return grad_args     # Multiple arguments case
        else:
            return grad_args, grad_kwargs  # Mixed args/kwargs case

    make_untraced_pytree(outputs)
    return outputs, vjp_fn
