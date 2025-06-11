from typing import Callable, Any, Sequence
from .vjp import vjp


def value_and_grad(
    fun: Callable = None,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence = (),
) -> Callable[..., Any]:
    """
    Creates a function that evaluates both the value and gradient of fun.

    This function uses VJP (Vector-Jacobian Product) directly with a cotangent
    of ones_like(output) to compute gradients for scalar-valued functions.
    This is simpler and more efficient than using jacrev/jacfwd for scalar outputs.

    Parameters:
        fun: Function to be differentiated. Should return a scalar.
        argnums: Which positional argument(s) to differentiate with respect to (default 0).
        has_aux: Whether fun returns (output, aux) pair (default False).
        holomorphic: Whether fun is holomorphic - currently ignored (default False).
        allow_int: Whether to allow integer inputs - currently ignored (default False).
        reduce_axes: Axes to reduce over - currently ignored (default ()).

    Returns:
        A function that computes both the value and gradient of fun.

    Examples:
        # As a function call
        value_and_grad_fn = value_and_grad(my_loss)
        value, grads = value_and_grad_fn(x)

        # As a decorator
        @value_and_grad
        def my_loss(x):
            return x**2

        value, grads = my_loss(3.0)
    """

    # Handle being used as a decorator without arguments
    if fun is None:
        return lambda f: value_and_grad(
            f,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        )

    def value_and_grad_fn(*args: Any) -> Any:
        """
        The actual value_and_grad function that gets returned.

        Validates that the function returns a scalar output, then computes
        both the value and gradient using VJP with ones_like cotangent.
        """
        # Compute VJP to get both output and pullback function
        if has_aux:
            output, vjp_fn, aux = vjp(fun, *args, has_aux=True)
        else:
            output, vjp_fn = vjp(fun, *args, has_aux=False)

        # Validate scalar output - handle both single scalars and pytrees with scalar leaves
        # JAX allows arrays with shape () or (1,) - both are considered "scalar-like"
        def validate_scalar_output(obj):
            """Recursively validate that all array leaves are scalars or scalar-like."""
            from ..core.array import Array
            
            if isinstance(obj, Array):
                # JAX behavior: allow both () and (1,) shapes as "scalar-like"
                if obj.shape != () and obj.shape != (1,):
                    raise ValueError(
                        f"Gradient only defined for scalar-output functions. "
                        f"Found array with shape: {obj.shape}"
                    )
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    validate_scalar_output(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    validate_scalar_output(value)
            else:
                # Handle non-Array outputs (like numpy arrays, Python scalars)
                import numpy as np
                test_array = np.asarray(obj)
                if test_array.shape != () and test_array.shape != (1,):
                    raise ValueError(
                        f"value_and_grad only defined for scalar-output functions. "
                        f"Found non-scalar with shape: {test_array.shape}"
                    )
        
        validate_scalar_output(output)

        # Create cotangent of ones_like(output) and compute gradients
        def create_ones_like_cotangent(obj):
            """Create a cotangent with the same structure as output, but with ones_like for each Array."""
            from ..core.array import Array
            from ..ops.creation import ones_like
            
            if isinstance(obj, Array):
                return ones_like(obj)
            elif isinstance(obj, (list, tuple)):
                return type(obj)(create_ones_like_cotangent(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: create_ones_like_cotangent(v) for k, v in obj.items()}
            else:
                # For non-Array leaves, we don't need cotangents
                return obj
        
        cotangent = create_ones_like_cotangent(output)
        
        # VJP computes gradients for all inputs, select based on argnums
        all_gradients = vjp_fn(cotangent)

        # Handle argnums selection - match JAX behavior exactly
        # The key insight: we need to distinguish between:
        # 1. Multiple separate arguments: func(a, b, c) -> vjp returns tuple of gradients
        # 2. Single argument that's a pytree: func([a, b, c]) -> vjp returns gradient in same structure
        
        num_inputs = len(args)
        
        if num_inputs == 1:
            # Single input case - but could be a single pytree (like a list)
            # In this case, all_gradients has the same structure as the input
            # We need to handle argnums as indices into that structure
            single_input = args[0]
            
            # Check if the single input is a list/tuple that can be indexed
            if isinstance(single_input, (list, tuple)) and isinstance(all_gradients, (list, tuple)):
                # This handles the case like: func([x, targets, param1, param2, ...]) 
                # where argnums=[2, 3, ...] should select param gradients
                if isinstance(argnums, int):
                    # Single argnum - select one gradient from the list
                    if argnums < 0 or argnums >= len(all_gradients):
                        raise ValueError(f"argnum {argnums} is out of bounds for function with {len(all_gradients)} elements in input")
                    selected_gradients = all_gradients[argnums]
                else:
                    # Multiple argnums - select multiple gradients from the list
                    max_index = len(all_gradients) - 1
                    for idx in argnums:
                        if idx < 0 or idx > max_index:
                            raise ValueError(f"argnum {idx} is out of bounds for function with {len(all_gradients)} elements in input")
                    selected_gradients = tuple(all_gradients[i] for i in argnums)
            else:
                # Single non-list input - traditional single argument case
                if isinstance(argnums, int):
                    if argnums != 0:
                        raise ValueError(f"argnum {argnums} is out of bounds for function with 1 argument")
                    selected_gradients = all_gradients
                else:
                    # Multiple argnums for single input - should only contain 0
                    if len(argnums) == 1 and argnums[0] == 0:
                        selected_gradients = (all_gradients,)
                    else:
                        raise ValueError(f"argnums {argnums} contains invalid indices for function with 1 argument")
        else:
            # Multiple input function case - all_gradients is a tuple
            if isinstance(argnums, int):
                # Single argnum case
                selected_gradients = all_gradients[argnums]
            else:
                # Multiple argnums (sequence)
                selected_gradients = tuple(all_gradients[i] for i in argnums)

        # Return based on has_aux - JAX returns ((value, aux), grad) when has_aux=True
        if has_aux:
            return (output, aux), selected_gradients
        else:
            return output, selected_gradients

    return value_and_grad_fn


def grad(
    fun: Callable = None,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence = (),
    mode: str = "reverse",  # Kept for API compatibility but ignored
) -> Callable[..., Any]:
    """
    Creates a function that evaluates the gradient of fun.

    This is implemented as a special case of value_and_grad that only returns
    the gradient part. Uses VJP directly for efficiency with scalar outputs.

    Parameters:
        fun: Function to be differentiated. Should return a scalar.
        argnums: Which positional argument(s) to differentiate with respect to (default 0).
        has_aux: Whether fun returns (output, aux) pair (default False).
        holomorphic: Whether fun is holomorphic - currently ignored (default False).
        allow_int: Whether to allow integer inputs - currently ignored (default False).
        reduce_axes: Axes to reduce over - currently ignored (default ()).
        mode: Kept for API compatibility but ignored (always uses reverse-mode VJP).

    Returns:
        A function that computes the gradient of fun.

    Examples:
        # As a function call
        grad_fn = grad(my_loss)
        grads = grad_fn(x)

        # As a decorator
        @grad
        def my_loss(x):
            return x**2

        grads = my_loss(3.0)  # Returns gradient, not function value
    """
    # Handle decorator pattern: if fun is None, return a decorator
    if fun is None:
        return lambda f: grad(f, argnums, has_aux, holomorphic, allow_int, reduce_axes, mode)

    # Get the value_and_grad function
    value_and_grad_fn = value_and_grad(
        fun=fun,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    def grad_fn(*args: Any) -> Any:
        """
        The actual gradient function that gets returned.
        
        Just calls value_and_grad and returns only the gradient part.
        """
        if has_aux:
            (value, aux), gradients = value_and_grad_fn(*args)
            return gradients, aux
        else:
            value, gradients = value_and_grad_fn(*args)
            return gradients

    return grad_fn
