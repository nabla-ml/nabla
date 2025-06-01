#!/usr/bin/env python3

"""Compare Nabla VJP behavior with JAX VJP behavior."""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import vjp as jax_vjp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available for comparison")

import nabla as nb


def test_single_arg_comparison():
    """Compare single argument behavior."""
    print("=== Single Argument Comparison ===")
    
    def func_jax(x):
        return x * x + 2.0 * x + 1.0
    
    def func_nabla(x):
        return x * x + nb.array([2.0]) * x + nb.array([1.0])
    
    # JAX test
    if JAX_AVAILABLE:
        x_jax = jnp.array([2.0, 3.0])
        outputs_jax, vjp_fn_jax = jax_vjp(func_jax, x_jax)
        gradient_jax = vjp_fn_jax(jnp.array([1.0, 1.0]))[0]  # JAX returns tuple
        
        print(f"JAX - Input: {x_jax}")
        print(f"JAX - Output: {outputs_jax}")
        print(f"JAX - Gradient: {gradient_jax}")
        print(f"JAX - Gradient type: {type(gradient_jax)}")
        print(f"JAX - VJP function returns: {type(vjp_fn_jax(jnp.array([1.0, 1.0])))}")
    
    # Nabla test
    x_nabla = nb.array([2.0, 3.0])
    outputs_nabla, vjp_fn_nabla = nb.vjp(func_nabla, x_nabla)
    gradient_nabla = vjp_fn_nabla(nb.array([1.0, 1.0]))
    
    print(f"Nabla - Input: {x_nabla}")
    print(f"Nabla - Output: {outputs_nabla}")
    print(f"Nabla - Gradient: {gradient_nabla}")
    print(f"Nabla - Gradient type: {type(gradient_nabla)}")
    print()


def test_multiple_args_comparison():
    """Compare multiple arguments behavior."""
    print("=== Multiple Arguments Comparison ===")
    
    def func_jax(x, y, z):
        return x * y + y * z + x * z
    
    def func_nabla(x, y, z):
        return x * y + y * z + x * z
    
    # JAX test
    if JAX_AVAILABLE:
        x_jax = jnp.array([1.0, 2.0])
        y_jax = jnp.array([3.0, 4.0])
        z_jax = jnp.array([5.0, 6.0])
        
        outputs_jax, vjp_fn_jax = jax_vjp(func_jax, x_jax, y_jax, z_jax)
        gradients_jax = vjp_fn_jax(jnp.array([1.0, 1.0]))
        
        print(f"JAX - Inputs: {x_jax}, {y_jax}, {z_jax}")
        print(f"JAX - Output: {outputs_jax}")
        print(f"JAX - Gradients: {gradients_jax}")
        print(f"JAX - Gradients type: {type(gradients_jax)}")
        print(f"JAX - Number of gradients: {len(gradients_jax)}")
    
    # Nabla test
    x_nabla = nb.array([1.0, 2.0])
    y_nabla = nb.array([3.0, 4.0])
    z_nabla = nb.array([5.0, 6.0])
    
    outputs_nabla, vjp_fn_nabla = nb.vjp(func_nabla, x_nabla, y_nabla, z_nabla)
    gradients_nabla = vjp_fn_nabla(nb.array([1.0, 1.0]))
    
    print(f"Nabla - Inputs: {x_nabla}, {y_nabla}, {z_nabla}")
    print(f"Nabla - Output: {outputs_nabla}")
    print(f"Nabla - Gradients: {gradients_nabla}")
    print(f"Nabla - Gradients type: {type(gradients_nabla)}")
    print(f"Nabla - Number of gradients: {len(gradients_nabla)}")
    print()


def test_nested_structures_comparison():
    """Compare nested structures behavior."""
    print("=== Nested Structures Comparison ===")
    
    def func_jax(data):
        x = data['x']
        y_list = data['y']
        return x * y_list[0] + x * y_list[1]
    
    def func_nabla(data):
        x = data['x']
        y_list = data['y']
        return x * y_list[0] + x * y_list[1]
    
    # JAX test
    if JAX_AVAILABLE:
        x_jax = jnp.array([2.0, 3.0])
        y1_jax = jnp.array([4.0, 5.0])
        y2_jax = jnp.array([6.0, 7.0])
        data_jax = {'x': x_jax, 'y': [y1_jax, y2_jax]}
        
        outputs_jax, vjp_fn_jax = jax_vjp(func_jax, data_jax)
        gradient_jax = vjp_fn_jax(jnp.array([1.0, 1.0]))[0]
        
        print(f"JAX - Input: {data_jax}")
        print(f"JAX - Output: {outputs_jax}")
        print(f"JAX - Gradient type: {type(gradient_jax)}")
        print(f"JAX - Gradient keys: {list(gradient_jax.keys())}")
        print(f"JAX - Gradient x: {gradient_jax['x']}")
        print(f"JAX - Gradient y[0]: {gradient_jax['y'][0]}")
        print(f"JAX - Gradient y[1]: {gradient_jax['y'][1]}")
    
    # Nabla test
    x_nabla = nb.array([2.0, 3.0])
    y1_nabla = nb.array([4.0, 5.0])
    y2_nabla = nb.array([6.0, 7.0])
    data_nabla = {'x': x_nabla, 'y': [y1_nabla, y2_nabla]}
    
    outputs_nabla, vjp_fn_nabla = nb.vjp(func_nabla, data_nabla)
    gradient_nabla = vjp_fn_nabla(nb.array([1.0, 1.0]))
    
    print(f"Nabla - Input: {data_nabla}")
    print(f"Nabla - Output: {outputs_nabla}")
    print(f"Nabla - Gradient type: {type(gradient_nabla)}")
    print(f"Nabla - Gradient keys: {list(gradient_nabla.keys())}")
    print(f"Nabla - Gradient x: {gradient_nabla['x']}")
    print(f"Nabla - Gradient y[0]: {gradient_nabla['y'][0]}")
    print(f"Nabla - Gradient y[1]: {gradient_nabla['y'][1]}")
    print()


def test_list_input_comparison():
    """Compare list input behavior."""
    print("=== List Input Comparison ===")
    
    def func_jax(inputs):
        return inputs[0] ** 3
    
    def func_nabla(inputs):
        return inputs[0] ** 3
    
    # JAX test
    if JAX_AVAILABLE:
        x_jax = jnp.array([2.0])
        inputs_jax = [x_jax]
        
        outputs_jax, vjp_fn_jax = jax_vjp(func_jax, inputs_jax)
        gradient_jax = vjp_fn_jax(jnp.array([1.0]))[0]
        
        print(f"JAX - Input: {inputs_jax}")
        print(f"JAX - Output: {outputs_jax}")
        print(f"JAX - Gradient: {gradient_jax}")
        print(f"JAX - Gradient type: {type(gradient_jax)}")
    
    # Nabla test
    x_nabla = nb.array([2.0])
    inputs_nabla = [x_nabla]
    
    outputs_nabla, vjp_fn_nabla = nb.vjp(func_nabla, inputs_nabla)
    gradient_nabla = vjp_fn_nabla([nb.array([1.0])])
    
    print(f"Nabla - Input: {inputs_nabla}")
    print(f"Nabla - Output: {outputs_nabla}")
    print(f"Nabla - Gradient: {gradient_nabla}")
    print(f"Nabla - Gradient type: {type(gradient_nabla)}")
    print()


def test_signature_behavior():
    """Test how JAX and Nabla handle different function signatures."""
    print("=== Function Signature Behavior ===")
    
    if JAX_AVAILABLE:
        print("JAX VJP function signature behavior:")
        
        # Single argument
        def f1(x): return x * 2
        _, vjp1 = jax_vjp(f1, 1.0)
        result1 = vjp1(1.0)
        print(f"Single arg vjp(f, x) returns: {type(result1)} with {len(result1)} elements")
        
        # Multiple arguments
        def f2(x, y): return x * y
        _, vjp2 = jax_vjp(f2, 1.0, 2.0)
        result2 = vjp2(1.0)
        print(f"Multi arg vjp(f, x, y) returns: {type(result2)} with {len(result2)} elements")
        
        # List as single argument
        def f3(inputs): return inputs[0] * 2
        _, vjp3 = jax_vjp(f3, [1.0])
        result3 = vjp3(1.0)
        print(f"List arg vjp(f, [x]) returns: {type(result3)} with {len(result3)} elements")
        print(f"Content: {result3[0]} (type: {type(result3[0])})")
        
        # Dict as single argument
        def f4(data): return data['x'] * 2
        _, vjp4 = jax_vjp(f4, {'x': 1.0})
        result4 = vjp4(1.0)
        print(f"Dict arg vjp(f, {{'x': x}}) returns: {type(result4)} with {len(result4)} elements")
        print(f"Content: {result4[0]} (type: {type(result4[0])})")
    
    print("\nNabla VJP function signature behavior:")
    
    # Single argument  
    def f1_nb(x): return x * nb.array([2.0])
    _, vjp1_nb = nb.vjp(f1_nb, nb.array([1.0]))
    result1_nb = vjp1_nb(nb.array([1.0]))
    print(f"Single arg vjp(f, x) returns: {type(result1_nb)}")
    
    # Multiple arguments
    def f2_nb(x, y): return x * y
    _, vjp2_nb = nb.vjp(f2_nb, nb.array([1.0]), nb.array([2.0]))
    result2_nb = vjp2_nb(nb.array([1.0]))
    print(f"Multi arg vjp(f, x, y) returns: {type(result2_nb)} with {len(result2_nb)} elements")
    
    # List as single argument
    def f3_nb(inputs): return inputs[0] * nb.array([2.0])
    _, vjp3_nb = nb.vjp(f3_nb, [nb.array([1.0])])
    result3_nb = vjp3_nb([nb.array([1.0])])
    print(f"List arg vjp(f, [x]) returns: {type(result3_nb)} with {len(result3_nb)} elements")
    print(f"Content: {result3_nb[0]} (type: {type(result3_nb[0])})")
    
    # Dict as single argument
    def f4_nb(data): return data['x'] * nb.array([2.0])
    _, vjp4_nb = nb.vjp(f4_nb, {'x': nb.array([1.0])})
    result4_nb = vjp4_nb({'x': nb.array([1.0])})
    print(f"Dict arg vjp(f, {{'x': x}}) returns: {type(result4_nb)}")
    print(f"Content: {result4_nb} (type: {type(result4_nb)})")
    print()


if __name__ == "__main__":
    print("🔍 JAX vs Nabla VJP Behavior Comparison")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("Note: JAX is not available, showing only Nabla behavior")
        print()
    
    test_single_arg_comparison()
    test_multiple_args_comparison()
    test_nested_structures_comparison()
    test_list_input_comparison()
    test_signature_behavior()
    
    print("📊 Summary:")
    if JAX_AVAILABLE:
        print("- JAX vjp function ALWAYS returns a tuple of gradients")
        print("- Even for single arguments, JAX returns (gradient,) not gradient")
        print("- JAX preserves pytree structure within the tuple")
    print("- Nabla vjp function returns gradients in the same structure as inputs")
    print("- Single arg: returns single gradient (not tuple)")
    print("- Multiple args: returns tuple of gradients")  
    print("- Nested structures: preserves exact input structure")
