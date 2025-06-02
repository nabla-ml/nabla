#!/usr/bin/env python3
"""Test script to demonstrate pytree support in vjp function."""

import nabla as nb

# Test 1: Simple array
print("=== Test 1: Simple Array ===")
x = nb.array([1.0, 2.0, 3.0])


def simple_func(x):
    return nb.sum(x**2)


out, vjp_fn = nb.vjp(simple_func, x)
grad = vjp_fn(nb.ones_like(out))
print(f"Input: {x}")
print(f"Output: {out}")
print(f"Gradient: {grad}")

# Test 2: Multiple arrays
print("\n=== Test 2: Multiple Arrays ===")
x = nb.array([1.0, 2.0])
y = nb.array([3.0, 4.0])


def multi_func(x, y):
    return nb.sum(x * y)


out, vjp_fn = nb.vjp(multi_func, x, y)
grad_x, grad_y = vjp_fn(nb.ones_like(out))
print(f"Inputs: x={x}, y={y}")
print(f"Output: {out}")
print(f"Gradients: grad_x={grad_x}, grad_y={grad_y}")

# Test 3: Dictionary input
print("\n=== Test 3: Dictionary Input ===")
params = {"weights": nb.array([1.0, 2.0]), "bias": nb.array([0.5])}


def dict_func(params):
    return nb.sum(params["weights"] ** 2) + nb.sum(params["bias"])


out, vjp_fn = nb.vjp(dict_func, params)
grad_dict = vjp_fn(nb.ones_like(out))
print(f"Input params: {params}")
print(f"Output: {out}")
print(f"Gradient dict: {grad_dict}")

# Test 4: Nested structure
print("\n=== Test 4: Nested Structure ===")
nested_params = {
    "layer1": [nb.array([1.0]), nb.array([2.0])],
    "layer2": {"w": nb.array([3.0]), "b": nb.array([4.0])},
}


def nested_func(params):
    l1_sum = nb.sum(params["layer1"][0]) + nb.sum(params["layer1"][1])
    l2_sum = nb.sum(params["layer2"]["w"]) + nb.sum(params["layer2"]["b"])
    return l1_sum + l2_sum


out, vjp_fn = nb.vjp(nested_func, nested_params)
grad_nested = vjp_fn(nb.ones_like(out))
print(f"Input nested params: {nested_params}")
print(f"Output: {out}")
print(f"Gradient nested: {grad_nested}")
