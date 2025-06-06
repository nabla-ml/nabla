#!/usr/bin/env python3
"""Test script to demonstrate pytree support in vjp function."""

import endia as nd

# Test 1: Simple array
print("=== Test 1: Simple Array ===")
x = nd.array([1.0, 2.0, 3.0])


def simple_func(x):
    return nd.sum(x**2)


out, vjp_fn = nd.vjp(simple_func, x)
grad = vjp_fn(nd.ones_like(out))
print(f"Input: {x}")
print(f"Output: {out}")
print(f"Gradient: {grad}")

# Test 2: Multiple arrays
print("\n=== Test 2: Multiple Arrays ===")
x = nd.array([1.0, 2.0])
y = nd.array([3.0, 4.0])


def multi_func(x, y):
    return nd.sum(x * y)


out, vjp_fn = nd.vjp(multi_func, x, y)
grad_x, grad_y = vjp_fn(nd.ones_like(out))
print(f"Inputs: x={x}, y={y}")
print(f"Output: {out}")
print(f"Gradients: grad_x={grad_x}, grad_y={grad_y}")

# Test 3: Dictionary input
print("\n=== Test 3: Dictionary Input ===")
params = {"weights": nd.array([1.0, 2.0]), "bias": nd.array([0.5])}


def dict_func(params):
    return nd.sum(params["weights"] ** 2) + nd.sum(params["bias"])


out, vjp_fn = nd.vjp(dict_func, params)
grad_dict = vjp_fn(nd.ones_like(out))
print(f"Input params: {params}")
print(f"Output: {out}")
print(f"Gradient dict: {grad_dict}")

# Test 4: Nested structure
print("\n=== Test 4: Nested Structure ===")
nested_params = {
    "layer1": [nd.array([1.0]), nd.array([2.0])],
    "layer2": {"w": nd.array([3.0]), "b": nd.array([4.0])},
}


def nested_func(params):
    l1_sum = nd.sum(params["layer1"][0]) + nd.sum(params["layer1"][1])
    l2_sum = nd.sum(params["layer2"]["w"]) + nd.sum(params["layer2"]["b"])
    return l1_sum + l2_sum


out, vjp_fn = nd.vjp(nested_func, nested_params)
grad_nested = vjp_fn(nd.ones_like(out))
print(f"Input nested params: {nested_params}")
print(f"Output: {out}")
print(f"Gradient nested: {grad_nested}")
