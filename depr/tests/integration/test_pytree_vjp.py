#!/usr/bin/env python3
"""Test script to demonstrate pytree support in vjp function."""

import nabla as nb


def test_pytree_vjp_simple_tensor():
    """Test VJP with simple tensor."""
    print("=== Test 1: Simple Tensor ===")
    x = nb.tensor([1.0, 2.0, 3.0])

    def simple_func(x):
        return nb.sum(x**2)

    out, vjp_fn = nb.vjp(simple_func, x)
    grad = vjp_fn(nb.ones_like(out))
    print(f"Input: {x}")
    print(f"Output: {out}")
    print(f"Gradient: {grad}")

    # Verify gradient is 2*x
    expected_grad = 2.0 * x
    assert nb.allclose(grad, expected_grad), f"Expected {expected_grad}, got {grad}"


def test_pytree_vjp_multiple_tensors():
    """Test VJP with multiple tensors."""
    print("\n=== Test 2: Multiple Tensors ===")
    x = nb.tensor([1.0, 2.0])
    y = nb.tensor([3.0, 4.0])

    def multi_func(x, y):
        return nb.sum(x * y)

    out, vjp_fn = nb.vjp(multi_func, x, y)
    grad_x, grad_y = vjp_fn(nb.ones_like(out))
    print(f"Inputs: x={x}, y={y}")
    print(f"Output: {out}")
    print(f"Gradients: grad_x={grad_x}, grad_y={grad_y}")

    # Verify gradients
    assert nb.allclose(grad_x, y), f"Expected grad_x={y}, got {grad_x}"
    assert nb.allclose(grad_y, x), f"Expected grad_y={x}, got {grad_y}"


def test_pytree_vjp_dict_input():
    """Test VJP with dictionary input."""
    print("\n=== Test 3: Dictionary Input ===")
    params = {"weights": nb.tensor([1.0, 2.0]), "bias": nb.tensor([0.5])}

    def dict_func(params):
        return nb.sum(params["weights"] * 2.0) + nb.sum(params["bias"])

    out, vjp_fn = nb.vjp(dict_func, params)
    grad_dict = vjp_fn(nb.ones_like(out))
    print(f"Input params: {params}")
    print(f"Output: {out}")
    print(f"Gradient dict: {grad_dict}")

    # Verify gradients
    expected_grad_weights = nb.tensor([2.0, 2.0])
    expected_grad_bias = nb.tensor([1.0])
    assert nb.allclose(grad_dict["weights"], expected_grad_weights), (
        f"Expected weights grad={expected_grad_weights}, got {grad_dict['weights']}"
    )
    assert nb.allclose(grad_dict["bias"], expected_grad_bias), (
        f"Expected bias grad={expected_grad_bias}, got {grad_dict['bias']}"
    )


def test_pytree_vjp_nested_structure():
    """Test VJP with nested structure."""
    print("\n=== Test 4: Nested Structure ===")
    nested_params = {
        "layer1": [nb.tensor([1.0]), nb.tensor([2.0])],
        "layer2": {"w": nb.tensor([3.0]), "b": nb.tensor([4.0])},
    }

    def nested_func(params):
        layer1_sum = nb.sum(params["layer1"][0]) + nb.sum(params["layer1"][1])
        layer2_sum = nb.sum(params["layer2"]["w"]) + nb.sum(params["layer2"]["b"])
        return layer1_sum + layer2_sum

    out, vjp_fn = nb.vjp(nested_func, nested_params)
    grad_nested = vjp_fn(nb.ones_like(out))
    print(f"Input nested params: {nested_params}")
    print(f"Output: {out}")
    print(f"Gradient nested: {grad_nested}")

    # Verify output value
    expected_out = 1.0 + 2.0 + 3.0 + 4.0
    assert abs(float(out.to_numpy()) - expected_out) < 1e-6, (
        f"Expected output {expected_out}, got {out}"
    )

    # Verify all gradients are 1.0 (since we're just summing everything)
    assert nb.allclose(grad_nested["layer1"][0], nb.tensor([1.0])), (
        f"Expected layer1[0] grad=1.0, got {grad_nested['layer1'][0]}"
    )
    assert nb.allclose(grad_nested["layer1"][1], nb.tensor([1.0])), (
        f"Expected layer1[1] grad=1.0, got {grad_nested['layer1'][1]}"
    )
    assert nb.allclose(grad_nested["layer2"]["w"], nb.tensor([1.0])), (
        f"Expected layer2.w grad=1.0, got {grad_nested['layer2']['w']}"
    )
    assert nb.allclose(grad_nested["layer2"]["b"], nb.tensor([1.0])), (
        f"Expected layer2.b grad=1.0, got {grad_nested['layer2']['b']}"
    )


if __name__ == "__main__":
    """Run all pytree VJP tests when executed as script."""
    print("=== Pytree VJP Tests ===")
    test_pytree_vjp_simple_tensor()
    test_pytree_vjp_multiple_tensors()
    test_pytree_vjp_dict_input()
    test_pytree_vjp_nested_structure()
    print("\n🎉 All pytree VJP tests passed!")
