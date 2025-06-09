"""
Test VJP structure behavior - ensuring gradients return in their natural structure.

This module tests the VJP (Vector-Jacobian Product) API to ensure that:
1. Single argument functions return gradients directly (not in tuples)
2. Multiple argument functions return gradients in tuples
3. PyTree structures are preserved in gradient outputs
4. Edge cases with various input/output structures work correctly
"""

import pytest
from jax import tree_util

import nabla as nb


def test_single_arg_returns_direct_gradient():
    """Test that VJP with single argument returns gradient directly, not in tuple."""

    def f(x):
        return nb.sum(x**2)

    x = nb.array([1.0, 2.0, 3.0])
    v = nb.array(1.0)

    out, vjp_fn = nb.vjp(f, x)
    grad = vjp_fn(v)

    # Should return gradient directly, not wrapped in tuple
    assert isinstance(grad, nb.Array)
    expected = 2.0 * x  # derivative of x^2 is 2x
    assert nb.allclose(grad, expected)


def test_multiple_args_return_tuple():
    """Test that VJP with multiple arguments returns gradients in tuple."""

    def f(x, y):
        return nb.sum(x * y)

    x = nb.array([1.0, 2.0])
    y = nb.array([3.0, 4.0])
    v = nb.array(1.0)

    out, vjp_fn = nb.vjp(f, x, y)
    grads = vjp_fn(v)

    # Should return tuple of gradients
    assert isinstance(grads, tuple)
    assert len(grads) == 2

    grad_x, grad_y = grads
    assert nb.allclose(grad_x, y)  # df/dx = y
    assert nb.allclose(grad_y, x)  # df/dy = x


def test_pytree_structure_preservation():
    """Test that PyTree structures are preserved in VJP gradients."""

    def f(tree):
        x, y_dict = tree
        return nb.sum(x) + nb.sum(y_dict["a"]) + nb.sum(y_dict["b"])

    x = nb.array([1.0, 2.0])
    y_dict = {"a": nb.array([3.0]), "b": nb.array([4.0, 5.0])}
    pytree = (x, y_dict)
    v = nb.array(1.0)

    out, vjp_fn = nb.vjp(f, pytree)
    grad_tree = vjp_fn(v)

    # Should preserve the original pytree structure
    assert isinstance(grad_tree, tuple)
    assert len(grad_tree) == 2

    grad_x, grad_y_dict = grad_tree
    assert isinstance(grad_x, nb.Array)
    assert isinstance(grad_y_dict, dict)
    assert "a" in grad_y_dict and "b" in grad_y_dict

    # Check gradient values
    assert nb.allclose(grad_x, nb.ones_like(x))
    assert nb.allclose(grad_y_dict["a"], nb.ones_like(y_dict["a"]))
    assert nb.allclose(grad_y_dict["b"], nb.ones_like(y_dict["b"]))


def test_nested_pytree_structure():
    """Test VJP with deeply nested PyTree structures."""

    def f(nested):
        outer_dict, inner_tuple = nested
        x = outer_dict["data"]
        y, z = inner_tuple
        return nb.sum(x**2) + nb.sum(y * z)

    outer_dict = {"data": nb.array([1.0, 2.0, 3.0])}
    inner_tuple = (nb.array([2.0, 3.0]), nb.array([4.0, 5.0]))
    nested = (outer_dict, inner_tuple)
    v = nb.array(1.0)

    out, vjp_fn = nb.vjp(f, nested)
    grad_nested = vjp_fn(v)

    # Verify structure preservation
    assert isinstance(grad_nested, tuple)
    assert len(grad_nested) == 2

    grad_outer_dict, grad_inner_tuple = grad_nested
    assert isinstance(grad_outer_dict, dict)
    assert "data" in grad_outer_dict
    assert isinstance(grad_inner_tuple, tuple)
    assert len(grad_inner_tuple) == 2

    # Check gradient values
    expected_x_grad = 2.0 * outer_dict["data"]  # d/dx of x^2
    assert nb.allclose(grad_outer_dict["data"], expected_x_grad)

    y, z = inner_tuple
    assert nb.allclose(grad_inner_tuple[0], z)  # d/dy of y*z = z
    assert nb.allclose(grad_inner_tuple[1], y)  # d/dz of y*z = y


def test_scalar_output_vector_input():
    """Test VJP structure with scalar output and vector input."""

    def f(x):
        return nb.sum(x * x)  # Use available nabla operations

    x = nb.array([1.0, 2.0, 3.0, 4.0])
    v = nb.array(1.0)  # scalar cotangent

    out, vjp_fn = nb.vjp(f, x)
    grad = vjp_fn(v)

    # Single input should return gradient directly
    assert isinstance(grad, nb.Array)
    assert grad.shape == x.shape
    assert nb.allclose(grad, 2.0 * x)


def test_vector_output_vector_input():
    """Test VJP structure with vector output and vector input."""

    def f(x):
        return x**2  # element-wise squaring

    x = nb.array([1.0, 2.0, 3.0])
    v = nb.array([1.0, 1.0, 1.0])  # vector cotangent

    out, vjp_fn = nb.vjp(f, x)
    grad = vjp_fn(v)

    # Single input should return gradient directly
    assert isinstance(grad, nb.Array)
    assert grad.shape == x.shape
    assert nb.allclose(grad, 2.0 * x)


def test_mixed_structure_consistency():
    """Test that VJP structure behavior is consistent across different scenarios."""

    # Test 1: Single array input
    def f1(x):
        return nb.sum(x)

    x1 = nb.array([1.0, 2.0])
    v1 = nb.array(1.0)

    out1, vjp_fn1 = nb.vjp(f1, x1)
    grad1 = vjp_fn1(v1)
    assert isinstance(grad1, nb.Array)  # Direct return, not tuple

    # Test 2: Two array inputs
    def f2(x, y):
        return nb.sum(x + y)

    x2, y2 = nb.array([1.0, 2.0]), nb.array([3.0, 4.0])
    v2 = nb.array(1.0)

    out2, vjp_fn2 = nb.vjp(f2, x2, y2)
    grads2 = vjp_fn2(v2)
    assert isinstance(grads2, tuple)  # Tuple return for multiple inputs
    assert len(grads2) == 2

    # Test 3: Single dict input (PyTree)
    def f3(d):
        return nb.sum(d["a"]) + nb.sum(d["b"])

    d3 = {"a": nb.array([1.0]), "b": nb.array([2.0, 3.0])}
    v3 = nb.array(1.0)

    out3, vjp_fn3 = nb.vjp(f3, d3)
    grad3 = vjp_fn3(v3)
    assert isinstance(grad3, dict)  # Direct return preserving structure


def test_empty_pytree_handling():
    """Test VJP with edge cases like empty structures."""

    def f(x):
        # Function that doesn't use all inputs
        return nb.sum(x["used"])

    x = {"used": nb.array([1.0, 2.0]), "unused": nb.array([3.0, 4.0])}
    v = nb.array(1.0)

    out, vjp_fn = nb.vjp(f, x)
    grad = vjp_fn(v)

    # Should preserve structure even for unused parts
    assert isinstance(grad, dict)
    assert "used" in grad and "unused" in grad
    assert nb.allclose(grad["used"], nb.ones((2,)))
    assert nb.allclose(grad["unused"], nb.zeros((2,)))  # unused should be zero


@pytest.mark.parametrize(
    "input_structure",
    [
        nb.array([1.0, 2.0]),  # Simple array
        {"a": nb.array([1.0])},  # Dict
        [nb.array([1.0]), nb.array([2.0])],  # List
        (nb.array([1.0]), nb.array([2.0])),  # Tuple
    ],
)
def test_parametrized_structure_preservation(input_structure):
    """Parametrized test to ensure structure preservation across different input types."""

    def f(x):
        return nb.sum(tree_util.tree_leaves(x)[0])  # Use first leaf

    v = nb.array(1.0)

    out, vjp_fn = nb.vjp(f, input_structure)
    grad = vjp_fn(v)

    # Gradient should have same structure as input
    assert tree_util.tree_structure(grad) == tree_util.tree_structure(input_structure)


if __name__ == "__main__":
    pytest.main([__file__])
