import numpy as np
import pytest

torch = pytest.importorskip("torch")

import nabla as nb
from nabla.transforms import backward as nabla_backward
from nabla.transforms import autograd


def _reset_grads(*tensors) -> None:
    for arr in tensors:
        arr.grad = None


def _assert_close(nabla_grad, torch_grad, *, rtol=1e-6, atol=1e-6) -> None:
    np.testing.assert_allclose(
        nabla_grad.to_numpy(), torch_grad.detach().cpu().numpy(), rtol=rtol, atol=atol
    )


def test_backward_scalar_matches_pytorch():
    nabla_a = nb.tensor([0.1, -0.2, 0.3])
    nabla_b = nb.tensor([1.5, -0.7, 2.2])
    nabla_a.requires_grad_()
    nabla_b.requires_grad_()

    nabla_out = nb.sum(nb.sin(nabla_a) * nabla_b + nb.exp(nabla_b))
    nabla_out.backward()

    torch_a = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
    torch_b = torch.tensor([1.5, -0.7, 2.2], requires_grad=True)
    torch_out = torch.sum(torch.sin(torch_a) * torch_b + torch.exp(torch_b))
    torch_out.backward()

    _assert_close(nabla_a.grad, torch_a.grad)
    _assert_close(nabla_b.grad, torch_b.grad)

    _reset_grads(nabla_a, nabla_b)
    if torch_a.grad is not None:
        torch_a.grad.zero_()
    if torch_b.grad is not None:
        torch_b.grad.zero_()


def test_backward_vector_with_cotangent_matches_pytorch():
    nabla_x = nb.tensor([[1.0, 2.0], [-1.0, 0.5]])
    nabla_y = nb.tensor([[0.3, -0.6], [1.2, 0.4]])
    nabla_x.requires_grad_()
    nabla_y.requires_grad_()

    nabla_out = nb.sin(nabla_x) * nb.cos(nabla_y) + nb.matmul(nabla_x, nabla_y)
    cotangent = nb.ones_like(nabla_out) * 0.25
    nabla_backward(nabla_out, cotangent)

    torch_x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], requires_grad=True)
    torch_y = torch.tensor([[0.3, -0.6], [1.2, 0.4]], requires_grad=True)
    torch_out = torch.sin(torch_x) * torch.cos(torch_y) + torch.matmul(torch_x, torch_y)
    torch.autograd.backward(torch_out, torch.full_like(torch_out, 0.25))

    _assert_close(nabla_x.grad, torch_x.grad)
    _assert_close(nabla_y.grad, torch_y.grad)

    _reset_grads(nabla_x, nabla_y)
    if torch_x.grad is not None:
        torch_x.grad.zero_()
    if torch_y.grad is not None:
        torch_y.grad.zero_()


def test_backward_accumulates_like_pytorch():
    """Test that gradients accumulate across multiple backward passes with retain_graph."""
    nabla_a = nb.tensor([0.4, -0.8])
    nabla_b = nb.tensor([-1.2, 0.9])
    nabla_a.requires_grad_()
    nabla_b.requires_grad_()

    nabla_out = nb.sum(nb.pow(nabla_a, 2) * nb.cos(nabla_b))
    nabla_out.backward(retain_graph=True)
    nabla_out.backward()

    torch_a = torch.tensor([0.4, -0.8], requires_grad=True)
    torch_b = torch.tensor([-1.2, 0.9], requires_grad=True)
    torch_out = torch.sum(torch.pow(torch_a, 2) * torch.cos(torch_b))
    torch_out.backward(retain_graph=True)
    torch_out.backward()

    _assert_close(nabla_a.grad, torch_a.grad)
    _assert_close(nabla_b.grad, torch_b.grad)


def test_autograd_single_input_scalar_output():
    """Test autograd with single input, scalar output."""
    x = nb.tensor([1.0, 2.0, 3.0])
    x.requires_grad_(True)
    y = nb.sum(x ** 2)
    
    grad_x = autograd(y, x)
    expected = [2.0, 4.0, 6.0]  # 2*x
    
    np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=1e-6, atol=1e-6)


def test_autograd_multiple_inputs():
    """Test autograd with multiple inputs."""
    a = nb.tensor([1.0, 2.0])
    b = nb.tensor([3.0, 4.0])
    a.requires_grad_(True)
    b.requires_grad_(True)
    z = nb.sum(a * b)
    
    grad_a, grad_b = autograd(z, [a, b])
    
    np.testing.assert_allclose(grad_a.to_numpy(), b.to_numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad_b.to_numpy(), a.to_numpy(), rtol=1e-6, atol=1e-6)


def test_autograd_custom_grad_outputs():
    """Test autograd with custom grad_outputs (cotangents)."""
    x = nb.tensor([1.0, 2.0, 3.0])
    x.requires_grad_(True)
    y = x ** 2
    
    grad_outputs = nb.tensor([0.5, 0.5, 0.5])
    grad_x = autograd(y, x, grad_outputs=grad_outputs)
    expected = [1.0, 2.0, 3.0]  # grad_outputs * 2*x
    
    np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=1e-6, atol=1e-6)


def test_autograd_no_accumulation():
    """Test that autograd doesn't accumulate gradients (unlike backward)."""
    x = nb.tensor([1.0, 2.0])
    x.requires_grad_(True)
    y = nb.sum(x ** 2)
    
    # Call autograd multiple times
    grad1 = autograd(y, x, create_graph=True)  # Keep graph for second call
    grad2 = autograd(y, x)  # This one cleans up
    
    # Both calls should return same values
    np.testing.assert_allclose(grad1.to_numpy(), grad2.to_numpy(), rtol=1e-6, atol=1e-6)
    # x.grad should remain None
    assert x.grad is None


def test_autograd_vs_backward():
    """Test that autograd and backward return the same gradients."""
    # Using autograd
    x1 = nb.tensor([1.0, 2.0, 3.0])
    x1.requires_grad_(True)
    y1 = nb.sum(x1 ** 2)
    grad_from_autograd = autograd(y1, x1)
    
    # Using backward
    x2 = nb.tensor([1.0, 2.0, 3.0])
    x2.requires_grad_(True)
    y2 = nb.sum(x2 ** 2)
    y2.backward()
    grad_from_backward = x2.grad
    
    np.testing.assert_allclose(
        grad_from_autograd.to_numpy(), grad_from_backward.to_numpy(), rtol=1e-6, atol=1e-6
    )


def test_autograd_multiple_outputs():
    """Test autograd with multiple outputs (requires grad_outputs)."""
    x = nb.tensor([1.0, 2.0])
    x.requires_grad_(True)
    y1 = nb.sum(x ** 2)
    y2 = nb.sum(x ** 3)
    
    grad_outputs = [nb.tensor(1.0), nb.tensor(1.0)]
    grad_x = autograd([y1, y2], x, grad_outputs=grad_outputs)
    expected = [2.0 + 3.0, 4.0 + 12.0]  # [2*1 + 3*1, 2*2 + 3*4]
    
    np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=1e-6, atol=1e-6)


def test_autograd_intermediate_nodes():
    """Test autograd with intermediate nodes using create_graph."""
    a = nb.tensor([1.0, 2.0])
    a.requires_grad_(True)
    b = a * 2.0
    b.requires_grad_(True)
    c = nb.sum(b ** 2)
    
    # Get gradient w.r.t intermediate node b
    grad_b = autograd(c, b, create_graph=True)
    expected_b = (2 * b).to_numpy()
    np.testing.assert_allclose(grad_b.to_numpy(), expected_b, rtol=1e-6, atol=1e-6)
    
    # Get gradient w.r.t original input a
    grad_a = autograd(c, a)
    expected_a = (4 * b).to_numpy()
    np.testing.assert_allclose(grad_a.to_numpy(), expected_a, rtol=1e-6, atol=1e-6)


def test_autograd_higher_order_derivatives():
    """Test autograd with higher-order derivatives using create_graph=True."""
    x = nb.tensor([1.0, 2.0, 3.0])
    x.requires_grad_(True)
    y = nb.sum(x ** 3)  # y = x1^3 + x2^3 + x3^3
    
    # First derivative: dy/dx = 3*x^2
    grad1 = autograd(y, x, create_graph=True)
    expected1 = (3 * x ** 2).to_numpy()
    np.testing.assert_allclose(grad1.to_numpy(), expected1, rtol=1e-6, atol=1e-6)
    
    # Second derivative for each element
    grad2_elem0 = autograd(grad1[0], x, create_graph=True)
    grad2_elem1 = autograd(grad1[1], x, create_graph=True)
    grad2_elem2 = autograd(grad1[2], x)
    
    # Check diagonal elements (non-zero parts)
    assert abs(grad2_elem0.to_numpy()[0] - 6*x.to_numpy()[0]) < 1e-6
    assert abs(grad2_elem1.to_numpy()[1] - 6*x.to_numpy()[1]) < 1e-6
    assert abs(grad2_elem2.to_numpy()[2] - 6*x.to_numpy()[2]) < 1e-6


def test_autograd_third_order():
    """Test third and fourth order derivatives."""
    x = nb.tensor([2.0])
    x.requires_grad_(True)
    y = x ** 4  # y = x^4
    
    # dy/dx = 4*x^3
    grad1 = autograd(y, x, create_graph=True)
    expected1 = 4 * x.to_numpy()[0]**3
    assert abs(grad1.to_numpy()[0] - expected1) < 1e-6
    
    # d²y/dx² = 12*x^2
    grad2 = autograd(grad1, x, create_graph=True)
    expected2 = 12 * x.to_numpy()[0]**2
    assert abs(grad2.to_numpy()[0] - expected2) < 1e-6
    
    # d³y/dx³ = 24*x
    grad3 = autograd(grad2, x, create_graph=True)
    expected3 = 24 * x.to_numpy()[0]
    assert abs(grad3.to_numpy()[0] - expected3) < 1e-6
    
    # d⁴y/dx⁴ = 24 (constant)
    grad4 = autograd(grad3, x)
    expected4 = 24
    assert abs(grad4.to_numpy()[0] - expected4) < 1e-6


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_backward_non_scalar_without_grad_raises_error():
    """Test that backward on non-scalar output without grad argument raises error."""
    x = nb.tensor([1.0, 2.0, 3.0])
    x.requires_grad_(True)
    y = x ** 2  # Non-scalar output
    
    with pytest.raises(ValueError, match="grad argument required for non-scalar"):
        y.backward()


def test_backward_with_invalid_grad_type_raises_error():
    """Test that backward with non-tensor grad raises TypeError."""
    x = nb.tensor([1.0, 2.0])
    x.requires_grad_(True)
    y = nb.sum(x ** 2)
    
    with pytest.raises(TypeError, match="grad must be a Nabla Tensor"):
        y.backward(grad=[1.0, 2.0])  # Pass list instead of tensor


def test_autograd_invalid_inputs_type_raises_error():
    """Test that autograd with invalid inputs type raises error."""
    x = nb.tensor([1.0, 2.0])
    x.requires_grad_(True)
    y = nb.sum(x ** 2)
    
    with pytest.raises(TypeError, match="inputs must be a Tensor or sequence"):
        autograd(y, "invalid_input")


def test_autograd_invalid_outputs_type_raises_error():
    """Test that autograd with invalid outputs type raises error."""
    x = nb.tensor([1.0, 2.0])
    x.requires_grad_(True)
    
    with pytest.raises(TypeError, match="outputs must be a Tensor or sequence"):
        autograd("invalid_output", x)


def test_autograd_mismatched_grad_outputs_length_raises_error():
    """Test that autograd raises error when grad_outputs length doesn't match outputs."""
    x = nb.tensor([1.0, 2.0])
    x.requires_grad_(True)
    y1 = nb.sum(x ** 2)
    y2 = nb.sum(x ** 3)
    
    # Provide wrong number of grad_outputs
    grad_outputs = [nb.tensor(1.0)]  # Only 1 but we have 2 outputs
    
    with pytest.raises(ValueError, match="Number of grad_outputs .* must match"):
        autograd([y1, y2], x, grad_outputs=grad_outputs)


# ============================================================================
# EDGE CASES
# ============================================================================


def test_mixed_requires_grad_tensors():
    """Test gradients with mixed requires_grad settings."""
    a = nb.tensor([1.0, 2.0])
    b = nb.tensor([3.0, 4.0])
    a.requires_grad_(True)
    # b does NOT require grad
    
    z = nb.sum(a * b)
    grad_a = autograd(z, a)
    
    # Should still compute gradient for a
    np.testing.assert_allclose(grad_a.to_numpy(), b.to_numpy(), rtol=1e-6, atol=1e-6)
    # b should not have gradient
    assert b.grad is None


def test_backward_with_zero_grad():
    """Test backward with zero gradient."""
    x = nb.tensor([1.0, 2.0, 3.0])
    x.requires_grad_(True)
    y = nb.sum(x ** 2)
    
    # Pass zero gradient explicitly
    zero_grad = nb.zeros_like(y)
    y.backward(grad=zero_grad)
    
    # Gradient should be zero
    np.testing.assert_allclose(x.grad.to_numpy(), [0.0, 0.0, 0.0], rtol=1e-6, atol=1e-6)


def test_multiple_backward_different_outputs():
    """Test multiple backward passes on different outputs from same inputs."""
    x = nb.tensor([1.0, 2.0])
    x.requires_grad_(True)
    
    # Compute both outputs from the traced input
    y1 = nb.sum(x ** 2)
    y2 = nb.sum(x ** 3)
    
    # First backward
    y1.backward(retain_graph=True)
    grad_after_first = x.grad.to_numpy().copy()
    
    # Second backward (should accumulate because same computation graph)
    y2.backward()
    grad_after_second = x.grad.to_numpy()
    
    # Gradient should accumulate: grad(x^2) + grad(x^3) = 2*x + 3*x^2
    expected = 2 * x.to_numpy() + 3 * x.to_numpy()**2
    np.testing.assert_allclose(grad_after_second, expected, rtol=1e-6, atol=1e-6)


def test_requires_grad_returns_self():
    """Test that requires_grad_() returns self for method chaining."""
    x = nb.tensor([1.0, 2.0, 3.0])
    result = x.requires_grad_(True)
    
    # Should return the same tensor object
    assert result is x
    assert x.requires_grad is True
    assert x.traced is True


def test_backward_clears_computation_graph():
    """Test that backward without retain_graph clears the computation graph."""
    x = nb.tensor([1.0, 2.0])
    x.requires_grad_(True)
    y = nb.sum(x ** 2)
    
    # First backward should work
    y.backward(retain_graph=False)
    assert x.grad is not None
    first_grad = x.grad.to_numpy().copy()
    
    # Reset gradient for clean test
    x.grad = None
    
    # Need to re-enable tracing for new computation
    x.requires_grad_(True)
    y2 = nb.sum(x ** 2)
    y2.backward()
    
    # Should get same gradient values
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.to_numpy(), first_grad, rtol=1e-6, atol=1e-6)


# ============================================================================
# PYTORCH PARITY TESTS
# ============================================================================


def test_grad_accumulation_matches_pytorch():
    """Test that gradient accumulation matches PyTorch exactly."""
    # Nabla
    nabla_x = nb.tensor([1.0, 2.0, 3.0])
    nabla_x.requires_grad_(True)
    
    nabla_y1 = nb.sum(nabla_x ** 2)
    nabla_y1.backward(retain_graph=True)
    
    nabla_y2 = nb.sum(nabla_x ** 3)
    nabla_y2.backward()
    
    # PyTorch
    torch_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    torch_y1 = torch.sum(torch_x ** 2)
    torch_y1.backward(retain_graph=True)
    
    torch_y2 = torch.sum(torch_x ** 3)
    torch_y2.backward()
    
    _assert_close(nabla_x.grad, torch_x.grad)


def test_complex_computation_graph_matches_pytorch():
    """Test complex computation graph with branching matches PyTorch."""
    # Nabla
    nabla_a = nb.tensor([1.0, 2.0])
    nabla_b = nb.tensor([3.0, 4.0])
    nabla_a.requires_grad_(True)
    nabla_b.requires_grad_(True)
    
    nabla_c = nabla_a * nabla_b
    nabla_d = nabla_a + nabla_b
    nabla_e = nb.sum(nabla_c * nabla_d)
    nabla_e.backward()
    
    # PyTorch
    torch_a = torch.tensor([1.0, 2.0], requires_grad=True)
    torch_b = torch.tensor([3.0, 4.0], requires_grad=True)
    
    torch_c = torch_a * torch_b
    torch_d = torch_a + torch_b
    torch_e = torch.sum(torch_c * torch_d)
    torch_e.backward()
    
    _assert_close(nabla_a.grad, torch_a.grad)
    _assert_close(nabla_b.grad, torch_b.grad)


def test_nested_operations_matches_pytorch():
    """Test nested operations match PyTorch."""
    # Nabla
    nabla_x = nb.tensor([0.5, 1.5, 2.5])
    nabla_x.requires_grad_(True)
    nabla_y = nb.sum(nb.sin(nb.exp(nabla_x)) * nb.cos(nabla_x ** 2))
    nabla_y.backward()
    
    # PyTorch
    torch_x = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
    torch_y = torch.sum(torch.sin(torch.exp(torch_x)) * torch.cos(torch_x ** 2))
    torch_y.backward()
    
    _assert_close(nabla_x.grad, torch_x.grad)
