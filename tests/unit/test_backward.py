import numpy as np
import pytest

torch = pytest.importorskip("torch")

import nabla as nb
from nabla.transforms import backward as nabla_backward


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
    nabla_a.requires_grad()
    nabla_b.requires_grad()

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
    nabla_x.requires_grad()
    nabla_y.requires_grad()

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
    nabla_a.requires_grad()
    nabla_b.requires_grad()

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
