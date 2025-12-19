"""
Comprehensive test suite for fold/unfold operations with full Jacobian computation.
Tests both jacfwd (forward-mode) and jacrev (reverse-mode) automatic differentiation.
All Nabla functions are JIT compiled for performance and correctness validation.
"""

import nabla as nb
import torch
import torch.nn.functional as F
import numpy as np


def test_fold_jacrev():
    """Test fold operation full Jacobian using jacrev (reverse-mode AD)."""
    print("\n" + "=" * 50)
    print("TEST 1: Fold Jacobian (jacrev - reverse mode)")
    print("=" * 50)

    # Smaller input for full Jacobian computation
    batch_size, channels, out_h, out_w = 1, 2, 3, 3
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 1, 1

    # Calculate col shape
    col_h = (out_h - kernel_h) // stride_h + 1
    col_w = (out_w - kernel_w) // stride_w + 1
    col_shape = (batch_size, channels * kernel_h * kernel_w, col_h * col_w)

    # Create input
    np_input = np.random.randn(*col_shape).astype(np.float32)

    # Nabla - JIT compiled jacrev
    def nb_fold_fn(x):
        return nb.sum(nb.fold(
            x,
            output_size=(out_h, out_w),
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
        ))

    nb_input = nb.tensor(np_input)
    nb_jacrev_fn = nb.jit(nb.grad(nb_fold_fn))
    nb_jac = nb_jacrev_fn(nb_input)

    print(nb_jac)

    # # PyTorch - compute full Jacobian manually
    # torch_input = torch.tensor(np_input, requires_grad=True)
    # torch_out = F.fold(
    #     torch_input,
    #     output_size=(out_h, out_w),
    #     kernel_size=(kernel_h, kernel_w),
    #     stride=(stride_h, stride_w),
    # )
    
    # # Compute Jacobian row by row
    # output_size = torch_out.numel()
    # input_size = torch_input.numel()
    # torch_jac = torch.zeros(output_size, input_size)
    
    # for i in range(output_size):
    #     if torch_input.grad is not None:
    #         torch_input.grad.zero_()
    #     torch_out_flat = torch_out.reshape(-1)
    #     torch_out_flat[i].backward(retain_graph=True)
    #     torch_jac[i] = torch_input.grad.reshape(-1)
    
    # torch_jac = torch_jac.reshape(torch_out.shape + torch_input.shape)

    # # Compare Jacobians
    # nb_result = nb_jac.numpy()
    # torch_result = torch_jac.numpy()

    # print(f"Nabla Jacobian shape: {nb_result.shape}")
    # print(f"PyTorch Jacobian shape: {torch_result.shape}")
    # print(f"Max difference: {np.max(np.abs(nb_result - torch_result))}")
    # match = np.allclose(nb_result, torch_result, rtol=1e-5, atol=1e-5)
    # print(f"Match: {match}")

    # assert match, "Fold jacrev Jacobian mismatch!"
    # print("✓ Fold jacrev test PASSED")


def test_fold_jacfwd():
    """Test fold operation full Jacobian using jacfwd (forward-mode AD)."""
    print("\n" + "=" * 50)
    print("TEST 2: Fold Jacobian (jacfwd - forward mode)")
    print("=" * 50)

    # Smaller input for full Jacobian computation
    batch_size, channels, out_h, out_w = 1, 2, 3, 3
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 1, 1

    # Calculate col shape
    col_h = (out_h - kernel_h) // stride_h + 1
    col_w = (out_w - kernel_w) // stride_w + 1
    col_shape = (batch_size, channels * kernel_h * kernel_w, col_h * col_w)

    # Create input
    np_input = np.random.randn(*col_shape).astype(np.float32)

    # Nabla - JIT compiled jacfwd
    def nb_fold_fn(x):
        return nb.fold(
            x,
            output_size=(out_h, out_w),
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
        )

    nb_input = nb.tensor(np_input)
    nb_jacfwd_fn = nb.jit(nb.jacfwd(nb_fold_fn))
    nb_jac = nb_jacfwd_fn(nb_input)

    # PyTorch - compute full Jacobian manually
    torch_input = torch.tensor(np_input, requires_grad=True)
    torch_out = F.fold(
        torch_input,
        output_size=(out_h, out_w),
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
    )
    
    # Compute Jacobian row by row
    output_size = torch_out.numel()
    input_size = torch_input.numel()
    torch_jac = torch.zeros(output_size, input_size)
    
    for i in range(output_size):
        if torch_input.grad is not None:
            torch_input.grad.zero_()
        torch_out_flat = torch_out.reshape(-1)
        torch_out_flat[i].backward(retain_graph=True)
        torch_jac[i] = torch_input.grad.reshape(-1)
    
    torch_jac = torch_jac.reshape(torch_out.shape + torch_input.shape)

    # Compare Jacobians
    nb_result = nb_jac.numpy()
    torch_result = torch_jac.numpy()

    print(f"Nabla Jacobian shape: {nb_result.shape}")
    print(f"PyTorch Jacobian shape: {torch_result.shape}")
    print(f"Max difference: {np.max(np.abs(nb_result - torch_result))}")
    match = np.allclose(nb_result, torch_result, rtol=1e-5, atol=1e-5)
    print(f"Match: {match}")

    assert match, "Fold jacfwd Jacobian mismatch!"
    print("✓ Fold jacfwd test PASSED")


def test_unfold_jacrev():
    """Test unfold operation full Jacobian using jacrev (reverse-mode AD)."""
    print("\n" + "=" * 50)
    print("TEST 3: Unfold Jacobian (jacrev - reverse mode)")
    print("=" * 50)

    # Smaller input for full Jacobian computation
    batch_size, channels, h, w = 1, 2, 4, 4
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 1, 1

    np_input = np.random.randn(batch_size, channels, h, w).astype(np.float32)

    # Nabla - JIT compiled jacrev
    def nb_unfold_fn(x):
        return nb.unfold(
            x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)
        )

    nb_input = nb.tensor(np_input)
    nb_jacrev_fn = nb.jit(nb.jacrev(nb_unfold_fn))
    nb_jac = nb_jacrev_fn(nb_input)

    # PyTorch - compute full Jacobian manually
    torch_input = torch.tensor(np_input, requires_grad=True)
    torch_out = F.unfold(
        torch_input, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)
    )
    
    # Compute Jacobian row by row
    output_size = torch_out.numel()
    input_size = torch_input.numel()
    torch_jac = torch.zeros(output_size, input_size)
    
    for i in range(output_size):
        if torch_input.grad is not None:
            torch_input.grad.zero_()
        torch_out_flat = torch_out.reshape(-1)
        torch_out_flat[i].backward(retain_graph=True)
        torch_jac[i] = torch_input.grad.reshape(-1)
    
    torch_jac = torch_jac.reshape(torch_out.shape + torch_input.shape)

    # Compare Jacobians
    nb_result = nb_jac.numpy()
    torch_result = torch_jac.numpy()

    print(f"Nabla Jacobian shape: {nb_result.shape}")
    print(f"PyTorch Jacobian shape: {torch_result.shape}")
    print(f"Max difference: {np.max(np.abs(nb_result - torch_result))}")
    match = np.allclose(nb_result, torch_result, rtol=1e-5, atol=1e-5)
    print(f"Match: {match}")

    assert match, "Unfold jacrev Jacobian mismatch!"
    print("✓ Unfold jacrev test PASSED")


def test_unfold_jacfwd():
    """Test unfold operation full Jacobian using jacfwd (forward-mode AD)."""
    print("\n" + "=" * 50)
    print("TEST 4: Unfold Jacobian (jacfwd - forward mode)")
    print("=" * 50)

    # Smaller input for full Jacobian computation
    batch_size, channels, h, w = 1, 2, 4, 4
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 1, 1

    np_input = np.random.randn(batch_size, channels, h, w).astype(np.float32)

    # Nabla - JIT compiled jacfwd
    def nb_unfold_fn(x):
        return nb.unfold(
            x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)
        )

    nb_input = nb.tensor(np_input)
    nb_jacfwd_fn = nb.jit(nb.jacfwd(nb_unfold_fn))
    nb_jac = nb_jacfwd_fn(nb_input)

    # PyTorch - compute full Jacobian manually
    torch_input = torch.tensor(np_input, requires_grad=True)
    torch_out = F.unfold(
        torch_input, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)
    )
    
    # Compute Jacobian row by row
    output_size = torch_out.numel()
    input_size = torch_input.numel()
    torch_jac = torch.zeros(output_size, input_size)
    
    for i in range(output_size):
        if torch_input.grad is not None:
            torch_input.grad.zero_()
        torch_out_flat = torch_out.reshape(-1)
        torch_out_flat[i].backward(retain_graph=True)
        torch_jac[i] = torch_input.grad.reshape(-1)
    
    torch_jac = torch_jac.reshape(torch_out.shape + torch_input.shape)

    # Compare Jacobians
    nb_result = nb_jac.numpy()
    torch_result = torch_jac.numpy()

    print(f"Nabla Jacobian shape: {nb_result.shape}")
    print(f"PyTorch Jacobian shape: {torch_result.shape}")
    print(f"Max difference: {np.max(np.abs(nb_result - torch_result))}")
    match = np.allclose(nb_result, torch_result, rtol=1e-5, atol=1e-5)
    print(f"Match: {match}")

    assert match, "Unfold jacfwd Jacobian mismatch!"
    print("✓ Unfold jacfwd test PASSED")


def test_fold_with_strides():
    """Test fold with different stride configurations."""
    print("\n" + "=" * 50)
    print("TEST 5: Fold with stride=2 (jacrev)")
    print("=" * 50)

    batch_size, channels, out_h, out_w = 1, 1, 5, 5
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 2, 2

    col_h = (out_h - kernel_h) // stride_h + 1
    col_w = (out_w - kernel_w) // stride_w + 1
    col_shape = (batch_size, channels * kernel_h * kernel_w, col_h * col_w)

    np_input = np.random.randn(*col_shape).astype(np.float32)

    # Nabla - JIT compiled jacrev
    def nb_fold_fn(x):
        return nb.fold(
            x,
            output_size=(out_h, out_w),
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
        )

    nb_input = nb.tensor(np_input)
    nb_jacrev_fn = nb.jit(nb.jacrev(nb_fold_fn))
    nb_jac = nb_jacrev_fn(nb_input)

    # PyTorch
    torch_input = torch.tensor(np_input, requires_grad=True)
    torch_out = F.fold(
        torch_input,
        output_size=(out_h, out_w),
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
    )
    
    output_size = torch_out.numel()
    input_size = torch_input.numel()
    torch_jac = torch.zeros(output_size, input_size)
    
    for i in range(output_size):
        if torch_input.grad is not None:
            torch_input.grad.zero_()
        torch_out_flat = torch_out.reshape(-1)
        torch_out_flat[i].backward(retain_graph=True)
        torch_jac[i] = torch_input.grad.reshape(-1)
    
    torch_jac = torch_jac.reshape(torch_out.shape + torch_input.shape)

    nb_result = nb_jac.numpy()
    torch_result = torch_jac.numpy()

    print(f"Nabla Jacobian shape: {nb_result.shape}")
    print(f"PyTorch Jacobian shape: {torch_result.shape}")
    print(f"Max difference: {np.max(np.abs(nb_result - torch_result))}")
    match = np.allclose(nb_result, torch_result, rtol=1e-5, atol=1e-5)
    print(f"Match: {match}")

    assert match, "Fold with stride=2 Jacobian mismatch!"
    print("✓ Fold stride=2 test PASSED")


def test_unfold_with_strides():
    """Test unfold with different stride configurations."""
    print("\n" + "=" * 50)
    print("TEST 6: Unfold with stride=2 (jacrev)")
    print("=" * 50)

    batch_size, channels, h, w = 1, 1, 5, 5
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 2, 2

    np_input = np.random.randn(batch_size, channels, h, w).astype(np.float32)

    # Nabla - JIT compiled jacrev
    def nb_unfold_fn(x):
        return nb.unfold(
            x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)
        )

    nb_input = nb.tensor(np_input)
    nb_jacrev_fn = nb.jit(nb.jacrev(nb_unfold_fn))
    nb_jac = nb_jacrev_fn(nb_input)

    # PyTorch
    torch_input = torch.tensor(np_input, requires_grad=True)
    torch_out = F.unfold(
        torch_input, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)
    )
    
    output_size = torch_out.numel()
    input_size = torch_input.numel()
    torch_jac = torch.zeros(output_size, input_size)
    
    for i in range(output_size):
        if torch_input.grad is not None:
            torch_input.grad.zero_()
        torch_out_flat = torch_out.reshape(-1)
        torch_out_flat[i].backward(retain_graph=True)
        torch_jac[i] = torch_input.grad.reshape(-1)
    
    torch_jac = torch_jac.reshape(torch_out.shape + torch_input.shape)

    nb_result = nb_jac.numpy()
    torch_result = torch_jac.numpy()

    print(f"Nabla Jacobian shape: {nb_result.shape}")
    print(f"PyTorch Jacobian shape: {torch_result.shape}")
    print(f"Max difference: {np.max(np.abs(nb_result - torch_result))}")
    match = np.allclose(nb_result, torch_result, rtol=1e-5, atol=1e-5)
    print(f"Match: {match}")

    assert match, "Unfold with stride=2 Jacobian mismatch!"
    print("✓ Unfold stride=2 test PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("COMPREHENSIVE FOLD/UNFOLD JACOBIAN TESTS")
    print("All Nabla functions are JIT compiled")
    print("=" * 50)

    test_fold_jacrev()
    # test_fold_jacfwd()
    # test_unfold_jacrev()
    # test_unfold_jacfwd()
    # test_fold_with_strides()
    # test_unfold_with_strides()

    # print("\n" + "=" * 50)
    # print("ALL TESTS PASSED! ✓")
    # print("=" * 50)
