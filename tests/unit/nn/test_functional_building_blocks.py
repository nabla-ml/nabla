# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for new functional building blocks: dropout, embedding,
scaled_dot_product_attention — validated against PyTorch / JAX."""

from __future__ import annotations

import math

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng

# ===----------------------------------------------------------------------=== #
# Dropout
# ===----------------------------------------------------------------------=== #


class TestFunctionalDropout:
    def test_dropout_eval_is_identity(self):
        rng = make_rng(1)
        x_np = rng.normal(size=(8, 16)).astype(np.float32)
        x = nb.Tensor.from_dlpack(x_np)
        y = nb.nn.functional.dropout(x, p=0.5, training=False)
        nb.testing.assert_allclose(y, x, rtol=0, atol=0)

    def test_dropout_p0_is_identity(self):
        rng = make_rng(2)
        x_np = rng.normal(size=(8, 16)).astype(np.float32)
        x = nb.Tensor.from_dlpack(x_np)
        y = nb.nn.functional.dropout(x, p=0.0, training=True)
        nb.testing.assert_allclose(y, x, rtol=0, atol=0)

    def test_dropout_p1_is_zeros(self):
        rng = make_rng(3)
        x_np = rng.normal(size=(8, 16)).astype(np.float32)
        x = nb.Tensor.from_dlpack(x_np)
        y = nb.nn.functional.dropout(x, p=1.0, training=True)
        expected = np.zeros_like(x_np)
        nb.testing.assert_allclose(y, expected, rtol=0, atol=0)

    def test_dropout_preserves_shape(self):
        x = nb.Tensor.from_dlpack(np.ones((4, 8), dtype=np.float32))
        y = nb.nn.functional.dropout(x, p=0.3, training=True)
        assert tuple(int(d) for d in y.shape) == (4, 8)

    def test_dropout_zeros_some_elements(self):
        """With p=0.5 on a large tensor, roughly half should be zero."""
        x = nb.Tensor.from_dlpack(np.ones((1000, 100), dtype=np.float32))
        y = nb.nn.functional.dropout(x, p=0.5, training=True)
        y_np = y.to_numpy()
        frac_zero = np.mean(y_np == 0.0)
        assert 0.3 < frac_zero < 0.7, f"Expected ~50% zeros, got {frac_zero:.2%}"

    def test_dropout_inverted_scaling(self):
        """Non-zero elements should be scaled by 1/(1-p)."""
        x = nb.Tensor.from_dlpack(np.ones((10000,), dtype=np.float32))
        p = 0.3
        y = nb.nn.functional.dropout(x, p=p, training=True)
        y_np = y.to_numpy()
        nonzero = y_np[y_np != 0.0]
        # Each surviving element should equal 1 / (1 - p)
        expected_val = 1.0 / (1.0 - p)
        np.testing.assert_allclose(nonzero, expected_val, rtol=1e-5)


# ===----------------------------------------------------------------------=== #
# Embedding
# ===----------------------------------------------------------------------=== #


class TestFunctionalEmbedding:
    def test_embedding_forward_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(10)
        weight_np = rng.normal(size=(10, 8)).astype(np.float32)
        indices_np = np.array([0, 3, 7, 1, 5], dtype=np.int64)

        # Nabla
        y_nb = nb.nn.functional.embedding(
            nb.Tensor.from_dlpack(indices_np),
            nb.Tensor.from_dlpack(weight_np),
        )
        # PyTorch reference
        y_pt = torch.nn.functional.embedding(
            torch.from_numpy(indices_np),
            torch.from_numpy(weight_np),
        )
        nb.testing.assert_allclose(y_nb, y_pt, rtol=1e-6, atol=1e-6)

    def test_embedding_shape(self):
        rng = make_rng(11)
        weight = nb.Tensor.from_dlpack(rng.normal(size=(20, 16)).astype(np.float32))
        indices = nb.Tensor.from_dlpack(np.array([1, 2, 3], dtype=np.int64))
        y = nb.nn.functional.embedding(indices, weight)
        assert tuple(int(d) for d in y.shape) == (3, 16)

    def test_embedding_correct_rows(self):
        """Embedding should return the exact rows of the weight matrix."""
        weight_np = np.eye(5, dtype=np.float32)
        indices_np = np.array([0, 2, 4], dtype=np.int64)
        y = nb.nn.functional.embedding(
            nb.Tensor.from_dlpack(indices_np),
            nb.Tensor.from_dlpack(weight_np),
        )
        expected = weight_np[[0, 2, 4]]
        nb.testing.assert_allclose(y, expected, rtol=0, atol=0)


# ===----------------------------------------------------------------------=== #
# Scaled dot-product attention
# ===----------------------------------------------------------------------=== #


class TestScaledDotProductAttention:
    def test_attention_shape(self):
        rng = make_rng(20)
        batch, heads, seq_q, seq_k, d_k, d_v = 2, 4, 6, 8, 16, 16
        q = nb.Tensor.from_dlpack(
            rng.normal(size=(batch, heads, seq_q, d_k)).astype(np.float32)
        )
        k = nb.Tensor.from_dlpack(
            rng.normal(size=(batch, heads, seq_k, d_k)).astype(np.float32)
        )
        v = nb.Tensor.from_dlpack(
            rng.normal(size=(batch, heads, seq_k, d_v)).astype(np.float32)
        )

        out = nb.nn.functional.scaled_dot_product_attention(q, k, v, training=False)
        assert tuple(int(d) for d in out.shape) == (batch, heads, seq_q, d_v)

    def test_attention_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(21)
        batch, heads, seq, d_k = 2, 2, 4, 8
        q_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)
        k_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)
        v_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)

        # Nabla
        out_nb = nb.nn.functional.scaled_dot_product_attention(
            nb.Tensor.from_dlpack(q_np),
            nb.Tensor.from_dlpack(k_np),
            nb.Tensor.from_dlpack(v_np),
            training=False,
        )

        # Manual PyTorch reference (not using F.scaled_dot_product_attention
        # because PyTorch's version may use different defaults)
        q_pt = torch.from_numpy(q_np)
        k_pt = torch.from_numpy(k_np)
        v_pt = torch.from_numpy(v_np)
        scale = 1.0 / math.sqrt(d_k)
        scores = torch.matmul(q_pt, k_pt.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out_pt = torch.matmul(weights, v_pt)

        nb.testing.assert_allclose(out_nb, out_pt, rtol=1e-4, atol=1e-5)

    def test_causal_attention_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(22)
        batch, heads, seq, d_k = 1, 1, 5, 4
        q_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)
        k_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)
        v_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)

        # Nabla with causal mask
        out_nb = nb.nn.functional.scaled_dot_product_attention(
            nb.Tensor.from_dlpack(q_np),
            nb.Tensor.from_dlpack(k_np),
            nb.Tensor.from_dlpack(v_np),
            is_causal=True,
            training=False,
        )

        # Manual PyTorch reference with causal mask
        q_pt = torch.from_numpy(q_np)
        k_pt = torch.from_numpy(k_np)
        v_pt = torch.from_numpy(v_np)
        scale = 1.0 / math.sqrt(d_k)
        scores = torch.matmul(q_pt, k_pt.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(seq, seq))
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out_pt = torch.matmul(weights, v_pt)

        nb.testing.assert_allclose(out_nb, out_pt, rtol=1e-4, atol=1e-5)

    def test_attention_with_additive_mask(self):
        """Additive mask should shift scores before softmax."""
        rng = make_rng(23)
        batch, heads, seq, d_k = 1, 1, 3, 4
        q_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)
        k_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)
        v_np = rng.normal(size=(batch, heads, seq, d_k)).astype(np.float32)

        # Zero mask => same as no mask
        zero_mask = nb.Tensor.from_dlpack(np.zeros((1, 1, seq, seq), dtype=np.float32))
        out_no_mask = nb.nn.functional.scaled_dot_product_attention(
            nb.Tensor.from_dlpack(q_np),
            nb.Tensor.from_dlpack(k_np),
            nb.Tensor.from_dlpack(v_np),
            training=False,
        )
        out_zero_mask = nb.nn.functional.scaled_dot_product_attention(
            nb.Tensor.from_dlpack(q_np),
            nb.Tensor.from_dlpack(k_np),
            nb.Tensor.from_dlpack(v_np),
            attn_mask=zero_mask,
            training=False,
        )
        nb.testing.assert_allclose(out_no_mask, out_zero_mask, rtol=1e-5, atol=1e-6)
