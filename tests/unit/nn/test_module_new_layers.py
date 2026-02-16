# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for new Module classes: Dropout, Embedding, Softmax,
MultiHeadAttention, TransformerEncoderLayer, TransformerDecoderLayer."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng, nb_from_np


# ===----------------------------------------------------------------------=== #
# Dropout module
# ===----------------------------------------------------------------------=== #


class TestDropoutModule:
    def test_dropout_train_drops_elements(self):
        x = nb.Tensor.from_dlpack(np.ones((1000,), dtype=np.float32))
        drop = nb.nn.Dropout(p=0.5)
        drop.train()
        y = drop(x).to_numpy()
        assert np.mean(y == 0.0) > 0.2  # some elements dropped

    def test_dropout_eval_is_identity(self):
        rng = make_rng(40)
        x_np = rng.normal(size=(8, 16)).astype(np.float32)
        x = nb.Tensor.from_dlpack(x_np)
        drop = nb.nn.Dropout(p=0.5)
        drop.eval()
        y = drop(x)
        nb.testing.assert_allclose(y, x, rtol=0, atol=0)

    def test_dropout_repr(self):
        d = nb.nn.Dropout(p=0.3)
        assert "0.3" in repr(d)

    def test_dropout_bad_p_raises(self):
        with pytest.raises(ValueError):
            nb.nn.Dropout(p=-0.1)
        with pytest.raises(ValueError):
            nb.nn.Dropout(p=1.5)


# ===----------------------------------------------------------------------=== #
# Embedding module
# ===----------------------------------------------------------------------=== #


class TestEmbeddingModule:
    def test_embedding_forward_shape(self):
        emb = nb.nn.Embedding(100, 32)
        indices = nb.Tensor.from_dlpack(np.array([0, 5, 99], dtype=np.int64))
        y = emb(indices)
        assert tuple(int(d) for d in y.shape) == (3, 32)

    def test_embedding_forward_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(50)
        weight_np = rng.normal(size=(20, 8)).astype(np.float32)
        indices_np = np.array([1, 3, 7, 0], dtype=np.int64)

        emb = nb.nn.Embedding(20, 8)
        emb.weight = nb_from_np(weight_np.copy(), requires_grad=True)

        y_nb = emb(nb.Tensor.from_dlpack(indices_np))

        pt_emb = torch.nn.Embedding(20, 8)
        pt_emb.weight = torch.nn.Parameter(torch.from_numpy(weight_np.copy()))
        y_pt = pt_emb(torch.from_numpy(indices_np))

        nb.testing.assert_allclose(y_nb, y_pt, rtol=1e-6, atol=1e-6)

    def test_embedding_has_gradient_param(self):
        emb = nb.nn.Embedding(10, 4)
        params = list(emb.parameters())
        assert len(params) == 1
        assert params[0].requires_grad

    def test_embedding_repr(self):
        emb = nb.nn.Embedding(100, 32)
        r = repr(emb)
        assert "100" in r
        assert "32" in r

    def test_embedding_pytree_roundtrip(self):
        emb = nb.nn.Embedding(10, 4)
        flat, treedef = nb.tree_flatten(emb)
        rebuilt = nb.tree_unflatten(treedef, flat)
        indices = nb.Tensor.from_dlpack(np.array([0, 1, 2], dtype=np.int64))
        nb.testing.assert_allclose(emb(indices), rebuilt(indices), rtol=1e-6, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Softmax module
# ===----------------------------------------------------------------------=== #


class TestSoftmaxModule:
    def test_softmax_module_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(60)
        x_np = rng.normal(size=(4, 8)).astype(np.float32)

        sm = nb.nn.Softmax(axis=-1)
        y_nb = sm(nb.Tensor.from_dlpack(x_np))
        y_pt = torch.softmax(torch.from_numpy(x_np), dim=-1)

        nb.testing.assert_allclose(y_nb, y_pt, rtol=1e-5, atol=1e-6)

    def test_softmax_sums_to_one(self):
        x = nb.Tensor.from_dlpack(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y = nb.nn.Softmax(axis=-1)(x).to_numpy()
        np.testing.assert_allclose(y.sum(axis=-1), 1.0, rtol=1e-6)


# ===----------------------------------------------------------------------=== #
# MultiHeadAttention module
# ===----------------------------------------------------------------------=== #


class TestMultiHeadAttention:
    def test_mha_output_shape(self):
        mha = nb.nn.MultiHeadAttention(d_model=16, num_heads=4)
        rng = make_rng(70)
        x = nb.Tensor.from_dlpack(rng.normal(size=(2, 5, 16)).astype(np.float32))
        out = mha(x, x, x)
        assert tuple(int(d) for d in out.shape) == (2, 5, 16)

    def test_mha_causal_output_shape(self):
        mha = nb.nn.MultiHeadAttention(d_model=8, num_heads=2)
        rng = make_rng(71)
        x = nb.Tensor.from_dlpack(rng.normal(size=(1, 4, 8)).astype(np.float32))
        out = mha(x, x, x, is_causal=True)
        assert tuple(int(d) for d in out.shape) == (1, 4, 8)

    def test_mha_cross_attention_shape(self):
        mha = nb.nn.MultiHeadAttention(d_model=16, num_heads=4)
        rng = make_rng(72)
        q = nb.Tensor.from_dlpack(rng.normal(size=(2, 3, 16)).astype(np.float32))
        kv = nb.Tensor.from_dlpack(rng.normal(size=(2, 7, 16)).astype(np.float32))
        out = mha(q, kv, kv)
        assert tuple(int(d) for d in out.shape) == (2, 3, 16)

    def test_mha_bad_dims_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            nb.nn.MultiHeadAttention(d_model=10, num_heads=3)

    def test_mha_parameters_count(self):
        mha = nb.nn.MultiHeadAttention(d_model=8, num_heads=2, bias=True)
        params = list(mha.parameters())
        # 4 Linear layers (q, k, v, out) × (weight + bias) = 8
        assert len(params) == 8

    def test_mha_eval_deterministic(self):
        """In eval mode, two forward passes should give identical results."""
        mha = nb.nn.MultiHeadAttention(d_model=8, num_heads=2, dropout=0.0)
        mha.eval()
        rng = make_rng(73)
        x = nb.Tensor.from_dlpack(rng.normal(size=(1, 3, 8)).astype(np.float32))
        y1 = mha(x, x, x).to_numpy()
        y2 = mha(x, x, x).to_numpy()
        np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)

    def test_mha_repr(self):
        mha = nb.nn.MultiHeadAttention(d_model=16, num_heads=4)
        r = repr(mha)
        assert "16" in r
        assert "4" in r


# ===----------------------------------------------------------------------=== #
# TransformerEncoderLayer module
# ===----------------------------------------------------------------------=== #


class TestTransformerEncoderLayer:
    def test_encoder_layer_shape(self):
        layer = nb.nn.TransformerEncoderLayer(
            d_model=16, num_heads=4, dim_feedforward=32, dropout=0.0
        )
        rng = make_rng(80)
        x = nb.Tensor.from_dlpack(rng.normal(size=(2, 5, 16)).astype(np.float32))
        out = layer(x)
        assert tuple(int(d) for d in out.shape) == (2, 5, 16)

    def test_encoder_layer_causal_shape(self):
        layer = nb.nn.TransformerEncoderLayer(
            d_model=8, num_heads=2, dim_feedforward=16, dropout=0.0
        )
        rng = make_rng(81)
        x = nb.Tensor.from_dlpack(rng.normal(size=(1, 4, 8)).astype(np.float32))
        out = layer(x, is_causal=True)
        assert tuple(int(d) for d in out.shape) == (1, 4, 8)

    def test_encoder_layer_residual_connection(self):
        """Output should differ from input (non-trivial transform)."""
        layer = nb.nn.TransformerEncoderLayer(
            d_model=8, num_heads=2, dim_feedforward=16, dropout=0.0
        )
        rng = make_rng(82)
        x = nb.Tensor.from_dlpack(rng.normal(size=(1, 3, 8)).astype(np.float32))
        out = layer(x)
        # Not identical (that would mean the layer does nothing)
        diff = np.abs(out.to_numpy() - x.to_numpy()).max()
        assert diff > 1e-6

    def test_encoder_layer_parameters(self):
        layer = nb.nn.TransformerEncoderLayer(
            d_model=8, num_heads=2, dim_feedforward=16, dropout=0.0
        )
        params = list(layer.parameters())
        # MHA: 4 Linear(8→8) each with weight+bias = 8 params
        # FFN: Linear(8→16, w+b) + Linear(16→8, w+b) = 4 params
        # LayerNorm × 2: each with weight + bias = 4 params
        # Total: 8 + 4 + 4 = 16
        assert len(params) == 16


# ===----------------------------------------------------------------------=== #
# TransformerDecoderLayer module
# ===----------------------------------------------------------------------=== #


class TestTransformerDecoderLayer:
    def test_decoder_layer_shape(self):
        layer = nb.nn.TransformerDecoderLayer(
            d_model=16, num_heads=4, dim_feedforward=32, dropout=0.0
        )
        rng = make_rng(90)
        tgt = nb.Tensor.from_dlpack(rng.normal(size=(2, 4, 16)).astype(np.float32))
        mem = nb.Tensor.from_dlpack(rng.normal(size=(2, 6, 16)).astype(np.float32))
        out = layer(tgt, mem)
        assert tuple(int(d) for d in out.shape) == (2, 4, 16)

    def test_decoder_layer_causal_shape(self):
        layer = nb.nn.TransformerDecoderLayer(
            d_model=8, num_heads=2, dim_feedforward=16, dropout=0.0
        )
        rng = make_rng(91)
        tgt = nb.Tensor.from_dlpack(rng.normal(size=(1, 3, 8)).astype(np.float32))
        mem = nb.Tensor.from_dlpack(rng.normal(size=(1, 5, 8)).astype(np.float32))
        out = layer(tgt, mem, is_causal=True)
        assert tuple(int(d) for d in out.shape) == (1, 3, 8)

    def test_decoder_layer_parameters(self):
        layer = nb.nn.TransformerDecoderLayer(
            d_model=8, num_heads=2, dim_feedforward=16, dropout=0.0
        )
        params = list(layer.parameters())
        # Self-attn MHA: 8 params, Cross-attn MHA: 8 params
        # FFN: 4 params, LayerNorm × 3: 6 params
        # Total: 8 + 8 + 4 + 6 = 26
        assert len(params) == 26
