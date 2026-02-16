# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for LoRA adapter utilities — validated against JAX as the functional
reference."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


# ===----------------------------------------------------------------------=== #
# init_lora_adapter
# ===----------------------------------------------------------------------=== #


class TestInitLoraAdapter:
    def test_shapes_and_keys(self):
        rng = make_rng(70)
        weight = nb.Tensor.from_dlpack(rng.normal(size=(12, 10)).astype(np.float32))
        adapter = nb.nn.finetune.init_lora_adapter(weight, rank=4)

        assert "A" in adapter and "B" in adapter
        assert tuple(int(d) for d in adapter["A"].shape) == (12, 4)
        assert tuple(int(d) for d in adapter["B"].shape) == (4, 10)

    def test_B_initialized_to_zero(self):
        rng = make_rng(71)
        weight = nb.Tensor.from_dlpack(rng.normal(size=(8, 6)).astype(np.float32))
        adapter = nb.nn.finetune.init_lora_adapter(weight, rank=2)
        b_np = np.asarray(adapter["B"])
        np.testing.assert_allclose(b_np, 0.0)


# ===----------------------------------------------------------------------=== #
# lora_delta
# ===----------------------------------------------------------------------=== #


class TestLoraDelta:
    def test_lora_delta_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(72)
        a_np = rng.normal(size=(8, 3)).astype(np.float32)
        b_np = rng.normal(size=(3, 6)).astype(np.float32)
        alpha = 6.0

        adapter = {
            "A": nb.Tensor.from_dlpack(a_np),
            "B": nb.Tensor.from_dlpack(b_np),
        }
        delta_nb = nb.nn.finetune.lora_delta(adapter, alpha=alpha)

        rank = 3
        delta_jax = (jnp.asarray(a_np) @ jnp.asarray(b_np)) * (alpha / rank)

        nb.testing.assert_allclose(delta_nb, delta_jax, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# lora_linear
# ===----------------------------------------------------------------------=== #


class TestLoraLinear:
    def test_lora_linear_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(73)
        x_np = rng.normal(size=(5, 8)).astype(np.float32)
        w_np = rng.normal(size=(8, 6)).astype(np.float32)
        a_np = rng.normal(size=(8, 2)).astype(np.float32)
        b_np = rng.normal(size=(2, 6)).astype(np.float32)
        alpha = 4.0

        adapter = {
            "A": nb.Tensor.from_dlpack(a_np),
            "B": nb.Tensor.from_dlpack(b_np),
        }

        y_nb = nb.nn.finetune.lora_linear(
            nb.Tensor.from_dlpack(x_np),
            nb.Tensor.from_dlpack(w_np),
            adapter,
            alpha=alpha,
        )

        rank = 2
        delta = (jnp.asarray(a_np) @ jnp.asarray(b_np)) * (alpha / rank)
        y_jax = jnp.asarray(x_np) @ jnp.asarray(w_np) + jnp.asarray(x_np) @ delta

        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# merge / unmerge roundtrip
# ===----------------------------------------------------------------------=== #


class TestMergeUnmerge:
    def test_roundtrip(self):
        rng = make_rng(7071)
        w_np = rng.normal(size=(12, 10)).astype(np.float32)
        weight = nb.Tensor.from_dlpack(w_np)
        adapter = nb.nn.finetune.init_lora_adapter(weight, rank=4, init_std=0.02)

        merged = nb.nn.finetune.merge_lora_weight(weight, adapter, alpha=8.0)
        restored = nb.nn.finetune.unmerge_lora_weight(merged, adapter, alpha=8.0)

        nb.testing.assert_allclose(restored, w_np, rtol=1e-5, atol=1e-5)

    def test_merge_value_vs_jax(self):
        jnp = pytest.importorskip("jax.numpy")
        rng = make_rng(74)
        w_np = rng.normal(size=(8, 6)).astype(np.float32)
        a_np = rng.normal(size=(8, 2)).astype(np.float32)
        b_np = rng.normal(size=(2, 6)).astype(np.float32)
        alpha = 4.0

        adapter = {
            "A": nb.Tensor.from_dlpack(a_np),
            "B": nb.Tensor.from_dlpack(b_np),
        }

        merged = nb.nn.finetune.merge_lora_weight(
            nb.Tensor.from_dlpack(w_np), adapter, alpha=alpha
        )

        rank = 2
        delta = (jnp.asarray(a_np) @ jnp.asarray(b_np)) * (alpha / rank)
        merged_jax = jnp.asarray(w_np) + delta

        nb.testing.assert_allclose(merged, merged_jax, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# tree_lora_delta
# ===----------------------------------------------------------------------=== #


class TestTreeLoraDelta:
    def test_tree_lora_delta_maps_single_adapter(self):
        """tree_lora_delta on a single adapter dict returns a Tensor delta."""
        rng = make_rng(75)
        adapter = {
            "A": nb.Tensor.from_dlpack(rng.normal(size=(8, 2)).astype(np.float32)),
            "B": nb.Tensor.from_dlpack(rng.normal(size=(2, 6)).astype(np.float32)),
        }

        def is_adapter_leaf(x):
            return isinstance(x, dict) and "A" in x and "B" in x

        delta = nb.nn.finetune.tree_lora_delta(adapter, alpha=6.0, is_leaf=is_adapter_leaf)
        assert isinstance(delta, nb.Tensor)
        assert tuple(int(d) for d in delta.shape) == (8, 6)

    def test_tree_lora_delta_maps_nested_adapters(self):
        """tree_lora_delta on a list/tuple of adapters maps over each."""
        rng = make_rng(76)
        adapters = [
            {
                "A": nb.Tensor.from_dlpack(rng.normal(size=(8, 2)).astype(np.float32)),
                "B": nb.Tensor.from_dlpack(rng.normal(size=(2, 6)).astype(np.float32)),
            },
            {
                "A": nb.Tensor.from_dlpack(rng.normal(size=(6, 3)).astype(np.float32)),
                "B": nb.Tensor.from_dlpack(rng.normal(size=(3, 4)).astype(np.float32)),
            },
        ]

        def is_adapter_leaf(x):
            return isinstance(x, dict) and "A" in x and "B" in x

        deltas = nb.nn.finetune.tree_lora_delta(adapters, alpha=6.0, is_leaf=is_adapter_leaf)
        assert isinstance(deltas, list)
        assert len(deltas) == 2
        assert isinstance(deltas[0], nb.Tensor)
        assert isinstance(deltas[1], nb.Tensor)
        assert tuple(int(d) for d in deltas[0].shape) == (8, 6)
        assert tuple(int(d) for d in deltas[1].shape) == (6, 4)
