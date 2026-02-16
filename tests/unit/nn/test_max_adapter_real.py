# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Integration tests adapting real classes from max.nn."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb


max_nn = pytest.importorskip("max.nn")


class TestRealMaxAdapter:
    def test_adapt_max_nn_core_helper(self):
        adapted = nb.nn.adapt_max_nn_core(max_nn)

        assert set(adapted.keys()) == {"Linear", "Embedding", "ModuleList", "Sequential"}

        Linear = adapted["Linear"]
        Sequential = adapted["Sequential"]

        model = Sequential(Linear(4, 5), Linear(5, 2))
        assert set(dict(model.named_parameters()).keys()) == {
            "0.weight",
            "0.bias",
            "1.weight",
            "1.bias",
        }

        x = nb.Tensor.from_dlpack(np.ones((2, 4), dtype=np.float32))
        x.requires_grad_(True)
        y = model(x)
        assert tuple(int(d) for d in y.shape) == (2, 2)

        loss = y.sum()
        model.backward(loss)
        assert model[0].weight.grad is not None
        assert model[1].weight.grad is not None

    def test_adapt_real_linear_forward_and_backward(self):
        NablaLinear = nb.nn.adapt_max_module_class(
            max_nn.Linear,
            source_module_base=max_nn.Module,
            class_name="AdaptedMaxLinear",
        )

        model = NablaLinear(4, 3)
        assert set(dict(model.named_parameters()).keys()) == {"weight", "bias"}

        x = nb.Tensor.from_dlpack(np.ones((5, 4), dtype=np.float32))
        x.requires_grad_(True)
        y = model(x)
        assert tuple(int(d) for d in y.shape) == (5, 3)

        loss = y.sum()
        model.backward(loss)

        assert model.weight.grad is not None
        assert model.bias.grad is not None

    def test_adapt_real_embedding_forward_and_backward(self):
        NablaEmbedding = nb.nn.adapt_max_module_class(
            max_nn.Embedding,
            source_module_base=max_nn.Module,
            class_name="AdaptedMaxEmbedding",
        )

        emb = NablaEmbedding(128, dim=16)
        assert set(dict(emb.named_parameters()).keys()) == {"weight"}

        indices = nb.Tensor.from_dlpack(np.array([3, 1, 7, 1], dtype=np.uint64))
        out = emb(indices)
        assert tuple(int(d) for d in out.shape) == (4, 16)

        loss = out.sum()
        emb.backward(loss)
        assert emb.weight.grad is not None

    def test_adapt_real_sequential_and_module_list(self):
        NablaModuleList = nb.nn.adapt_max_module_class(
            max_nn.ModuleList,
            source_module_base=max_nn.Module,
            class_name="AdaptedMaxModuleList",
        )
        NablaSequential = nb.nn.adapt_max_module_class(
            max_nn.Sequential,
            base_overrides={max_nn.ModuleList: NablaModuleList},
            class_name="AdaptedMaxSequential",
            global_overrides={"ModuleList": NablaModuleList},
        )
        NablaLinear = nb.nn.adapt_max_module_class(
            max_nn.Linear,
            source_module_base=max_nn.Module,
            class_name="AdaptedMaxLinearInSequential",
        )

        seq = NablaSequential(
            NablaLinear(4, 6),
            NablaLinear(6, 2),
        )

        assert set(dict(seq.named_parameters()).keys()) == {
            "0.weight",
            "0.bias",
            "1.weight",
            "1.bias",
        }

        x = nb.Tensor.from_dlpack(np.ones((3, 4), dtype=np.float32))
        x.requires_grad_(True)
        y = seq(x)
        assert tuple(int(d) for d in y.shape) == (3, 2)

        loss = y.sum()
        seq.backward(loss)
        assert seq[0].weight.grad is not None
        assert seq[1].weight.grad is not None
