# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for Module core infrastructure: pytree registration, state_dict,
repr, train/eval, parameters/buffers iteration, and register_buffer."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng, nb_from_np


# ===----------------------------------------------------------------------=== #
# Pytree semantics
# ===----------------------------------------------------------------------=== #


class TestModulePytree:
    """Module instances are pytree nodes with tensor leaves."""

    def test_linear_has_two_tensor_leaves(self):
        model = nb.nn.Linear(4, 3)
        leaves = nb.tree_leaves(model)
        tensor_leaves = [x for x in leaves if isinstance(x, nb.Tensor)]
        assert len(tensor_leaves) == 2  # weight + bias

    def test_linear_no_bias_has_one_tensor_leaf(self):
        model = nb.nn.Linear(4, 3, bias=False)
        leaves = nb.tree_leaves(model)
        tensor_leaves = [x for x in leaves if isinstance(x, nb.Tensor)]
        assert len(tensor_leaves) == 1

    def test_flatten_unflatten_roundtrip_preserves_forward(self):
        rng = make_rng(12)
        model = nb.nn.Linear(4, 3)
        x_np = rng.normal(size=(5, 4)).astype(np.float32)
        x = nb.Tensor.from_dlpack(x_np)

        y_ref = model(x).to_numpy()
        flat, treedef = nb.tree_flatten(model)
        rebuilt = nb.tree_unflatten(treedef, flat)
        y_rebuilt = rebuilt(x).to_numpy()
        np.testing.assert_allclose(y_rebuilt, y_ref, rtol=1e-6, atol=1e-6)

    def test_sequential_pytree_leaves_count(self):
        model = nb.nn.Sequential(
            nb.nn.Linear(4, 6),
            nb.nn.ReLU(),
            nb.nn.Linear(6, 3),
        )
        leaves = nb.tree_leaves(model)
        tensor_leaves = [x for x in leaves if isinstance(x, nb.Tensor)]
        # Linear(4,6): weight+bias, Linear(6,3): weight+bias => 4
        assert len(tensor_leaves) == 4


# ===----------------------------------------------------------------------=== #
# Parameters / buffers / modules iteration
# ===----------------------------------------------------------------------=== #


class TestModuleIteration:
    def test_parameters_yields_all_params(self):
        model = nb.nn.Linear(5, 3)
        params = list(model.parameters())
        assert len(params) == 2  # weight, bias
        assert all(isinstance(p, nb.Tensor) for p in params)

    def test_named_parameters_includes_names(self):
        model = nb.nn.Linear(5, 3)
        named = dict(model.named_parameters())
        assert "weight" in named
        assert "bias" in named

    def test_named_parameters_nested_prefixes(self):
        model = nb.nn.Sequential(
            nb.nn.Linear(4, 6),
            nb.nn.ReLU(),
            nb.nn.Linear(6, 3),
        )
        named = dict(model.named_parameters())
        assert "0.weight" in named
        assert "0.bias" in named
        assert "2.weight" in named
        assert "2.bias" in named

    def test_modules_iterator(self):
        model = nb.nn.Sequential(
            nb.nn.Linear(4, 6),
            nb.nn.ReLU(),
            nb.nn.Linear(6, 3),
        )
        mods = list(model.modules())
        # Sequential, Linear, ReLU, Linear => 4
        assert len(mods) == 4
        assert isinstance(mods[0], nb.nn.Sequential)

    def test_register_buffer(self):
        model = nb.nn.Linear(4, 3)
        buf = nb.Tensor.from_dlpack(np.zeros((1, 3), dtype=np.float32))
        model.register_buffer("running_mean", buf)
        buffers = dict(model.named_buffers())
        assert "running_mean" in buffers


# ===----------------------------------------------------------------------=== #
# Train / eval mode
# ===----------------------------------------------------------------------=== #


class TestTrainEval:
    def test_train_mode_propagates(self):
        model = nb.nn.Sequential(
            nb.nn.Linear(4, 3),
            nb.nn.ReLU(),
        )
        model.eval()
        for m in model.modules():
            assert not m._training
        model.train()
        for m in model.modules():
            assert m._training


# ===----------------------------------------------------------------------=== #
# state_dict / load_state_dict
# ===----------------------------------------------------------------------=== #


class TestStateDict:
    def test_state_dict_roundtrip_simple(self):
        model = nb.nn.Linear(4, 3)
        sd = model.state_dict()
        assert "weight" in sd and "bias" in sd

    def test_load_state_dict_changes_weights(self):
        rng = make_rng(909)
        src = nb.nn.Sequential(nb.nn.Linear(4, 6), nb.nn.ReLU(), nb.nn.Linear(6, 3))
        dst = nb.nn.Sequential(nb.nn.Linear(4, 6), nb.nn.ReLU(), nb.nn.Linear(6, 3))

        x_np = rng.normal(size=(11, 4)).astype(np.float32)
        x = nb.Tensor.from_dlpack(x_np)

        y_src = src(x)

        # Perturb dst so it differs
        first = getattr(dst, "0")
        first.weight = first.weight + 0.123

        y_dst_before = dst(x)
        dst.load_state_dict(src.state_dict())
        y_dst_after = dst(x)

        src_np = np.asarray(y_src)
        before_np = np.asarray(y_dst_before)
        after_np = np.asarray(y_dst_after)

        assert not np.allclose(src_np, before_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(src_np, after_np, rtol=1e-6, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# __repr__
# ===----------------------------------------------------------------------=== #


class TestRepr:
    def test_linear_repr_contains_features(self):
        model = nb.nn.Linear(4, 3)
        r = repr(model)
        assert "Linear" in r
        assert "4" in r and "3" in r

    def test_sequential_repr_contains_children(self):
        model = nb.nn.Sequential(
            nb.nn.Linear(4, 6),
            nb.nn.ReLU(),
            nb.nn.Linear(6, 3),
        )
        r = repr(model)
        assert "Sequential" in r
        assert "Linear" in r
        assert "ReLU" in r


# ===----------------------------------------------------------------------=== #
# Sequential with OrderedDict
# ===----------------------------------------------------------------------=== #


class TestSequentialOrderedDict:
    def test_ordered_dict_constructor(self):
        model = nb.nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nb.nn.Linear(4, 6)),
                    ("act", nb.nn.ReLU()),
                    ("fc2", nb.nn.Linear(6, 3)),
                ]
            )
        )
        # Named children should be accessible by name
        assert hasattr(model, "fc1")
        assert hasattr(model, "act")
        assert hasattr(model, "fc2")

        rng = make_rng(77)
        x_np = rng.normal(size=(5, 4)).astype(np.float32)
        x = nb.Tensor.from_dlpack(x_np)
        y = model(x)
        assert tuple(int(d) for d in y.shape) == (5, 3)
