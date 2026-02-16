# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for weight initializers in nabla.nn.functional."""

from __future__ import annotations

import numpy as np

import nabla as nb


class TestXavierNormal:
    def test_shape_and_dtype(self):
        shape = (512, 256)
        w = nb.nn.functional.xavier_normal(shape)
        assert tuple(int(d) for d in w.shape) == shape
        assert w.dtype == nb.DType.float32

    def test_statistics(self):
        shape = (512, 256)
        w = nb.nn.functional.xavier_normal(shape)
        w_np = np.asarray(w)
        expected_std = (2.0 / (shape[0] + shape[1])) ** 0.5
        assert abs(float(np.mean(w_np))) < 0.02
        assert abs(float(np.std(w_np)) - expected_std) < 0.02


class TestHeNormal:
    def test_shape_and_dtype(self):
        shape = (512, 256)
        w = nb.nn.functional.he_normal(shape)
        assert tuple(int(d) for d in w.shape) == shape
        assert w.dtype == nb.DType.float32

    def test_statistics(self):
        shape = (512, 256)
        w = nb.nn.functional.he_normal(shape)
        w_np = np.asarray(w)
        expected_std = (2.0 / shape[0]) ** 0.5
        assert abs(float(np.mean(w_np))) < 0.02
        assert abs(float(np.std(w_np)) - expected_std) < 0.02
