# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import gc
import weakref
import nabla
from nabla.core import Tensor

class TestLazyExecution:
    def test_dead_tensor_elimination(self, spy_execution):
        def build_graph():
            x = Tensor.constant(2.0)
            y = x * x
            z = y + 1.0
            return z, id(y)
        z, y_id = build_graph()
        z.realize()
        assert spy_execution.called
        unrealized = spy_execution.call_args[0][1]
        target_ids = {id(t) for t in unrealized}
        assert id(z) in target_ids
        assert y_id not in target_ids

    def test_alive_intermediate_evaluation(self, spy_execution):
        x = Tensor.constant(10.0)
        y = x * 2.0
        z = y + 5.0
        z.realize()
        args = spy_execution.call_args[0]
        unrealized = args[1]
        target_ids = {id(t) for t in unrealized}
        assert id(z) in target_ids
        assert id(y) in target_ids

    def test_unnecessary_recomputation(self, spy_execution):
        x = Tensor.constant(5.0)
        y = x * 2.0
        y.realize()
        assert spy_execution.call_count == 1
        y.realize()
        assert spy_execution.call_count == 1
        z = y + 1.0
        z.realize()
        assert spy_execution.call_count == 2
        args_2 = spy_execution.call_args[0]
        unrealized_2 = args_2[1]
        target_ids = {id(t) for t in unrealized_2}
        assert id(z) in target_ids
        assert len(target_ids) == 1

    def test_multi_output_graph_execution(self, spy_execution):
        x = Tensor.constant(3.0)
        a = x + 1.0
        b = x + 2.0
        a.realize()
        args = spy_execution.call_args[0]
        unrealized = args[1]
        target_ids = {id(t) for t in unrealized}
        assert id(a) in target_ids
        assert id(b) in target_ids
