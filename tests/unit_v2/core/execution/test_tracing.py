# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import nabla
from nabla.core import Tensor, trace
from nabla.core.graph.tracing import Trace

class TestTracing:
    def test_trace_capture_structure(self):
        def logic(x, y):
            a = x + y
            b = a * 2.0
            c = b - x
            return c
        x = Tensor.constant(1.0)
        y = Tensor.constant(2.0)
        t = trace(logic, x, y)
        assert isinstance(t, Trace)
        assert len(t.nodes) == 4
        op_names = [n.op.name for n in t.nodes]
        assert "add" in op_names
        assert "mul" in op_names
        assert "sub" in op_names

    def test_traced_execution_via_realize(self, spy_execution):
        def logic(x):
            return x * x
        x = Tensor.constant(3.0)
        t = trace(logic, x)
        out = t.outputs
        assert isinstance(out, Tensor)
        out.realize()
        assert spy_execution.called
        import numpy as np
        np.testing.assert_allclose(out.to_numpy(), 9.0)

    def test_output_ref_integrity(self):
        def logic(x):
            return x + 1.0
        x = Tensor.constant(1.0)
        t = trace(logic, x)
        out = t.outputs
        assert out._impl.output_refs is not None
        assert out._impl.output_refs.op.name == "add"
