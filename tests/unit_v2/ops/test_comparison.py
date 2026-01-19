# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla
from nabla.ops import equal, not_equal, greater, greater_equal, less, less_equal

@pytest.mark.parametrize("shape_a, shape_b", [
    ((10,), (10,)),
    ((10, 10), (10, 10)),
    ((10, 5), (5,)),
    ((1,), (10,)),
])
class TestComparisonOps:
    def test_equal(self, op_verifier, shape_a, shape_b):
        op_verifier.verify(equal, np.equal, [shape_a, shape_b])

    def test_not_equal(self, op_verifier, shape_a, shape_b):
        op_verifier.verify(not_equal, np.not_equal, [shape_a, shape_b])

    def test_greater(self, op_verifier, shape_a, shape_b):
        op_verifier.verify(greater, np.greater, [shape_a, shape_b])

    def test_greater_equal(self, op_verifier, shape_a, shape_b):
        op_verifier.verify(greater_equal, np.greater_equal, [shape_a, shape_b])

    def test_less(self, op_verifier, shape_a, shape_b):
        op_verifier.verify(less, np.less, [shape_a, shape_b])

    def test_less_equal(self, op_verifier, shape_a, shape_b):
        op_verifier.verify(less_equal, np.less_equal, [shape_a, shape_b])
