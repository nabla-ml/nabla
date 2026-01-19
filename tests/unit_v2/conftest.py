# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import weakref
from unittest.mock import MagicMock, patch
import nabla
from nabla.core import Tensor, GRAPH
from nabla.core.graph.engine import ComputeGraph

@pytest.fixture
def clean_graph():
    GRAPH._reset(None, 0)
    yield
    GRAPH._reset(None, 0)
    import gc
    gc.collect()

@pytest.fixture
def spy_execution(clean_graph):
    target = "nabla.core.graph.engine.ComputeGraph._compile_and_execute_with_map"
    real_method = ComputeGraph._compile_and_execute_with_map
    with patch(target, autospec=True) as mock_method:
        mock_method.side_effect = lambda self, unrealized, value_map, return_model: real_method(self, unrealized, value_map, return_model)
        yield mock_method
