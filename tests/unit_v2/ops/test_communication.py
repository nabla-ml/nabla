# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
from nabla import DeviceMesh, DimSpec, ShardingSpec, Tensor
from nabla.ops.communication import all_gather, all_reduce, axis_index

class TestCommunicationOps:
    @pytest.fixture
    def mesh(self):
        return DeviceMesh("comm_mesh", (4,), ("d",))

    def test_all_gather(self, mesh):
        x_np = np.arange(4).astype(np.float32)
        x = Tensor.from_dlpack(x_np).shard(mesh, [DimSpec(["d"])])
        res = all_gather(x, axis=0)
        np.testing.assert_allclose(res.to_numpy(), x_np)

    def test_axis_index(self, mesh):
        dummy = Tensor.zeros((4,)).shard(mesh, [DimSpec(["d"])])
        res = axis_index(mesh, "d")
        # axis_index returns a tensor that is logically [0, 1, 2, 3] across the mesh
        # When moving to numpy, it should gather
        np.testing.assert_allclose(res.to_numpy(), np.arange(4).astype(np.int32))

    def test_all_reduce(self, mesh):
        x_np = np.arange(4).astype(np.float32)
        x = Tensor.from_dlpack(x_np).shard(mesh, [DimSpec(["d"])])
        res = all_reduce(x)
        # 0+1+2+3 = 6
        np.testing.assert_allclose(res.to_numpy(), [6.0])
