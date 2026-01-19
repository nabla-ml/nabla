# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla
from nabla import shard_map, DeviceMesh
from nabla.core import Tensor
from nabla.core.sharding.spec import ShardingSpec, DimSpec

def test_shard_map_matmul_e2e():
    mesh = DeviceMesh("test_mesh", (4,), ("d",), devices=[0, 1, 2, 3])
    M, K, N = 128, 128, 128
    sharded_fn = shard_map(lambda a, b: a @ b, mesh, in_specs={0: None, 1: None}, out_specs=None, auto_sharding=True)
    a_np, b_np = np.random.randn(M, K).astype(np.float32), np.random.randn(K, N).astype(np.float32)
    a, b = Tensor.from_dlpack(a_np), Tensor.from_dlpack(b_np)
    res = sharded_fn(a, b)
    np.testing.assert_allclose(res.to_numpy(), a_np @ b_np, rtol=1e-4)

def test_shard_map_manual_specs():
    mesh = DeviceMesh("test_mesh", (2,), ("x",), devices=[0, 1])
    spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
    sharded_fn = shard_map(lambda x: x + 1.0, mesh, in_specs={0: spec}, out_specs={0: spec}, auto_sharding=False)
    x_np = np.random.randn(4, 4).astype(np.float32)
    x = Tensor.from_dlpack(x_np)
    res = sharded_fn(x)
    np.testing.assert_allclose(res.to_numpy(), x_np + 1.0, rtol=1e-5)
    assert res.is_sharded
    assert res.sharding.mesh == mesh
    assert res.sharding.dim_specs[0].axes == ["x"]
