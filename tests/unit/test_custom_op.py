# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import numpy as np

import nabla
from nabla.core import trace
from nabla.core.sharding.propagation import OpShardingRuleTemplate
from nabla.ops.base import UnaryOperation, ensure_tensor
from nabla.ops.custom_op import call_custom_kernel


class AddOneCustomOp(UnaryOperation):
    @property
    def name(self) -> str:
        return "add_one_custom"

    def maxpr(self, *args, **kwargs):
        return call_custom_kernel(
            func_name="add_one_custom",
            kernel_path=Path(__file__).parent / "custom_kernels",
            values=list(args),
            out_types=args[0].type,
        )

    def sharding_rule(self, input_shapes, output_shapes, **kwargs):
        rank = len(input_shapes[0])
        mapping = {i: [f"d{i}"] for i in range(rank)}
        return OpShardingRuleTemplate([mapping], [mapping]).instantiate(
            input_shapes, output_shapes
        )

    def __call__(self, x: nabla.Tensor) -> nabla.Tensor:
        return super().__call__(ensure_tensor(x))


add_one_custom = AddOneCustomOp()


def test_custom_kernel_execution():
    x = nabla.Tensor.ones((4, 8))
    print("\nEager Trace:", trace(add_one_custom, x), sep="\n")

    y = add_one_custom(x)
    np.testing.assert_allclose(y.to_numpy(), np.ones((4, 8)) + 1.0)
    print("✅ Custom kernel execution verified")


def test_custom_kernel_sharding():
    mesh = nabla.DeviceMesh("mesh", [2], ["x"])
    x = nabla.Tensor.ones((4, 8))
    x_sharded = nabla.shard(x, mesh, nabla.P("x", None))

    print("\nSharding Trace:", trace(add_one_custom, x_sharded), sep="\n")

    y = add_one_custom(x_sharded)
    assert y.is_sharded and y.sharding.mesh == mesh
    assert y.sharding.dim_specs[0].axes == ["x"]
    assert y.sharding.dim_specs[1].axes == []

    np.testing.assert_allclose(y.to_numpy(), np.ones((4, 8)) + 1.0)
    print("✅ Custom kernel sharding verified")


def test_custom_kernel_vmap():
    x = nabla.Tensor.ones((2, 4, 8))
    batched_add_one = nabla.vmap(add_one_custom)

    print("\nVmap Trace:", trace(batched_add_one, x), sep="\n")

    y = batched_add_one(x)
    assert tuple(int(d) for d in y.shape) == (2, 4, 8)
    np.testing.assert_allclose(y.to_numpy(), np.ones((2, 4, 8)) + 1.0)
    print("✅ Custom kernel vmap verified")
