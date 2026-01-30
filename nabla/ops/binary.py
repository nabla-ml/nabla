# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import BinaryOperation, Operation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


class AddOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "add"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.add(args[0], args[1])

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for addition: ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1."""
        return (cotangent, cotangent)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for addition: tangent_x + tangent_y."""
        from . import add

        return add(tangents[0], tangents[1])


class MulOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "mul"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.mul(args[0], args[1])

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for multiplication: ∂(x*y)/∂x = y, ∂(x*y)/∂y = x."""
        x, y = primals
        from . import mul

        return (mul(cotangent, y), mul(cotangent, x))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for multiplication: x * tangent_y + y * tangent_x."""
        x, y = primals
        tx, ty = tangents
        from . import add, mul

        return add(mul(x, ty), mul(tx, y))


class SubOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "sub"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.sub(args[0], args[1])

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for subtraction: ∂(x-y)/∂x = 1, ∂(x-y)/∂y = -1."""
        from ..ops.unary import neg

        return (cotangent, neg(cotangent))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for subtraction: tangent_x - tangent_y."""
        from . import sub

        return sub(tangents[0], tangents[1])


class DivOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "div"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.div(args[0], args[1])

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for division: ∂(x/y)/∂x = 1/y, ∂(x/y)/∂y = -x/y²."""
        x, y = primals
        from . import div, mul
        from ..ops.unary import neg

        grad_x = div(cotangent, y)
        grad_y = neg(mul(cotangent, div(x, mul(y, y))))
        return (grad_x, grad_y)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for division: (y*tangent_x - x*tangent_y) / y²."""
        x, y = primals
        tx, ty = tangents
        from . import div, mul, sub

        return div(sub(mul(y, tx), mul(x, ty)), mul(y, y))


class MatmulOp(Operation):
    """Matmul with 1D promotion handling."""

    @property
    def name(self) -> str:
        return "matmul"

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Estimate FLOPs for matmul: 2 * M * N * K."""

        if not input_shapes or len(input_shapes) < 2:
            return 0.0

        shape_a = input_shapes[0]
        shape_b = input_shapes[1]

        if len(shape_a) < 2 or len(shape_b) < 2:
            return 0.0

        m = shape_a[-2]
        k = shape_a[-1]
        n = shape_b[-1]

        batch_size = 1
        for d in shape_a[:-2]:
            batch_size *= d

        return 2.0 * batch_size * m * n * k

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.matmul(args[0], args[1])

    def physical_execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for Matmul."""
        from ..core import GRAPH
        from ..core.sharding import spmd

        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.maxpr, args, kwargs, mesh, op=self
            )

        return (shard_results, None, mesh)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        from . import view as view_ops
        from .base import ensure_tensor

        x = ensure_tensor(x)
        y = ensure_tensor(y)

        x_was_1d = len(x.shape) == 1
        y_was_1d = len(y.shape) == 1

        if x_was_1d:
            x = view_ops.unsqueeze(x, axis=0)
        if y_was_1d:
            y = view_ops.unsqueeze(y, axis=-1)

        # 1. Logical Broadcasting
        x_logical = tuple(int(d) for d in x.shape)
        y_logical = tuple(int(d) for d in y.shape)

        # Matmul broadcasting: align batch prefixes
        x_batch_logical = x_logical[:-2]
        y_batch_logical = y_logical[:-2]

        if x_batch_logical != y_batch_logical:
            # Standard numpy matmul broadcasting for batch dims
            # We can use BinaryOperation's helper if we had access, or just simple align
            target_batch = self._broadcast_batch_shapes(
                x_batch_logical, y_batch_logical
            )

            if x_batch_logical != target_batch:
                x = view_ops.broadcast_to(x, target_batch + x_logical[-2:])
            if y_batch_logical != target_batch:
                y = view_ops.broadcast_to(y, target_batch + y_logical[-2:])

        x_physical_batch = x.physical_global_shape[:-2]
        y_physical_batch = y.physical_global_shape[:-2]

        if x_physical_batch != y_physical_batch:
            target_batch = self._broadcast_batch_shapes(
                x_physical_batch, y_physical_batch
            )
            if x_physical_batch != target_batch:
                x = view_ops.broadcast_to_physical(x, target_batch + x_logical[-2:])
            if y_physical_batch != target_batch:
                y = view_ops.broadcast_to_physical(y, target_batch + y_logical[-2:])

        result = super().__call__(x, y)

        if x_was_1d and y_was_1d:
            result = view_ops.squeeze(result, axis=-1)
            result = view_ops.squeeze(result, axis=-1)
        elif x_was_1d:
            result = view_ops.squeeze(result, axis=0)
        elif y_was_1d:
            result = view_ops.squeeze(result, axis=-1)

        return result

    def _broadcast_batch_shapes(
        self, s1: tuple[int, ...], s2: tuple[int, ...]
    ) -> tuple[int, ...]:
        s1 = tuple(s1)
        s2 = tuple(s2)
        if len(s1) > len(s2):
            s2 = (1,) * (len(s1) - len(s2)) + s2
        elif len(s2) > len(s1):
            s1 = (1,) * (len(s2) - len(s1)) + s1
        res = []
        for d1, d2 in zip(s1, s2):
            if d1 == d2:
                res.append(d1)
            elif d1 == 1:
                res.append(d2)
            elif d2 == 1:
                res.append(d1)
            else:
                raise ValueError(f"Batch shapes {s1} and {s2} incompatible")
        return tuple(res)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ) -> Any:
        # print(f"[Matmul] sharding_rule: {input_shapes} -> {output_shapes}")
        """Matmul: (batch..., m, k) @ (batch..., k, n) -> (batch..., m, n)."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        return OpShardingRuleTemplate.parse(
            "... m k, ... k n -> ... m n", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for matmul: ∂(X@W)/∂X = cot@W.T, ∂(X@W)/∂W = X.T@cot."""
        x, y = primals
        from ..ops.view.axes import swap_axes
        from . import matmul

        # ∂L/∂X = ∂L/∂out @ W.T
        grad_x = matmul(cotangent, swap_axes(y, axis1=-2, axis2=-1))
        # ∂L/∂W = X.T @ ∂L/∂out
        grad_y = matmul(swap_axes(x, axis1=-2, axis2=-1), cotangent)
        return (grad_x, grad_y)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for matmul: X @ tangent_W + tangent_X @ W."""
        x, y = primals
        tx, ty = tangents
        from . import add, matmul

        return add(matmul(x, ty), matmul(tx, y))


add = AddOp()
mul = MulOp()
sub = SubOp()
div = DivOp()
matmul = MatmulOp()


__all__ = [
    "AddOp",
    "MulOp",
    "SubOp",
    "DivOp",
    "MatmulOp",
    "add",
    "mul",
    "sub",
    "div",
    "matmul",
]
