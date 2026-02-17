# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import ops

from .base import BinaryOperation, OpArgs, Operation, OpKwargs, OpResult, OpTensorValues

if TYPE_CHECKING:
    from ..core.tensor import Tensor


class AddOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "add"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.add(args[0], args[1])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for addition: ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1."""
        return [cotangents[0], cotangents[0]]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for addition: tangent_x + tangent_y."""
        from . import add

        return [add(tangents[0], tangents[1])]


class MulOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "mul"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.mul(args[0], args[1])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for multiplication: ∂(x*y)/∂x = y, ∂(x*y)/∂y = x."""
        x, y = primals[0], primals[1]
        from . import mul

        return [mul(cotangents[0], y), mul(cotangents[0], x)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for multiplication: x * tangent_y + y * tangent_x."""
        x, y = primals[0], primals[1]
        tx, ty = tangents[0], tangents[1]
        from . import add, mul
        return [add(mul(x, ty), mul(tx, y))]


class SubOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "sub"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.sub(args[0], args[1])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for subtraction: ∂(x-y)/∂x = 1, ∂(x-y)/∂y = -1."""
        from ..ops.unary import neg

        return [cotangents[0], neg(cotangents[0])]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for subtraction: tangent_x - tangent_y."""
        from . import sub

        return [sub(tangents[0], tangents[1])]


class DivOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "div"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.div(args[0], args[1])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for division: ∂(x/y)/∂x = 1/y, ∂(x/y)/∂y = -x/y²."""
        x, y = primals[0], primals[1]
        cotangent = cotangents[0]
        from ..ops.unary import neg
        from . import div, mul

        grad_x = div(cotangent, y)
        grad_y = neg(mul(cotangent, div(x, mul(y, y))))
        return [grad_x, grad_y]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for division: (y*tangent_x - x*tangent_y) / y²."""
        x, y = primals[0], primals[1]
        tx, ty = tangents[0], tangents[1]
        from . import div, mul, sub

        return [div(sub(mul(y, tx), mul(x, ty)), mul(y, y))]


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

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.matmul(args[0], args[1])]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for Matmul: (... M K) @ (... K N) -> (... M N)."""
        from ..core.sharding import spmd

        x = args[0]
        y = args[1]

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx_x = i if i < x.num_shards else 0
            idx_y = i if i < y.num_shards else 0

            sx = x.physical_local_shape_ints(idx_x)
            sy = y.physical_local_shape_ints(idx_y)

            if sx is not None and sy is not None:
                res_shape = sx[:-1] + sy[-1:]
                shapes.append(res_shape)
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)

        return shapes, dtypes, devices

    _infer_output_sharding: bool = False

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        from . import view as view_ops
        from .base import ensure_tensor

        x = ensure_tensor(args[0])
        y = ensure_tensor(args[1])

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
            target_batch = self._broadcast_shapes(x_batch_logical, y_batch_logical)

            if x_batch_logical != target_batch:
                x = view_ops.broadcast_to(x, target_batch + x_logical[-2:])
            if y_batch_logical != target_batch:
                y = view_ops.broadcast_to(y, target_batch + y_logical[-2:])

        x_physical_batch = x.physical_global_shape[:-2]
        y_physical_batch = y.physical_global_shape[:-2]

        if x_physical_batch != y_physical_batch:
            target_batch = self._broadcast_shapes(x_physical_batch, y_physical_batch)
            if x_physical_batch != target_batch:
                x = view_ops.broadcast_to_physical(x, target_batch + x_logical[-2:])
            if y_physical_batch != target_batch:
                y = view_ops.broadcast_to_physical(y, target_batch + y_logical[-2:])

        result = super().__call__([x, y], kwargs)[0]

        if x_was_1d and y_was_1d:
            result = view_ops.squeeze(result, axis=-1)
            result = view_ops.squeeze(result, axis=-1)
        elif x_was_1d:
            result = view_ops.squeeze(result, axis=0)
        elif y_was_1d:
            result = view_ops.squeeze(result, axis=-1)

        return [result]

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

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for matmul: ∂(X@W)/∂X = cot@W.T, ∂(X@W)/∂W = X.T@cot."""
        x, y = primals[0], primals[1]
        cotangent = cotangents[0]
        from ..ops.view.axes import swap_axes
        from . import matmul

        grad_x = matmul(cotangent, swap_axes(y, axis1=-2, axis2=-1))
        grad_y = matmul(swap_axes(x, axis1=-2, axis2=-1), cotangent)
        return [grad_x, grad_y]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for matmul: X @ tangent_W + tangent_X @ W."""
        x, y = primals[0], primals[1]
        tx, ty = tangents[0], tangents[1]
        from . import add, matmul

        return [add(matmul(x, ty), matmul(tx, y))]


_add_op = AddOp()
_mul_op = MulOp()
_sub_op = SubOp()
_div_op = DivOp()
_matmul_op = MatmulOp()


def add(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _add_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def mul(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _mul_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def sub(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _sub_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def div(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _div_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def matmul(x: Tensor, y: Tensor) -> Tensor:
    from .base import ensure_tensor

    return _matmul_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


class ModOp(BinaryOperation):
    """Elementwise modulus (remainder)."""

    @property
    def name(self) -> str:
        return "mod"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.mod(args[0], args[1])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP: (cotangent, -cotangent * floor(lhs / rhs))."""
        lhs, rhs = primals[0], primals[1]
        cotangent = cotangents[0]
        from ..ops.unary import floor, neg
        from . import div, mul

        grad_lhs = cotangent
        grad_rhs = neg(mul(cotangent, floor(div(lhs, rhs))))
        return [grad_lhs, grad_rhs]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP: tangent_lhs - tangent_rhs * floor(lhs / rhs)."""
        lhs, rhs = primals[0], primals[1]
        tl, tr = tangents[0], tangents[1]
        from ..ops.unary import floor
        from . import div, mul, sub

        return [sub(tl, mul(tr, floor(div(lhs, rhs))))]


class PowOp(BinaryOperation):
    """Elementwise exponentiation: lhs ^ rhs."""

    @property
    def name(self) -> str:
        return "pow"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.pow(args[0], args[1])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP: (cotangent * rhs * lhs^(rhs-1), cotangent * output * log(lhs))."""
        lhs, rhs = primals[0], primals[1]
        output = outputs[0]
        cotangent = cotangents[0]
        from ..ops.unary import log
        from . import mul, pow, sub

        grad_lhs = mul(cotangent, mul(rhs, pow(lhs, sub(rhs, 1.0))))
        grad_rhs = mul(cotangent, mul(output, log(lhs)))
        return [grad_lhs, grad_rhs]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP: rhs * lhs^(rhs-1) * tangent_lhs + output * log(lhs) * tangent_rhs."""
        lhs, rhs = primals[0], primals[1]
        tl, tr = tangents[0], tangents[1]
        output = outputs[0]
        from ..ops.unary import log
        from . import add, mul, pow, sub

        term1 = mul(rhs, mul(pow(lhs, sub(rhs, 1.0)), tl))
        term2 = mul(output, mul(log(lhs), tr))
        return [add(term1, term2)]


class OuterOp(BinaryOperation):
    """Outer product of two vectors."""

    @property
    def name(self) -> str:
        return "outer"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x, y = args
        if len(x.shape) > 1 or len(y.shape) > 1:
            # Handle vmapped case: x is (B, N), y is (B, M) -> output (B, N, M)
            # We want x.unsqueeze(-1) * y.unsqueeze(-2)
            x_up = ops.unsqueeze(x, axis=-1)
            y_up = ops.unsqueeze(y, axis=-2)
            return [ops.mul(x_up, y_up)]
        return [ops.outer(x, y)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        lhs, rhs = args
        num_shards = max(lhs.num_shards, rhs.num_shards)
        shapes = []
        for i in range(num_shards):
            idx_l = i if i < lhs.num_shards else 0
            idx_r = i if i < rhs.num_shards else 0
            sl = lhs.physical_local_shape_ints(idx_l)
            sr = rhs.physical_local_shape_ints(idx_r)

            # Use global shape if local is missing (replicated fallback)
            if sl is None:
                sl = tuple(int(d) for d in (lhs.physical_global_shape or lhs.shape))
            if sr is None:
                sr = tuple(int(d) for d in (rhs.physical_global_shape or rhs.shape))

            shapes.append(sl[:-1] + (sl[-1], sr[-1]))
        return shapes, [lhs.dtype] * num_shards, [lhs.device] * num_shards

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        lhs, rhs = input_shapes
        prefix = lhs[:-1]
        return prefix + (lhs[-1], rhs[-1])

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank_l = len(input_shapes[0])
        rank_r = len(input_shapes[1])
        prefix_rank = rank_l - 1
        prefix_factors = [f"p{i}" for i in range(prefix_rank)]
        in_l = prefix_factors + ["i"]
        in_r = prefix_factors + ["j"]
        out = prefix_factors + ["i", "j"]

        in_mapping_l = {i: [in_l[i]] for i in range(rank_l)}
        in_mapping_r = {i: [in_r[i]] for i in range(rank_r)}
        out_mapping = {i: [out[i]] for i in range(len(out))}

        return OpShardingRuleTemplate(
            [in_mapping_l, in_mapping_r], [out_mapping]
        ).instantiate(input_shapes, output_shapes)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP: (matmul(cotangent, rhs), matmul(transpose(cotangent), lhs))."""
        lhs, rhs = primals[0], primals[1]
        cotangent = cotangents[0]
        from ..ops.view.axes import swap_axes
        from . import matmul

        grad_lhs = matmul(cotangent, rhs)
        grad_rhs = matmul(swap_axes(cotangent, -2, -1), lhs)
        return [grad_lhs, grad_rhs]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP: outer(tangent_lhs, rhs) + outer(lhs, tangent_rhs)."""
        lhs, rhs = primals[0], primals[1]
        tl, tr = tangents[0], tangents[1]
        from . import add

        return [add(outer(tl, rhs), outer(lhs, tr))]


_mod_op = ModOp()
_pow_op = PowOp()
_outer_op = OuterOp()


def mod(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _mod_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def pow(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _pow_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def outer(x: Tensor, y: Tensor) -> Tensor:
    from .base import ensure_tensor

    return _outer_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


__all__ = [
    "add",
    "mul",
    "sub",
    "div",
    "matmul",
    "mod",
    "pow",
    "outer",
]
