# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from max import graph
from max.graph import ops

from ..core import GRAPH, pytree
from .base import Operation, ensure_tensor

if TYPE_CHECKING:
    from ..core.sharding.spec import DeviceMesh
    from ..core.tensor import Tensor

if TYPE_CHECKING:
    from ..core.sharding.spec import DeviceMesh


def _unwrap_tensor(x: Any) -> Any:
    """Unwrap Tensor to TensorValue for MAX ops."""
    from ..core.tensor import Tensor

    if isinstance(x, Tensor):
        if not x._values:
            pass
        if hasattr(x, "_impl") and x._values:
            return x._values[0]
        return x
    return x


def _wrap_tensor(x: Any, like: Tensor | None = None) -> Tensor:
    """Wrap TensorValue from MAX op back to Tensor."""
    if isinstance(x, (graph.TensorValue, graph.BufferValue)):
        from ..core.tensor import Tensor

        return Tensor(values=[x])
    return x


class WhereOp(Operation):
    """Element-wise conditional selection: where(cond, x, y)."""

    def __call__(self, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
        from . import view as view_ops
        from .base import ensure_tensor

        condition = ensure_tensor(condition)
        x = ensure_tensor(x)
        y = ensure_tensor(y)

        # 1. Logical Broadcasting
        c_shape = tuple(int(d) for d in condition.shape)
        x_shape = tuple(int(d) for d in x.shape)
        y_shape = tuple(int(d) for d in y.shape)

        from .binary import mul  # Just to get access to _broadcast_shapes or similar

        # Actually BinaryOperation._broadcast_shapes is not static.
        def broadcast_shapes(*shapes):
            res = shapes[0]
            for s in shapes[1:]:
                # Simplified broadcast logic
                if len(res) < len(s):
                    res = (1,) * (len(s) - len(res)) + res
                if len(s) < len(res):
                    s = (1,) * (len(res) - len(s)) + s
                new_res = []
                for d1, d2 in zip(res, s):
                    if d1 == d2:
                        new_res.append(d1)
                    elif d1 == 1:
                        new_res.append(d2)
                    elif d2 == 1:
                        new_res.append(d1)
                    else:
                        raise ValueError(f"Incompatible shapes {res} and {s}")
                res = tuple(new_res)
            return res

        target_logical = broadcast_shapes(c_shape, x_shape, y_shape)

        if c_shape != target_logical:
            condition = view_ops.broadcast_to(condition, target_logical)
        if x_shape != target_logical:
            x = view_ops.broadcast_to(x, target_logical)
        if y_shape != target_logical:
            y = view_ops.broadcast_to(y, target_logical)

        # 2. Physical Broadcasting (Batch Dims)
        max_bd = max(condition.batch_dims, x.batch_dims, y.batch_dims)

        # Get batch shape
        batch_shape = ()
        for t in [condition, x, y]:
            if t.batch_dims == max_bd:
                global_phys = t.physical_global_shape or t.local_shape
                if global_phys:
                    batch_shape = tuple(int(d) for d in global_phys[:max_bd])
                    break

        target_physical = batch_shape + target_logical

        def align(t):
            t_phys = t.physical_global_shape or t.local_shape
            if t_phys is None or tuple(int(d) for d in t_phys) != target_physical:
                return view_ops.broadcast_to_physical(t, target_physical)
            return t

        condition = align(condition)
        x = align(x)
        y = align(y)

        return super().__call__(condition, x, y)

    @property
    def name(self) -> str:
        return "where"

    def maxpr(
        self, condition: graph.TensorValue, x: graph.TensorValue, y: graph.TensorValue
    ) -> graph.TensorValue:
        return ops.where(condition, x, y)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for where(cond, x, y): masked cotangent for x and y, None for cond."""
        from .creation import zeros_like
        from .control_flow import where

        condition, x, y = primals

        # grad_x is cotangent where condition is True, else 0
        grad_x = where(condition, cotangent, zeros_like(x))

        # grad_y is cotangent where condition is False, else 0 (or 0 where cond is True)
        grad_y = where(condition, zeros_like(y), cotangent)

        return (None, grad_x, grad_y)


class CondOp(Operation):
    @property
    def name(self) -> str:
        return "cond"

    def __call__(
        self,
        pred: Tensor | bool,
        true_fn: Callable[..., Any],
        false_fn: Callable[..., Any],
        *operands: Any,
    ) -> Any:
        from ..core.tensor import Tensor

        operands = pytree.tree_map(ensure_tensor, operands)
        if not isinstance(pred, Tensor):
            pred = ensure_tensor(pred)
        return super().__call__(pred, true_fn, false_fn, *operands)

    def infer_sharding_spec(self, args: tuple, mesh: DeviceMesh, kwargs: dict = None):
        """Cond: Output sharding is determined by operands/branches."""
        return None, [], False

    def maxpr(self, pred_shard, true_fn, false_fn, *operand_shards):
        def wrapped_fn(fn, input_tensors):
            return fn(*input_tensors)

        from ..core.tensor import Tensor

        wrapped_operand_shards = pytree.tree_map(
            lambda x: Tensor(value=x), operand_shards
        )

        def max_true_fn():
            res = true_fn(*wrapped_operand_shards)
            return pytree.tree_map(_unwrap_tensor, res)

        def max_false_fn():
            res = false_fn(*wrapped_operand_shards)
            return pytree.tree_map(_unwrap_tensor, res)

        from max.graph import Graph

        out_types = []
        with Graph("scratch"):
            scratch_res = max_true_fn()

            def extract_type(x):
                return x.type

            out_types = pytree.tree_map(extract_type, scratch_res)

        flat_out_types = pytree.tree_leaves(out_types)

        def flat_max_true_fn():
            r = max_true_fn()
            return pytree.tree_leaves(r)

        def flat_max_false_fn():
            r = max_false_fn()
            return pytree.tree_leaves(r)

        res_flat = ops.cond(
            pred_shard, flat_out_types, flat_max_true_fn, flat_max_false_fn
        )

        return pytree.tree_unflatten(pytree.tree_structure(scratch_res), res_flat)


class WhileLoopOp(Operation):
    @property
    def name(self) -> str:
        return "while_loop"

    def __call__(self, cond_fn: Callable, body_fn: Callable, init_val: Any) -> Any:
        from max import graph as g

        from ..core import pytree
        from ..core.sharding import spmd
        from ..core.tensor import Tensor

        args = (cond_fn, body_fn, init_val)

        leaves = pytree.tree_leaves(init_val)
        any_traced = any(x.traced for x in leaves if isinstance(x, Tensor))
        max_batch_dims = max(
            (x.batch_dims for x in leaves if isinstance(x, Tensor)), default=0
        )
        any_sharded = any(x.is_sharded for x in leaves if isinstance(x, Tensor))

        mesh = spmd.get_mesh_from_args(leaves) if any_sharded else None

        leaf_specs = []
        for x in leaves:
            if isinstance(x, Tensor) and x.sharding:
                leaf_specs.append(x.sharding)
            else:
                if mesh:
                    rank = len(x.shape) if isinstance(x, Tensor) else 0
                    if isinstance(x, Tensor):
                        rank = len(x.shape) + x.batch_dims
                        from ..core.sharding.spmd import create_replicated_spec

                        leaf_specs.append(create_replicated_spec(mesh, rank))
                    else:
                        leaf_specs.append(None)
                else:
                    leaf_specs.append(None)

        num_shards = len(mesh.devices) if mesh else 1
        shard_results = []

        with GRAPH.graph:
            for shard_idx in range(num_shards):
                shard_init_val = spmd.get_shard_args(
                    init_val, shard_idx, leaf_specs, g, Tensor, pytree
                )
                res = self.maxpr(cond_fn, body_fn, shard_init_val)
                shard_results.append(res)

        if not shard_results:
            return None

        flat_results_per_shard = [pytree.tree_leaves(res) for res in shard_results]
        treedef = pytree.tree_structure(shard_results[0])
        num_leaves = len(flat_results_per_shard[0])

        if num_leaves != len(leaf_specs):
            pass

        output_leaves = []
        for i in range(num_leaves):
            leaf_shards = [shard[i] for shard in flat_results_per_shard]
            spec = leaf_specs[i] if i < len(leaf_specs) else None

            tensor = spmd.create_sharded_output(
                leaf_shards, spec, any_traced, max_batch_dims, mesh=mesh
            )
            output_leaves.append(tensor)

        return pytree.tree_unflatten(treedef, output_leaves)

    def infer_sharding_spec(self, args: tuple, mesh: DeviceMesh, kwargs: dict = None):
        """Unused by custom __call__."""
        return None, [], False

    def maxpr(self, cond_fn, body_fn, *init_shards):
        init_val_shard = init_shards[0]

        from ..core.tensor import Tensor

        def wrap(x):
            return Tensor(value=x)

        def unwrap(x):
            return x._values[0] if isinstance(x, Tensor) else x

        def max_cond_fn(*args_flat):
            args_struct = pytree.tree_unflatten(
                pytree.tree_structure(init_val_shard), args_flat
            )
            wrapped_args = pytree.tree_map(wrap, args_struct)
            res = cond_fn(wrapped_args)
            return unwrap(res)

        def max_body_fn(*args_flat):
            args_struct = pytree.tree_unflatten(
                pytree.tree_structure(init_val_shard), args_flat
            )
            wrapped_args = pytree.tree_map(wrap, args_struct)
            res = body_fn(wrapped_args)
            return pytree.tree_leaves(pytree.tree_map(unwrap, res))

        init_flat = pytree.tree_leaves(init_val_shard)
        res_flat = ops.while_loop(init_flat, max_cond_fn, max_body_fn)

        return pytree.tree_unflatten(pytree.tree_structure(init_val_shard), res_flat)


class ScanOp(Operation):
    @property
    def name(self) -> str:
        return "scan"

    def maxpr(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "ScanOp is currently a macro and does not map to a single MAX op."
        )

    def __call__(
        self,
        f: Callable,
        init: Any,
        xs: Any,
        length: int | None = None,
        reverse: bool = False,
    ) -> tuple[Any, Any]:
        """Scan implementation using loop unrolling (MVP).

        Args:
            f: Function (carry, x) -> (carry, y)
            init: Initial carry value
            xs: Inputs with leading dimension being scanned
            length: Scan length (optional, inferred from xs)
            reverse: If True, scan in reverse order

        Returns:
            (final_carry, stacked_outputs)
        """
        if reverse:
            raise NotImplementedError("Reverse scan not implemented yet")

        xs_flat = pytree.tree_leaves(xs)
        if not xs_flat:
            raise ValueError("scan requires non-empty xs")

        first_x = xs_flat[0]
        inferred_length = int(first_x.shape[0])

        if length is not None and length != inferred_length:
            raise ValueError(
                f"Explicit length {length} != inferred length {inferred_length}"
            )
        length = inferred_length

        if length == 0:
            raise NotImplementedError("Zero-length scan not implemented")

        from ..ops import view

        carry = init
        ys_list = []

        for i in range(length):

            def _slice_at_i(x):
                local_shape = x.local_shape
                start = [i] + [0] * (x.rank - 1)
                size = [1] + [int(d) for d in local_shape[1:]]
                slc = view.slice_tensor(x, start=start, size=size)
                return view.squeeze(slc, 0)

            x_i = pytree.tree_map(_slice_at_i, xs)

            carry, y_i = f(carry, x_i)
            ys_list.append(y_i)

        def _stack_outputs(ys_leaves_list):

            return view.stack(ys_leaves_list, axis=0)

        if ys_list:

            first_y_flat = pytree.tree_leaves(ys_list[0])
            treedef = pytree.tree_structure(ys_list[0])
            num_leaves = len(first_y_flat)

            stacked_leaves = []
            for leaf_idx in range(num_leaves):
                leaves_for_this_idx = [pytree.tree_leaves(y)[leaf_idx] for y in ys_list]
                stacked = _stack_outputs(leaves_for_this_idx)
                stacked_leaves.append(stacked)

            stacked_ys = pytree.tree_unflatten(treedef, stacked_leaves)
        else:
            stacked_ys = None

        return carry, stacked_ys


_where_op = WhereOp()
_cond_op = CondOp()
_while_loop_op = WhileLoopOp()
_scan_op = ScanOp()


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return _where_op(condition, x, y)


def cond(pred: Tensor, true_fn: Callable, false_fn: Callable, *operands: Any) -> Any:
    return _cond_op(pred, true_fn, false_fn, *operands)


def while_loop(cond_fn: Callable, body_fn: Callable, init_val: Any) -> Any:
    return _while_loop_op(cond_fn, body_fn, init_val)


def scan(
    f: Callable, init: Any, xs: Any, length: int | None = None, reverse: bool = False
) -> tuple[Any, Any]:
    return _scan_op(f, init, xs, length=length, reverse=reverse)


__all__ = ["where", "cond", "while_loop", "scan"]
