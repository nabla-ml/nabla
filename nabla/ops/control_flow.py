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
from .base import OpArgs, OpKwargs, OpResult, OpTensorValues, Operation, ensure_tensor

if TYPE_CHECKING:
    from ..core.sharding.spec import DeviceMesh
    from ..core.tensor import Tensor


class WhereOp(Operation):
    """Element-wise conditional selection: where(cond, x, y)."""

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        from . import view as view_ops
        from .base import ensure_tensor

        condition = ensure_tensor(args[0])
        x = ensure_tensor(args[1])
        y = ensure_tensor(args[2])

        # 1. Logical Broadcasting (Ported from original)
        c_shape = tuple(int(d) for d in condition.shape)
        x_shape = tuple(int(d) for d in x.shape)
        y_shape = tuple(int(d) for d in y.shape)

        def broadcast_shapes(*shapes):
            res = shapes[0]
            for s in shapes[1:]:
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

        # 2. Physical Alignment (Ported from original)
        max_bd = max(condition.batch_dims, x.batch_dims, y.batch_dims)
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

        return super().__call__([condition, x, y], kwargs)

    @property
    def name(self) -> str:
        return "where"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ):
        from ..core.sharding import spmd

        condition, x, y = args
        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                shapes.append(tuple(int(d) for d in s))
            else:
                # Fallback: use global shape
                shapes.append(tuple(int(d) for d in x.shape))

        dtypes = [x.dtype] * num_shards
        devices = [x.device] * num_shards if not mesh else list(mesh.device_refs)

        return shapes, dtypes, devices

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        condition, x, y = args[0], args[1], args[2]
        return [ops.where(condition, x, y)]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from .creation import zeros_like
        from .control_flow import where

        condition, x, y = primals[0], primals[1], primals[2]
        grad_x = where(condition, cotangents[0], zeros_like(x))
        grad_y = where(condition, zeros_like(y), cotangents[0])
        return [None, grad_x, grad_y]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from .control_flow import where

        condition = primals[0]
        t_x = tangents[1]
        t_y = tangents[2]
        return [where(condition, t_x, t_y)]


class CondOp(Operation):
    @property
    def name(self) -> str:
        return "cond"

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        from ..core.tensor import Tensor

        pred = args[0]
        true_fn = args[1]
        false_fn = args[2]
        operands = args[3:]

        operands = pytree.tree_map(ensure_tensor, operands)
        if not isinstance(pred, Tensor):
            pred = ensure_tensor(pred)
        return super().__call__([pred, true_fn, false_fn] + list(operands), kwargs)

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ):
        """Infer output shapes by tracing true_fn symbolically (no graph building).

        Since cond requires both branches to return identical shapes/dtypes,
        we only need to trace one branch. The trace creates promise tensors
        with shape metadata - we read that metadata directly without building
        any MAX graph operations.
        """
        from ..core.sharding import spmd
        from ..core.tensor import Tensor

        pred, true_fn, false_fn, *operands = args
        mesh = spmd.get_mesh_from_args(operands)
        num_shards = len(mesh.devices) if mesh else 1

        # Call true_fn to get output structure with shape metadata.
        # This creates promise tensors via our lazy tracing system.
        # We do NOT call _replay_trace_to_build_graph - just read the metadata.
        res = true_fn(*operands)
        flat_res, _ = pytree.tree_flatten(res, is_leaf=lambda x: isinstance(x, Tensor))

        all_shapes, all_dtypes, all_devices = [], [], []
        for leaf in flat_res:
            if not isinstance(leaf, Tensor):
                # Non-tensor output (rare, but handle it)
                all_shapes.append([()])
                all_dtypes.append([None])
                all_devices.append([None])
                continue

            # Read shape from the tensor's physical metadata (set during __call__)
            leaf_shapes = leaf._impl._physical_shapes
            leaf_dtypes = leaf._impl._shard_dtypes
            leaf_devices = leaf._impl._shard_devices

            if leaf_shapes and leaf_dtypes:
                all_shapes.append(leaf_shapes)
                all_dtypes.append(leaf_dtypes)
                all_devices.append(leaf_devices or [None] * num_shards)
            else:
                # Fallback: use logical shape replicated across shards
                shape = tuple(int(d) for d in leaf.shape)
                all_shapes.append([shape] * num_shards)
                all_dtypes.append([leaf.dtype] * num_shards)
                all_devices.append([None] * num_shards)

        if len(flat_res) == 1:
            return all_shapes[0], all_dtypes[0], all_devices[0]
        return all_shapes, all_dtypes, all_devices

    def infer_sharding_spec(self, args: list, mesh, kwargs: dict = None):

        # Trace true_fn to get output structure
        res = true_fn(*operands)
        flat_res = pytree.tree_leaves(res)

        # Collect specs from output tensors
        output_specs = []
        for leaf in flat_res:
            if isinstance(leaf, Tensor) and leaf.sharding:
                output_specs.append(leaf.sharding.clone())
            else:
                output_specs.append(None)

        # Input specs: just for operands (pred and fns don't have sharding)
        input_specs = [None, None, None]  # pred, true_fn, false_fn
        for op in operands:
            if isinstance(op, Tensor) and op.sharding:
                input_specs.append(op.sharding.clone())
            else:
                input_specs.append(None)

        if len(output_specs) == 1:
            return output_specs[0], input_specs, False
        return output_specs, input_specs, False

    def sharding_rule(self, input_shapes, output_shapes, **kwargs):
        from ..core.sharding.propagation import OpShardingRuleTemplate

        mappings = [
            {j: [f"t0_d{j}"] for j in range(len(shape))} for shape in input_shapes
        ]
        return OpShardingRuleTemplate(mappings, mappings).instantiate(
            input_shapes, output_shapes
        )

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        from ..core.tensor import Tensor

        pred_shard = args[0]
        true_fn = args[1]
        false_fn = args[2]
        operand_shards = args[3:]

        def wrap(v):
            if not hasattr(v, "type"):
                return v
            t = Tensor(value=v)
            t._impl.graph_values_epoch = GRAPH.epoch
            return t

        wrapped_operands = [wrap(v) for v in operand_shards]

        def trace_and_replay(fn):
            res = fn(*wrapped_operands)
            flat_res, _ = pytree.tree_flatten(
                res, is_leaf=lambda x: isinstance(x, Tensor)
            )
            GRAPH._replay_trace_to_build_graph(flat_res)
            return [
                t._impl.primary_value if isinstance(t, Tensor) else t for t in flat_res
            ]

        # Get output structure and types by tracing true_fn.
        # We DON'T call _replay_trace_to_build_graph here - just read metadata.
        # The actual graph building happens inside trace_and_replay when ops.cond calls the lambdas.
        scratch_res = true_fn(*wrapped_operands)
        out_structure = pytree.tree_structure(scratch_res)
        flat_scratch = pytree.tree_flatten(
            scratch_res, is_leaf=lambda x: isinstance(x, Tensor)
        )[0]

        # Build out_types from the promise tensors' metadata (shapes/dtypes set during __call__)
        out_types = []
        for t in flat_scratch:
            if isinstance(t, Tensor):
                # Use physical shape from the traced tensor
                phys_shapes = t._impl._physical_shapes
                if phys_shapes:
                    shape = phys_shapes[0]  # Use first shard's shape
                else:
                    shape = tuple(int(d) for d in t.shape)
                # Get device - use tensor's device or default to CPU
                device = t.device if t.device else graph.DeviceRef.CPU()
                out_types.append(graph.TensorType(t.dtype, shape, device))
            else:
                out_types.append(graph.Type(type(t), ()))

        res_flat = ops.cond(
            pred_shard,
            out_types,
            lambda: trace_and_replay(true_fn),
            lambda: trace_and_replay(false_fn),
        )
        result = pytree.tree_unflatten(out_structure, res_flat)
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]


class WhileLoopOp(Operation):
    @property
    def name(self) -> str:
        return "while_loop"

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        cond_fn = args[0]
        body_fn = args[1]
        init_val = args[2]
        return super().__call__([cond_fn, body_fn, init_val], kwargs)

    def infer_sharding_spec(self, args: list, mesh, kwargs: dict = None):
        """While loop: output sharding matches init_val sharding."""
        from ..core.tensor import Tensor

        cond_fn, body_fn, init_val = args
        flat_init = pytree.tree_leaves(init_val)

        # Collect specs from init tensors
        output_specs = []
        input_specs = []
        for leaf in flat_init:
            if isinstance(leaf, Tensor) and leaf.sharding:
                output_specs.append(leaf.sharding.clone())
                input_specs.append(leaf.sharding.clone())
            else:
                output_specs.append(None)
                input_specs.append(None)

        # Single output: return single spec, multi: return list
        if len(output_specs) == 1:
            return output_specs[0], input_specs, False
        return output_specs, input_specs, False

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ):
        from ..core.tensor import Tensor
        from ..core.sharding import spmd

        cond_fn, body_fn, init_val = args
        flat_init, _ = pytree.tree_flatten(
            init_val, is_leaf=lambda x: isinstance(x, Tensor)
        )

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        all_shapes, all_dtypes, all_devices = [], [], []
        for leaf in flat_init:
            if isinstance(leaf, Tensor):
                phys_shapes = leaf._impl._physical_shapes
                shard_dtypes = leaf._impl._shard_dtypes
                shard_devices = leaf._impl._shard_devices

                # Handle replication for scalar/unsharded inputs on mesh
                if phys_shapes and len(phys_shapes) == 1 and num_shards > 1:
                    phys_shapes = phys_shapes * num_shards
                    shard_dtypes = shard_dtypes * num_shards
                    shard_devices = (
                        shard_devices * num_shards
                        if shard_devices
                        else [None] * num_shards
                    )

                all_shapes.append(phys_shapes)
                all_dtypes.append(shard_dtypes)
                all_devices.append(shard_devices)
            else:
                all_shapes.append([(1,)] * num_shards)
                all_dtypes.append([None] * num_shards)
                all_devices.append([None] * num_shards)

        if len(flat_init) == 1:
            return all_shapes[0], all_dtypes[0], all_devices[0]
        return all_shapes, all_dtypes, all_devices

    def sharding_rule(self, input_shapes, output_shapes, **kwargs):
        from ..core.sharding.propagation import OpShardingRuleTemplate

        mappings = [
            {j: [f"t0_d{j}"] for j in range(len(shape))} for shape in input_shapes
        ]
        return OpShardingRuleTemplate(mappings, mappings).instantiate(
            input_shapes, output_shapes
        )

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        from ..core.tensor import Tensor

        cond_fn = args[0]
        body_fn = args[1]
        init_val_shard = args[2]

        def wrap(v):
            if not hasattr(v, "type"):
                return v
            t = Tensor(value=v)
            t._impl.graph_values_epoch = GRAPH.epoch
            return t

        def unwrap(t):
            return t._impl.primary_value if isinstance(t, Tensor) else t

        # Get the structure of init_val_shard for later reconstruction
        init_structure = pytree.tree_structure(init_val_shard)
        init_flat = pytree.tree_leaves(init_val_shard)

        def trace_and_replay_cond(*args_flat):
            # MAX passes args unpacked, we need to reconstruct the structure
            args_struct = pytree.tree_unflatten(init_structure, list(args_flat))
            wrapped_args = pytree.tree_map(wrap, args_struct)
            res = cond_fn(wrapped_args)
            if isinstance(res, Tensor):
                GRAPH._replay_trace_to_build_graph([res])
            return unwrap(res)

        def trace_and_replay_body(*args_flat):
            # MAX passes args unpacked, we need to reconstruct the structure
            args_struct = pytree.tree_unflatten(init_structure, list(args_flat))
            wrapped_args = pytree.tree_map(wrap, args_struct)
            res = body_fn(wrapped_args)
            flat_res, _ = pytree.tree_flatten(
                res, is_leaf=lambda x: isinstance(x, Tensor)
            )
            tensors = [t for t in flat_res if isinstance(t, Tensor)]
            if tensors:
                GRAPH._replay_trace_to_build_graph(tensors)
            return [unwrap(t) for t in flat_res]

        res_flat = ops.while_loop(
            init_flat, trace_and_replay_cond, trace_and_replay_body
        )
        result = pytree.tree_unflatten(init_structure, res_flat)
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]


class ScanOp(Operation):
    @property
    def name(self) -> str:
        return "scan"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        return None, None, None

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        raise NotImplementedError("ScanOp is unrolled via __call__.")

    def __call__(
        self,
        args: OpArgs,
        kwargs: OpKwargs,
    ) -> OpResult:
        f = args[0]
        init = args[1]
        xs = args[2]
        length = kwargs.get("length", None)
        reverse = kwargs.get("reverse", False)

        if reverse:
            raise NotImplementedError("Reverse scan not implemented")
        from ..ops import view

        xs_flat = pytree.tree_leaves(xs)
        if not xs_flat:
            raise ValueError("scan requires non-empty xs")
        length = length or int(xs_flat[0].shape[0])

        carry = init
        ys_list = []
        for i in range(length):

            def _slice_at_i(x):
                slc = view.slice_tensor(
                    x,
                    start=[i] + [0] * (x.rank - 1),
                    size=[1] + [int(d) for d in x.shape[1:]],
                )
                return view.squeeze(slc, 0)

            x_i = pytree.tree_map(_slice_at_i, xs)
            carry, y_i = f(carry, x_i)
            ys_list.append(y_i)

        if not ys_list:
            return [carry, None]
        treedef = pytree.tree_structure(ys_list[0])
        stacked_leaves = [
            view.stack([pytree.tree_leaves(y)[j] for y in ys_list], axis=0)
            for j in range(len(pytree.tree_leaves(ys_list[0])))
        ]
        return [carry, pytree.tree_unflatten(treedef, stacked_leaves)]


_where_op = WhereOp()
_cond_op = CondOp()
_while_loop_op = WhileLoopOp()
_scan_op = ScanOp()


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return _where_op([condition, x, y], {})[0]


def cond(pred: Tensor, true_fn: Callable[..., Any], false_fn: Callable[..., Any], *operands: Any) -> Any:
    result = _cond_op([pred, true_fn, false_fn] + list(operands), {})
    if len(result) == 1:
        return result[0]
    return tuple(result)


def while_loop(cond_fn: Callable[..., bool], body_fn: Callable[..., Any], init_val: Any) -> Any:
    result = _while_loop_op([cond_fn, body_fn, init_val], {})
    if len(result) == 1:
        return result[0]
    return tuple(result)


def scan(
    f: Callable[[Any, Any], tuple[Any, Any]], init: Any, xs: Any, length: int | None = None, reverse: bool = False
) -> tuple[Any, Any]:
    return _scan_op([f, init, xs], {"length": length, "reverse": reverse})


__all__ = ["where", "cond", "while_loop", "scan"]
