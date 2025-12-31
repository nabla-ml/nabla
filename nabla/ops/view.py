# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Logical View Operations
# ===----------------------------------------------------------------------=== #

"""Logical view operations.

These operations work on the LOGICAL shape (user's view).
Integer arguments are translated by adding batch_dims offset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from .operation import LogicalAxisOperation, LogicalShapeOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


class UnsqueezeOp(LogicalAxisOperation):
    axis_offset_for_insert = True
    
    @property
    def name(self) -> str:
        return "unsqueeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Unsqueeze: insert new dimension at axis position."""
        from ..sharding.propagation import unsqueeze_template
        in_rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        return unsqueeze_template(in_rank, axis).instantiate(input_shapes, output_shapes)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Insert new dimension of size 1 at axis position."""
        in_shape = input_shapes[0]
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis = len(in_shape) + 1 + axis
        return in_shape[:axis] + (1,) + in_shape[axis:]


class SqueezeOp(LogicalAxisOperation):
    axis_offset_for_insert = False
    
    @property
    def name(self) -> str:
        return "squeeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Squeeze: remove dimension at axis position."""
        from ..sharding.propagation import squeeze_template
        in_rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        return squeeze_template(in_rank, axis).instantiate(input_shapes, output_shapes)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Remove dimension of size 1 at axis position."""
        in_shape = input_shapes[0]
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis = len(in_shape) + axis
        return in_shape[:axis] + in_shape[axis+1:]


class SwapAxesOp(LogicalAxisOperation):
    @property
    def name(self) -> str:
        return "swap_axes"
    
    def maxpr(self, x: TensorValue, *, axis1: int, axis2: int) -> TensorValue:
        return ops.transpose(x, axis1, axis2)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """SwapAxes: swap two dimensions and their shardings."""
        from ..sharding.propagation import swap_axes_template
        in_rank = len(input_shapes[0])
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        return swap_axes_template(in_rank, axis1, axis2).instantiate(input_shapes, output_shapes)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Swap dimensions at axis1 and axis2."""
        in_shape = list(input_shapes[0])
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        if axis1 < 0:
            axis1 = len(in_shape) + axis1
        if axis2 < 0:
            axis2 = len(in_shape) + axis2
        in_shape[axis1], in_shape[axis2] = in_shape[axis2], in_shape[axis1]
        return tuple(in_shape)


class BroadcastToOp(LogicalShapeOperation):
    @property
    def name(self) -> str:
        return "broadcast_to"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Broadcast: input dims align to output SUFFIX (numpy semantics).
        
        Uses shape-aware template to handle dimension expansion (size 1 -> N).
        """
        from ..sharding.propagation import broadcast_with_shapes_template
        in_shape = input_shapes[0]
        out_shape = output_shapes[0]
        return broadcast_with_shapes_template(in_shape, out_shape).instantiate(input_shapes, output_shapes)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Output shape is the target shape."""
        return kwargs.get("shape", input_shapes[0])
    
    def _call_spmd(self, *args, **kwargs):
        """SPMD execution for broadcast_to: compute local target shape per shard.
        
        Like ReshapeOp, broadcast_to needs to compute the LOCAL target shape
        based on the output sharding.
        """
        from ..core.tensor import Tensor
        from ..core.compute_graph import GRAPH
        from ..core import pytree
        from ..sharding.spec import ShardingSpec, DimSpec, compute_local_shape
        from ..sharding import spmd
        from max import graph as g
        
        mesh = spmd.get_mesh_from_args(args)
        if mesh is None:
            raise ValueError("SPMD dispatch called but no mesh found")
        
        num_shards = len(mesh.devices)
        global_target_shape = kwargs.get('shape')
        
        # Annotate unsharded inputs as replicated
        def ensure_sharding(x):
            if isinstance(x, Tensor) and x._impl.sharding is None:
                rank = len(x.shape)
                x._impl.sharding = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
            return x
        
        args = pytree.tree_map(ensure_sharding, args)
        
        # Collect metadata
        traced, batch_dims = self._collect_input_metadata(args)
        
        # Infer output sharding using propagation
        output_sharding, input_shardings, needs_allreduce = spmd.infer_output_sharding(self, args, mesh, kwargs)
        
        # Compute LOCAL target shape based on output sharding
        # If output is replicated (None sharding), local shape = global shape
        if output_sharding is None:
            local_target_shape = global_target_shape
        else:
            local_target_shape = compute_local_shape(global_target_shape, output_sharding, device_id=0)
        
        # Execute per-shard with LOCAL target shape
        with GRAPH.graph:
            shard_results = []
            for shard_idx in range(num_shards):
                shard_args = spmd.get_shard_args(
                    args, shard_idx, input_shardings, g, Tensor, pytree
                )
                # Use LOCAL target shape for each shard
                local_kwargs = dict(kwargs)
                local_kwargs['shape'] = local_target_shape
                shard_results.append(self.maxpr(*shard_args, **local_kwargs))
            
            if needs_allreduce and len(shard_results) > 1:
                from .communication import all_reduce_op
                shard_results = all_reduce_op.maxpr(shard_results, reduction="sum")
        
        # Create output
        output = spmd.create_sharded_output(
            shard_results, output_sharding, traced, batch_dims, mesh=mesh
        )
        
        return self._check_and_reshard_output(output, output_sharding, kwargs)


class ReshapeOp(LogicalShapeOperation):
    @property
    def name(self) -> str:
        return "reshape"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.reshape(x, shape)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Output shape is the target shape."""
        return kwargs.get("shape")

    def sharding_rule(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]], **kwargs):
        """Create sharding rule for reshape using compound factors."""
        from ..sharding.propagation import reshape_template
        
        target_shape = kwargs.get('shape')
        if target_shape is None:
            return None
            
        return reshape_template(input_shapes[0], target_shape).instantiate(input_shapes, output_shapes)
    
    def _call_spmd(self, *args, **kwargs):
        """SPMD execution for reshape: compute local target shape per shard.
        
        The key insight: when reshaping a sharded tensor, each shard has a 
        LOCAL shape that differs from the GLOBAL shape. We need to compute
        the LOCAL target shape based on the output sharding.
        
        Example:
            Global: (8, 4) sharded on dim 0 with 2 shards
            Local per shard: (4, 4)
            Target global: (2, 4, 4)
            Target output sharding: sharded on dim 0
            Target local per shard: (1, 4, 4)
        """
        from ..core.tensor import Tensor
        from ..core.compute_graph import GRAPH
        from ..core import pytree
        from ..sharding.spec import ShardingSpec, DimSpec, compute_local_shape
        from ..sharding import spmd
        from max import graph as g
        
        mesh = spmd.get_mesh_from_args(args)
        if mesh is None:
            raise ValueError("SPMD dispatch called but no mesh found")
        
        num_shards = len(mesh.devices)
        global_target_shape = kwargs.get('shape')
        
        # Annotate unsharded inputs as replicated
        def ensure_sharding(x):
            if isinstance(x, Tensor) and x._impl.sharding is None:
                rank = len(x.shape)
                x._impl.sharding = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
            return x
        
        args = pytree.tree_map(ensure_sharding, args)
        
        # Collect metadata
        traced, batch_dims = self._collect_input_metadata(args)
        
        # Infer output sharding using propagation
        output_sharding, input_shardings, needs_allreduce = spmd.infer_output_sharding(self, args, mesh, kwargs)
        
        # Compute LOCAL target shape based on output sharding
        # If output is replicated (None sharding), local shape = global shape
        if output_sharding is None:
            local_target_shape = global_target_shape
        else:
            local_target_shape = compute_local_shape(global_target_shape, output_sharding, device_id=0)
        
        # Execute per-shard with LOCAL target shape
        with GRAPH.graph:
            shard_results = []
            for shard_idx in range(num_shards):
                shard_args = spmd.get_shard_args(
                    args, shard_idx, input_shardings, g, Tensor, pytree
                )
                # Use LOCAL target shape for each shard
                local_kwargs = dict(kwargs)
                local_kwargs['shape'] = local_target_shape
                shard_results.append(self.maxpr(*shard_args, **local_kwargs))
            
            # AllReduce if needed (unlikely for reshape but keep for consistency)
            if needs_allreduce and len(shard_results) > 1:
                from .communication import all_reduce_op
                shard_results = all_reduce_op.maxpr(shard_results, reduction="sum")
        
        # Create output
        output = spmd.create_sharded_output(
            shard_results, output_sharding, traced, batch_dims, mesh=mesh
        )
        
        return self._check_and_reshard_output(output, output_sharding, kwargs)


_unsqueeze_op = UnsqueezeOp()
_squeeze_op = SqueezeOp()
_swap_axes_op = SwapAxesOp()
_broadcast_to_op = BroadcastToOp()
_reshape_op = ReshapeOp()


def unsqueeze(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_op(x, axis=axis)

def squeeze(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_op(x, axis=axis)

def swap_axes(x: Tensor, axis1: int, axis2: int) -> Tensor:
    return _swap_axes_op(x, axis1=axis1, axis2=axis2)

def broadcast_to(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _broadcast_to_op(x, shape=shape)

def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _reshape_op(x, shape=shape)


__all__ = [
    "UnsqueezeOp", "SqueezeOp", "SwapAxesOp", "BroadcastToOp", "ReshapeOp",
    "unsqueeze", "squeeze", "swap_axes", "broadcast_to", "reshape",
]
