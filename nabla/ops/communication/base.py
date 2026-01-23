# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import Operation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, ShardingSpec


class CollectiveOperation(Operation):
    """Base class for collective communication operations.

    Handles value hydration, graph execution (maxpr), and output wrapping/sharding update.
    """

    def maxpr_all(
        self,
        args: tuple,
        kwargs: dict,
        output_sharding: Any,
        mesh: Any,
        any_traced: bool,
        max_batch_dims: int,
        original_kwargs: dict | None = None,
    ) -> Any:
        from ...core import GRAPH, Tensor
        from ...core.sharding import spmd

        if not args:
            return None

        # Collective operations operate on the whole set of shards at once.
        # We assume the first argument is the sharded tensor.
        sharded_tensor = args[0]
        if isinstance(sharded_tensor, Tensor):
            sharded_tensor.hydrate()
            values = sharded_tensor.values
        else:
            # Handle list of values if passed directly (unlikely in tracing)
            values = sharded_tensor if isinstance(sharded_tensor, list) else [sharded_tensor]

        # Filter kwargs to match what maxpr expects (consistent with execute)
        maxpr_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("mesh", "reduce_axes")
        }

        with GRAPH.graph:
            result_values = self.maxpr(values, mesh=mesh, **maxpr_kwargs)

        # Re-infer output sharding if it was None (though rehydrate usually has it)
        if output_sharding is None and isinstance(sharded_tensor, Tensor):
            output_sharding = self._compute_output_spec(
                sharded_tensor, result_values, **kwargs
            )

        output = spmd.create_sharded_output(
            result_values,
            output_sharding,
            any_traced,
            max_batch_dims,
            mesh=mesh,
        )

        self._setup_output_refs(output, args, original_kwargs or kwargs, any_traced)
        return output

    def execute(self, sharded_tensor, **kwargs):
        from ...core import GRAPH, Tensor

        if not self._should_proceed(sharded_tensor):
            return sharded_tensor

        mesh = sharded_tensor.sharding.mesh if sharded_tensor.sharding else None

        with GRAPH.graph:
            sharded_tensor.hydrate()
            maxpr_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("mesh", "reduce_axes")
            }
            result_values = self.maxpr(sharded_tensor.values, mesh=mesh, **maxpr_kwargs)

        output_spec = self._compute_output_spec(sharded_tensor, result_values, **kwargs)

        output = Tensor._create_unsafe(
            values=result_values,
            traced=sharded_tensor.traced,
            batch_dims=sharded_tensor.batch_dims,
        )
        output.sharding = output_spec

        self._setup_output_refs(
            output, (sharded_tensor,), kwargs, sharded_tensor.traced
        )

        return output

    def _should_proceed(self, tensor):
        """Check if operation should proceed (has sharding and potentially multiple shards)."""
        if not tensor.sharding:
            return False
        if (tensor._values and len(tensor._values) > 1) or (
            tensor._storages and len(tensor._storages) > 1
        ):
            return True
        return False

    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Compute output sharding spec. Default: preserve input spec."""
        return input_tensor.sharding

    def communication_cost(
        self,
        input_specs: list[ShardingSpec],
        output_specs: list[ShardingSpec],
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        mesh: DeviceMesh,
    ) -> float:
        """Unified communication cost estimation."""
        if not input_shapes:
            return 0.0

        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        size_bytes = num_elements * 4

        axes = []
        if input_specs and input_specs[0]:
            for dim_spec in input_specs[0].dim_specs:
                axes.extend(dim_spec.axes)

        return self.estimate_cost(size_bytes, mesh, axes, input_specs, output_specs)

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: DeviceMesh,
        axes: list[str],
        input_specs: list[ShardingSpec] = None,
        output_specs: list[ShardingSpec] = None,
    ) -> float:
        """Estimate cost of the collective operation."""
        return 0.0

    def _group_shards_by_axes(self, shard_values, mesh, group_by_axes):
        """Group shards by coordinates on specific axes."""
        groups = {}
        for shard_idx, val in enumerate(shard_values):
            key_parts = []
            for axis_name in group_by_axes:
                key_parts.append(mesh.get_coordinate(shard_idx, axis_name))

            key = tuple(key_parts)
            if key not in groups:
                groups[key] = []
            groups[key].append((shard_idx, val))

        return groups
