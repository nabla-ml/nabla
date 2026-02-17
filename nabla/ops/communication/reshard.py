# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from .shard import create_replicated_spec, shard


def reshard_tensor(tensor, from_spec, to_spec, mesh):
    """Adapter for legacy spmd.reshard_tensor signature.

    Legacy: (tensor, from_spec, to_spec, mesh)
    New Shard: (tensor, mesh, dim_specs, replicated_axes=...)
    """
    return shard(
        tensor, mesh, to_spec.dim_specs, replicated_axes=to_spec.replicated_axes
    )


def reshard(*args, **kwargs):
    """Deprecated: Use ops.shard instead.

    This function delegates strictly to ops.shard, which now handles
    smart resharding transitions (AllGather/AllReduce) automatically.
    """
    return shard(*args, **kwargs)


__all__ = ["reshard", "reshard_tensor", "create_replicated_spec"]
