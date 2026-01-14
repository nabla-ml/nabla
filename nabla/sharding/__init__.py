"""Sharding infrastructure for distributed execution."""

from .spec import DeviceMesh, ShardingSpec, DimSpec, compute_local_shape, get_num_shards
from .propagation import OpShardingRule, OpShardingRuleTemplate, PropagationStrategy, OpPriority, FactorSharding
from .partition_spec import P, PartitionSpec
from .cost_model import allreduce_cost, allgather_cost, reduce_scatter_cost, resharding_cost
from . import spmd

__all__ = [
    # Core types
    "DeviceMesh",
    "ShardingSpec",
    "DimSpec",
    # Syntax helpers
    "P",
    "PartitionSpec",
    # Utilities
    "compute_local_shape",
    "get_num_shards",
    # Propagation
    "OpShardingRule",
    "OpShardingRuleTemplate",
    "PropagationStrategy",
    "OpPriority",
    "FactorSharding",
    # Cost Model
    "allreduce_cost",
    "allgather_cost",
    "reduce_scatter_cost",
    "resharding_cost",
    "spmd",
]
