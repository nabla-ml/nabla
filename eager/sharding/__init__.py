"""Sharding infrastructure for distributed execution."""

from .spec import DeviceMesh, ShardingSpec, DimSpec, compute_local_shape, get_num_shards
from .propagation import OpShardingRule, OpShardingRuleTemplate, PropagationStrategy, OpPriority, FactorSharding

__all__ = [
    "DeviceMesh",
    "ShardingSpec",
    "DimSpec",
    "compute_local_shape",
    "get_num_shards",
    "OpShardingRule",
    "OpShardingRuleTemplate",
    "PropagationStrategy",
    "OpPriority",
    "FactorSharding",
]
