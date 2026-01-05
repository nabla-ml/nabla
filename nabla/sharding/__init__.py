"""Sharding infrastructure for distributed execution."""

from .spec import DeviceMesh, ShardingSpec, DimSpec, compute_local_shape, get_num_shards
from .propagation import OpShardingRule, OpShardingRuleTemplate, PropagationStrategy, OpPriority, FactorSharding
from .partition_spec import P, PartitionSpec
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
    "spmd",
]

