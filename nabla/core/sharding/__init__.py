# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from .spec import DeviceMesh, ShardingSpec, DimSpec, compute_local_shape, get_num_shards, P, PartitionSpec
from .propagation import OpShardingRule, OpShardingRuleTemplate, PropagationStrategy, OpPriority, FactorSharding
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
    "FactorSharding",
    "spmd",
]
