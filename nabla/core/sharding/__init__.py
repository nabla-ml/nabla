# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from . import spmd
from .propagation import (
    OpShardingRule,
    OpShardingRuleTemplate,
    infer_from_rule,
    propagate_sharding,
)
from .spec import (
    DeviceMesh,
    DimSpec,
    P,
    PartitionSpec,
    ShardingSpec,
    compute_local_shape,
    get_num_shards,
)

__all__ = [
    "DeviceMesh",
    "ShardingSpec",
    "DimSpec",
    "P",
    "PartitionSpec",
    "compute_local_shape",
    "get_num_shards",
    "OpShardingRule",
    "OpShardingRuleTemplate",
    "infer_from_rule",
    "propagate_sharding",
    "spmd",
]
