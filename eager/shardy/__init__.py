"""
Shardy: A Factor-based Sharding Representation and Propagation System
=====================================================================

Exposes the core data structures and propagation algorithms.
"""

from .core import (
    DeviceMesh,
    DimSpec,
    ShardingSpec,
    DistributedTensor,
    GraphTensor,
    # Helper functions useful for debugging or advanced mesh setup
    parse_sub_axis,
    validate_sub_axes_non_overlapping,
    check_sub_axes_maximality
)

from .propagation import (
    # Enums
    OpPriority,
    PropagationStrategy,
    
    # Intermediate Representations
    FactorSharding,
    FactorShardingState,
    
    # Rule System
    OpShardingRule,
    OpShardingRuleTemplate,
    
    # Graph & Pass
    Operation,
    DataFlowEdge,
    ShardingPass,
    propagate_sharding,
    
    # Rule Templates
    matmul_template,
    elementwise_template,
    unary_template,
    transpose_template,
    reduce_template,
    gather_template,
    attention_template,
    embedding_template,
)

__all__ = [
    "DeviceMesh",
    "DimSpec",
    "ShardingSpec",
    "DistributedTensor",
    "GraphTensor",
    "OpPriority",
    "PropagationStrategy",
    "FactorSharding",
    "FactorShardingState",
    "OpShardingRule",
    "OpShardingRuleTemplate",
    "Operation",
    "DataFlowEdge",
    "ShardingPass",
    "propagate_sharding",
    "matmul_template",
    "elementwise_template",
    "unary_template",
    "transpose_template",
    "reduce_template",
    "gather_template",
    "attention_template",
    "embedding_template",
]