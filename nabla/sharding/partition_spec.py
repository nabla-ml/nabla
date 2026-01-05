"""PartitionSpec helper for cleaner sharding syntax.

Provides JAX-like P() syntax as shorthand for DimSpec lists.

Example:
    # Instead of:
    x.shard(mesh, [DimSpec(["dp"]), DimSpec(["tp"]), DimSpec([])])
    
    # Use:
    x.shard(mesh, P("dp", "tp", None))
"""

from typing import List, Optional, Tuple, Union

from .spec import DimSpec

# Type alias for axis specification
AxisSpec = Optional[Union[str, Tuple[str, ...]]]


def P(*specs: AxisSpec) -> List[DimSpec]:
    """Create DimSpec list from PartitionSpec-style arguments.
    
    Args:
        *specs: One per tensor dimension. Each is:
            - None: replicated (no sharding on this dimension)
            - str: sharded on that single mesh axis  
            - tuple of str: sharded on multiple axes (major to minor)
    
    Returns:
        List[DimSpec] suitable for tensor.shard() or ShardingSpec
        
    Examples:
        >>> P("x", None)
        [DimSpec(axes=['x']), DimSpec(axes=[])]
        
        >>> P(("dp", "tp"), None)
        [DimSpec(axes=['dp', 'tp']), DimSpec(axes=[])]
        
        >>> P()  # For scalar or explicit empty
        []
    """
    result = []
    for spec in specs:
        if spec is None:
            result.append(DimSpec([]))
        elif isinstance(spec, str):
            result.append(DimSpec([spec]))
        elif isinstance(spec, tuple):
            result.append(DimSpec(list(spec)))
        else:
            raise TypeError(
                f"Invalid spec element: {spec!r}. "
                f"Expected None, str, or tuple of str."
            )
    return result


# Alias for documentation clarity
PartitionSpec = P

__all__ = ["P", "PartitionSpec"]
