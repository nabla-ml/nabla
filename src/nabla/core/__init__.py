"""Core components of the Nabla framework."""

from .array import Array
from .execution_context import ThreadSafeExecutionContext, global_execution_context

# Import graph execution functions - handle potential import issues gracefully
try:
    from .graph_execution import realize_, get_trace, compute_node_hash
except ImportError:
    # If there are import issues, we'll import these later
    pass

__all__ = ["Array", "ThreadSafeExecutionContext", "global_execution_context"]
