"""
Nabla: A modular deep learning framework.

Import the clean, refactored implementation by default.
For backward compatibility, the original graph.py is still available.
"""

from max.dtype import DType

# Import from the clean modular implementation
from .nabla import *

# Also provide access to the original implementation if needed
# from . import graph as original_graph  # Original graph.py no longer exists

# Provide the clean implementation as graph_improved for test compatibility
from . import nabla as graph_improved

__version__ = "0.1.0"
