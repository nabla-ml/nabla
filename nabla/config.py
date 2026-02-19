# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Global configuration for Nabla."""

import os

# Eager MAX Graph Building
# If True, operations build the MAX graph immediately during __call__.
# This allows for immediate shape verification and graph inspection,
# but execution (realizing values) is still deferred to evaluate().
# Note: evaluate() may rebuild the graph from trace for optimization.
EAGER_MAX_GRAPH: bool = os.environ.get("EAGER_MAX_GRAPH", "0") == "1"

# Verify Eager Shapes
# If True (and EAGER_MAX_GRAPH is True), verifies that computed physical shapes
# match the actual shapes produced by eager execution.
VERIFY_EAGER_SHAPES: bool = os.environ.get("VERIFY_EAGER_SHAPES", "0") == "1"

# Internal Tracing Flag
# Set to True during compile() or jit() trace to suppress side-effecting operations.
TRACING: bool = False
