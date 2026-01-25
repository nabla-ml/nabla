# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Automatic differentiation for Nabla."""

from .utils import backward_on_trace
from .api import grad, value_and_grad

__all__ = ["backward_on_trace", "grad", "value_and_grad"]
