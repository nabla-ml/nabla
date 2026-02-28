# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Global configuration for Nabla.

Configuration flags are stored as ``ContextVar`` instances so that
concurrent async tasks or threads each get isolated values.  Convenience
accessor properties (``EAGER_MAX_GRAPH``, ``VERIFY_EAGER_SHAPES``,
``TRACING``) are provided at module level for backwards-compatible
``nabla_config.EAGER_MAX_GRAPH`` reads.  Writes must go through the
``ContextVar.set()`` / ``ContextVar.reset(token)`` API, or through the
provided context manager helpers.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from contextvars import ContextVar

# --- ContextVar-backed configuration flags ---

_EAGER_MAX_GRAPH: ContextVar[bool] = ContextVar(
    "_EAGER_MAX_GRAPH",
    default=os.environ.get("EAGER_MAX_GRAPH", "0") == "1",
)

_VERIFY_EAGER_SHAPES: ContextVar[bool] = ContextVar(
    "_VERIFY_EAGER_SHAPES",
    default=os.environ.get("VERIFY_EAGER_SHAPES", "0") == "1",
)

_TRACING: ContextVar[bool] = ContextVar(
    "_TRACING",
    default=False,
)


# --- Module-level property shims for backward-compatible reads ---
#
# Existing code does ``from .. import config as nabla_config`` then
# ``if nabla_config.EAGER_MAX_GRAPH:``.  We support that pattern by
# making the module act as a property namespace via __getattr__ / __setattr__.

def __getattr__(name: str):
    _map = {
        "EAGER_MAX_GRAPH": _EAGER_MAX_GRAPH,
        "VERIFY_EAGER_SHAPES": _VERIFY_EAGER_SHAPES,
        "TRACING": _TRACING,
    }
    if name in _map:
        return _map[name].get()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# __setattr__ at module level isn't natively supported in Python < 3.7,
# but from 3.7+ it works via the same mechanism as __getattr__.
# However, for modules, __setattr__ is *not* called automatically.
# Instead, writes like ``nabla_config.EAGER_MAX_GRAPH = True`` bypass
# __setattr__ and go straight to the module __dict__.  We therefore
# need a different strategy: we intercept reads (via __getattr__) and
# provide explicit set functions for writes.

def set_eager_max_graph(value: bool) -> None:
    """Set EAGER_MAX_GRAPH for the current context."""
    _EAGER_MAX_GRAPH.set(value)

def set_verify_eager_shapes(value: bool) -> None:
    """Set VERIFY_EAGER_SHAPES for the current context."""
    _VERIFY_EAGER_SHAPES.set(value)

def set_tracing(value: bool) -> None:
    """Set TRACING for the current context."""
    _TRACING.set(value)


# --- Context manager helpers for scoped config changes ---

@contextlib.contextmanager
def eager_max_graph_context(value: bool) -> Generator[None, None, None]:
    """Temporarily set EAGER_MAX_GRAPH within a ``with`` block."""
    token = _EAGER_MAX_GRAPH.set(value)
    try:
        yield
    finally:
        _EAGER_MAX_GRAPH.reset(token)


@contextlib.contextmanager
def verify_eager_shapes_context(value: bool) -> Generator[None, None, None]:
    """Temporarily set VERIFY_EAGER_SHAPES within a ``with`` block."""
    token = _VERIFY_EAGER_SHAPES.set(value)
    try:
        yield
    finally:
        _VERIFY_EAGER_SHAPES.reset(token)


@contextlib.contextmanager
def tracing_context(value: bool) -> Generator[None, None, None]:
    """Temporarily set TRACING within a ``with`` block."""
    token = _TRACING.set(value)
    try:
        yield
    finally:
        _TRACING.reset(token)
