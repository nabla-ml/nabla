# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for ContextVar-based concurrency safety.

Verifies that ComputeGraph, config flags, module call depth, and graph epochs
are properly isolated across async tasks / threads via ContextVar.

NOTE: Python's asyncio.gather() shares the parent context across child tasks by
default. To get true per-task isolation we use contextvars.copy_context() to
run each task in a fresh copy of the context. This mirrors what a real async
server framework (e.g. Starlette, aiohttp) does for each incoming request.
"""

import asyncio
import contextvars
from contextvars import ContextVar

import numpy as np
import pytest

import nabla
import nabla.nn as nn
from nabla import config as nabla_config
from nabla.config import _EAGER_MAX_GRAPH, _TRACING, _VERIFY_EAGER_SHAPES
from nabla.core.graph.engine import (
    GRAPH,
    ComputeGraph,
    _CURRENT_GRAPH,
    _GRAPH_EPOCH,
    _GraphProxy,
    _get_current_graph,
)
from nabla.nn.modules.base import _MODULE_CALL_DEPTH


# ---------------------------------------------------------------------------
# Helpers — run coroutines in isolated contexts
# ---------------------------------------------------------------------------


def _run_in_fresh_context(coro_fn, *args):
    """Run an async function in a fresh copy of the current context.

    This simulates what real async frameworks do per-request: each handler
    runs in an isolated copy of the context so ContextVars don't leak.
    """
    ctx = contextvars.copy_context()
    return ctx.run(asyncio.run, coro_fn(*args))


async def _gather_isolated(*coros):
    """Like asyncio.gather but each coroutine runs in its own context copy.

    Standard asyncio.gather shares the parent context. This helper creates a
    fresh context copy per task and resets _CURRENT_GRAPH so each task lazily
    creates its own ComputeGraph — matching real-world server behaviour where
    each request handler starts with a clean context.
    """
    loop = asyncio.get_event_loop()
    tasks = []
    for coro in coros:
        ctx = contextvars.copy_context()
        # Reset graph state so each task gets its own ComputeGraph
        ctx.run(_CURRENT_GRAPH.set, None)
        tasks.append(loop.create_task(coro, context=ctx))
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Graph proxy basics
# ---------------------------------------------------------------------------


def test_graph_is_proxy():
    """GRAPH should be a _GraphProxy, not a bare ComputeGraph."""
    assert isinstance(GRAPH, _GraphProxy)


def test_graph_proxy_delegates_to_current_context():
    """Proxy attributes must match the underlying ComputeGraph."""
    underlying = _get_current_graph()
    assert isinstance(underlying, ComputeGraph)
    assert GRAPH.epoch == underlying.epoch
    assert GRAPH.graph is underlying.graph
    assert GRAPH.sources is underlying.sources


def test_lazy_graph_creation():
    """_get_current_graph() creates a ComputeGraph lazily on first access."""
    g = _get_current_graph()
    assert isinstance(g, ComputeGraph)
    assert g.epoch > 0


# ---------------------------------------------------------------------------
# _GRAPH_EPOCH
# ---------------------------------------------------------------------------


def test_graph_epoch_is_contextvar():
    """_GRAPH_EPOCH must be a ContextVar, not a plain int."""
    assert isinstance(_GRAPH_EPOCH, ContextVar)


# ---------------------------------------------------------------------------
# Config ContextVar reads / writes
# ---------------------------------------------------------------------------


def test_config_reads_via_module_getattr():
    """Reading nabla_config.X should return the ContextVar's current value."""
    assert nabla_config.EAGER_MAX_GRAPH == _EAGER_MAX_GRAPH.get()
    assert nabla_config.VERIFY_EAGER_SHAPES == _VERIFY_EAGER_SHAPES.get()
    assert nabla_config.TRACING == _TRACING.get()


def test_config_token_set_reset():
    """ContextVar token-based set/reset must restore the original value."""
    original = _TRACING.get()
    token = _TRACING.set(not original)
    assert nabla_config.TRACING == (not original)
    _TRACING.reset(token)
    assert nabla_config.TRACING == original


def test_config_context_managers_nested():
    """Nested context managers must scope config changes correctly."""
    from nabla.config import tracing_context

    assert not nabla_config.TRACING
    with tracing_context(True):
        assert nabla_config.TRACING
        with tracing_context(False):
            assert not nabla_config.TRACING
        assert nabla_config.TRACING
    assert not nabla_config.TRACING


# ---------------------------------------------------------------------------
# Module call depth
# ---------------------------------------------------------------------------


def test_module_call_depth_is_contextvar():
    """Module call depth tracker must be a ContextVar."""
    assert isinstance(_MODULE_CALL_DEPTH, ContextVar)


def test_module_call_depth_tracking():
    """Nested module calls must correctly increment/decrement depth."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self._AUTO_REALIZE_TOPLEVEL_FORWARD = False
            self.seen_depth = None

        def forward(self, x):
            self.seen_depth = _MODULE_CALL_DEPTH.get()
            return x

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self._AUTO_REALIZE_TOPLEVEL_FORWARD = False
            self.inner = Inner()
            self.seen_depth = None

        def forward(self, x):
            self.seen_depth = _MODULE_CALL_DEPTH.get()
            return self.inner(x)

    model = Outer()
    x = nabla.Tensor.constant([1.0])
    model(x)

    assert model.seen_depth == 1, f"Outer depth: expected 1, got {model.seen_depth}"
    assert model.inner.seen_depth == 2, (
        f"Inner depth: expected 2, got {model.inner.seen_depth}"
    )
    assert _MODULE_CALL_DEPTH.get() == 0, "Depth must be 0 after call completes"


# ---------------------------------------------------------------------------
# Basic ops through the proxy
# ---------------------------------------------------------------------------


def test_tensor_ops_through_proxy():
    """Basic tensor arithmetic must work through the GRAPH proxy."""
    x = nabla.Tensor.constant([1.0, 2.0, 3.0])
    y = nabla.Tensor.constant([10.0, 20.0, 30.0])
    z = x * y + x
    z.realize()
    np.testing.assert_allclose(z.to_numpy(), [11.0, 42.0, 93.0], atol=1e-5)


def test_gradient_through_proxy():
    """Gradient computation must work end-to-end through the proxy."""

    def f(x):
        return nabla.reduce_sum(x**3)

    df = nabla.grad(f)
    x = nabla.Tensor.constant([2.0, 3.0])
    grad_np = df(x).to_numpy()
    # d/dx(sum(x^3)) = 3*x^2 = [12.0, 27.0]
    np.testing.assert_allclose(grad_np, [12.0, 27.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Async isolation  (use _gather_isolated for true per-task context copies)
# ---------------------------------------------------------------------------


def test_async_graph_isolation():
    """Each async task must get its own isolated ComputeGraph."""

    graph_ids: dict[str, int] = {}

    async def task(name: str, val: float):
        g = _get_current_graph()
        graph_ids[name] = id(g)
        x = nabla.Tensor.constant([val])
        y = x * x
        y.realize()
        return y.to_numpy()[0]

    async def run():
        r = await _gather_isolated(task("A", 5.0), task("B", 7.0))
        return r

    results = asyncio.run(run())

    np.testing.assert_allclose(results[0], 25.0, atol=1e-5)
    np.testing.assert_allclose(results[1], 49.0, atol=1e-5)
    assert len(set(graph_ids.values())) == 2, (
        f"Expected 2 unique graphs, got {len(set(graph_ids.values()))}: {graph_ids}"
    )


def test_async_config_isolation():
    """Config changes in one async task must not leak to another."""

    results: dict[str, bool] = {}

    async def setter():
        token = _TRACING.set(True)
        await asyncio.sleep(0)
        results["setter"] = nabla_config.TRACING
        _TRACING.reset(token)

    async def reader():
        await asyncio.sleep(0)
        results["reader"] = nabla_config.TRACING

    async def run():
        _TRACING.set(False)
        await _gather_isolated(setter(), reader())

    asyncio.run(run())

    assert results["setter"] is True, "Setter should see TRACING=True"
    assert results["reader"] is False, "Reader should see TRACING=False (no leak)"


def test_async_module_depth_isolation():
    """Module call depth must not leak between async tasks."""

    depths: dict[str, int] = {}

    class Probe(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.probe_name = name
            self._AUTO_REALIZE_TOPLEVEL_FORWARD = False

        def forward(self, x):
            depths[self.probe_name] = _MODULE_CALL_DEPTH.get()
            return x

    class Wrapper(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.inner = Probe(name)
            self._AUTO_REALIZE_TOPLEVEL_FORWARD = False

        def forward(self, x):
            return self.inner(x)

    async def run_model(name):
        Wrapper(name)(nabla.Tensor.constant([1.0]))
        await asyncio.sleep(0)

    async def run():
        await _gather_isolated(run_model("A"), run_model("B"))

    asyncio.run(run())

    assert depths["A"] == 2, f"A depth: expected 2, got {depths['A']}"
    assert depths["B"] == 2, f"B depth: expected 2, got {depths['B']}"
    assert _MODULE_CALL_DEPTH.get() == 0, "Depth must be 0 after async tasks"


def test_async_concurrent_gradients():
    """Multiple async tasks can compute gradients concurrently."""

    async def compute_grad(values: list[float]) -> tuple[list[float], list[float]]:
        x = nabla.Tensor.constant(values)

        def f(x):
            return nabla.reduce_sum(x**2 + 2 * x)

        grad_np = nabla.grad(f)(x).to_numpy()
        return values, list(grad_np)

    async def run():
        return await _gather_isolated(
            compute_grad([1.0, 2.0, 3.0]),
            compute_grad([10.0, 20.0]),
            compute_grad([0.5, -1.0, 2.5, 4.0]),
        )

    results = asyncio.run(run())

    for values, actual_grad in results:
        expected_grad = [2 * v + 2 for v in values]
        np.testing.assert_allclose(
            actual_grad,
            expected_grad,
            atol=1e-4,
            err_msg=f"Gradient mismatch for input {values}",
        )
