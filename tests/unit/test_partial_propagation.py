"""Diagnostic: Partial-tensor propagation through the SPMD sharding system.

Explores whether nabla correctly handles the "distributive property"
optimization — i.e., linear ops applied to Partial tensors should NOT
trigger an AllReduce, while non-linear ops SHOULD.

Run:
    python tests/unit/test_partial_propagation.py
"""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType

import nabla as nb
from nabla.core import trace
from nabla.core.sharding.spec import DeviceMesh, DimSpec


def header(title: str) -> None:
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Setup: 2-device mesh simulated on CPU
# ---------------------------------------------------------------------------
mesh = DeviceMesh("tp", shape=(2,), axis_names=("tp",))


def make_tensor(shape: tuple[int, ...], seed: int = 42) -> nb.Tensor:
    """Create a nabla Tensor from random numpy data."""
    rng = np.random.RandomState(seed)
    data = rng.randn(*shape).astype(np.float32)
    return nb.Tensor.constant(data, dtype=DType.float32)


def shard_with_specs(t: nb.Tensor, dim_specs: list[DimSpec]) -> nb.Tensor:
    """Shard tensor with explicit DimSpecs."""
    return nb.shard(t, mesh=mesh, dim_specs=dim_specs)


def replicated(t: nb.Tensor) -> nb.Tensor:
    """Make a tensor fully replicated on the mesh."""
    rank = len(t.shape)
    return nb.shard(t, mesh=mesh, dim_specs=[DimSpec([], is_open=True) for _ in range(rank)])


# ---------------------------------------------------------------------------
# Scenario A: matmul with k-dim sharded → Partial output → linear op (mul)
#
# X: (4, 8) sharded on dim 1 (the k-dim) across tp
# W: (8, 4) replicated
# Result of matmul: Partial (each shard has partial product)
# Then: mul by scalar 2.0 — should NOT trigger AllReduce
# ---------------------------------------------------------------------------
header("A: matmul(X[k sharded], W[replicated]) → mul(result, 2.0)")
print("Expected: Partial propagates through mul, NO all_reduce in graph")

X_a = shard_with_specs(
    make_tensor((4, 8), seed=1),
    [DimSpec([], is_open=False), DimSpec(["tp"], is_open=False)],  # shard k on tp
)
@pytest.fixture(scope="module")
def mesh_2d() -> DeviceMesh:
    return DeviceMesh("2d", shape=(2, 2), axis_names=("tp", "dp"))


W_a = shard_with_specs(
    make_tensor((8, 4), seed=2),
    [DimSpec([], is_open=False), DimSpec([], is_open=False)],  # replicated
)
scalar = replicated(make_tensor((1,), seed=3))


def scenario_a(x, w, s):
    y = nb.matmul(x, w)  # → Partial (k contracted, was sharded on tp)
    return nb.mul(y, s)


tr_a = trace(scenario_a, X_a, W_a, scalar)
print(tr_a)

# ---------------------------------------------------------------------------
# Scenario B: matmul → relu (NON-LINEAR — should force AllReduce first!)
# ---------------------------------------------------------------------------
header("B: matmul(X[k sharded], W[replicated]) → relu(result)")
print("Expected: all_reduce BEFORE relu (currently MISSING — this is the bug)")


def scenario_b(x, w):
    y = nb.matmul(x, w)  # → Partial
    return nb.relu(y)  # nonlinear! relu(P0+P1) ≠ relu(P0)+relu(P1)


tr_b = trace(scenario_b, X_a, W_a)
print(tr_b)

# ---------------------------------------------------------------------------
# Scenario C: Partial input → second matmul (the "Gemini" scenario)
#
# If matmul(X, W1) produces Partial, and we then do matmul(result, W2)
# with W2 replicated, the distributive property says:
#   (P0 + P1) @ W2 = P0 @ W2 + P1 @ W2
# So the second matmul's result is ALSO Partial — no AllReduce needed!
# ---------------------------------------------------------------------------
header("C: matmul(Partial, W2_replicated) — chained matmuls, no reduce")
print("Expected: Two matmuls, Partial propagates, NO all_reduce")

W2_a = shard_with_specs(
    make_tensor((4, 4), seed=4),
    [DimSpec([], is_open=False), DimSpec([], is_open=False)],
)


def scenario_c(x, w1, w2):
    y = nb.matmul(x, w1)  # → Partial
    return nb.matmul(y, w2)  # → still Partial


tr_c = trace(scenario_c, X_a, W_a, W2_a)
print(tr_c)

# ---------------------------------------------------------------------------
# Scenario D: add(Partial, replicated_bias) — linear, should stay Partial
# ---------------------------------------------------------------------------
header("D: matmul → add(result, bias) — linear chain")
print("Expected: Partial propagates through add, NO all_reduce")

bias = shard_with_specs(
    make_tensor((4,), seed=5),
    [DimSpec([], is_open=False)],
)


def scenario_d(x, w, b):
    y = nb.matmul(x, w)  # → Partial
    return nb.add(y, b)  # linear — should stay Partial


tr_d = trace(scenario_d, X_a, W_a, bias)
print(tr_d)

# ---------------------------------------------------------------------------
# Scenario E: softmax(Partial) — non-linear, needs AllReduce
# ---------------------------------------------------------------------------
header("E: matmul → softmax(result) — non-linear")
print("Expected: all_reduce BEFORE softmax")


def scenario_e(x, w):
    y = nb.matmul(x, w)
    return nb.softmax(y, axis=-1)


tr_e = trace(scenario_e, X_a, W_a)
print(tr_e)

# ---------------------------------------------------------------------------
# Scenario F: neg(Partial) — linear (neg distributes), should stay Partial
# ---------------------------------------------------------------------------
header("F: matmul → neg(result) — linear")
print("Expected: Partial propagates through neg, NO all_reduce")


def scenario_f(x, w):
    y = nb.matmul(x, w)
    return nb.neg(y)


tr_f = trace(scenario_f, X_a, W_a)
print(tr_f)

# ---------------------------------------------------------------------------
# Scenario G: exp(Partial) — non-linear, needs AllReduce
# ---------------------------------------------------------------------------
header("G: matmul → exp(result) — non-linear")
print("Expected: all_reduce BEFORE exp")


def scenario_g(x, w):
    y = nb.matmul(x, w)
    return nb.exp(y)  # exp(P0+P1) ≠ exp(P0)+exp(P1)


tr_g = trace(scenario_g, X_a, W_a)
print(tr_g)

# ---------------------------------------------------------------------------
# Scenario H: Full Megatron-style MLP column-parallel
# Linear1 (column parallel) → ReLU → Linear2 (row parallel) → AllReduce
# ---------------------------------------------------------------------------
header("H: Column-parallel Linear → ReLU → Row-parallel Linear")
print("Expected: matmul(col-par) → relu (no issue, output is Shard not Partial) → matmul(row-par) → Partial result")

# Column-parallel: W1 sharded on output dim (n), so each shard computes
# a slice of the output. No Partial sum here.
W1_col = shard_with_specs(
    make_tensor((8, 4), seed=6),
    [DimSpec([], is_open=False), DimSpec(["tp"], is_open=False)],  # shard n on tp
)
# Row-parallel: W2 sharded on input dim (k), producing Partial
W2_row = shard_with_specs(
    make_tensor((2, 4), seed=7),
    [DimSpec(["tp"], is_open=False), DimSpec([], is_open=False)],  # shard k on tp
)
X_h = replicated(make_tensor((4, 8), seed=8))


def scenario_h(x, w1, w2):
    y = nb.matmul(x, w1)  # column-parallel: output sharded on tp (dim 1)
    y = nb.relu(y)  # elementwise on sharded tensor — fine
    y = nb.matmul(y, w2)  # row-parallel: k sharded → Partial
    return y  # Return Partial — caller would all_reduce


tr_h = trace(scenario_h, X_h, W1_col, W2_row)
print(tr_h)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header("SUMMARY")
print("""
Scenarios where Partial should propagate (NO all_reduce):
  A: matmul → mul(scalar)      — linear (distributive)
  C: matmul → matmul           — linear (distributive)
  D: matmul → add(bias)        — linear (distributive)
  F: matmul → neg              — linear (distributive)

Scenarios where all_reduce MUST appear before the op:
  B: matmul → relu             — non-linear
  E: matmul → softmax          — non-linear
  G: matmul → exp              — non-linear

Inspect the traces above to see if the system behaves correctly.
Look for 'all_reduce' nodes and 'partial={...}' annotations.
""")
