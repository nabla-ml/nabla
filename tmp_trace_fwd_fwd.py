"""Trace jacfwd(jacfwd(f)) and jacrev(jacrev(f)) to compare op sequences."""
import nabla as nb
from nabla.core.graph.tracing import trace
from tests.unit.common import make_jax_array, tensor_from_jax

x_jax = make_jax_array(3, seed=5)
x = tensor_from_jax(x_jax)

f = lambda x: nb.reduce_sum(nb.exp(x))

# ── fwd_fwd ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("jacfwd(jacfwd(f))")
print("=" * 70)
h_ff = nb.jacfwd(nb.jacfwd(f))
t_ff = trace(h_ff, x)
print(t_ff)

# ── rev_rev ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("jacrev(jacrev(f))")
print("=" * 70)
h_rr = nb.jacrev(nb.jacrev(f))
t_rr = trace(h_rr, x)
print(t_rr)

# ── fwd_rev ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("jacfwd(jacrev(f))")
print("=" * 70)
h_fr = nb.jacfwd(nb.jacrev(f))
t_fr = trace(h_fr, x)
print(t_fr)

# ── rev_fwd ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("jacrev(jacfwd(f))")
print("=" * 70)
h_rf = nb.jacrev(nb.jacfwd(f))
t_rf = trace(h_rf, x)
print(t_rf)
