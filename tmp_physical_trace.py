import numpy as np
import nabla as nb
from nabla.ops.view import broadcast_batch_dims, moveaxis_physical
from nabla.core.graph.tracing import trace


def print_ops(label, fn, *args):
    t = trace(fn, *args)
    names = [getattr(n.op, "name", type(n.op).__name__) for n in t.nodes]
    uniq = []
    for name in names:
        if name not in uniq:
            uniq.append(name)
    print(f"\n{label}")
    print(f"total_nodes={len(names)}")
    print("unique_ops=" + ",".join(uniq))


x = nb.Tensor.from_dlpack(np.array(np.random.randn(2, 3, 4), dtype=np.float32))


def f(x):
    y = moveaxis_physical(x, 0, 2)
    return nb.reduce_sum(y * y)


print_ops("primal_f", f, x)
print_ops("jacrev_f", nb.jacrev(f), x)
print_ops("jacfwd_f", nb.jacfwd(f), x)

x2 = nb.Tensor.from_dlpack(np.array(np.random.randn(2, 4), dtype=np.float32))


def g(x):
    y = broadcast_batch_dims(x, (3,))
    y = nb.reduce_sum_physical(y * y, axis=0, keepdims=False)
    return nb.reduce_sum(y)


print_ops("primal_g", g, x2)
print_ops("jacrev_g", nb.jacrev(g), x2)
