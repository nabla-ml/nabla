import numpy as np
import importlib
import nabla as nb

jvp_mod = importlib.import_module("nabla.transforms.jvp")
x = nb.Tensor.from_dlpack(np.array([2.0], dtype=np.float32))
inner = nb.Tensor.from_dlpack(np.array([1.0], dtype=np.float32))
outer = nb.Tensor.from_dlpack(np.array([1.0], dtype=np.float32))


def vals(t):
    if t is None:
        return None, None
    t1 = nb.Tensor(impl=t)
    v1 = t1.to_numpy()
    v2 = nb.Tensor(impl=t1.tangent).to_numpy() if t1.tangent is not None else None
    return v1, v2


saved_outer = jvp_mod._save_and_attach_tangents((x,), (outer,))
try:
    saved_inner = jvp_mod._save_and_attach_tangents((x,), (inner,))
    try:
        a = x * x
        b = a * x
    finally:
        jvp_mod._restore_tangents(saved_inner)

    print("x=", x.to_numpy(), "x.tangent=", vals(x.tangent))
    print("a=", a.to_numpy(), "a.tangent=", vals(a.tangent))
    print("b=", b.to_numpy(), "b.tangent=", vals(b.tangent))
finally:
    jvp_mod._restore_tangents(saved_outer)
