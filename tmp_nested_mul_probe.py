import numpy as np
import importlib
import nabla as nb

jvp_mod = importlib.import_module("nabla.transforms.jvp")

x = nb.Tensor.from_dlpack(np.array([2.0], dtype=np.float32))
inner = nb.Tensor.from_dlpack(np.array([1.0], dtype=np.float32))
outer = nb.Tensor.from_dlpack(np.array([1.0], dtype=np.float32))

def g(z):
    return z * z * z

saved_outer = jvp_mod._save_and_attach_tangents((x,), (outer,))
try:
    saved_inner = jvp_mod._save_and_attach_tangents((x,), (inner,))
    try:
        y = g(x)
    finally:
        jvp_mod._restore_tangents(saved_inner)

    y_tan = nb.Tensor(impl=y.tangent) if y.tangent is not None else None
    print("y=", y.to_numpy())
    print("y_tan=", y_tan.to_numpy() if y_tan is not None else None)
    print("y_tan_has_tangent=", y_tan.tangent is not None if y_tan is not None else None)
    if y_tan is not None and y_tan.tangent is not None:
        print("y_tan_tan=", nb.Tensor(impl=y_tan.tangent).to_numpy())
finally:
    jvp_mod._restore_tangents(saved_outer)
