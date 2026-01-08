
from nabla.core.tensor import Tensor
from nabla import vmap
import numpy as np

def debug_shape():
    from nabla import DeviceMesh, P
    mesh = DeviceMesh("mesh", (4,), ("dp",))
    t = Tensor.from_dlpack(np.array([1, 2, 3, 4]))
    t = t.shard(mesh, P("dp"))
    
    print(f"Tensor shape type: {type(t.shape)}")
    print(f"Tensor shape: {t.shape}")
    print(f"Rank: {t.shape.rank}")

    try:
        from nabla.transforms.vmap import _get_batch_size
        print("Calling _get_batch_size...")
        _get_batch_size(t, 0)
    except Exception as e:
        print(f"_get_batch_size failed: {e}")

if __name__ == "__main__":
    debug_shape()
