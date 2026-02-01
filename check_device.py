
from max.graph import DeviceRef
import max.graph as g
from max.dtype import DType

d1 = DeviceRef.CPU()
d2 = DeviceRef.CPU()

print(f"d1: {d1}, type: {type(d1)}")
print(f"d2: {d2}, type: {type(d2)}")
print(f"d1 == d2: {d1 == d2}")

graph = g.Graph("test")
with graph:
    v = g.ops.constant(1.0, DType.float32, d1)
    print(f"v.device: {v.device}, type: {type(v.device)}")
    print(f"v.device == d1: {v.device == d1}")
