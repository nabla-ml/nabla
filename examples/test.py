from dataclasses import dataclass
import numpy as np
from max.dtype import DType
from max.graph import Graph, TensorType, TensorValue, ops, DeviceRef
from max.engine import InferenceSession
from max.driver import Tensor, driver

device = driver.CPU() if driver.accelerator_count() == 0 else driver.Accelerator()
print(f"Using {device} device")


# The following code works just fine on a T4 GPU (default runtime in Google Colab).

@dataclass
class MulGraph:

    def __call__(self, x: TensorValue, y: TensorValue) -> TensorValue:
        return ops.mul(x, y)

try:
    Mul_graph = Graph(
        "Mul_graph",
        MulGraph(),
        input_types=[TensorType(DType.float32, (2, 3), device=DeviceRef.from_device(device)), TensorType(DType.float32, (2, 3), device=DeviceRef.from_device(device))]
    )

    session = InferenceSession(devices=[device])
    model = session.load(Mul_graph)

    x = Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3)).to(device)
    print("x:", x)
    print(x.to_numpy())

    y = Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3)).to(device)
    print("y:", y)
    print(y.to_numpy())

    res = model.execute(x, y)[0]
    print("res:", res)
    print(res.to_numpy())
except Exception as e:
    print("An error occurred:")
    print(e)


# The following code (using the ops.amtmul op) does not work on a T4 GPU, getting:
# LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16

@dataclass
class MatmulGraph:

    def __call__(self, x: TensorValue, y: TensorValue) -> TensorValue:
        return ops.matmul(x, y)

try:
    Matmul_graph = Graph(
        "Matmul_graph",
        MatmulGraph(),
        input_types=[TensorType(DType.float32, (2, 3), device=DeviceRef.from_device(device)), TensorType(DType.float32, (3, 4), device=DeviceRef.from_device(device))]
    )

    session = InferenceSession(devices=[device])
    model = session.load(Matmul_graph)

    x = Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3)).to(device)
    print("x:", x)
    print(x.to_numpy())

    y = Tensor.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4)).to(device)
    print("y:", y)
    print(y.to_numpy())

    res = model.execute(x, y)[0]
    print("res:", res)
    print(res.to_numpy())
except Exception as e:
    print("An error occurred:")
    print(e)