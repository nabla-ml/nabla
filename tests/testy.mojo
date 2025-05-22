from nabla.compiler.driver import accelerator, cpu, Tensor, Device, AnyMemory
from nabla.compiler.engine import InferenceSession, SessionOptions
from nabla.compiler.graph import Graph, TensorType, ops
from nabla.compiler.driver import DeviceTensor
from nabla.compiler.tensor import TensorSpec
import nabla


def test_device_tensor():
    host_device = cpu()
    gpu_device = cpu()  # accelerator()

    graph = Graph.__init__(TensorType(DType.float32, 6))
    result = ops.sin(graph[0]) * 100
    graph.output(result)

    options = SessionOptions(gpu_device)
    session = InferenceSession(options)
    model = session.load(graph)

    # input_tensor = Tensor[DType.float32, 1]((6), host_device)
    # input_tensor = DeviceTensor(TensorSpec(DType.float32, 1, 6), host_device, String(""))
    # for i in range(6):
    #     input_tensor.unsafe_ptr().bitcast[Scalar[DType.float32]]().store(i, 1.25)
    var input_tensor = nabla.arange((1, 6)).to_device_tensor()

    results = model.execute(List(AnyMemory(input_tensor)))#^.move_to(gpu_device))))
    output = (
        results[0]
        .take()
        .to_device_tensor()
        .move_to(host_device)
        .to_tensor[DType.float32, 1]()
    )
    print(output)
