# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from nabla.compiler.driver import accelerator, cpu, Tensor, Device, AnyMemory
from nabla.compiler.engine import InferenceSession, SessionOptions
from nabla.compiler.graph import Graph, TensorType, ops


def test_new_model_execute_method():
    host_device = cpu()
    gpu_device = cpu()  # accelerator()

    graph = Graph.__init__(TensorType(DType.float32, 6))
    result = ops.sin(graph[0])
    graph.output(result)

    options = SessionOptions(gpu_device)
    session = InferenceSession(options)
    model = session.load(graph)

    input_tensor = Tensor[DType.float32, 1]((6), host_device)
    for i in range(6):
        input_tensor[i] = 1.25

    results = model.execute(List(AnyMemory(input_tensor^.move_to(gpu_device))))
    output = (
        results[0]
        .take()
        .to_device_tensor()
        .move_to(host_device)
        .to_tensor[DType.float32, 1]()
    )
    print(output)
