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

import math
from time import perf_counter
import nabla


def test_simple_nn():
    batch_size = 128

    layers = [1, 64, 128, 128, 64, 1]
    every = 100
    iterations = 400

    periods = 8
    lr = 0.01
    momentum = 0.9
    avg_loss = 0.0
    avg_time = 0.0

    weights = List[nabla.Array]()
    biases = List[nabla.Array]()
    weight_velocities = List[nabla.Array]()
    bias_velocities = List[nabla.Array]()
    ctx = nabla.ExecutionContext()

    for i in range(len(layers) - 1):
        w = nabla.randn((layers[i + 1], layers[i]), DType.float32) * math.sqrt(
            2.0 / layers[i]
        )
        w.requires_grad_(True)
        weights.append(w)
        biases.append(nabla.zeros((layers[i + 1], 1), DType.float32, True))
        weight_velocities.append(
            nabla.zeros((layers[i + 1], layers[i]), DType.float32)
        )
        bias_velocities.append(nabla.zeros((layers[i + 1], 1), DType.float32))

    def forward(_input: nabla.Array) capturing -> nabla.Array:
        x = _input
        for i in range(len(layers) - 1):
            x = weights[i] @ x + biases[i]
            if i % 2 == 0:
                x.checkpoint()
            if i < len(layers) - 2:
                x = nabla.relu(x)
                # _ = x.load(0, ctx)
        return x

    for iteration in range(1, iterations + 1):
        start = perf_counter()
        input = nabla.rand(
            (1, batch_size),
            DType.float32,
            min=0.0,
            max=1.0,
        )
        y = nabla.sin((periods * 2.0 * math.pi) * input) / 2.0 + 0.5

        prediction = forward(input)
        loss = nabla.sum((prediction - y) ** 2.0) / batch_size

        loss.backward()

        for i in range(len(layers) - 1):
            # Update velocities
            weight_velocities[i] = (
                weight_velocities[i] * momentum - weights[i].grad() * lr
            )
            bias_velocities[i] = (
                bias_velocities[i] * momentum - biases[i].grad() * lr
            )

            # zero grad before updating parameters, otherwise the tangents will accumulate and the memory will explode
            weights[i].zero_grad()
            biases[i].zero_grad()

            # # Update parameters
            weights[i] = weights[i] + weight_velocities[i]
            biases[i] = biases[i] + bias_velocities[i]

            # set the new parameters to the leaf nodes
            weights[i].requires_grad_(True)
            biases[i].requires_grad_(True)

        avg_loss += loss.item(ctx)

        end = perf_counter()
        avg_time += end - start

        if iteration % every == 0:
            print("\nITERATION:", iteration)
            print("LOSS:", avg_loss / every)
            avg_loss = 0.0
            print("TIME:", avg_time / Float64(every))
            avg_time = 0.0
