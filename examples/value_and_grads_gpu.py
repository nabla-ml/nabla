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

import nabla as nb

device = nb.cpu() if nb.accelerator_count() == 0 else nb.accelerator()
print(f"Using {device} device")


def foo(x, y):
    return nb.sin(x @ y)


@nb.jit(show_graph=False)
def value_and_grads(x, y):
    value, vjp_fn = nb.vjp(foo, x, y)
    print("Value device:", value.device)
    return value, vjp_fn(nb.ones_like(value))


if __name__ == "__main__":
    a = nb.ndarange(
        (
            2,
            3,
        )
    ).to(device)
    print(a, a.device, a.device)

    b = nb.ndarange(
        (
            3,
            4,
        )
    ).to(device)
    print(b, b.device)

    value, grads = value_and_grads(a, b)
    print("Value:", value)

    print("Gradients:")
    for grad in grads:
        print(grad, grad.device)
