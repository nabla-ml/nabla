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


def simple_add(args):
    return [args[0] + args[1]]


if __name__ == "__main__":
    # Test simple vmap
    a = nb.arange((3, 4), nb.DType.float32)  # shape (3, 4)
    b = nb.arange((4,), nb.DType.float32)  # shape (4,)

    print("a.shape:", a.shape)
    print("b.shape:", b.shape)

    # This should vectorize over the first axis of a, and broadcast b
    vmapped_add = nb.vmap(simple_add, [0, None])
    result = vmapped_add([a, b])

    print("result.shape:", result[0].shape)
    print("result:", result[0])
