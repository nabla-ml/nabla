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

if __name__ == "__main__":
    device = nb.device("gpu:0")  # Change to "cpu" for CPU testing

    for iter in range(10000):
        n0 = nb.randn((8, 8)).to(device)
        n1 = nb.randn((4, 8, 8)).to(device)
        res = nb.reduce_sum(
            nb.reshape(nb.sin(n0 + n1 * n1 + n0), shape=(2, 2, 4, 2, 8)),
            axes=(4),
            keep_dims=False,
        )
        res.realize()
        # print(res)
        if iter % 1000 == 0:
            print(f"Iteration {iter}: Result shape {res.shape}, dtype {res.dtype}")
            # print(res)
