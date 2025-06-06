# ===----------------------------------------------------------------------=== #
# Endia 2025
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
import endia


fn test_motree() raises:
    var tree = endia.motree(
        (
            String("params"),
            endia.motree(
                (String("weight_1"), endia.arange((2, 4))),
                (String("bias_1"), endia.arange((2, 5))),
            ),
        ),
        (
            String("velocities"),
            endia.motree(
                (String("weight_1"), endia.arange((2, 4))),
                (String("bias_1"), endia.arange((2, 5))),
            ),
        ),
        (
            String("tangents"),
            endia.motree(
                (String("weight_1"), endia.arange((2, 4))),
                (String("bias_1"), endia.arange((2, 5))),
            ),
        ),
    )

    var flattened_tree = tree.flatten()
    for leaf in flattened_tree:
        print(leaf[])


fn test_motree_func() raises:
    fn foo(args: endia.MoTree) raises -> endia.MoTree:
        var params = args["params"][List[endia.Array]]
        var grads = args["grads"][List[endia.Array]]

        var updated_params = List[endia.Array]()
        for i in range(len(params)):
            var updated_param = params[i] + 0.1 * grads[i]
            updated_params.append(updated_param)

        var outputs = endia.motree(
            (String("params"), updated_params),
        )
        return outputs

    var params = List[endia.Array]()
    var grads = List[endia.Array]()

    for _ in range(3):
        params.append(endia.arange((2, 4)))
        grads.append(endia.arange((2, 4)))

    var args = endia.motree(
        (String("params"), params),
        (String("grads"), grads),
    )

    var outputs = foo(args)
    var flattened_outputs = outputs.flatten()
    for i in range(len(flattened_outputs)):
        print("\nparam", i)
        print(flattened_outputs[i])
