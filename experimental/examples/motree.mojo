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


fn test_motree() raises:
    var tree = nabla.motree(
        (
            String("params"),
            nabla.motree(
                (String("weight_1"), nabla.ndarange((2, 4))),
                (String("bias_1"), nabla.ndarange((2, 5))),
            ),
        ),
        (
            String("velocities"),
            nabla.motree(
                (String("weight_1"), nabla.ndarange((2, 4))),
                (String("bias_1"), nabla.ndarange((2, 5))),
            ),
        ),
        (
            String("tangents"),
            nabla.motree(
                (String("weight_1"), nabla.ndarange((2, 4))),
                (String("bias_1"), nabla.ndarange((2, 5))),
            ),
        ),
    )

    var flattened_tree = tree.flatten()
    for leaf in flattened_tree:
        print(leaf[])


fn test_motree_func() raises:
    fn foo(args: nabla.MoTree) raises -> nabla.MoTree:
        var params = args["params"][List[nabla.Array]]
        var grads = args["grads"][List[nabla.Array]]

        var updated_params = List[nabla.Array]()
        for i in range(len(params)):
            var updated_param = params[i] + 0.1 * grads[i]
            updated_params.append(updated_param)

        var outputs = nabla.motree(
            (String("params"), updated_params),
        )
        return outputs

    var params = List[nabla.Array]()
    var grads = List[nabla.Array]()

    for _ in range(3):
        params.append(nabla.ndarange((2, 4)))
        grads.append(nabla.ndarange((2, 4)))

    var args = nabla.motree(
        (String("params"), params),
        (String("grads"), grads),
    )

    var outputs = foo(args)
    var flattened_outputs = outputs.flatten()
    for i in range(len(flattened_outputs)):
        print("\nparam", i)
        print(flattened_outputs[i])
