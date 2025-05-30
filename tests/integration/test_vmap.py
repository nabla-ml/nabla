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

import nabla

if __name__ == "__main__":

    def foo(args: list[nabla.Array]) -> list[nabla.Array]:
        a = args[0]

        a = nabla.incr_batch_dim_ctr(a)

        c = nabla.arange((2, 3, 4))

        res = nabla.reduce_sum(a * a * a)

        return [res]

    a = nabla.arange((2, 3, 4))

    res = foo([a])

    print("Result:", res[0])

    values, jacobian = nabla.vjp(foo, [a])
    print("Values:", values)

    cotangents = [nabla.array([1.0])]
    print("\nJacobian:", nabla.xpr(jacobian, cotangents))

    grads = jacobian(cotangents)
    print("Grads:", grads)
