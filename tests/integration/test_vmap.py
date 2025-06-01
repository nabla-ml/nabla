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

    def foo(args: list[nb.Array]) -> list[nb.Array]:
        a = args[0]

        c = nb.arange((2, 3, 4))

        res = nb.reduce_sum(c * a * a, axes=[0])

        return [res]

    a = nb.arange((2, 3, 4))

    res = foo([a])

    print("\nResult:", res[0])

    foo_vmapped = nb.vmap(foo)
    print("\nFoo Vmapped:", nb.xpr(foo_vmapped, [a]))
    res = foo_vmapped([a])
    print("\nVmapped Result:", res[0])
