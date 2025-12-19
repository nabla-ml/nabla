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

import compiler
from utils.index import IndexList
from nabla.compiler.tensor import (
    OutputTensor,
    InputTensor,
    foreach,
    ManagedTensorSlice,
)
from runtime.asyncrt import DeviceContextPtr


@compiler.register("custom_op")
struct Negate:
    @staticmethod
    fn execute[
        # "gpu" or "cpu"
        target: StaticString,
    ](
        # the first argument is the output
        out: OutputTensor,
        # starting here is the list of inputs
        x: InputTensor[type = out.type, rank = out.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            return -10 * x.load[width](idx)

        foreach[func, target=target](out, ctx)
