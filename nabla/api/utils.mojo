# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from memory import ArcPointer
from collections import Dict
from nabla.engine.utils import TrafoMeta, GraphRepr, Callable
from nabla.compiler.engine import Model


alias none: Int = -55555


@value
struct ExecutionContext(Copyable, Movable):
    var dict: ArcPointer[Dict[Int, ArcPointer[Model]]]

    fn __init__(out self):
        self.dict = ArcPointer(Dict[Int, ArcPointer[Model]]())

    fn __getitem__(self, key: Int) raises -> ArcPointer[Model]:
        return self.dict[][key]

    fn __setitem__(mut self, key: Int, value: ArcPointer[Model]) -> None:
        if key in self.dict[]:
            print("Warning: key-value pair alrey present in model cache.")
            return
        self.dict[][key] = value

    fn __contains__(self, key: Int) -> Bool:
        return key in self.dict[]

    fn clear(mut self) -> None:
        self.dict[].clear()


fn xpr(callable: Callable) raises -> GraphRepr:
    return GraphRepr(callable)


fn xpr(func: fn (List[Array]) raises -> List[Array]) raises -> GraphRepr:
    return GraphRepr(jit(func))
