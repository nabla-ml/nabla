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

from utils import Variant

alias ShapeType = Variant[
    List[Int],
    Tuple[Int],
    Tuple[Int, Int],
    Tuple[Int, Int, Int],
    Tuple[Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int, Int, Int],
]


fn getshape(shape: ShapeType) raises -> List[Int]:
    var listshape = List[Int]()
    if shape.isa[List[Int]]():
        listshape = shape[List[Int]]
    elif shape.isa[Tuple[Int]]():
        var s = shape[Tuple[Int]]
        listshape.append(s[0])
    elif shape.isa[Tuple[Int, Int]]():
        var s = shape[Tuple[Int, Int]]
        listshape.append(s[0])
        listshape.append(s[1])
    elif shape.isa[Tuple[Int, Int, Int]]():
        var s = shape[Tuple[Int, Int, Int]]
        listshape.append(s[0])
        listshape.append(s[1])
        listshape.append(s[2])
    elif shape.isa[Tuple[Int, Int, Int, Int]]():
        var s = shape[Tuple[Int, Int, Int, Int]]
        listshape.append(s[0])
        listshape.append(s[1])
        listshape.append(s[2])
        listshape.append(s[3])
    elif shape.isa[Tuple[Int, Int, Int, Int, Int]]():
        var s = shape[Tuple[Int, Int, Int, Int, Int]]
        listshape.append(s[0])
        listshape.append(s[1])
        listshape.append(s[2])
        listshape.append(s[3])
        listshape.append(s[4])
    elif shape.isa[Tuple[Int, Int, Int, Int, Int, Int]]():
        var s = shape[Tuple[Int, Int, Int, Int, Int, Int]]
        listshape.append(s[0])
        listshape.append(s[1])
        listshape.append(s[2])
        listshape.append(s[3])
        listshape.append(s[4])
        listshape.append(s[5])
    elif shape.isa[Tuple[Int, Int, Int, Int, Int, Int, Int]]():
        var s = shape[Tuple[Int, Int, Int, Int, Int, Int, Int]]
        listshape.append(s[0])
        listshape.append(s[1])
        listshape.append(s[2])
        listshape.append(s[3])
        listshape.append(s[4])
        listshape.append(s[5])
        listshape.append(s[6])
    elif shape.isa[Tuple[Int, Int, Int, Int, Int, Int, Int, Int]]():
        var s = shape[Tuple[Int, Int, Int, Int, Int, Int, Int, Int]]
        listshape.append(s[0])
        listshape.append(s[1])
        listshape.append(s[2])
        listshape.append(s[3])
        listshape.append(s[4])
        listshape.append(s[5])
        listshape.append(s[6])
        listshape.append(s[7])
    else:
        raise "Unsupported shape"

    return listshape


fn compact_dtype_repr(dtype: DType) -> String:
    if dtype == DType.uint8:
        return "u8"
    elif dtype == DType.uint16:
        return "u16"
    elif dtype == DType.uint32:
        return "u32"
    elif dtype == DType.uint64:
        return "u64"
    elif dtype == DType.int8:
        return "i8"
    elif dtype == DType.int16:
        return "i16"
    elif dtype == DType.int32:
        return "i32"
    elif dtype == DType.int64:
        return "i64"
    elif dtype == DType.float16:
        return "f16"
    elif dtype == DType.float32:
        return "f32"
    elif dtype == DType.float64:
        return "f64"
    else:
        return "N/A"
