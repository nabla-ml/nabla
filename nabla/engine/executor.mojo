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
from nabla.compiler.graph import Symbol, Graph, Type, Dim, TensorType
from nabla.compiler.engine import InferenceSession
from memory import ArcPointer
from utils import Variant
from nabla.core.device_array import DeviceArray
from nabla.core.utils import compact_dtype_repr
from nabla.api.utils import ExecutionContext


@value
struct NameDict(Copyable, Movable):
    var names: Dict[Int, String]
    alias alphabet: String = "abcdefghijklmnopqrstuvwxyz"
    var counter: Int
    var prefix_ctr: Int
    var curr_prefix: String

    fn __init__(out self):
        self.names = Dict[Int, String]()
        self.prefix_ctr = 0
        self.curr_prefix = ""
        self.counter = 0

    fn get_name(mut self, key: Int) -> String:
        try:
            if key in self.names:
                return self.names[key]
            else:
                var name = self.curr_prefix + self.alphabet[self.counter]
                if self.counter == len(self.alphabet) - 1:
                    self.counter = 0
                    self.prefix_ctr += 1
                    self.curr_prefix = self.alphabet[self.prefix_ctr]
                else:
                    self.counter += 1
                self.names[key] = name
                return name
        except:
            return String(key)


@value
struct Executor(Copyable, Movable, Stringable, Writable):
    var inputs: List[DeviceArray]
    var trace: List[DeviceArray]
    var outputs: List[DeviceArray]
    var execution_context: ExecutionContext

    fn __init__(
        out self,
        mut outputs: List[DeviceArray],
        execution_context: Optional[ExecutionContext],
    ) raises:
        self.execution_context = (
            execution_context.value() if execution_context else ExecutionContext()
        )
        self.inputs = List[DeviceArray]()
        self.trace = List[DeviceArray]()
        self.outputs = List[DeviceArray]()

        for output in outputs:
            if output[].num_elements() > 0:
                continue

            output[].is_tmp_output_(True)
            self.setup_trace_recursively(output[])

        self.reset_visited()

        # print("Trace:")
        # print(self)

    fn setup_trace_recursively(mut self, mut array: DeviceArray) raises -> None:
        if array.visited():
            return

        array.visited_(True)

        if array.num_elements() > 0:
            array.is_tmp_input_(True)
            self.inputs.append(array)
            array.id_(len(self.trace))
            self.trace.append(array)
            return

        for arg in array.args():
            var parent = arg[]
            self.setup_trace_recursively(parent)

        array.id_(len(self.trace))

        if not array.not_to_be_materialized() or array.is_tmp_output():
            array.not_to_be_materialized_(False)
            array.is_tmp_output_(True)
            self.outputs.append(array)

        self.trace.append(array)

    fn reset_visited(mut self) raises -> None:
        for array in self.trace:
            array[].visited_(False)

    fn setup_output[
        dtype: DType
    ](mut self, i: Int, max_outputs: compiler.engine.TensorMap) raises:
        var max_output = max_outputs.get[dtype]("output" + String(i))
        self.outputs[i].impl[]._data.free()
        var shape = List[Int]()
        for i in range(max_output.rank()):
            var dim = max_output.shape()[i]
            shape.append(dim)
        var ptr = max_output._take_data_ptr().bitcast[Scalar[DType.uint8]]()
        var spec = compiler.tensor.TensorSpec(dtype, shape)
        self.outputs[i].impl[]._data = ptr
        self.outputs[i].impl[].spec = spec
        self.outputs[i].dtype_(dtype)
        self.outputs[i].impl[].shape = shape

    fn map[
        dtype: DType
    ](
        self,
        input: DeviceArray,
        name: String,
        array_map: compiler.engine.TensorMap,
    ) raises -> None:
        var ptr = input.impl[]._data.bitcast[SIMD[dtype, 1]]()
        array_map.borrow[dtype](
            name,
            input.impl[].spec,
            ptr,
        )

    fn execute_trace(
        mut self,
        read max_model: compiler.engine.Model,
    ) raises -> None:
        var array_map = compiler.engine.TensorMap(
            max_model._ctx,
            max_model._lib,
            max_model._session,
        )

        for i in range(len(self.inputs)):
            var name = "input" + String(i)
            var dtype = self.inputs[i].impl[].spec.dtype()
            if dtype == DType.float16:
                self.map[DType.float16](self.inputs[i], name, array_map)
            elif dtype == DType.float32:
                self.map[DType.float32](self.inputs[i], name, array_map)
            elif dtype == DType.float64:
                self.map[DType.float64](self.inputs[i], name, array_map)
            elif dtype == DType.int8:
                self.map[DType.int8](self.inputs[i], name, array_map)
            elif dtype == DType.int16:
                self.map[DType.int16](self.inputs[i], name, array_map)
            elif dtype == DType.int32:
                self.map[DType.int32](self.inputs[i], name, array_map)
            elif dtype == DType.int64:
                self.map[DType.int64](self.inputs[i], name, array_map)
            elif dtype == DType.uint8:
                self.map[DType.uint8](self.inputs[i], name, array_map)
            elif dtype == DType.uint16:
                self.map[DType.uint16](self.inputs[i], name, array_map)
            elif dtype == DType.uint32:
                self.map[DType.uint32](self.inputs[i], name, array_map)
            elif dtype == DType.uint64:
                self.map[DType.uint64](self.inputs[i], name, array_map)
            else:
                raise "Unsupported dtype: " + String(dtype)

        if len(self.inputs) != max_model.num_model_inputs():
            raise "Number of inputs does not match the model"

        var max_outputs = max_model.execute(array_map)

        for i in range(len(self.outputs)):
            var dtype = self.outputs[i].dtype()
            if dtype == DType.float16:
                self.setup_output[DType.float16](i, max_outputs)
            elif dtype == DType.float32:
                self.setup_output[DType.float32](i, max_outputs)
            elif dtype == DType.float64:
                self.setup_output[DType.float64](i, max_outputs)
            elif dtype == DType.int8:
                self.setup_output[DType.int8](i, max_outputs)
            elif dtype == DType.int16:
                self.setup_output[DType.int16](i, max_outputs)
            elif dtype == DType.int32:
                self.setup_output[DType.int32](i, max_outputs)
            elif dtype == DType.int64:
                self.setup_output[DType.int64](i, max_outputs)
            elif dtype == DType.uint8:
                self.setup_output[DType.uint8](i, max_outputs)
            elif dtype == DType.uint16:
                self.setup_output[DType.uint16](i, max_outputs)
            elif dtype == DType.uint32:
                self.setup_output[DType.uint32](i, max_outputs)
            elif dtype == DType.uint64:
                self.setup_output[DType.uint64](i, max_outputs)
            else:
                raise "Unsupported dtype: " + String(dtype)

    fn realize(mut self) raises -> None:
        # print("Trace:")
        # print(self)

        var nothing_to_realize = True
        for output in self.outputs:
            if output[].num_elements() == 0:
                nothing_to_realize = False
                break

        if nothing_to_realize:
            return

        var key: Int = 0
        for array in self.trace:
            var node_hash: Int
            if array[].is_tmp_input():
                node_hash = hash(array[].impl[].spec.__str__())
            else:
                node_hash = hash(array[].impl[].name)
            key = key ^ (node_hash + 0x9E3779B9 + (key << 6) + (key >> 2))

        key = key % 1000000007

        if key not in self.execution_context:
            var model = self.create_model()
            self.execution_context[key] = model

        var max_model = self.execution_context[key]
        self.execute_trace(max_model[])

        for input in self.inputs:
            input[].is_tmp_input_(False)

        for output in self.outputs:
            if (
                not output[].impl[]._diffable
                or output[].impl[].requires_pullback
            ):
                output[].clear_args()
            output[].is_tmp_output_(False)

    fn realize_staticexecutor(mut self) raises -> None:
        var keys = self.execution_context.dict[].keys()
        if len(keys) != 1:
            raise "Only one model should be in the cache" + len(keys).__str__()
        var key = -1
        for k in keys:
            key = k[]
        var max_model = self.execution_context[key]
        self.execute_trace(max_model[])

    fn create_model(self) raises -> ArcPointer[compiler.engine.Model]:
        var in_types = List[Type]()
        for input in self.inputs:
            var shape_dim = List[Dim]()
            var shape = input[].impl[].spec.shape
            for i in range(input[].impl[].spec.rank()):
                shape_dim.append(Dim(shape[i]))
            in_types.append(
                Type(
                    TensorType(input[].dtype(), shape_dim)
                )
            )
        var graph = Graph(in_types)

        for i in range(len(self.inputs)):
            var input = self.inputs[i]

            input.impl[]._max_symbol = graph[i]

        for array in self.trace:
            var arg_ids = List[Int]()
            for arg in array[].args():
                arg_ids.append(arg[].id())

            if array[].impl[]._max_symbol:
                continue
            else:
                var _args__max_symbol = List[Symbol]()
                for arg in array[].args():
                    if arg[].impl[]._max_symbol:
                        _args__max_symbol.append(
                            arg[].impl[]._max_symbol.value()
                        )
                    else:
                        raise "Max array not found for array with id: " + String(
                            arg[].id()
                        )
                if array[].impl[]._maxpr:
                    array[].impl[]._max_symbol = (
                        array[]
                        .impl[]
                        ._maxpr.value()(_args__max_symbol, array[])
                    )
                else:
                    raise "Execute max function not found for array with id: " + String(
                        array[].id()
                    )

        var output_arrays = List[Symbol]()
        for output in self.outputs:
            if output[].impl[]._max_symbol:
                output_arrays.append(output[].impl[]._max_symbol.value())
            else:
                raise "Max array not found for array with id: " + String(
                    output[].id()
                )

        graph.output(output_arrays)
        graph.verify()

        var session = InferenceSession()
        max_model = ArcPointer(session.load(graph))

        for array in self.trace:
            array[].impl[]._max_symbol = None

        return max_model^

    fn __str__(self) -> String:
        var name_dict = NameDict()

        var out: String = "{ \033[1;94mlambda \033[0m"

        for i in range(len(self.inputs)):
            var array = self.inputs[i]
            out += (
                " "
                + name_dict.get_name(array.id())
                + "\033[35m:"
                + compact_dtype_repr(array.dtype())
                + array.shape().__str__()
                + "\033[0m"
            )

        out += ". \033[1;94mlet\033[0m\n"
        for array in self.trace:
            if array[].is_tmp_input():
                continue

            var id = array[].id()
            var name = array[].impl[].name
            var batch_dim_ctr = 0
            try:
                start_idx = name.find("{")
                end_idx = name.find("}")
                batch_dim_ctr = (
                    (name[start_idx + 1 : end_idx]).__float__().__int__()
                )
                name = name[end_idx + 1 :]
            except e:
                print("Error in executor __str__ method:", e, "String:", name)

            var dtype_str = "\033[35m:" + compact_dtype_repr(array[].dtype())
            var shape_str: String = "\033[35m["
            var shape = array[].shape()
            var first_element_in_shape = True

            if batch_dim_ctr > 0:
                shape_str += "\033[38;5;242m"
                for i in range(batch_dim_ctr):
                    if not first_element_in_shape:
                        shape_str += ","
                    shape_str += shape[i].__str__()
                    first_element_in_shape = False
                shape_str += "\033[0m"

            if batch_dim_ctr < len(shape):
                shape_str += "\033[35m"
                if not first_element_in_shape:
                    shape_str += ","
                for i in range(batch_dim_ctr, len(shape)):
                    if i > batch_dim_ctr:
                        shape_str += ","
                    shape_str += shape[i].__str__()
                    # first_element_in_shape = False
                shape_str += "\033[0m"

            shape_str += "\033[35m]\033[0m"

            for input in self.inputs:
                if input[].id() == id:
                    # is_input = True
                    break
            for output in self.outputs:
                if output[].id() == id:
                    # is_output = True
                    break
            var arg_ids = String("")
            for arg in array[].impl[]._args:
                arg_ids += name_dict.get_name(arg[][].id) + " "
            # if is_input:
            #     colored_id = "\033[33m%" + String(id) + "\033[0m"
            # elif is_output:
            #     colored_id = "\033[38;5;215m%" + String(id) + "\033[0m"
            # else:
            #     colored_id = "%" + String(id)

            out += (
                "    "
                + name_dict.get_name(id)
                + dtype_str
                + shape_str
                + " = "
                + name
                + " "
                + arg_ids
            )
            if array[].impl[].is_checkpoint:
                out += " \033[92m•\033[0m"

            out += "\n"

        out += "\033[1;94m  in\033[0m ("
        for i in range(len(self.outputs)):
            var array = self.outputs[i]
            out += name_dict.get_name(array.id())
            if i < len(self.outputs) - 1:
                out += ","
        out += ") }\n"
        return out

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())
