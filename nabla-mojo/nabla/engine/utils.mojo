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

from collections import Optional
from utils import Variant
from collections import Dict
from memory import ArcPointer

from nabla.api.array import Array
from nabla.core.device_array import DeviceArray
from nabla.engine.trafos.vjp_trafo import compute_cotangent
from nabla.api.utils import ExecutionContext
from nabla.engine.executor import Executor
from nabla.api.array import zeros
from nabla.api.utils import none


alias axes_type = Variant[
    Int,
    List[Int],
    List[Optional[Int]],
    Tuple[Int],
    Tuple[Int, Int],
    Tuple[Int, Int, Int],
    Tuple[Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int, Int, Int],
]


def get_axes(axes: axes_type) -> List[Int]:
    if axes.isa[Int]():
        var res = List[Int]()
        res.append(axes[Int])
        return res
    elif axes.isa[List[Int]]():
        var res = List[Int]()
        for axis in axes[List[Int]]:
            res.append(axis[])
        return res
    elif axes.isa[List[Optional[Int]]]():
        var res = List[Int]()
        for axis in axes[List[Optional[Int]]]:
            if axis[]:
                res.append(axis[].value())
            else:
                res.append(none)
        return res
    elif axes.isa[Tuple[Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int]]
        res.append(tuple[0])
        return res
    elif axes.isa[Tuple[Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        return res
    elif axes.isa[Tuple[Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        res.append(tuple[4])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        res.append(tuple[4])
        res.append(tuple[5])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        res.append(tuple[4])
        res.append(tuple[5])
        res.append(tuple[6])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int, Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int, Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        res.append(tuple[4])
        res.append(tuple[5])
        res.append(tuple[6])
        res.append(tuple[7])
        return res

    else:
        raise "Error: Invalid axes type. Use _None alias to use Tuples."


@value
struct TrafoMeta(Copyable, Movable):
    var data: ArcPointer[Dict[String, List[Int]]]

    fn __init__(out self) raises:
        self.data = ArcPointer(Dict[String, List[Int]]())

    fn __getitem__(self, key: String) raises -> List[Int]:
        return self.data[][key]

    fn __setitem__(mut self, key: String, value: List[Int]) raises:
        self.data[][key] = value

    fn __contains__(self, key: String) raises -> Bool:
        return key in self.data[]


fn reset_full_trace_recursively_jvp(mut array: DeviceArray) raises -> None:
    if not array.impl[]._compute_jvp:
        return

    array.impl[]._compute_jvp = False

    for arg in array.args():
        var parent = arg[]
        reset_full_trace_recursively_jvp(parent)


fn get_full_trace_recursively_jvp(
    mut trace: List[DeviceArray], mut array: DeviceArray
) raises -> None:
    if array.impl[]._compute_jvp or not array.impl[]._jvp:
        return

    array.impl[]._compute_jvp = True

    for arg in array.args():
        var parent = arg[]
        get_full_trace_recursively_jvp(trace, parent)

    trace.append(array)


fn default_start_rule(
    mut args: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    return args


fn default_call(
    meta: TrafoMeta,
    args: List[Array],
) raises -> List[Array]:
    return args


fn default_end_rule(
    mut args: List[Array],
    mut res: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    return res


@value
struct Callable(Copyable, Movable):
    var func: Optional[fn (List[Array]) raises -> List[Array]]
    var pre: Optional[fn (mut List[Array], mut TrafoMeta) raises -> List[Array]]
    var call: Optional[
        fn (
            meta: TrafoMeta,
            args: List[Array],
        ) raises -> List[Array]
    ]
    var post: Optional[
        fn (
            mut List[Array], mut List[Array], mut TrafoMeta
        ) raises -> List[Array]
    ]
    var meta: TrafoMeta
    var trafos: List[ArcPointer[Self]]
    var const_args: List[Array]
    var execution_context: Optional[ExecutionContext]

    fn __init__(
        out self,
        callable: Self,
        mut meta: TrafoMeta,
        pre: fn (mut List[Array], mut TrafoMeta) raises -> List[
            Array
        ] = default_start_rule,
        call: fn (
            TrafoMeta,
            List[Array],
        ) raises -> List[Array] = default_call,
        post: fn (
            mut List[Array], mut List[Array], mut TrafoMeta
        ) raises -> List[Array] = default_end_rule,
        const_args: List[Array] = List[Array](),
    ) raises:
        self.func = None
        self.pre = pre
        self.call = call
        self.post = post
        self.meta = meta
        self.trafos = List[ArcPointer[Self]](callable)
        self.const_args = const_args
        self.execution_context = None

    fn __init__(
        out self,
        func: fn (List[Array]) raises -> List[Array],
        mut meta: TrafoMeta,
        pre: fn (mut List[Array], mut TrafoMeta) raises -> List[
            Array
        ] = default_start_rule,
        call: fn (
            TrafoMeta,
            List[Array],
        ) raises -> List[Array] = default_call,
        post: fn (
            mut List[Array], mut List[Array], mut TrafoMeta
        ) raises -> List[Array] = default_end_rule,
        const_args: List[Array] = List[Array](),
    ) raises:
        self.func = func
        self.pre = pre
        self.call = call
        self.post = post
        self.meta = meta
        self.trafos = List[ArcPointer[Self]]()
        self.const_args = const_args
        self.execution_context = None

    fn __call__(self, args: List[Array] = List[Array]()) raises -> List[Array]:
        var adapted_args = self.const_args + args
        var meta = self.meta
        var res: List[Array]

        if self.pre:
            adapted_args = self.pre.value()(adapted_args, meta)

        if self.execution_context:
            for arg in adapted_args:
                arg[].device_array[].impl[].execution_context = (
                    self.execution_context.value()
                )

        if self.func:
            var _adapted_args = self.call.value()(meta, adapted_args)
            res = self.func.value()(_adapted_args)

        elif len(self.trafos) == 1:
            var child_trafo = self.trafos[0]
            var _adapted_args = self.call.value()(meta, adapted_args)
            res = child_trafo[](_adapted_args)
        else:
            raise "Error in Callable struct."

        if self.post:
            res = self.post.value()(adapted_args, res, meta)

        return res

    fn __call__(self, arg0: Array, arg1: Array) raises -> List[Array]:
        return self(List[Array](arg0, arg1))

    fn __call__(
        self, arg0: List[Array], arg1: List[Array]
    ) raises -> List[Array]:
        return self(arg0 + arg1)


@value
struct GraphRepr(Copyable, Movable):
    var callable: ArcPointer[Callable]

    fn __init__(out self, callable: Callable) raises:
        self.callable = ArcPointer(callable)

    fn __call__(self, args: List[Array]) raises -> String:
        var res = self.callable[](args)
        var device_arrays = List[DeviceArray]()
        for array in res:
            device_arrays.append(array[].device_array[])
        var ctx = ExecutionContext()
        var executor = Executor(device_arrays, ctx)
        return executor.__str__()


fn callable(
    func: Variant[fn (List[Array]) raises -> List[Array], Callable],
) raises -> Callable:
    if func.isa[Callable]():
        return func[Callable]
    else:
        var meta = TrafoMeta()
        return Callable(
            func[fn (List[Array]) raises -> List[Array]],
            meta,
            pre=default_start_rule,
            call=default_call,
            post=default_end_rule,
        )


fn std_basis(
    args: List[Array],
) raises -> Tuple[List[Int], List[Array]]:
    var num_total_arg_elements = 0
    var max_rank = 0
    for arg in args:
        var num_elements = 1
        var batch_dim_ctr = arg[].batch_dim_ctr()
        batch_dim_ctr = batch_dim_ctr if batch_dim_ctr != none else 0
        for dim in arg[].shape()[batch_dim_ctr:]:
            num_elements *= dim[]
        num_total_arg_elements += num_elements
        var rank = len(arg[].shape()[batch_dim_ctr:])
        if rank > max_rank:
            max_rank = rank

    var batch_ctr = 0
    var sizes = List[Int]()

    var tangents = List[Array]()

    for arg in args:
        var num_elements = 1

        var batch_dim_ctr = arg[].batch_dim_ctr()
        batch_dim_ctr = batch_dim_ctr if batch_dim_ctr != none else 0
        for dim in arg[].shape()[batch_dim_ctr:]:
            num_elements *= dim[]

        arg[].device_array[].impl[]._compute_jvp = True
        var arg_batch_ctr = arg[].batch_dim_ctr()
        arg_batch_ctr = arg_batch_ctr if arg_batch_ctr != none else 0
        var batched_shape = arg[].shape()
        for _ in range(max_rank - len(batched_shape)):
            batched_shape = List(1) + batched_shape

        batched_shape = (
            batched_shape[:arg_batch_ctr]
            + List(num_total_arg_elements)
            + batched_shape[arg_batch_ctr:]
        )
        var dtype = arg[].dtype()
        var tangent = zeros(batched_shape, dtype)

        var total_elements = 1
        for dim in batched_shape:
            total_elements *= dim[]

        var num_els_batch_dims = 1
        for dim in arg[].shape()[:batch_dim_ctr]:
            num_els_batch_dims *= dim[]

        for i in range(num_els_batch_dims):
            var offset = batch_ctr + num_total_arg_elements * num_elements * i

            for j in range(num_elements):
                idx = offset + j
                tangent.store(idx, 1.0)
                offset += num_elements

        batch_ctr += num_elements * num_elements
        tangent.batch_dim_ctr_((arg[].batch_dim_ctr()))
        tangents.append(tangent)
        sizes.append(num_elements)

    return sizes, tangents


fn get_full_trace_recursively(
    mut trace: List[DeviceArray], mut array: DeviceArray
) raises -> None:
    # TODO: The folloing loop is to maintain materialization of certain arrays,
    # however in fact it should NOT be needed, if we remove it the code breaks
    # when we compute the loss after the updates of the weights and biases.
    if not array.not_to_be_materialized():
        array.is_tmp_output_(True)

    if array.visited() or not array.impl[]._diffable:
        return

    array.visited_(True)

    for arg in array.args():
        var parent = arg[]
        get_full_trace_recursively(trace, parent)

    array.id_(len(trace))
    trace.append(array)


fn reset_visited(mut trace: List[DeviceArray]) raises -> None:
    for array in trace:
        array[].visited_(False)


fn cotangent_with_remat(
    outs: List[DeviceArray], keep_graph: Bool = True
) raises -> List[DeviceArray]:
    var trace = List[DeviceArray]()

    for output in outs:
        var parent = output[]
        get_full_trace_recursively(trace, parent)

    reset_visited(trace)

    for array in trace:
        if (
            not array[].impl[].is_checkpoint
            and (not array[].impl[].requires_pullback)
            and array[].impl[]._diffable
            and (not array[].is_tmp_output())
        ):
            var dual_args = array[].args()

            for i in range(len(dual_args)):
                var arg = dual_args[i]
                if arg.has_dual():
                    dual_args[i] = arg.dual()

            var dual = DeviceArray(ArcPointer(array[].impl[]))
            dual.name_("dual_" + array[].name())
            dual.args_(dual_args)
            array[].dual_(dual)

    var cotangents = List[DeviceArray]()

    for i in range(len(trace) - 1, -1, -1):
        var p_array = trace[i]
        var array = p_array

        if (
            len(array.args()) == 0
            or not array.impl[]._diffable
            or array.impl[].requires_pullback
        ):
            continue

        if len(array.impl[]._dual) == 1:
            array = DeviceArray(p_array.impl[]._dual[0])
            if (
                len(p_array.impl[].cotangent) == 1
                and len(array.impl[].cotangent) == 0
            ):
                array.impl[].cotangent = p_array.impl[].cotangent

        compute_cotangent(array, cotangents)
        array.impl[].cotangent.clear()
        p_array.impl[].cotangent.clear()

    for array in trace:
        array[].impl[]._dual.clear()

    if not keep_graph:
        for cotangent in cotangents:
            cotangent[].requires_pullback_(False)

    return cotangents
