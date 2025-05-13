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


from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    get_full_trace_recursively_jvp,
    Callable,
    callable,
)
from nabla.api.utils import ExecutionContext
from nabla.engine.trafos.jacfwd_trafo import jacfwd_end_rule
from nabla.engine.trafos.jvp_trafo import jvp_call, jvp_end_rule
from nabla.engine.trafos.jacrev_trafo import (
    jacrev_start_rule,
    jacrev_call,
    jacrev_end_rule,
)
from nabla.engine.trafos.vjp_trafo import vjp_call, vjp_end_rule, backward
from nabla.engine.trafos.jit_trafo import set_execution_context_recursively
from nabla.engine.trafos.vmap_trafo import vmap_start_rule, vmap_end_rule
from nabla.engine.trafos.grad_trafo import grad_call, grad_end_rule
from nabla.engine.utils import GraphRepr
from memory import ArcPointer


fn jacfwd(
    callable: Callable,
) raises -> Callable:
    var meta = TrafoMeta()
    return Callable(
        callable,
        meta,
        post=jacfwd_end_rule,
    )


fn jacfwd(
    func: fn (List[Array]) raises -> List[Array],
) raises -> Callable:
    return jacfwd(callable(func))


fn jacrev(callable: Callable, remat: Bool = False) raises -> Callable:
    var meta = TrafoMeta()
    meta["with_remat"] = List[Int](remat)
    return Callable(
        callable,
        meta,
        jacrev_start_rule,
        jacrev_call,
        jacrev_end_rule,
    )


fn jacrev(
    func: fn (List[Array]) raises -> List[Array],
    remat: Bool = False,
) raises -> Callable:
    return jacrev(callable(func), remat)


fn grad(callable: Callable, remat: Bool = False) raises -> Callable:
    var meta = TrafoMeta()
    meta["with_remat"] = List[Int](remat)
    return Callable(
        callable,
        meta,
        call=grad_call,
        post=grad_end_rule,
    )


fn grad(
    func: fn (List[Array]) raises -> List[Array], remat: Bool = False
) raises -> Callable:
    return grad(callable(func), remat)


fn jit(func: Callable) raises -> Callable:
    var meta = TrafoMeta()
    var execution_context = ExecutionContext()
    callable_ref = ArcPointer(Callable(func, meta=meta))
    set_execution_context_recursively(callable_ref, execution_context)
    return callable_ref[]


fn jit(func: fn (List[Array]) raises -> List[Array]) raises -> Callable:
    return jit(callable(func))


fn jvp(
    func: Callable,
    primals: List[Array],
    tangents: List[Array],
) raises -> Tuple[List[Array], List[Array]]:
    var meta = TrafoMeta()
    var res = Callable(
        func,
        meta,
        call=jvp_call,
        post=jvp_end_rule,
        const_args=primals + tangents,
    )()
    var num_res = meta["num_res"][0]
    return res[:num_res], res[num_res:]


fn jvp(
    func: fn (List[Array]) raises -> List[Array],
    primals: List[Array],
    tangents: List[Array],
) raises -> Tuple[List[Array], List[Array]]:
    return jvp(callable(func), primals, tangents)


fn vjp(
    func: Callable,
    primals: List[Array],
    remat: Bool = False,
) raises -> Tuple[List[Array], Callable]:
    var meta = TrafoMeta()
    meta["with_remat"] = List[Int](remat)
    meta["num_primals"] = List(len(primals))
    return func(primals), Callable(
        func,
        meta,
        call=vjp_call,
        post=vjp_end_rule,
        const_args=primals,
    )


fn vjp(
    func: fn (List[Array]) raises -> List[Array],
    primals: List[Array],
    remat: Bool = False,
) raises -> Tuple[List[Array], Callable]:
    return vjp(callable(func), primals, remat)


fn vmap(
    func: Callable,
    in_axes: List[Int] = List[Int](),
    out_axes: List[Int] = List[Int](),
) raises -> Callable:
    var meta = TrafoMeta()
    meta["in_axes"] = in_axes
    meta["out_axes"] = out_axes
    return Callable(
        func,
        meta,
        pre=vmap_start_rule,
        post=vmap_end_rule,
    )


fn vmap(
    func: fn (List[Array]) raises -> List[Array],
    in_axes: List[Int] = List[Int](),
    out_axes: List[Int] = List[Int](),
) raises -> Callable:
    return vmap(callable(func), in_axes, out_axes)
