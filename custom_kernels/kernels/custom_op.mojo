import compiler
from utils.index import IndexList
from nabla.compiler.tensor import OutputTensor, InputTensor, foreach, ManagedTensorSlice
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
            return  - 10 * x.load[width](idx)

        foreach[func, target=target](out, ctx)