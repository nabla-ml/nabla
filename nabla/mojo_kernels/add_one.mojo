import compiler
from tensor import OutputTensor, InputTensor, foreach
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList


@compiler.register("add_one_custom")
struct AddOneCustom:
    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        # the first argument is the "output"
        out: OutputTensor,
        # this is followed by the "inputs"
        x: InputTensor[type = out.type, rank = out.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn elementwise_add_one[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            return x.load[width](idx) + 1

        foreach[elementwise_add_one, target=target](out, ctx)