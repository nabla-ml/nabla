# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Custom Reduction Kernel
# ===----------------------------------------------------------------------=== #

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from utils.index import IndexList


@compiler.register("custom_sum_reduce")
struct CustomSumReduce:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        # Custom sum reduction over axis 1: (N, M) -> (N)
        # Optimized for readability, not performance
        for i in range(output.dim_size(0)):
            var sum_val = x.load[1](IndexList[2](i, 0)) - x.load[1](
                IndexList[2](i, 0)
            )  # Zero init (hacky but type safe)
            for j in range(x.dim_size(1)):
                sum_val += x.load[1](IndexList[2](i, j))
            output.store[1](IndexList[1](i), sum_val)
