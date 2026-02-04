# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList


@compiler.register("my_kernel")
struct MyKernel:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ):
        @parameter
        fn add_one[width: Int](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[add_one, target=target](output, ctx)
