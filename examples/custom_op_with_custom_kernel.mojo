import nabla
from nabla.core.device_array import DeviceArray
from nabla.ops.utils import register_unary_op
from nabla.compiler.graph import Symbol


struct CustomOp:
    @staticmethod
    fn maxpr(args: List[Symbol], array: DeviceArray) raises -> Symbol:
        return nabla.compiler.graph.ops.custom["custom_op"](
            args[0], args[0].type()
        )

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for CustomOp"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(custom_op(tangent))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return custom_op(tangents[0])


fn custom_op(arg: DeviceArray) raises -> DeviceArray:
    return register_unary_op[
        CustomOp.maxpr, CustomOp.vjp, CustomOp.jvp, CustomOp.eagerxpr
    ](
        arg,
        name="custom_op",
        custom_kernel_path=String("./examples/custom_kernels/kernels.mojopkg"),
    )


fn custom_op(arg: nabla.Array) raises -> nabla.Array:
    return nabla.Array(custom_op(arg.device_array[]))


fn test_custom_op_with_custom_kernel() raises -> None:
    # Test the custom op with a custom kernel
    x = nabla.arange((2, 3))
    res = custom_op(x)
    print(res)
