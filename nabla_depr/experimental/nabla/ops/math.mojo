from nabla.core.tensor import Tensor
from nabla.utils.max_bindings import MaxTensorValue, MaxDevice, graph_ops
from nabla.utils.debug import err_loc
from nabla.ops.interface import Operation, BinaryOp, UnaryOp


struct AddOp(BinaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "add"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().add(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        return [cotangent, cotangent]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        return tangents[0] + tangents[1]


@always_inline
fn add(a: Tensor, b: Tensor) raises -> Tensor:
    try:
        return AddOp.execute([a, b], {})
    except err:
        raise Error("\nError in Add operation: " + String(err) + err_loc())


struct MulOp(BinaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "mul"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().mul(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        var grad_a = cotangent * primals[1]
        var grad_b = cotangent * primals[0]
        return [grad_a, grad_b]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        var term1 = tangents[0] * primals[1]
        var term2 = primals[0] * tangents[1]
        return term1 + term2


@always_inline
fn mul(a: Tensor, b: Tensor) raises -> Tensor:
    try:
        return MulOp.execute([a, b], {})
    except err:
        raise Error("\nError in Mul operation: " + String(err) + err_loc())


struct Matmul(Operation):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "matmul"

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("\nMatmul requires exactly 2 input tensors" + err_loc())
        var a_shape = inputs[0].shape()
        var b_shape = inputs[1].shape()
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise Error("\nInput tensors must be at least 2D for Matmul" + err_loc())
        if a_shape[-1] != b_shape[-2]:
            raise Error(
                "\nInner dimensions must match for Matmul, but got shapes "
                + a_shape.__str__()
                + " and "
                + b_shape.__str__()
                + err_loc()
            )
        var result_shape = List(a_shape[0:-1]) + List(b_shape[0:-2]) + List(b_shape[-1])
        return result_shape^

    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("\nMatmul requires exactly 2 input tensors" + err_loc())
        if inputs[0].batch_dims() != inputs[1].batch_dims():
            raise Error(
                "\nInput tensors must have the same batch_dims for Matmul" + err_loc()
            )
        return inputs[0].batch_dims()

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 2:
            raise Error("\nMatmul requires exactly 2 input tensors" + err_loc())
        if inputs[0].dtype() != inputs[1].dtype():
            raise Error(
                "\nInput tensors must have the same dtype for Matmul" + err_loc()
            )
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 2:
            raise Error("\nMatmul requires exactly 2 input tensors" + err_loc())
        if inputs[0].device() != inputs[1].device():
            raise Error(
                "\nInput tensors must be on the same device for Matmul" + err_loc()
            )
        return inputs[0].device()

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().matmul(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        raise Error("\nVJP rule for Matmul not implemented yet" + err_loc())

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        raise Error("\nJVP rule for Matmul not implemented yet" + err_loc())


@always_inline
fn matmul(a: Tensor, b: Tensor) raises -> Tensor:
    var err_loc = err_loc()
    try:
        return Matmul.execute([a, b], {})
    except err:
        raise Error("\nError in Matmul operation: " + String(err) + err_loc)


struct SubOp(BinaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "sub"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().sub(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        return [cotangent, -cotangent]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        return sub(tangents[0], tangents[1])


@always_inline
fn sub(a: Tensor, b: Tensor) raises -> Tensor:
    try:
        return SubOp.execute([a, b], {})
    except err:
        raise Error("\nError in Sub operation: " + String(err))


struct DivOp(BinaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "div"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(
            graph_ops().div(inputs[0].to_python(), inputs[1].to_python())
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        return [
            cotangent / primals[1],
            -(cotangent * primals[0]) / (primals[1] * primals[1]),
        ]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        return tangents[0] / primals[1] - (primals[0] * tangents[1]) / (
            primals[1] * primals[1]
        )


@always_inline
fn div(a: Tensor, b: Tensor) raises -> Tensor:
    try:
        return DivOp.execute([a, b], {})
    except err:
        raise Error("\nError in Div operation: " + String(err))


struct NegOp(UnaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "neg"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(graph_ops().neg(inputs[0].to_python()))

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        return [-cotangent]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        return -tangents[0]


@always_inline
fn neg(a: Tensor) raises -> Tensor:
    try:
        return NegOp.execute([a], {})
    except err:
        raise Error("\nError in Neg operation: " + String(err))
