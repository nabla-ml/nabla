from nabla.core.tensor import Tensor
from nabla.utils.max_bindings import MaxTensorValue, graph_ops
from nabla.utils.debug import err_loc
from nabla.ops.interface import UnaryOp


struct ReluOp(UnaryOp):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "relu"

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        return MaxTensorValue(graph_ops().relu(inputs[0].to_python()))

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        raise Error("\nVJP rule for Relu not implemented yet" + err_loc())

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        raise Error("\nJVP rule for Relu not implemented yet" + err_loc())


@always_inline
fn relu(x: Tensor) raises -> Tensor:
    try:
        return ReluOp.execute([x], {})
    except err:
        raise Error("\nError in ReLU operation: " + String(err) + err_loc())
