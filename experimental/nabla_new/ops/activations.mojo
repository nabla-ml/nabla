from nabla_new.core.tensor import Tensor
from nabla_new.max_bindings import MaxTensorValue, graph_ops
from nabla_new.utils import err_loc
from nabla_new.ops.interface import UnaryOp


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
