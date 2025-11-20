from nabla.core.tensor import Tensor
from nabla.utils.max_bindings import (
    MaxTensorValue,
    MaxDevice,
    graph_ops,
    _list_to_py_tuple,
)
from nabla.utils.debug import err_loc
from nabla.ops.interface import Operation, is_broadcastable


struct ReshapeOp(Operation):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "reshape"

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nReshape requires exactly 1 input tensor" + err_loc())
        if not "target_shape" in kwargs:
            var available_keys = List[String]()
            for key in kwargs.keys():
                available_keys.append(key)
            raise Error(
                "\nReshape requires 'target_shape' in kwargs, but it was not found."
                " Available keys: "
                + available_keys.__str__()
                + err_loc()
            )
        return kwargs["target_shape"].copy()

    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nReshape requires exactly 1 input tensor" + err_loc())
        return inputs[0].batch_dims()

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 1:
            raise Error("\nReshape requires exactly 1 input tensor" + err_loc())
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 1:
            raise Error("\nReshape requires exactly 1 input tensor" + err_loc())
        return inputs[0].device()

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        if not "target_shape" in kwargs:
            var available_keys = List[String]()
            for key in kwargs.keys():
                available_keys.append(key)
            raise Error(
                "\nReshape.maxpr requires 'target_shape' in kwargs, but it was not"
                " found. Available keys: "
                + available_keys.__str__()
                + err_loc()
            )
        var target_shape = kwargs["target_shape"].copy()
        return MaxTensorValue(
            graph_ops().reshape(inputs[0].to_python(), _list_to_py_tuple(target_shape))
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        # Gradient of reshape is just reshaping back to original shape
        var original_shape = primals[0].shape()
        var grad_kwargs = Dict[String, List[Int]]()
        grad_kwargs["target_shape"] = original_shape.copy()
        var grad = ReshapeOp.execute([cotangent], grad_kwargs)
        return [grad]

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        # Forward-mode AD for reshape is just reshaping the tangent
        return ReshapeOp.execute([tangents[0]], kwargs)


@always_inline
fn reshape(x: Tensor, target_shape: List[Int]) raises -> Tensor:
    try:
        var kwargs = Dict[String, List[Int]]()
        kwargs["target_shape"] = target_shape.copy()
        return ReshapeOp.execute([x], kwargs)
    except err:
        raise Error("\nError in Reshape operation: " + String(err))


struct BroadcastOp(Operation):
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        return "broadcast"

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nBroadcast requires exactly 1 input tensor" + err_loc())
        if not "target_shape" in kwargs:
            raise Error("\nBroadcast requires 'target_shape' in kwargs" + err_loc())

        var input_shape = inputs[0].shape()
        var target_shape = kwargs["target_shape"].copy()

        # Check if input can be broadcast to target
        if not is_broadcastable(input_shape, target_shape):
            raise Error(
                "\nInput shape "
                + input_shape.__str__()
                + " cannot be broadcast to target shape "
                + target_shape.__str__()
                + err_loc()
            )

        return target_shape.copy()

    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nBroadcast requires exactly 1 input tensor" + err_loc())
        return inputs[0].batch_dims()

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 1:
            raise Error("\nBroadcast requires exactly 1 input tensor" + err_loc())
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 1:
            raise Error("\nBroadcast requires exactly 1 input tensor" + err_loc())
        return inputs[0].device()

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        if not "target_shape" in kwargs:
            raise Error("\nBroadcast requires 'target_shape' in kwargs" + err_loc())
        var target_shape = kwargs["target_shape"].copy()
        # return bindings_broadcast_to(inputs[0], target_shape)

        return MaxTensorValue(
            graph_ops().broadcast_to(
                inputs[0].to_python(), _list_to_py_tuple(target_shape)
            )
        )

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        # Gradient of broadcast is sum reduction over the broadcast dimensions
        # For now, just return the cotangent (simplified)
        raise Error("\nVJP rule for Broadcast not implemented yet" + err_loc())

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        # Forward-mode AD for broadcast is just broadcasting the tangent
        return BroadcastOp.execute([tangents[0]], kwargs)


@always_inline
fn broadcast(x: Tensor, target_shape: List[Int]) raises -> Tensor:
    try:
        var kwargs = Dict[String, List[Int]]()
        kwargs["target_shape"] = target_shape.copy()
        return BroadcastOp.execute([x], kwargs)
    except err:
        raise Error("\nError in Broadcast operation: " + String(err))
