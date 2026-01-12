from nabla.core.tensor import Tensor
from nabla.utils.max_bindings import MaxTensorValue, MaxDevice
from nabla.utils.debug import err_loc


fn is_broadcastable(shape_a: List[Int], shape_b: List[Int]) raises -> Bool:
    """Check if two shapes are broadcastable and return True if they are."""
    var ndim_a = len(shape_a)
    var ndim_b = len(shape_b)
    var max_ndim = max(ndim_a, ndim_b)

    for i in range(max_ndim):
        var dim_a = shape_a[ndim_a - max_ndim + i] if i >= (max_ndim - ndim_a) else 1
        var dim_b = shape_b[ndim_b - max_ndim + i] if i >= (max_ndim - ndim_b) else 1

        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False

    return True


fn broadcast_shapes(shape_a: List[Int], shape_b: List[Int]) raises -> List[Int]:
    """Compute the broadcasted shape of two input shapes."""
    if not is_broadcastable(shape_a, shape_b):
        var msg = (
            "Shapes "
            + shape_a.__str__()
            + " and "
            + shape_b.__str__()
            + " are not broadcastable"
        )
        raise Error(msg + err_loc())

    var ndim_a = len(shape_a)
    var ndim_b = len(shape_b)
    var max_ndim = max(ndim_a, ndim_b)

    var result_shape = List[Int]()
    for i in range(max_ndim):
        var dim_a = shape_a[ndim_a - max_ndim + i] if i >= (max_ndim - ndim_a) else 1
        var dim_b = shape_b[ndim_b - max_ndim + i] if i >= (max_ndim - ndim_b) else 1
        result_shape.append(max(dim_a, dim_b))

    return result_shape^


trait Operation:
    @staticmethod
    fn name(kwargs: Dict[String, List[Int]]) -> String:
        pass

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        pass

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        pass

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        pass

    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        pass

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        pass

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        pass

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        pass

    @staticmethod
    fn stage_realization(inputs: List[Tensor]) raises -> Bool:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())
        return inputs[0].stage_realization() or inputs[1].stage_realization()

    @staticmethod
    fn custom_kernel_path() -> String:
        return String("")

    @staticmethod
    fn execute(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> Tensor:
        try:
            var res = Tensor(
                shape=Self.shape(inputs, kwargs),
                dtype=Self.dtype(inputs, kwargs),
                stage_realization=True,
            )
            res.set_name(Self.name(kwargs))
            res.set_parents(inputs)
            res.set_maxpr(Self.maxpr)
            res.set_vjp_rule(Self.vjp_rule)
            res.set_jvp_rule(Self.jvp_rule)
            res.set_custom_kernel_path(Self.custom_kernel_path())
            res.set_parents(inputs)
            # Store kwargs on the tensor so they can be used during realization
            # BUT: don't copy "is_arg" metadata to child tensors
            for key in kwargs.keys():
                if key != "is_arg":
                    res[key] = kwargs[key].copy()
            return res
        except err:
            raise Error(
                "\nError during execution of operation '"
                + Self.name(kwargs)
                + "': "
                + String(err)
            )


trait BinaryOp(Operation):
    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())
        if inputs[0].batch_dims() != inputs[1].batch_dims():
            raise Error(
                "\nInput tensors must have the same batch_dims for BinaryOp, but got "
                + inputs[0].batch_dims().__str__()
                + " and "
                + inputs[1].batch_dims().__str__()
                + err_loc()
            )
        return inputs[0].batch_dims()

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())

        var shape_a = inputs[0].shape()
        var shape_b = inputs[1].shape()

        # Use the helper function to compute broadcasted shape
        return broadcast_shapes(shape_a, shape_b)

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())
        if inputs[0].dtype() != inputs[1].dtype():
            raise Error(
                "\nInput tensors must have the same dtype for BinaryOp, but got "
                + String(inputs[0].dtype())
                + " and "
                + String(inputs[1].dtype())
                + err_loc()
            )
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 2:
            raise Error("\nBinaryOp requires exactly 2 input tensors" + err_loc())
        if inputs[0].device() != inputs[1].device():
            raise Error(
                "\nInput tensors must be on the same device for BinaryOp, but got "
                + inputs[0].device().label()
                + " and "
                + inputs[1].device().label()
                + err_loc()
            )
        return inputs[0].device()

    @staticmethod
    fn maxpr(
        inputs: List[MaxTensorValue], kwargs: Dict[String, List[Int]]
    ) raises -> MaxTensorValue:
        pass

    @staticmethod
    fn vjp_rule(
        primals: List[Tensor], cotangent: Tensor, kwargs: Dict[String, List[Int]]
    ) raises -> List[Tensor]:
        pass

    @staticmethod
    fn jvp_rule(
        primals: List[Tensor], tangents: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> Tensor:
        pass


trait UnaryOp(Operation):
    @staticmethod
    fn batch_dims(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nUnaryOp requires exactly 1 input tensor" + err_loc())
        return inputs[0].batch_dims()

    @staticmethod
    fn shape(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> List[Int]:
        if len(inputs) != 1:
            raise Error("\nUnaryOp requires exactly 1 input tensor" + err_loc())
        return inputs[0].shape()

    @staticmethod
    fn dtype(inputs: List[Tensor], kwargs: Dict[String, List[Int]]) raises -> DType:
        if len(inputs) != 1:
            raise Error("\nUnaryOp requires exactly 1 input tensor" + err_loc())
        return inputs[0].dtype()

    @staticmethod
    fn device(
        inputs: List[Tensor], kwargs: Dict[String, List[Int]]
    ) raises -> MaxDevice:
        if len(inputs) != 1:
            raise Error("\nUnaryOp requires exactly 1 input tensor" + err_loc())
        return inputs[0].device()
