from memory import ArcPointer
from nabla.utils.max_bindings import (
    MaxModel,
    MaxGraph,
    MaxTensorType,
    MaxDType,
    MaxTensorValue,
    MaxInferenceSession,
    MaxDevice,
    MaxTensor,
    graph_ops,
    _list_to_py_tuple,
)
from .tensor import Tensor
from nabla.utils.debug import err_loc


struct ExecutionContext(ImplicitlyCopyable, Movable):
    var model_dict_ptr: ArcPointer[Dict[UInt64, ArcPointer[MaxModel]]]

    fn __init__(out self) raises:
        self.model_dict_ptr = ArcPointer(Dict[UInt64, ArcPointer[MaxModel]]())

    fn __getitem__(self, key: UInt64) raises -> ArcPointer[MaxModel]:
        return self.model_dict_ptr[][key]

    fn __setitem__(mut self, key: UInt64, val: ArcPointer[MaxModel]) raises:
        self.model_dict_ptr[][key] = val


fn realize(
    mut unmaterialized_output: Tensor, _ctx: Optional[ExecutionContext] = None
) raises:
    var unmaterialized_outputs = [unmaterialized_output]
    return realize(unmaterialized_outputs, _ctx)


fn realize(
    mut unmaterialized_outputs: List[Tensor], _ctx: Optional[ExecutionContext] = None
) raises:
    """Realize all tensors' data."""
    # get trace with constants separated
    var trace_tuple = get_unmaterialized_trace_with_constants(unmaterialized_outputs)
    var trace = trace_tuple[0].copy()
    var constants = trace_tuple[2].copy()

    var key: UInt64 = 0
    for var constant in constants:
        var tensor_hash: UInt64 = (
            hash(constant.name())
            + hash(constant.shape().__str__())
            + hash(String(constant.dtype()))
        )
        key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
    for tensor in trace:
        var tensor_hash: UInt64 = (
            hash(tensor.name())
            + hash(tensor.shape().__str__())
            + hash(String(tensor.dtype()))
        )
        key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
    key = key % 1000000007

    # creat or use exection context
    if not _ctx:
        ctx = ExecutionContext()
    else:
        ctx = _ctx[]

    # check for available compiled model
    if not key in ctx.model_dict_ptr[]:
        # compile new model using inputs (args + constants) as graph inputs
        var input_types = List[MaxTensorType]()
        # Then add constants
        for constant in constants:
            input_types.append(
                MaxTensorType(
                    MaxDType.from_dtype(constant.dtype()),
                    constant.shape(),
                    constant.device(),
                )
            )

        var graph = MaxGraph("compiled_model", input_types)

        with graph:
            var graph_inputs = graph.inputs()

            # Set tensor values for all inputs (args first, then constants)
            var idx = 0
            for i in range(len(constants)):
                constants[i].set_tensor_value(graph_inputs[idx])
                idx += 1

            # Now build the computation graph
            try:
                for var tensor in trace:
                    var parent_tvs = List[MaxTensorValue]()
                    if tensor.has_parents():
                        var parents = tensor.parents()
                        for i in range(len(parents)):
                            parent_tvs.append(parents[i].tensor_value())
                    if tensor.has_maxpr():
                        var maxpr_fn = tensor.maxpr()
                        tensor.set_tensor_value(maxpr_fn(parent_tvs, tensor.kwargs()))
                        if not tensor.has_tensor_value():
                            raise Error(
                                "maxpr did not set tensor_value for tensor: "
                                + tensor.name()
                            )
                    else:
                        raise Error("No maxpr defined for tensor: " + tensor.name())
                var outputs_tvs = List[MaxTensorValue]()
                for output in unmaterialized_outputs:
                    outputs_tvs.append(output.tensor_value())
                graph.output(outputs_tvs)
            except err:
                raise Error("Failed to compile function '" + "': " + String(err))

        # Create MAX inference session and load the compiled graph
        var session = MaxInferenceSession([MaxDevice.cpu()])
        ctx[key] = ArcPointer(session.load(graph))

        # Clean up tensor values from constants after compilation
        for var constant in constants:
            constant.remove_tensor_value()

    # execute compiled model with args + constants
    var max_inputs = List[MaxTensor]()
    # Add the constants data
    for i in range(len(constants)):
        max_inputs.append(constants[i].data())

    max_outputs = ctx[key][].execute(max_inputs)
    for i in range(len(unmaterialized_outputs)):
        unmaterialized_outputs[i].set_data(max_outputs[i])

    # cleanups: reset tensor values in the graph
    for var tensor in trace:
        tensor.remove_tensor_value()
    for var output in unmaterialized_outputs:
        output.remove_tensor_value()


fn reset_visited(mut tensors: List[Tensor]) raises:
    """Reset visited flag for entire graph."""
    for var tensor in tensors:
        tensor.set_visited(False)
        var parents = tensor.parents()
        reset_visited(parents)


fn print_trace(trace: List[Tensor]) raises -> None:
    for t in trace:
        print(t.name())


fn get_dependencies_recursive(
    mut outputs: List[Tensor], mut trace: List[Tensor], mut inputs: List[Tensor]
) raises -> None:
    for var output in outputs:
        if not output.visited():
            if output.has_parents():
                output.set_visited(True)
                var parents = output.parents()
                get_dependencies_recursive(parents, trace, inputs)
                trace.append(output)
            else:
                inputs.append(output)


fn get_unmaterialized_recursive(
    mut outputs: List[Tensor], mut trace: List[Tensor], mut inputs: List[Tensor]
) raises -> None:
    for var output in outputs:
        if not output.visited():
            if output.has_data():
                inputs.append(output)
            else:
                output.set_visited(True)
                var parents = output.parents()
                get_unmaterialized_recursive(parents, trace, inputs)
                trace.append(output)


fn get_unmaterialized_recursive_with_constants(
    mut outputs: List[Tensor],
    mut trace: List[Tensor],
    mut inputs: List[Tensor],
    mut constants: List[Tensor],
) raises -> None:
    """DFS that separates materialized inputs into args (marked with 'is_arg') and constants.
    """
    for var output in outputs:
        if not output.visited():
            if output.has_data():
                # Check if this tensor is marked as an arg
                if "is_arg" in output.kwargs():
                    inputs.append(output)
                else:
                    constants.append(output)
            else:
                output.set_visited(True)
                var parents = output.parents()
                get_unmaterialized_recursive_with_constants(
                    parents, trace, inputs, constants
                )
                trace.append(output)


fn get_dependency_trace(
    outputs: List[Tensor],
) raises -> Tuple[List[Tensor], List[Tensor]]:
    var trace: List[Tensor] = []
    var inputs: List[Tensor] = []
    var _outputs = outputs.copy()
    reset_visited(_outputs)
    get_dependencies_recursive(_outputs, trace, inputs)
    reset_visited(_outputs)
    return (trace^, inputs^)


fn get_unmaterialized_trace(
    outputs: List[Tensor],
) raises -> Tuple[List[Tensor], List[Tensor]]:
    var trace: List[Tensor] = []
    var inputs: List[Tensor] = []
    var _outputs = outputs.copy()
    reset_visited(_outputs)
    get_unmaterialized_recursive(_outputs, trace, inputs)
    reset_visited(_outputs)
    return (trace^, inputs^)


fn get_unmaterialized_trace_with_constants(
    outputs: List[Tensor],
) raises -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Get trace separating inputs (marked with 'is_arg') from constants (created inside function).
    """
    var trace: List[Tensor] = []
    var inputs: List[Tensor] = []
    var constants: List[Tensor] = []
    var _outputs = outputs.copy()
    reset_visited(_outputs)
    get_unmaterialized_recursive_with_constants(_outputs, trace, inputs, constants)
    reset_visited(_outputs)
    return (trace^, inputs^, constants^)
