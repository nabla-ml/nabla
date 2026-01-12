from nabla.core.tensor import Tensor
from nabla.core.motree import MoTree
from nabla.core.execution import get_unmaterialized_trace_with_constants
from nabla.utils.max_bindings import (
    MaxTensorType,
    MaxDType,
    MaxGraph,
    MaxModel,
    MaxTensorValue,
    MaxInferenceSession,
    MaxDevice,
    MaxTensor,
)
from nabla.utils.debug import err_loc
from memory import ArcPointer


struct Callable(Copyable, Movable):
    var func: fn (args: MoTree) raises -> MoTree
    var name: String
    var compiled_model: Dict[UInt64, ArcPointer[MaxModel]]
    var constants: Dict[UInt64, List[Tensor]]
    var compiled: Bool

    fn __init__(
        out self,
        func: fn (args: MoTree) raises -> MoTree,
        name: String,
        compiled: Bool = False,
    ) raises:
        self.func = func
        self.name = name
        self.compiled_model = {}
        self.constants = {}
        self.compiled = compiled

    fn compile(mut self) raises:
        self.compiled = True

    fn __call__(mut self, all_args: MoTree) raises -> MoTree:
        if not self.compiled:
            return self.func(all_args)
        else:
            var args = all_args.get_all_tensors()
            for ref arg in args:
                if not arg.has_data():
                    raise Error(
                        "All input tensors must be materialized for compiled execution"
                        + err_loc()
                    )
                arg["is_arg"] = List[Int]()

            var _unmaterialized_outputs = self.func(all_args)
            var unmaterialized_outputs = _unmaterialized_outputs.get_all_tensors()
            # get trace with constants separated
            var trace_tuple = get_unmaterialized_trace_with_constants(
                unmaterialized_outputs
            )
            var trace = trace_tuple[0].copy()
            # var inputs = trace_tuple[1].copy()
            var constants = trace_tuple[2].copy()

            var key: UInt64 = 0
            # Hash based on the original args
            for arg in args:
                var tensor_hash: UInt64 = (
                    hash(arg.name())
                    + hash(arg.shape().__str__())
                    + hash(String(arg.dtype()))
                )
                key = key ^ (tensor_hash + 0x9E3779B9 + (key << 6) + (key >> 2))
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

            # check for available compiled model
            if not key in self.compiled_model:
                # Store constants for this compiled model
                self.constants[key] = constants.copy()

                # compile new model using inputs (args + constants) as graph inputs
                var input_types = List[MaxTensorType]()
                # First add args
                for arg in args:
                    input_types.append(
                        MaxTensorType(
                            MaxDType.from_dtype(arg.dtype()), arg.shape(), arg.device()
                        )
                    )
                # Then add constants
                for constant in constants:
                    input_types.append(
                        MaxTensorType(
                            MaxDType.from_dtype(constant.dtype()),
                            constant.shape(),
                            constant.device(),
                        )
                    )

                var graph = MaxGraph(self.name + "_compiled_model", input_types)

                with graph:
                    var graph_inputs = graph.inputs()

                    # Set tensor values for all inputs (args first, then constants)
                    var idx = 0
                    var _args = args.copy()
                    for i in range(len(args)):
                        _args[i].set_tensor_value(graph_inputs[idx])
                        idx += 1
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
                                tensor.set_tensor_value(
                                    maxpr_fn(parent_tvs, tensor.kwargs())
                                )
                                if not tensor.has_tensor_value():
                                    raise Error(
                                        "maxpr did not set tensor_value for tensor: "
                                        + tensor.name()
                                    )
                            else:
                                raise Error(
                                    "No maxpr defined for tensor: " + tensor.name()
                                )
                        var outputs_tvs = List[MaxTensorValue]()
                        for output in unmaterialized_outputs:
                            outputs_tvs.append(output.tensor_value())
                        graph.output(outputs_tvs)
                    except err:
                        raise Error(
                            "Failed to compile function '"
                            + self.name
                            + "': "
                            + String(err)
                        )

                # Create MAX inference session and load the compiled graph
                var session = MaxInferenceSession([MaxDevice.cpu()])
                self.compiled_model[key] = ArcPointer(session.load(graph))

                # Clean up tensor values from inputs and constants after compilation
                for var arg in args:
                    arg.remove_tensor_value()
                for var constant in constants:
                    constant.remove_tensor_value()

            # execute compiled model with args + constants
            var max_inputs = List[MaxTensor]()
            # First add the args data
            for arg in args:
                max_inputs.append(arg.data())
            # Then add the constants data
            if key in self.constants:
                for i in range(len(self.constants[key])):
                    max_inputs.append(self.constants[key][i].data())

            max_outputs = self.compiled_model[key][].execute(max_inputs)
            for i in range(len(unmaterialized_outputs)):
                unmaterialized_outputs[i].set_data(max_outputs[i])

            # cleanups: reset tensor values in the graph
            for var tensor in trace:
                tensor.remove_tensor_value()
            for var output in unmaterialized_outputs:
                output.remove_tensor_value()

            return _unmaterialized_outputs^  # now materialized!


fn jit(func: fn (args: MoTree) raises -> MoTree) raises -> Callable:
    return Callable(func, "func", True)
