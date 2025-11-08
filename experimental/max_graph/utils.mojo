
from python import Python, PythonObject
from memory import ArcPointer
from .types import Tensor, TensorType, TensorValue, DType, Device

# ============================================================================
# PYTHON BRIDGE - Centralized Python module management
# ============================================================================

struct PythonBridge:
    """Singleton-like struct for managing Python module imports."""
    
    @staticmethod
    fn setup() raises:
        """Initialize Python path for MAX interop."""
        Python.add_to_path("./max_graph/max_python")
    
    @staticmethod
    fn get_module(name: String) raises -> PythonObject:
        """Get a Python module by name."""
        return Python.import_module(name)
    
    @staticmethod
    fn get_builtins() raises -> PythonObject:
        """Get Python builtins module."""
        return Python.import_module("builtins")
    
    @staticmethod
    fn shape_to_python(shape: List[Int]) raises -> PythonObject:
        """Convert List[Int] shape to Python tuple."""
        var builtins = Self.get_builtins()
        var py_list = builtins.list()
        for i in range(len(shape)):
            _ = py_list.append(shape[i])
        return builtins.tuple(py_list)



# ============================================================================
# OPERATION REGISTRY - Extensible operations
# ============================================================================

struct OpRegistry:
    """Registry for graph operations."""
    
    @staticmethod
    fn check_same_context(tensors: List[TensorValue]) raises:
        """Verify all tensors belong to the same graph context."""
        if len(tensors) < 2:
            return
        
        var first_ctx = tensors[0].get_graph()
        for i in range(1, len(tensors)):
            if first_ctx[] != tensors[i].get_graph()[]:
                raise Error("TensorValues must belong to the same graph context")
    
    @staticmethod
    fn binary_op(op_name: String, a: TensorValue, b: TensorValue) raises -> TensorValue:
        """Execute a binary operation on two tensors.
        
        Args:
            op_name: Name of the operation (add, matmul, mul, etc.).
            a: First tensor operand.
            b: Second tensor operand.
        
        Returns:
            Result tensor in the same graph context.
        """
        Self.check_same_context([a, b])
        
        var max_graph = PythonBridge.get_module("main")
        var op_func = max_graph.__getattr__(op_name)
        
        var result_tensor = op_func(
            a.get_graph()[].graph, 
            a.to_python(), 
            b.to_python()
        )
        
        return TensorValue(a.get_graph(), result_tensor)
    
    @staticmethod
    fn unary_op(op_name: String, x: TensorValue) raises -> TensorValue:
        """Execute a unary operation on a tensor.
        
        Args:
            op_name: Name of the operation (abs, relu, exp, etc.).
            x: Input tensor operand.
        
        Returns:
            Result tensor in the same graph context.
        """
        var max_graph = PythonBridge.get_module("main")
        var op_func = max_graph.__getattr__(op_name)
        
        var result_tensor = op_func(
            x.get_graph()[].graph,
            x.to_python()
        )
        
        return TensorValue(x.get_graph(), result_tensor)



# ============================================================================
# GRAPH BUILDER - Graph construction API
# ============================================================================

struct Graph(Copyable, Movable):
    """MAX computation graph builder."""
    var graph: PythonObject
    var name: String

    fn __init__(out self, name: String, input_types: List[TensorType], 
                custom_extensions: List[String] = List[String]()) raises:
        """Create a new computation graph.
        
        Args:
            name: Unique name for this graph.
            input_types: List of input tensor type specifications.
            custom_extensions: List of paths to custom op extensions (.mojopkg or .mojo files). Defaults to empty list.
        """
        PythonBridge.setup()
        self.name = name
        
        var max_graph = PythonBridge.get_module("main")
        var builtins = PythonBridge.get_builtins()
        
        # Convert input types to Python list
        var python_input_types = builtins.list()
        for i in range(len(input_types)):
            _ = python_input_types.append(input_types[i].to_python())
        
        # Get the MaxGraph class from max_graph
        var MaxGraphClass = max_graph.__getattr__("MaxGraph")
        
        # Create graph with or without custom extensions
        if len(custom_extensions) > 0:
            var pathlib = Python.import_module("pathlib")
            var python_extensions = builtins.list()
            for i in range(len(custom_extensions)):
                # Convert string path to Python Path object
                var path_obj = pathlib.Path(custom_extensions[i])
                _ = python_extensions.append(path_obj)
            self.graph = MaxGraphClass(name, python_input_types, custom_extensions=python_extensions)
        else:
            self.graph = MaxGraphClass(name, python_input_types)

    fn inputs(self) raises -> List[TensorValue]:
        """Get symbolic input tensors for this graph."""
        var python_inputs = self.graph.inputs
        var input_list = List[TensorValue]()
        
        for inp in python_inputs:
            input_list.append(TensorValue(ArcPointer[Graph](self.copy()), inp))
        
        return input_list^

    fn output(self, outputs: List[TensorValue]) raises:
        """Set the output tensors for this graph.
        
        Args:
            outputs: List of tensor values to be graph outputs.
        """
        var builtins = PythonBridge.get_builtins()
        var python_outputs = builtins.list()
        for i in range(len(outputs)):
            _ = python_outputs.append(outputs[i].to_python())
        
        self.graph.output(python_outputs)

    fn compile(self) raises -> MaxModel:
        """Compile the graph into an executable model."""
        var compiled = self.graph.compile()
        return MaxModel(compiled)
    
    fn __ne__(self, other: Graph) raises -> Bool:
        """Check if two graphs are different."""
        return Bool(self.graph != other.graph)


# ============================================================================
# MODEL EXECUTION - Compiled model wrapper
# ============================================================================

struct MaxModel:
    """Compiled MAX model for inference."""
    var model: PythonObject
    
    fn __init__(out self, model: PythonObject) raises:
        """Wrap a compiled Python model object."""
        self.model = model
        
    fn execute(self, inputs: List[Tensor]) raises -> List[Tensor]:
        """Execute the model with input tensors.
        
        Args:
            inputs: List of Tensor objects.
        
        Returns:
            List of output Tensors.
        """
        var builtins = PythonBridge.get_builtins()
        var python_inputs = builtins.list()
        for i in range(len(inputs)):
            _ = python_inputs.append(inputs[i].to_python())
        
        var results = self.model.execute(python_inputs)
        
        # Convert results to Tensor list
        var output_tensors = List[Tensor]()
        for result in results:
            output_tensors.append(Tensor(result))
        
        return output_tensors^
    
    fn num_inputs(self) raises -> Int:
        """Get the number of expected inputs."""
        return Int(self.model.num_inputs)
    
    fn input_metadata(self) raises -> PythonObject:
        """Get metadata about expected inputs."""
        return self.model.input_metadata




# ============================================================================
# DeviceRef Support Functions
# ============================================================================

struct DeviceRef:
    """Device reference for specifying device types in operations."""
    var device_str: String
    
    fn __init__(out self, device_type: String, id: Int = 0):
        """Create a device reference.
        
        Args:
            device_type: Type of device ("cpu" or "gpu").
            id: Device ID (default: 0).
        """
        self.device_str = device_type + ":" + String(id)
    
    @staticmethod
    fn CPU(id: Int = 0) -> DeviceRef:
        """Create a CPU device reference."""
        return DeviceRef("cpu", id)
    
    @staticmethod
    fn GPU(id: Int = 0) -> DeviceRef:
        """Create a GPU device reference."""
        return DeviceRef("gpu", id)
    
    fn to_string(self) -> String:
        """Convert to string representation."""
        return self.device_str