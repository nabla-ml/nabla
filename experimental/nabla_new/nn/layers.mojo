from nabla_new.core.tensor import Tensor
from nabla_new.core.execution import ExecutionContext
from nabla_new.ops.creation import randn
from nabla_new.ops.math import matmul, add
from nabla_new.ops.activations import relu
from .module import Module


struct Linear(Module):
    var weight: Tensor
    var bias: Tensor
    var _ctx: ExecutionContext

    fn __init__(out self, in_dim: Int, out_dim: Int) raises:
        self.weight = randn([in_dim, out_dim])
        self.bias = randn([out_dim])
        self._ctx = ExecutionContext()

    fn __call__(self, args: List[Tensor]) raises -> List[Tensor]:
        # args[0] @ self.weight + self.bias
        return [relu(add(matmul(args[0], self.weight), self.bias))]

    fn params(self) raises -> List[Tensor]:
        return [self.weight, self.bias]

    fn ctx(self) raises -> ExecutionContext:
        return self._ctx


struct MLP(Module):
    var layers: List[Linear]
    var _ctx: ExecutionContext

    fn __init__(out self, layer_dims: List[Int]) raises:
        self.layers = []
        for i in range(1, len(layer_dims)):
            self.layers.append(Linear(layer_dims[i - 1], layer_dims[i]))
        self._ctx = ExecutionContext()

    fn __call__(self, args: List[Tensor]) raises -> List[Tensor]:
        var hidden = args.copy()
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden^

    fn params(self) raises -> List[Tensor]:
        var layer_params = List[Tensor]()
        for layer in self.layers:
            layer_params.extend(layer.params())
        return layer_params^

    fn ctx(self) raises -> ExecutionContext:
        return self._ctx
