from nabla.all import Tensor, randn, Callable, full, relu, err_loc, matmul, MLP, Linear, ndarange
from time import perf_counter_ns
from utils import Variant


fn main() raises:
    var model = MLP([1, 16, 32, 16, 1])

    for it in range(500):
        var input = randn([4, 1])
        var output = model.execute([input])[0]
        if it % 10 == 0:
            print("\nIteration", it)
            print(output)