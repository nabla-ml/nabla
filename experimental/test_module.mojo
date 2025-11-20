from nabla_new import (
    Tensor,
    randn,
    Callable,
    relu,
    MLP,
    ndarange,
)
from time import perf_counter_ns


fn main() raises:
    var model = MLP([1, 16, 32, 16, 1])

    for it in range(500):
        var input = randn([4, 1])
        var output = model.execute([input])[0]
        if it % 10 == 0:
            print("\nIteration", it)
            print(output)
