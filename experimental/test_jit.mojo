from nabla.all import Tensor, randn, jit, full, relu, err_loc, matmul, MoTree
from time import perf_counter_ns


fn mlp(args: MoTree) raises -> MoTree:
    var h1 = relu(args[0].as_tensor() @ args[1].as_tensor() + args[2].as_tensor())
    var h2 = relu(h1 @ args[3].as_tensor() + args[4].as_tensor())
    var h3 = relu(h2 @ args[5].as_tensor() + args[6].as_tensor())
    var h4 = relu(h3 @ args[7].as_tensor() + args[8].as_tensor())
    var h5 = relu(h4 @ args[9].as_tensor() + args[10].as_tensor())
    var h6 = relu(matmul(h5, args[11].as_tensor()) + args[12].as_tensor())
    var h7 = relu(h6 @ args[13].as_tensor() + args[14].as_tensor())
    var h8 = relu(h7 @ args[15].as_tensor() + args[16].as_tensor())
    var output = h8 @ args[17].as_tensor() + args[18].as_tensor()
    return output


fn main() raises:
    var params = [
        randn([512, 2048], 0.0, 0.02),
        randn([2048], 0.0, 0.01),
        randn([2048, 4096], 0.0, 0.02),
        randn([4096], 0.0, 0.01),
        randn([4096, 4096], 0.0, 0.02),
        randn([4096], 0.0, 0.01),
        randn([4096, 4096], 0.0, 0.02),
        randn([4096], 0.0, 0.01),
        randn([4096, 2048], 0.0, 0.02),
        randn([2048], 0.0, 0.01),
        randn([2048, 1024], 0.0, 0.02),
        randn([1024], 0.0, 0.01),
        randn([1024, 512], 0.0, 0.02),
        randn([512], 0.0, 0.01),
        randn([512, 256], 0.0, 0.02),
        randn([256], 0.0, 0.01),
        randn([256, 10], 0.0, 0.02),
        randn([10], 0.0, 0.01)
    ]

    var mlp_jit = jit(mlp)

    for it in range(20000):
        var t_iter_start = perf_counter_ns()
        
        var input = randn([4, 512])

        var output = mlp_jit([input] + params.copy())
        var t_iter_end = perf_counter_ns()
        var iter_time_ms = (t_iter_end - t_iter_start) / 1_000_000
        
        if it % 100 == 0:
            print("Iteration", it, "| Time:", iter_time_ms, "ms")
            print(output.as_tensor())
