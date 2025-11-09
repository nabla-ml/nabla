from nabla import Tensor, randn, Callable, full, relu 
from time import perf_counter_ns


fn main() raises:
    fn mlp(args: List[Tensor]) raises -> List[Tensor]:
        var h1 = relu(args[0] @ args[1] + args[2])
        var h2 = relu(h1 @ args[3] + args[4])
        var h3 = relu(h2 @ args[5] + args[6])
        var h4 = relu(h3 @ args[7] + args[8])
        var h5 = relu(h4 @ args[9] + args[10])
        var h6 = relu(h5 @ args[11] + args[12])
        var h7 = relu(h6 @ args[13] + args[14])
        var h8 = relu(h7 @ args[15] + args[16])
        var output = h8 @ args[17] + args[18]        
        return [output]
    
    var w1 = randn([512, 2048], 0.0, 0.02)
    var b1 = randn([2048], 0.0, 0.01)
    var w2 = randn([2048, 4096], 0.0, 0.02)
    var b2 = randn([4096], 0.0, 0.01)
    var w3 = randn([4096, 4096], 0.0, 0.02)
    var b3 = randn([4096], 0.0, 0.01)
    var w4 = randn([4096, 4096], 0.0, 0.02)
    var b4 = randn([4096], 0.0, 0.01)
    var w5 = randn([4096, 2048], 0.0, 0.02)
    var b5 = randn([2048], 0.0, 0.01)
    var w6 = randn([2048, 1024], 0.0, 0.02)
    var b6 = randn([1024], 0.0, 0.01)
    var w7 = randn([1024, 512], 0.0, 0.02)
    var b7 = randn([512], 0.0, 0.01)
    var w8 = randn([512, 256], 0.0, 0.02)
    var b8 = randn([256], 0.0, 0.01)
    var w_out = randn([256, 10], 0.0, 0.02)
    var b_out = randn([10], 0.0, 0.01)

    var mlp_callable = Callable(mlp, "mlp", True)
    for it in range(20000):
        var t_iter_start = perf_counter_ns()
        
        var input = full(it, [64, 512])
        
        # Pack all arguments: input + all weights and biases
        var all_args = List[Tensor]()
        all_args.append(input)
        all_args.append(w1)
        all_args.append(b1)
        all_args.append(w2)
        all_args.append(b2)
        all_args.append(w3)
        all_args.append(b3)
        all_args.append(w4)
        all_args.append(b4)
        all_args.append(w5)
        all_args.append(b5)
        all_args.append(w6)
        all_args.append(b6)
        all_args.append(w7)
        all_args.append(b7)
        all_args.append(w8)
        all_args.append(b8)
        all_args.append(w_out)
        all_args.append(b_out)
        
        var res = mlp_callable(all_args)
        
        var t_iter_end = perf_counter_ns()
        var iter_time_ms = (t_iter_end - t_iter_start) / 1_000_000
        
        if it % 100 == 0:
            print("Iteration", it, "| Time:", iter_time_ms, "ms")