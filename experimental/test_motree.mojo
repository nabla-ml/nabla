from nabla.all import (
    MoTree,
    ExecutionContext,
    ndarange,
    randn,
    realize,
    relu,
    full,
    jit,
)


fn foo(args: MoTree) raises -> MoTree:
    x = args.arg.as_tensor()
    w1 = args.w1.as_tensor()
    b1 = args.b1.as_tensor()
    sth = args.sth.as_float32() * 2 + 5
    return relu(x @ w1 + b1 + full(sth, []))


fn main() raises:
    var foo_jit = jit(foo)

    for it in range(10000):
        var args = MoTree()
        args.arg = ndarange([4, 1])
        args.w1 = randn([1, 4])
        args.b1 = randn([4])
        args.sth = Float32(4.0)

        var tensors = args.get_all_tensors()
        _ = tensors.copy()

        var res = foo_jit(args).as_tensor()

        if it % 100 == 0:
            print("\nIteration", it)
            for tensor in tensors:
                print(tensor)

            print("res:")
            print(res)
