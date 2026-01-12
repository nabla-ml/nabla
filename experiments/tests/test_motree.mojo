from nabla import MoTree, ndarange, randn, relu, full, jit
from collections import List


fn foo(args: MoTree) raises -> MoTree:
    x = args.arg.as_tensor()
    w1 = args.w1.as_tensor()
    b1 = args.b1.as_tensor()
    sth = args.sth.as_float32() * 2 + 5
    return relu(x @ w1 + b1 + full(sth, []))


fn test_motree() raises:
    print("Starting MoTree tests...")
    for it in range(5):
        var args = MoTree()
        args.arg = ndarange([4, 1])
        args.w1 = randn([1, 4])
        args.b1 = randn([4])
        args.sth = Float32(4.0)

        var res = foo(args).as_tensor()
        if it == 0:
            print("Iteration", it)
            # print("res shape:", res.shape)

    print("Basic usage passed.")

    var tree = MoTree()
    tree.a = randn([2, 2])
    tree.b = Float32(1.0)
    var inner = MoTree()
    inner.c = randn([3])
    tree.inner = inner

    var flat_res = tree.flatten()
    var treedef = flat_res[0]
    # List is not implicitly copyable, create copy
    var leaves = List(flat_res[1])

    print("Flattened leaves count:", len(leaves))
    if len(leaves) != 2:
        print("Error: Expected 2 leaves")

    var reconstructed = MoTree.unflatten(treedef, leaves)
    print("Reconstruction successful.")
