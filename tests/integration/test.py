

import nabla as nb 


def test1():

    def foo(x):
        x = nb.broadcast_to(x, (4, 2, 3))
        return x

    x = nb.arange((2, 3))
    print("x:")
    print(x)

    values, vjp_fn = nb.vjp(foo, x)
    print("values:")
    print(values)

    cotangent = nb.ones_like(values)
    print("cotangent:")
    print(cotangent)

    grad = vjp_fn(cotangent)
    print("grad:")
    print(grad)


def test2():

    def foo(x):
        # x = nb.broadcast_batch_dims(x, (2, 3))
        # x = x * y
        x = nb.broadcast_to(x, (4, 2, 3))
        x = nb.incr_batch_dim_ctr(x)
        x = nb.decr_batch_dim_ctr(x)
        x = nb.unsqueeze(x, [-2])
        x = nb.squeeze(x, [-2])
        x = nb.unsqueeze_batch_dims(x, [-2])
        x = nb.squeeze_batch_dims(x, [-2])
        # x = nb.pad(x, [slice(1, 5)], (6, 2, 3))
        # x = nb.sum(x, [0, 1, 2], keep_dims=False)
        # x = nb.sum_batch_dims(x, [0,])
        return x

    foo = nb.vmap(nb.vmap(foo))

    x = nb.arange((2, 3))
    # y = nb.arange((2, 3))
    print("\nx:")
    print(x)
    # print("\ny:")
    # print(y)

    print("XPR:")
    print(nb.xpr(foo, x))


    # values, vjp_fn = nb.vjp(foo, x, y)
    # print("\nvalues:")
    # print(values)

    # cotangent = nb.ones_like(values)
    # cotangent = nb.broadcast_batch_dims(cotangent, (2, 2, 3))
    # print("cotangent:")
    # print(cotangent)

    # print("XPR:")
    # print(nb.xpr(vjp_fn, cotangent))

    # grad = vjp_fn(cotangent)
    # print("grad:")
    # print(grad)


def test3():

    print("Testing _std_basis...")
    a = nb.arange((3, 2,))
    b = nb.arange((2, 4,))
    c = nb.arange((4, 1))

    sizes, tangents = nb.core.trafos._std_basis([a, b, c])
    print("sizes:", sizes)
    print("tangents:")
    for t in tangents:
        print(t)


if __name__ == "__main__":
    # test1()
    # test2()
    test3()
