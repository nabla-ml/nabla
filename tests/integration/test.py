

import endia as nb 


def test1():

    def foo(x):
        x = nd.broadcast_to(x, (4, 2, 3))
        return x

    x = nd.arange((2, 3))
    print("x:")
    print(x)

    values, vjp_fn = nd.vjp(foo, x)
    print("values:")
    print(values)

    cotangent = nd.ones_like(values)
    print("cotangent:")
    print(cotangent)

    grad = vjp_fn(cotangent)
    print("grad:")
    print(grad)


def test2():

    def foo(x):
        # x = nd.broadcast_batch_dims(x, (2, 3))
        # x = x * y
        x = nd.broadcast_to(x, (4, 2, 3))
        x = nd.incr_batch_dim_ctr(x)
        x = nd.decr_batch_dim_ctr(x)
        x = nd.unsqueeze(x, [-2])
        x = nd.squeeze(x, [-2])
        x = nd.unsqueeze_batch_dims(x, [-2])
        x = nd.squeeze_batch_dims(x, [-2])
        # x = nd.pad(x, [slice(1, 5)], (6, 2, 3))
        # x = nd.sum(x, [0, 1, 2], keep_dims=False)
        # x = nd.sum_batch_dims(x, [0,])
        return x

    foo = nd.vmap(nd.vmap(foo))

    x = nd.arange((2, 3))
    # y = nd.arange((2, 3))
    print("\nx:")
    print(x)
    # print("\ny:")
    # print(y)

    print("XPR:")
    print(nd.xpr(foo, x))


    # values, vjp_fn = nd.vjp(foo, x, y)
    # print("\nvalues:")
    # print(values)

    # cotangent = nd.ones_like(values)
    # cotangent = nd.broadcast_batch_dims(cotangent, (2, 2, 3))
    # print("cotangent:")
    # print(cotangent)

    # print("XPR:")
    # print(nd.xpr(vjp_fn, cotangent))

    # grad = vjp_fn(cotangent)
    # print("grad:")
    # print(grad)




if __name__ == "__main__":
    # test1()
    test2()
