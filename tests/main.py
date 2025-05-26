import nabla as nb

if __name__ == "__main__":

    # Basic tests and benchmarks
    # a = nb.arange(shape=(4, 8, 8), dtype=nb.DType.float32)  # .to(Accelerator())
    # print("\na:")
    # print(a)

    # b = nb.arange(shape=(3, 4, 8, 8), dtype=nb.DType.float32)  # .to(Accelerator())
    # print("\nb:")
    # print(b)

    # for iter in range(1000):
    #     c = nb.mul(a, b)
    #     res = nb.arange(
    #         shape=(2, 3, 4, 8, 8), dtype=nb.DType.float32
    #     )  # .to(Accelerator())

    #     for i in range(10):
    #         res = nb.add_one_custom(nb.sin(nb.cos(nb.mul(res, c))))

    #     res.realize()  # Trigger realization

    #     if iter % 100 == 0:
    #         print(f"Iteration {iter} completed.")
    #         print(res)

    n0 = nb.randn((8, 8))
    n1 = nb.randn((4, 8, 8))
    n = nb.sin(nb.negate(nb.add(n0, n1)))
    # print(n)

    # res = nb.unsqueeze(n, axis=2)
    # # print(res)
    # print(res)
    # print(res.shape)

    # res2 = nb.squeeze(res, axis=2)
    # # print(res2)
    # print(res2)
    # print(res2.shape)

    res3 = nb.sum(n, axes=2, keep_dims=False)
    print(res3)
    print(res3.shape)

    res4 = nb.transpose(nb.reshape(res3, shape=(2, 2, 8)))
    print(res4)
    print(res4.shape)

    # print(res.shape)
    # print(res2.shape)
    # print(res3.shape)
    # print(res4.shape)
