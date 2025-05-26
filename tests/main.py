import nabla as nb

if __name__ == "__main__":

    # Basic tests and benchmarks
    a = nb.arange(shape=(4, 8, 8), dtype=nb.DType.float32)  # .to(Accelerator())
    # print("\na:")
    # print(a)

    b = nb.arange(shape=(3, 4, 8, 8), dtype=nb.DType.float32)  # .to(Accelerator())
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

    res = nb.unsqueeze(a, axis=2)
    # print(res)
    print(res)

    res2 = nb.squeeze(res, axis=2)
    # print(res2)
    print(res2)

    res3 = nb.sum(res2, axes=2)
    print(res3)

    res4 = nb.reshape(res3, shape=(2, 2, 4, 2))
    print(res4)

    print(res.shape)
    print(res2.shape)
    print(res3.shape)
    print(res4.shape)
