import nabla as nb

if __name__ == "__main__":

    # Basic tests and benchmarks
    a = nb.arange(shape=(4, 8, 8), dtype=nb.DType.float32)  # .to(Accelerator())
    # print("\na:")
    # print(a)

    b = nb.arange(shape=(3, 4, 8, 8), dtype=nb.DType.float32)  # .to(Accelerator())
    # print("\nb:")
    # print(b)

    for iter in range(1000):
        c = nb.mul(a, b)
        res = nb.arange(
            shape=(2, 3, 4, 8, 8), dtype=nb.DType.float32
        )  # .to(Accelerator())

        for i in range(1000):
            res = nb.sin(nb.cos(nb.mul(res, c)))

        res.realize()  # Trigger realization

        if iter % 100 == 0:
            print(f"Iteration {iter} completed.")
            # print(res)
