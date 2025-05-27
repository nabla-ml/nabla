import nabla as nb

if __name__ == "__main__":
    n0 = nb.randn((8, 8))
    n1 = nb.randn((4, 8, 8))
    n = nb.reduce_sum(
        nb.reshape(nb.sin(n0 + n1 * n1 + n0), shape=(2, 2, 8, 8)),
        axes=(0, 1, 2),
        keep_dims=False,
    )
    n.realize()
    print(n)
   