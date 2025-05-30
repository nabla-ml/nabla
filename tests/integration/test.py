import nabla

if __name__ == "__main__":
    a = nabla.arange((2, 3, 4))
    print(a)

    b = nabla.incr_batch_dim_ctr(a)
    print(b)

    c = nabla.broadcast_to(b, (1, 3, 4))
    print(c)

    d = nabla.unsqueeze(c, [0])
    print(d)

    e = nabla.squeeze(d, [0])
    print(e)
