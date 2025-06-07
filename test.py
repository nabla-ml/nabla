import nabla as nb 


if __name__ == "__main__":
    
    a = nb.arange((2, 3, 4))
    print("\na:")
    print(a)

    a = nb.incr_batch_dim_ctr(a)

    b = nb.sum(a, [0])
    print("\nb:")
    print(b)