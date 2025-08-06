import nabla as nb 

device = nb.cpu() if nb.accelerator_count() == 0 else nb.accelerator()
print(f"Using {device} device")


if __name__ == "__main__":

    # Example usage of the Nabla library
    a = nb.ndarange((2, 3)).to(device)
    b = nb.ndarange((2, 3)).to(device)

    print("Add")
    res = nb.jit(nb.add)(a, b)
    print(res, res.device)

    print("Multiply")
    res = nb.jit(nb.mul)(a, b)
    print(res, res.device)

    print("Subtract")
    res = nb.jit(nb.sub)(a, b)
    print(res, res.device)

    print("Divide")
    res = nb.jit(nb.div)(a, b + 1)
    print(res, res.device)

    print("Power")
    res = nb.jit(nb.pow)(a, b)
    print(res, res.device)

    print("Sin")
    res = nb.jit(nb.sin)(a)
    print(res, res.device)

