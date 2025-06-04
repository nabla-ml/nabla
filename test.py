import nabla as nb


if __name__ == "__main__":

    def func(x, y):
        return x * y
    
    a = nb.arange((2, 3))
    b = nb.arange((2, 3))

    # Test default behavior (gradient w.r.t. first argument)
    gradient = nb.jacrev(func)(a, b)
    print("Gradient w.r.t. first argument:")
    print(gradient)
    
    # Test gradient w.r.t. both arguments
    gradients = nb.jacrev(func, argnums=(0, 1))(a, b)
    print("\nGradients w.r.t. both arguments:")
    for i, grad in enumerate(gradients):
        print(f"Gradient {i}:")
        print(grad)