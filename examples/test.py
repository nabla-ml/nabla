import nabla as nb


@nb.sjit
def foo(x, i, y):
    """Simple function to test JIT compilation."""
    return x * i * y


a = nb.arange((2, 3))
b = nb.arange((2, 3))

for i in range(5):
    res = foo(a, i, b)
    print(res)
