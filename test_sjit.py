import nabla as nb

if __name__ == "__main__":

    def foo(x, y):
        const = nb.arange((2, 4))
        return nb.sin(x @ y + const)

    a = nb.arange((3, 5, 1, 4, 2, 3))
    b = nb.arange((1, 5, 3, 4, 3, 4))

    foo_sjitted = nb.sjit(foo, show_graph=True)

    res = foo_sjitted(a, b)
    print(res)

    res = foo_sjitted(a, b)
    print(res)
