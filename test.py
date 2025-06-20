import nabla as nb

a = nb.array(3)


def f(x):
    return nb.abs(x)


b = f(a)
print(b)

_, f_vjp = nb.vjp(f, a)
c = f_vjp(nb.array(1.0))
print(c)
