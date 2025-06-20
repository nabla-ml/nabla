import nabla as nb

# Test basic relu
a = nb.array(3.0)
b = nb.relu(a)
print("relu(3.0):", b)


# Test relu VJP
def f_relu(x):
    return nb.relu(x)


_, f_vjp = nb.vjp(f_relu, a)
c = f_vjp(nb.array(1.0))
print("relu VJP at 3.0:", c)

# Test negative value
a_neg = nb.array(-2.0)
b_neg = nb.relu(a_neg)
print("relu(-2.0):", b_neg)

_, f_vjp_neg = nb.vjp(f_relu, a_neg)
c_neg = f_vjp_neg(nb.array(1.0))
print("relu VJP at -2.0:", c_neg)

# Test relu JVP
f_jvp = nb.jvp(f_relu, (a,), (nb.array(1.0),))
print("relu JVP at 3.0:", f_jvp)
